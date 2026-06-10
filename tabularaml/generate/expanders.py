"""Deterministic base-table expansion applied before the genetic search.

Datetime decomposition and row-wise statistics are cheap, always worth
offering to the search as parent features, and must be reproducible at
transform time. A fitted expander (column lists are the only state; every
emitted value is a stateless function of the row) is the single integration
point used by search, replay, fit and transform.
"""
import warnings
import numpy as np
import pandas as pd

_DT_EPOCH = pd.Timestamp("1970-01-01")


class BaselineFeatureExpander:
    """Expand the base table with datetime decompositions and row statistics.

    - Datetime: detects datetime64 columns and parseable object/string columns
      (sampled parse rate >= min_parse_frac; numeric dtypes are never parsed),
      emits year/month/day/dayofweek/hour/is_weekend/cyclical encodings and an
      epoch-based day count, then drops the raw column (models reject
      datetime64). Constant outputs on the fit data are pruned.
    - Row stats: mean/std/max/min over the original numeric block (skipped
      when fewer than 3 numeric columns) and a NaN count over all original
      columns.
    """

    def __init__(self, datetime_features=True, row_stats=True, exclude_cols=(),
                 min_parse_frac=0.95):
        self.datetime_features = datetime_features
        self.row_stats = row_stats
        self.exclude_cols = tuple(exclude_cols)
        self.min_parse_frac = min_parse_frac
        self.datetime_cols_ = []
        self.dt_outputs_ = {}
        self.row_stat_cols_ = []
        self.nan_count_cols_ = []
        self.row_stat_outputs_ = []
        self.added_cols_ = []

    # --- datetime helpers ---

    def _parse_datetime(self, s: pd.Series):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.to_datetime(s, errors="coerce", format="mixed")

    def _detect_datetime(self, s: pd.Series):
        """Return the parsed series if s is/encodes datetimes, else None."""
        if pd.api.types.is_datetime64_any_dtype(s):
            return s
        is_str_like = (s.dtype == object or pd.api.types.is_string_dtype(s)
                       or isinstance(s.dtype, pd.CategoricalDtype))
        if not is_str_like:
            return None  # numeric dtypes are never parsed (int YYYYMMDD is a documented miss)
        sample = s.dropna()
        if len(sample) == 0:
            return None
        sample = sample.sample(min(200, len(sample)), random_state=0).astype(str)
        parsed_sample = self._parse_datetime(sample)
        if parsed_sample.notna().mean() < self.min_parse_frac:
            return None
        parsed = self._parse_datetime(s.astype(str))
        if parsed.notna().mean() < 0.8:
            return None
        return parsed

    @staticmethod
    def _dt_parts(col: str, parsed: pd.Series) -> dict:
        dt = parsed.dt
        month = dt.month.astype(float)
        dow = dt.dayofweek.astype(float)
        return {
            f"{col}_year": dt.year.astype(float),
            f"{col}_month": month,
            f"{col}_day": dt.day.astype(float),
            f"{col}_dayofweek": dow,
            f"{col}_hour": dt.hour.astype(float),
            f"{col}_is_weekend": (dow >= 5).astype(float),
            f"{col}_month_sin": np.sin(2 * np.pi * (month - 1) / 12.0),
            f"{col}_month_cos": np.cos(2 * np.pi * (month - 1) / 12.0),
            f"{col}_dow_sin": np.sin(2 * np.pi * dow / 7.0),
            f"{col}_dow_cos": np.cos(2 * np.pi * dow / 7.0),
            f"{col}_days_since_epoch": (parsed - _DT_EPOCH).dt.days.astype(float),
        }

    # --- API ---

    def fit(self, X: pd.DataFrame) -> "BaselineFeatureExpander":
        self.datetime_cols_, self.dt_outputs_ = [], {}
        existing = set(X.columns)

        if self.datetime_features:
            for col in X.columns:
                if col in self.exclude_cols:
                    continue
                parsed = self._detect_datetime(X[col])
                if parsed is None:
                    continue
                parts = self._dt_parts(col, parsed)
                keep = [name for name, vals in parts.items()
                        if name not in existing and pd.Series(vals).nunique(dropna=True) > 1]
                if keep:
                    self.datetime_cols_.append(col)
                    self.dt_outputs_[col] = keep

        self.row_stat_cols_, self.nan_count_cols_, self.row_stat_outputs_ = [], [], []
        if self.row_stats:
            num_cols = [c for c in X.select_dtypes("number").columns
                        if c not in self.exclude_cols and c not in self.datetime_cols_]
            outputs = []
            if len(num_cols) >= 3:
                self.row_stat_cols_ = num_cols
                outputs += ["row_mean", "row_std", "row_max", "row_min"]
            self.nan_count_cols_ = [c for c in X.columns if c not in self.datetime_cols_]
            outputs.append("row_nan_count")
            self.row_stat_outputs_ = [o for o in outputs if o not in existing]

        self.added_cols_ = [name for outs in self.dt_outputs_.values() for name in outs]
        self.added_cols_ += self.row_stat_outputs_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.added_cols_ and not self.datetime_cols_:
            return X
        X = X.copy()
        new_cols = {}

        for col in self.datetime_cols_:
            if col not in X.columns:
                continue
            s = X[col]
            parsed = s if pd.api.types.is_datetime64_any_dtype(s) else self._parse_datetime(s.astype(str))
            parts = self._dt_parts(col, parsed)
            for name in self.dt_outputs_.get(col, []):
                new_cols[name] = np.asarray(parts[name], dtype=float)

        if self.row_stat_outputs_:
            if self.row_stat_cols_:
                block = X.reindex(columns=self.row_stat_cols_).apply(pd.to_numeric, errors="coerce")
                if "row_mean" in self.row_stat_outputs_:
                    new_cols["row_mean"] = block.mean(axis=1).values
                if "row_std" in self.row_stat_outputs_:
                    new_cols["row_std"] = block.std(axis=1).values
                if "row_max" in self.row_stat_outputs_:
                    new_cols["row_max"] = block.max(axis=1).values
                if "row_min" in self.row_stat_outputs_:
                    new_cols["row_min"] = block.min(axis=1).values
            if "row_nan_count" in self.row_stat_outputs_:
                present = [c for c in self.nan_count_cols_ if c in X.columns]
                new_cols["row_nan_count"] = X[present].isna().sum(axis=1).astype(float).values

        for name, vals in new_cols.items():
            X[name] = vals
        X = X.drop(columns=[c for c in self.datetime_cols_ if c in X.columns])
        return X

    def summary(self) -> str:
        n_dt = sum(len(v) for v in self.dt_outputs_.values())
        return (f"{n_dt} datetime parts from {len(self.datetime_cols_)} col(s), "
                f"{len(self.row_stat_outputs_)} row stats")
