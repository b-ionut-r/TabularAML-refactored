"""FeatureForge: fast, wide, residual-guided feature search with an honest gate.

Search loop (per round):

1. **Base model.** K-fold LightGBM on the current feature set gives out-of-fold
   raw margins and split-gain importances.
2. **Candidates.** Thousands of features from the most important columns:
   pairwise arithmetic (add/sub/mul/div), frequency encodings, out-of-fold
   target encodings of single keys and key pairs, and group statistics
   (mean/std/min/max/deviation/z-score of a numeric within a categorical or
   low-cardinality key). Round 2+ also combine the features selected so far
   with raw columns, so useful second-order features (e.g. ``a*b - c``) can
   emerge.
3. **Residual screening.** Each candidate alone trains a tiny LightGBM whose
   ``init_score`` is the base margin, so its validation gain measures only
   the information the current model lacks. Successive halving over row
   subsets keeps this to milliseconds per candidate; candidates run in
   parallel threads.
4. **Selection.** Survivors are ranked by split gain in a joint model, then the
   best prefix (3, 6, 12, ... features) is chosen by K-fold CV loss.

Finally a **gate**: rows held out before any search compare raw vs. raw+new
features with the same model. If the new features do not lower held-out
loss, nothing is added. This keeps the search honest: a CV gain that only
exists on the folds it was selected on is rejected.

Typical use::

    forge = FeatureForge(time_budget=300).fit(X_train, y_train)
    X_train_fe = forge.transform_train(X_train)   # OOF target encodings
    X_test_fe = forge.transform(X_test)
"""
from __future__ import annotations

import time
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

TE_SMOOTHING = 20.0
GROUP_STATS = ("mean", "std", "min", "max", "dev", "z")
ARITH_OPS = ("add", "sub", "mul", "div", "rdiv")


# ============================================================================
# Column helpers
# ============================================================================

def _is_cat(s: pd.Series) -> bool:
    return (s.dtype == object or s.dtype == bool or isinstance(s.dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(s))


def _as_str(s: pd.Series) -> pd.Series:
    """String view of a column with missing values mapped to an explicit level."""
    return s.astype(object).where(s.notna(), "__NA__").astype(str)


def _key_values(s: pd.Series) -> np.ndarray:
    """Hashable, NaN-safe values used to index key categories."""
    if _is_cat(s):
        return _as_str(s).to_numpy()
    return s.astype(float).fillna(-1.234567e300).to_numpy()


class Context:
    """Shared task info passed to specs."""

    def __init__(self, task: str, n_classes: int, seed: int):
        self.task = task
        self.n_classes = n_classes
        self.seed = seed


def _fit_vocab(df: pd.DataFrame, cols: Sequence[str]) -> List[pd.Index]:
    return [pd.Index(pd.unique(_key_values(df[c]))) for c in cols]


def _codes(df: pd.DataFrame, cols: Sequence[str], vocabs: List[pd.Index]) -> np.ndarray:
    """Combined int64 code of key columns; values unseen at fit time get their own code."""
    out = np.zeros(len(df), dtype=np.int64)
    for c, vocab in zip(cols, vocabs):
        idx = vocab.get_indexer(_key_values(df[c]))
        idx[idx < 0] = len(vocab)
        out = out * (len(vocab) + 1) + idx
    return out


def X_nunique(df: pd.DataFrame, c: str) -> int:
    return int(df[c].nunique(dropna=False))


# ============================================================================
# Feature specs
# ============================================================================

class Spec:
    """A generated feature. ``transform`` returns an (n,) or (n, k) float array."""
    target_dep = False
    n_out = 1

    def __init__(self, parents: Sequence[str]):
        self.parents = list(parents)

    def fit(self, df, y, ctx):
        return self

    def transform(self, df, ctx) -> np.ndarray:
        raise NotImplementedError

    def _keys(self):
        return self.parents

    def _fit_codes(self, df) -> np.ndarray:
        self.vocabs_ = _fit_vocab(df, self._keys())
        return _codes(df, self._keys(), self.vocabs_)

    def _codes(self, df) -> np.ndarray:
        return _codes(df, self._keys(), self.vocabs_)

    def fit_transform_oof(self, df, y, ctx, folds) -> np.ndarray:
        self.fit(df, y, ctx)
        return self.transform(df, ctx)

    def out_names(self) -> List[str]:
        return [self.name] if self.n_out == 1 else [f"{self.name}_{k}" for k in range(self.n_out)]

    def __repr__(self):
        return self.name


class Arith(Spec):
    def __init__(self, op: str, a: str, b: str):
        super().__init__([a, b])
        self.op = op
        self.name = f"{a}__{op}__{b}"

    def transform(self, df, ctx):
        a = df[self.parents[0]].to_numpy(dtype=float)
        b = df[self.parents[1]].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            if self.op == "add":
                r = a + b
            elif self.op == "sub":
                r = a - b
            elif self.op == "mul":
                r = a * b
            elif self.op == "div":
                r = a / np.where(b == 0, np.nan, b)
            else:
                r = b / np.where(a == 0, np.nan, a)
        r[~np.isfinite(r)] = np.nan
        return r


class Count(Spec):
    def __init__(self, keys: Sequence[str]):
        super().__init__(keys)
        self.name = "count__" + "__".join(keys)

    def fit(self, df, y, ctx):
        k = self._fit_codes(df)
        self.table_ = pd.Series(k).value_counts()
        return self

    def transform(self, df, ctx):
        k = self._codes(df)
        return self.table_.reindex(k).fillna(0).to_numpy(dtype=float)


class TargetEnc(Spec):
    """Smoothed target mean of a key (or key pair); out-of-fold on training rows."""
    target_dep = True

    def __init__(self, keys: Sequence[str], n_classes: int = 0):
        super().__init__(keys)
        self.name = "te__" + "__".join(keys)
        self.n_classes = n_classes
        self.n_out = n_classes if n_classes > 2 else 1

    def _targets(self, y):
        y = np.asarray(y, dtype=float)
        if self.n_out > 1:
            return np.eye(self.n_classes)[y.astype(int)]
        return y[:, None]

    def _stats(self, k, Y):
        frame = pd.DataFrame(Y)
        frame["_k"] = k
        g = frame.groupby("_k")
        return g.sum(), g.size(), Y.mean(axis=0)

    def _apply(self, k, sums, counts, prior):
        s = sums.reindex(k).to_numpy()
        c = counts.reindex(k).to_numpy(dtype=float)[:, None]
        s = np.where(np.isnan(s), 0.0, s)
        c = np.where(np.isnan(c), 0.0, c)
        out = (s + TE_SMOOTHING * prior) / (c + TE_SMOOTHING)
        return out[:, 0] if self.n_out == 1 else out

    def fit(self, df, y, ctx):
        k = self._fit_codes(df)
        self.sums_, self.counts_, self.prior_ = self._stats(k, self._targets(y))
        return self

    def transform(self, df, ctx):
        return self._apply(self._codes(df), self.sums_, self.counts_, self.prior_)

    def fit_transform_oof(self, df, y, ctx, folds):
        k = self._fit_codes(df)
        Y = self._targets(y)
        out = np.zeros((len(df), self.n_out))
        for tr, va in folds:
            sums, counts, prior = self._stats(k[tr], Y[tr])
            r = self._apply(k[va], sums, counts, prior)
            out[va] = r if r.ndim == 2 else r[:, None]
        self.fit(df, y, ctx)
        return out[:, 0] if self.n_out == 1 else out


class KNNTarget(Spec):
    """Mean target of the k nearest training rows in a standardised numeric subspace.

    A classic contest meta-feature: it injects local, smooth structure that
    axis-aligned trees approximate poorly. Training rows are encoded out of
    fold (a row is never its own neighbour); new rows query all training rows.
    Multiclass targets give per-class neighbour frequencies.
    """
    target_dep = True

    def __init__(self, cols: Sequence[str], ks: Sequence[int], n_classes: int, label: str):
        super().__init__(cols)
        self.ks = tuple(ks)
        self.n_classes = n_classes
        self.per = n_classes if n_classes > 2 else 1
        self.n_out = len(self.ks) * self.per
        self.name = f"knn__{label}"

    def _matrix(self, df, fit=False):
        M = df[self.parents].to_numpy(dtype=float)
        if fit:
            lo, hi = np.nanpercentile(M, 1, axis=0), np.nanpercentile(M, 99, axis=0)
            self.lo_, self.hi_ = lo, hi
            Mc = np.clip(M, lo, hi)
            self.med_ = np.nanmedian(Mc, axis=0)
            Mc = np.where(np.isnan(Mc), self.med_, Mc)
            self.mu_, self.sd_ = Mc.mean(0), Mc.std(0) + 1e-12
        Mc = np.clip(M, self.lo_, self.hi_)
        Mc = np.where(np.isnan(Mc), self.med_, Mc)
        return ((Mc - self.mu_) / self.sd_).astype(np.float32)

    def _targets(self, y):
        y = np.asarray(y, dtype=float)
        return np.eye(self.n_classes)[y.astype(int)] if self.per > 1 else y[:, None]

    def _query(self, M_ref, Y_ref, M_q):
        from sklearn.neighbors import NearestNeighbors
        kmax = min(max(self.ks), len(M_ref))
        nn = NearestNeighbors(n_neighbors=kmax).fit(M_ref)
        _, idx = nn.kneighbors(M_q)
        out = np.empty((len(M_q), self.n_out))
        csum = np.cumsum(Y_ref[idx], axis=1)  # (n, kmax, per)
        for i, k in enumerate(self.ks):
            k = min(k, kmax)
            out[:, i * self.per:(i + 1) * self.per] = csum[:, k - 1, :] / k
        return out

    def fit(self, df, y, ctx):
        self.M_ = self._matrix(df, fit=True)
        self.Y_ = self._targets(y)
        return self

    def transform(self, df, ctx):
        return self._query(self.M_, self.Y_, self._matrix(df))

    def fit_transform_oof(self, df, y, ctx, folds):
        self.fit(df, y, ctx)
        out = np.empty((len(df), self.n_out))
        for tr, va in folds:
            out[va] = self._query(self.M_[tr], self.Y_[tr], self.M_[va])
        return out


class RowStat(Spec):
    """Row-wise statistic over a family of columns (e.g. one-hot blocks, repeated measurements)."""

    def __init__(self, cols: Sequence[str], stat: str, label: str):
        super().__init__(cols)
        self.stat = stat
        self.name = f"row_{stat}__{label}"

    def transform(self, df, ctx):
        M = df[self.parents].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            if self.stat == "sum":
                return np.nansum(M, axis=1)
            if self.stat == "mean":
                return np.nanmean(M, axis=1)
            if self.stat == "std":
                return np.nanstd(M, axis=1)
            if self.stat == "max":
                return np.nanmax(M, axis=1)
            if self.stat == "min":
                return np.nanmin(M, axis=1)
            if self.stat == "argmax":
                r = np.argmax(np.where(np.isnan(M), -np.inf, M), axis=1).astype(float)
                r[np.all(np.isnan(M), axis=1)] = np.nan
                return r
            if self.stat == "nonzero":
                return (np.nan_to_num(M) != 0).sum(axis=1).astype(float)
            if self.stat == "nan":
                return np.isnan(M).sum(axis=1).astype(float)
        raise ValueError(self.stat)


def column_families(cols: Sequence[str], min_size: int = 3) -> Dict[str, List[str]]:
    """Group columns sharing a name stem that ends in a running index (``Soil_Type_12``, ``px3``)."""
    import re
    fam: Dict[str, List[str]] = {}
    for c in cols:
        m = re.match(r"^(.*?)[_.\-]?(\d+)$", str(c))
        if m and m.group(1):
            fam.setdefault(m.group(1), []).append(c)
    return {k: v for k, v in fam.items() if len(v) >= min_size}


class GroupStat(Spec):
    """Statistic of numeric ``num`` within groups of ``key``."""

    def __init__(self, key: str, num: str, stat: str):
        super().__init__([key, num])
        self.stat = stat
        self.name = f"grp_{stat}__{num}__by__{key}"

    def _keys(self):
        return self.parents[:1]

    def fit(self, df, y, ctx):
        k = self._fit_codes(df)
        v = pd.Series(df[self.parents[1]].to_numpy(dtype=float))
        g = v.groupby(k)
        if self.stat in ("mean", "dev"):
            self.a_ = g.mean()
        elif self.stat == "std":
            self.a_ = g.std()
        elif self.stat == "min":
            self.a_ = g.min()
        elif self.stat == "max":
            self.a_ = g.max()
        elif self.stat == "z":
            self.a_ = g.mean()
            self.b_ = g.std()
        return self

    def transform(self, df, ctx):
        k = self._codes(df)
        a = self.a_.reindex(k).to_numpy(dtype=float)
        if self.stat == "dev":
            return df[self.parents[1]].to_numpy(dtype=float) - a
        if self.stat == "z":
            b = self.b_.reindex(k).to_numpy(dtype=float)
            with np.errstate(all="ignore"):
                r = (df[self.parents[1]].to_numpy(dtype=float) - a) / np.where(b > 0, b, np.nan)
            return r
        return a


# ============================================================================
# LightGBM helpers
# ============================================================================

def _lgb_objective(task: str, n_classes: int) -> dict:
    if task == "regression":
        return {"objective": "regression", "metric": "l2"}
    if task == "binary":
        return {"objective": "binary", "metric": "binary_logloss"}
    return {"objective": "multiclass", "metric": "multi_logloss", "num_class": n_classes}


def _loss(task, y, margin) -> float:
    """Loss of raw margins in the LightGBM objective's own units."""
    y = np.asarray(y)
    if task == "regression":
        return float(np.mean((y - margin) ** 2))
    if task == "binary":
        p = 1 / (1 + np.exp(-margin))
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    m = margin - margin.max(axis=1, keepdims=True)
    logp = m - np.log(np.exp(m).sum(axis=1, keepdims=True))
    return float(-np.mean(logp[np.arange(len(y)), y.astype(int)]))


# ============================================================================
# FeatureForge
# ============================================================================

class FeatureForge:
    """Residual-guided automated feature engineering.

    Parameters
    ----------
    task : "regression" | "binary" | "multiclass" | None
    log_target : bool
        Search on ``log1p(y)`` (use for RMSLE-scored regression).
    time_budget : float
        Soft wall-clock budget in seconds for the whole search.
    n_rounds : int
        Search rounds; round r>1 composes previously selected features.
    max_new_features : int
        Cap on features added in total.
    top_numeric, top_keys : int
        How many of the most important numeric / key columns seed candidates.
    gate_frac : float
        Fraction of rows held out (before search) to confirm the final gain.
        0 disables the gate.
    """

    def __init__(self, task: Optional[str] = None, log_target: bool = False,
                 time_budget: float = 300, n_rounds: int = 3, max_new_features: int = 60,
                 top_numeric: int = 24, top_keys: int = 10, max_key_cardinality: int = 200,
                 arith_ops: Sequence[str] = ARITH_OPS, group_stats: Sequence[str] = GROUP_STATS,
                 gate_frac: float = 0.2, min_rel_gain: float = 0.001, cv: int = 3,
                 random_state: int = 0, n_jobs: int = -1, verbose: bool = True):
        self.task = task
        self.log_target = log_target
        self.time_budget = time_budget
        self.n_rounds = n_rounds
        self.max_new_features = max_new_features
        self.top_numeric = top_numeric
        self.top_keys = top_keys
        self.max_key_cardinality = max_key_cardinality
        self.arith_ops = tuple(arith_ops)
        self.group_stats = tuple(group_stats)
        self.gate_frac = gate_frac
        self.min_rel_gain = min_rel_gain
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    # ------------------------------------------------------------- utilities
    def _log(self, msg):
        if self.verbose:
            print(f"[FeatureForge {time.time() - self._t0:6.1f}s] {msg}", flush=True)

    def _time_left(self) -> float:
        return self.time_budget - (time.time() - self._t0)

    def _threads(self) -> int:
        import os
        return os.cpu_count() if self.n_jobs in (-1, None) else max(1, self.n_jobs)

    def _folds(self, n_rows, y, k, seed):
        if self.task_ == "regression":
            return list(KFold(k, shuffle=True, random_state=seed).split(np.zeros(n_rows)))
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(n_rows), y))

    def _model_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.cat_cols_:
            if c in X.columns:
                X[c] = pd.Categorical(_as_str(X[c]), categories=self.cat_levels_[c])
        return X

    def _lgb_params(self, lr=0.1, threads=None, **kw):
        p = dict(_lgb_objective(self.task_, self.n_classes_), learning_rate=lr, num_leaves=31,
                 min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                 lambda_l2=1.0, verbosity=-1, seed=self.random_state,
                 num_threads=threads or self._threads(), max_cat_to_onehot=8)
        p.update(kw)
        return p

    def _fit_eval(self, Xtr, ytr, Xva, yva, lr=0.1, rounds=2000, es=50):
        import lightgbm as lgb
        dtr = lgb.Dataset(Xtr, ytr, free_raw_data=False)
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        b = lgb.train(self._lgb_params(lr), dtr, rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(es, verbose=False)])
        return b

    def _cv(self, X: pd.DataFrame, y, repeats, lr=0.1):
        """Repeated K-fold: OOF raw margins (averaged over repeats), mean loss, summed gain importance."""
        n = len(X)
        oof_mean = np.zeros((n, self.n_classes_)) if self.task_ == "multiclass" else np.zeros(n)
        imp = pd.Series(0.0, index=X.columns)
        losses = []
        for folds in repeats:
            oof = np.zeros_like(oof_mean)
            for tr, va in folds:
                b = self._fit_eval(X.iloc[tr], y[tr], X.iloc[va], y[va], lr=lr)
                oof[va] = b.predict(X.iloc[va], num_iteration=b.best_iteration, raw_score=True)
                imp += pd.Series(b.feature_importance("gain"), index=X.columns)
            losses.append(_loss(self.task_, y, oof))
            oof_mean += oof / len(repeats)
        return oof_mean, float(np.mean(losses)), imp

    # ------------------------------------------------------- candidate pool
    def _generate(self, W: pd.DataFrame, imp: pd.Series, selected: List[Spec], round_idx: int) -> List[Spec]:
        num_cols = [c for c in W.columns if c not in self.cat_cols_]
        num_rank = [c for c in imp.sort_values(ascending=False).index if c in num_cols]
        keys_rank = [c for c in imp.sort_values(ascending=False).index if c in self.key_cols_]
        top_num = num_rank[:self.top_numeric]
        top_keys = keys_rank[:self.top_keys]
        existing = set(W.columns)
        cands: List[Spec] = []

        def add(spec):
            if spec.name not in existing:
                existing.add(spec.name)
                cands.append(spec)

        sel_names = [s.name for s in selected if s.n_out == 1 and s.name in num_cols]
        if round_idx == 0:
            raw_num = [c for c in self.raw_cols_ if c not in self.cat_cols_]
            for stem, cols in column_families(raw_num).items():
                binary = all(set(pd.unique(W[c].dropna())) <= {0, 1} for c in cols)
                stats = ("argmax", "sum") if binary else ("sum", "mean", "std", "max", "min", "argmax")
                for st in stats:
                    add(RowStat(cols, st, stem))
            if len(raw_num) >= 3:
                add(RowStat(raw_num, "nonzero", "all"))
                if W[raw_num].isna().any().any():
                    add(RowStat(raw_num, "nan", "all"))
            dense_num = [c for c in top_num if W[c].nunique() > 10]
            if len(W) <= 300_000:
                for d in (4, 12, 32):
                    if len(dense_num) >= max(2, d // 2):
                        ks = (5, 20, 100) if self.n_classes_ <= 2 else (10, 50)
                        add(KNNTarget(dense_num[:d], ks, self.n_classes_, f"top{d}"))
            for a, b in combinations(top_num, 2):
                for op in self.arith_ops:
                    add(Arith(op, a, b))
            for k in top_keys:
                add(Count([k]))
                if X_nunique(W, k) > 2:
                    add(TargetEnc([k], self.n_classes_))
            for k1, k2 in combinations(top_keys, 2):
                add(Count([k1, k2]))
                add(TargetEnc([k1, k2], self.n_classes_))
            for c in top_num:
                if c not in self.key_cols_:
                    add(Count([c]))
            for k in top_keys:
                for c in top_num:
                    if c == k:
                        continue
                    for st in self.group_stats:
                        add(GroupStat(k, c, st))
        else:
            # Compose selected features with the strongest raw columns.
            partners = top_num[: max(8, self.top_numeric // 2)]
            for s in sel_names:
                for c in partners:
                    if c == s:
                        continue
                    for op in self.arith_ops:
                        add(Arith(op, s, c))
                for k in top_keys[:6]:
                    for st in ("mean", "dev", "z"):
                        add(GroupStat(k, s, st))
            for k1, k2, k3 in list(combinations(top_keys[:6], 3)):
                add(Count([k1, k2, k3]))
                add(TargetEnc([k1, k2, k3], self.n_classes_))
        return cands

    def _materialize(self, specs: List[Spec], W: pd.DataFrame, y, folds) -> Dict[str, np.ndarray]:
        out = {}
        for s in specs:
            try:
                v = s.fit_transform_oof(W, y, self.ctx_, folds) if s.target_dep else s.fit(W, y, self.ctx_).transform(W, self.ctx_)
            except Exception:
                continue
            v = np.asarray(v, dtype=np.float32)
            v2 = v if v.ndim == 2 else v[:, None]
            col = v2[:, 0]
            finite = np.isfinite(col)
            if finite.mean() < 0.05 or np.nanstd(np.where(finite, col, np.nan)) == 0:
                continue
            out[s.name] = v
        return out

    # ------------------------------------------------------------ screening
    def _grad_hess(self, y, margin):
        if self.task_ == "regression":
            return margin - y, np.ones_like(margin)
        if self.task_ == "binary":
            p = 1 / (1 + np.exp(-margin))
            return p - y, np.maximum(p * (1 - p), 1e-6)
        m = margin - margin.max(axis=1, keepdims=True)
        P = np.exp(m)
        P /= P.sum(axis=1, keepdims=True)
        Y = np.eye(self.n_classes_)[y.astype(int)]
        return P - Y, np.maximum(P * (1 - P), 1e-6)

    def _residual_gain(self, x, g, h, lam, n_bins) -> float:
        """Cross-fitted loss reduction from a histogram Newton correction on one feature.

        The candidate is cut into quantile bins (missing values get their own
        bin). For each fold, every bin receives the regularised Newton step of
        the current model's gradients computed on the *other* folds, i.e. a
        one-feature, ``n_bins``-leaf tree fitted to the residuals; the step is
        applied to the held-out fold and the loss is measured on all rows.
        """
        finite = np.isfinite(x)
        if finite.sum() < 20:
            return -np.inf
        edges = np.unique(np.nanquantile(x[finite], np.linspace(0, 1, n_bins + 1)[1:-1]))
        nb = len(edges) + 2
        bins = np.where(finite, np.searchsorted(edges, np.where(finite, x, 0.0), side="right"), nb - 1)
        K = self._n_probe_folds
        cell = self._probe_fold * nb + bins
        if g.ndim == 1:
            G = np.bincount(cell, g, minlength=K * nb).reshape(K, nb)
            H = np.bincount(cell, h, minlength=K * nb).reshape(K, nb)
            step = -(G.sum(0) - G) / (H.sum(0) - H + lam)
            corr = step[self._probe_fold, bins]
        else:
            corr = np.empty_like(g)
            for k in range(g.shape[1]):
                G = np.bincount(cell, g[:, k], minlength=K * nb).reshape(K, nb)
                H = np.bincount(cell, h[:, k], minlength=K * nb).reshape(K, nb)
                step = -(G.sum(0) - G) / (H.sum(0) - H + lam)
                corr[:, k] = step[self._probe_fold, bins]
        return self._probe_base_loss - _loss(self.task_, self._probe_y, self._probe_margin + corr)

    def _screen(self, values: Dict[str, np.ndarray], specs: Dict[str, Spec], W: pd.DataFrame,
                y, margin, idx_a, idx_b, keep: int) -> List[str]:
        """Rank candidates by *novel* residual gain.

        A candidate's raw residual gain is compared with the gain of its own
        parent columns under the same probe: an early-stopped model leaves some
        residual signal in existing columns, and any re-expression of such a
        column (``a + const``-like transforms, group statistics of it, ...) would
        otherwise look useful. Only gain beyond the best parent counts.
        """
        n = len(y)
        self._n_probe_folds = 5
        self._probe_fold = np.random.default_rng(self.random_state).permutation(n) % self._n_probe_folds
        self._probe_y, self._probe_margin = y, margin
        self._probe_base_loss = _loss(self.task_, y, margin)
        g, h = self._grad_hess(y, margin)
        lam = 10.0 * float(h.mean())
        n_bins = int(np.clip(n // 50, 8, 64))

        def probe(v):
            v = np.asarray(v)
            cols = [v] if v.ndim == 1 else [v[:, j] for j in range(v.shape[1])]
            return max(self._residual_gain(c.astype(float), g, h, lam, n_bins) for c in cols)

        parent_gain = {}
        for c in W.columns:
            col = W[c]
            x = pd.factorize(col)[0].astype(float) if c in self.cat_cols_ else col.to_numpy(dtype=float)
            parent_gain[c] = max(probe(x), 0.0)
        gains, novelty = {}, {}
        for name, v in values.items():
            gains[name] = probe(v)
            novelty[name] = gains[name] - max(parent_gain.get(p, 0.0) for p in specs[name].parents)
        ranked = sorted(novelty, key=novelty.get, reverse=True)
        alive = [nm for nm in ranked if novelty[nm] > 0][:keep]
        self._log(f"  screened {len(values)} candidates on {n} rows (5-fold cross-fitted) "
                  f"-> {sum(g > 0 for g in novelty.values())} novel, {len(alive)} kept")
        self._last_gains, self._last_novelty = gains, novelty
        return alive

    # ------------------------------------------------------------------ fit
    def fit(self, X: pd.DataFrame, y, X_unlabeled: Optional[pd.DataFrame] = None):
        """Search features on ``(X, y)``.

        ``X_unlabeled`` (e.g. the competition test set) is optional: when given,
        the final count and group statistics of the selected features are
        computed over training plus unlabeled rows, a standard transductive
        contest trick. Target statistics only ever use labeled rows.
        """
        self._t0 = time.time()
        X = X.reset_index(drop=True).copy()
        y = pd.Series(np.asarray(y))
        if self.task is None:
            from tabularaml.contest.solver import infer_task
            self.task_ = infer_task(y)
        else:
            self.task_ = self.task
        if self.task_ == "regression":
            self.n_classes_ = 0
            y_np = y.to_numpy(dtype=float)
            if self.log_target:
                y_np = np.log1p(y_np)
        else:
            self.classes_, y_np = np.unique(y.to_numpy(), return_inverse=True)
            self.n_classes_ = len(self.classes_)
        self.cat_cols_ = [c for c in X.columns if _is_cat(X[c])]
        for c in self.cat_cols_:
            X[c] = _as_str(X[c])
        self.cat_levels_ = {c: pd.Index(sorted(pd.unique(X[c]))) for c in self.cat_cols_}
        self.raw_cols_ = list(X.columns)
        self.key_cols_ = [c for c in X.columns
                          if c in self.cat_cols_ or 2 <= X[c].nunique() <= self.max_key_cardinality]
        self.ctx_ = Context(self.task_, self.n_classes_, self.random_state)

        # Gate split: these rows never influence the search.
        n = len(X)
        idx = np.arange(n)
        strat = y_np if self.task_ != "regression" else None
        if self.gate_frac and n >= 200:
            try:
                idx_sel, idx_gate = train_test_split(idx, test_size=self.gate_frac,
                                                     random_state=self.random_state, stratify=strat)
            except ValueError:
                idx_sel, idx_gate = train_test_split(idx, test_size=self.gate_frac,
                                                     random_state=self.random_state)
            idx_sel, idx_gate = np.sort(idx_sel), np.sort(idx_gate)
        else:
            idx_sel, idx_gate = idx, np.array([], dtype=int)

        W = X.iloc[idx_sel].reset_index(drop=True)
        yW = y_np[idx_sel]
        # Small tables get repeated CV so that selection is not driven by fold noise.
        n_rep = int(np.clip(round(12_000 / max(len(W), 1)), 1, 3))
        folds = [self._folds(len(W), yW, self.cv, self.random_state + 100 * r) for r in range(n_rep)]
        te_folds = self._folds(len(W), yW, 5, self.random_state + 1)
        # Screening split (A trains the residual boosters, B measures them).
        try:
            idx_a, idx_b = train_test_split(np.arange(len(W)), test_size=0.3, random_state=self.random_state,
                                            stratify=yW if self.task_ != "regression" else None)
        except ValueError:
            idx_a, idx_b = train_test_split(np.arange(len(W)), test_size=0.3, random_state=self.random_state)

        selected: List[Spec] = []
        self.history_ = []
        Wm = self._model_frame(W)
        margin, cur_loss, imp = self._cv(Wm, yW, folds)
        self.base_cv_loss_ = cur_loss
        self._log(f"task={self.task_} rows={n} (search {len(W)}, gate {len(idx_gate)}) "
                  f"cols={X.shape[1]} base CV loss={cur_loss:.6f}")

        for r in range(self.n_rounds):
            if self._time_left() <= 0 or len(selected) >= self.max_new_features:
                break
            cands = self._generate(W, imp, selected, r)
            values = self._materialize(cands, W, yW, te_folds)
            self._log(f"round {r + 1}: {len(cands)} candidates generated, {len(values)} valid")
            if not values:
                break
            spec_by_name = {s.name: s for s in cands}
            survivors = self._screen(values, spec_by_name, W, yW, margin, idx_a, idx_b,
                                     keep=min(80, 3 * self.max_new_features))
            if not survivors:
                break
            # Joint model ranks survivors by split gain alongside current features.
            joint = Wm.copy()
            for nm in survivors:
                for j, col in enumerate(spec_by_name[nm].out_names()):
                    v = values[nm]
                    joint[col] = v if v.ndim == 1 else v[:, j]
            b = self._fit_eval(joint.iloc[idx_a], yW[idx_a], joint.iloc[idx_b], yW[idx_b])
            gain = pd.Series(b.feature_importance("gain"), index=joint.columns)
            rank = sorted(survivors, key=lambda nm: -sum(gain.get(c, 0) for c in spec_by_name[nm].out_names()))
            rank = [nm for nm in rank if sum(gain.get(c, 0) for c in spec_by_name[nm].out_names()) > 0]

            room = self.max_new_features - len(selected)
            ladder = [k for k in (3, 6, 12, 25, 50, 80) if k < min(len(rank), room)] + [min(len(rank), room)]
            best_k, best_loss, best_fit = 0, cur_loss, None
            for k in sorted(set(ladder)):
                if k <= 0:
                    continue
                cols = list(Wm.columns) + [c for nm in rank[:k] for c in spec_by_name[nm].out_names()]
                oof_k, loss_k, imp_k = self._cv(joint[cols], yW, folds)
                self._log(f"  top-{k:<3d} CV loss={loss_k:.6f} ({100 * (cur_loss - loss_k) / cur_loss:+.2f}%)")
                if loss_k < best_loss:
                    best_k, best_loss, best_fit = k, loss_k, (oof_k, imp_k, cols)
                if self._time_left() < 0:
                    break
            if best_k == 0 or (cur_loss - best_loss) / cur_loss < self.min_rel_gain:
                self._log(f"round {r + 1}: no prefix beats current CV loss by {self.min_rel_gain:.2%}; stopping")
                break
            chosen = [spec_by_name[nm] for nm in rank[:best_k]]
            for sp in chosen:
                sp.round_ = r
            selected.extend(chosen)
            Wm = joint[best_fit[2]]
            for s in chosen:
                for j, col in enumerate(s.out_names()):
                    v = values[s.name]
                    W[col] = v if v.ndim == 1 else v[:, j]
            margin, imp = best_fit[0], best_fit[1]
            self.history_.append(dict(round=r + 1, n_candidates=len(values), n_added=best_k,
                                      cv_loss_before=cur_loss, cv_loss_after=best_loss))
            self._log(f"round {r + 1}: +{best_k} features, CV loss {cur_loss:.6f} -> {best_loss:.6f}")
            cur_loss = best_loss

        self.search_cv_loss_ = cur_loss
        self.selected_ = selected
        self.gate_passed_ = None
        if selected and len(idx_gate):
            best_round, self.gate_raw_loss_, self.gate_fe_loss_ = self._gate(X, y_np, idx_sel, idx_gate)
            self.gate_passed_ = best_round is not None
            if self.gate_passed_:
                self.selected_ = [s for s in self.selected_ if s.round_ <= best_round]
            self._log(f"gate: raw={self.gate_raw_loss_:.6f} fe={self.gate_fe_loss_:.6f} "
                      f"({100 * (self.gate_raw_loss_ - self.gate_fe_loss_) / self.gate_raw_loss_:+.2f}%) "
                      f"-> {'PASS' if self.gate_passed_ else 'REJECT'}")
            if not self.gate_passed_:
                self.selected_ = []

        # Refit every spec's statistics on all training rows.
        self._fit_full(X, y_np, None if X_unlabeled is None else self._prep(X_unlabeled))
        self.elapsed_ = time.time() - self._t0
        self._log(f"done: {len(self.selected_)} features added in {self.elapsed_:.1f}s")
        return self

    def _gate(self, X, y, idx_sel, idx_gate):
        Xs = X.iloc[idx_sel].reset_index(drop=True)
        Xg = X.iloc[idx_gate].reset_index(drop=True)
        ys, yg = y[idx_sel], y[idx_gate]
        folds = self._folds(len(Xs), ys, 5, self.random_state + 7)
        Fs, Fg = Xs.copy(), Xg.copy()
        for s in self.selected_:
            vs = s.fit_transform_oof(Fs, ys, self.ctx_, folds) if s.target_dep else s.fit(Fs, ys, self.ctx_).transform(Fs, self.ctx_)
            vg = s.transform(Fg, self.ctx_)
            for j, col in enumerate(s.out_names()):
                Fs[col] = vs if np.ndim(vs) == 1 else vs[:, j]
                Fg[col] = vg if np.ndim(vg) == 1 else vg[:, j]
        import lightgbm as lgb
        es_split = self._folds(len(Xs), ys, 5, self.random_state + 11)[0]

        def gate_loss(cols):
            A, G = self._model_frame(Fs[cols]), self._model_frame(Fg[cols])
            tr, va = es_split
            b = self._fit_eval(A.iloc[tr], ys[tr], A.iloc[va], ys[va], lr=0.05, es=100)
            # Refit on all search rows at the early-stopped size, then score the gate rows.
            full = lgb.train(self._lgb_params(0.05), lgb.Dataset(A, ys), max(1, b.best_iteration))
            return _loss(self.task_, yg, full.predict(G, raw_score=True))

        raw_l = gate_loss(self.raw_cols_)
        # Candidate feature sets are the cumulative rounds; the best one on the gate wins.
        best_round, best_l = None, raw_l
        for r in sorted({s.round_ for s in self.selected_}):
            cols = self.raw_cols_ + [c for s in self.selected_ if s.round_ <= r for c in s.out_names()]
            l = gate_loss(cols)
            self._log(f"  gate rounds<={r + 1}: loss={l:.6f} vs raw {raw_l:.6f}")
            if l < best_l:
                best_round, best_l = r, l
        return best_round, raw_l, best_l

    def _fit_full(self, X, y, U=None):
        folds = self._folds(len(X), y, 5, self.random_state + 1)
        F = X.copy()
        U = None if U is None else U[self.raw_cols_].copy()
        for s in self.selected_:
            if s.target_dep:
                v = s.fit_transform_oof(F, y, self.ctx_, folds)
            elif U is not None:
                s.fit(pd.concat([F, U], ignore_index=True), None, self.ctx_)
                v = s.transform(F, self.ctx_)
            else:
                v = s.fit(F, y, self.ctx_).transform(F, self.ctx_)
            if U is not None:
                vu = s.transform(U, self.ctx_)
                for j, col in enumerate(s.out_names()):
                    U[col] = vu if np.ndim(vu) == 1 else vu[:, j]
            for j, col in enumerate(s.out_names()):
                F[col] = v if np.ndim(v) == 1 else v[:, j]
        self.new_columns_ = [c for s in self.selected_ for c in s.out_names()]
        self._train_frame = F[self.new_columns_].astype(np.float32)

    # ------------------------------------------------------------- transform
    def _prep(self, X):
        X = X.reset_index(drop=True).copy()
        for c in self.cat_cols_:
            X[c] = _as_str(X[c])
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add features to new rows using statistics from all training rows."""
        F = self._prep(X)
        for s in self.selected_:
            v = s.transform(F, self.ctx_)
            for j, col in enumerate(s.out_names()):
                F[col] = v if np.ndim(v) == 1 else v[:, j]
        out = X.reset_index(drop=True).copy()
        for c in self.new_columns_:
            out[c] = F[c].astype(np.float32).to_numpy()
        return out

    def transform_train(self, X: pd.DataFrame) -> pd.DataFrame:
        """Training rows with features as fitted (target encodings out-of-fold)."""
        out = X.reset_index(drop=True).copy()
        if len(out) != len(self._train_frame):
            raise ValueError("transform_train expects the exact frame passed to fit")
        for c in self.new_columns_:
            out[c] = self._train_frame[c].to_numpy()
        return out

    def fit_transform(self, X, y, X_unlabeled=None):
        return self.fit(X, y, X_unlabeled).transform_train(X)

    def report(self) -> pd.DataFrame:
        return pd.DataFrame(self.history_)
