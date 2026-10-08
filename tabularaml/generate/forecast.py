"""Forecasting features: what is known about an entity's target at forecast time.

Switches on from the data's structure: a date column, entity keys whose rows repeat
over time (a store, a store x item), and unlabeled rows that lie after the labelled
period, as a sales-forecasting contest's test file does. Everything is computed as of
a forecast origin. For the test rows the origin is the last labelled period ``T`` and
the horizon ``h = t - T`` runs over the test file's range. Training rows get the same
horizons: the training period is cut into blocks as long as the test period, aligned
so the last one ends at ``T``, and every row's origin is the period before its block
starts. A training row at horizon 3 then sees labels up to 3 periods back, exactly as
a test row at horizon 3 does, and never its own label or a later one; no out-of-fold
scheme is needed, and held-out labels never enter.

Per entity (as of the origin): the horizon; mean target over trailing windows; spread,
zero share and mean of non-zero values when the target is intermittent; the mean of
the last 1 / 4 / 8 values at the same phase of the week (``t - 7j`` up to the origin);
last year's values around ``t`` and last year's change from the origin's date to
``t``'s (a naive seasonal forecast follows from both); age; recent trend. Per group of
entities (a store type, an item family, the whole panel): the group's mean target over
a few windows, same-phase mean and last year's change. Per known-in-advance covariate
that varies within an entity (promotion, holiday, closure flags): periods since its last
and until its next event (until is cut at the end of what the test file would show:
the origin plus the longest horizon) and the entity's recent target difference between
event and non-event periods. Covariates are label-free and read from every row, the
test rows included, as in the contest.
"""
from __future__ import annotations

import warnings
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

_DAY = 86400.0


def _to_days(s: pd.Series) -> np.ndarray:
    if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object:
        s = s.astype(str)
    d = pd.to_datetime(s, errors="coerce")
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    v = d.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float) / (_DAY * 1e9)
    v[d.isna().to_numpy()] = np.nan
    return v


def _is_date(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    v = s.dropna().astype(str).head(500)
    if len(v) < 20 or v.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").mean() < 0.95:
        return False
    return pd.to_datetime(v, errors="coerce").notna().mean() > 0.95


def _canon(s: pd.Series) -> pd.Series:
    """Values as text, numbers read alike whether int or float (1 and 1.0; train and test
    files often differ only in that)."""
    if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object:
        num = pd.to_numeric(s.astype(object), errors="coerce")
        if num.notna().sum() == s.notna().sum():
            s = num
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return pd.Series(np.where(s.isna(), None, s.astype(float).astype(str)), index=s.index, dtype=object)
    return s.astype(object).where(s.notna(), None).astype(str).where(s.notna(), None)


def _keytext(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    key = _canon(df[cols[0]]).fillna("\x00")
    for c in cols[1:]:
        key = key + "\x1f" + _canon(df[c]).fillna("\x00")
    return key


def _codes(s: pd.Series) -> np.ndarray:
    return pd.factorize(s.astype(str) if isinstance(s.dtype, pd.CategoricalDtype) else s, use_na_sentinel=True)[0]


def _combo(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    k = np.zeros(len(df), dtype=np.int64)
    for c in cols:
        x = _codes(df[c]).astype(np.int64)
        k = k * (int(x.max()) + 2) + (x + 1)
    return pd.factorize(k)[0]


def _integral(s: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return True
    x = s.dropna().to_numpy(dtype=float)[:20000]
    return len(x) > 0 and bool(np.all(x == np.round(x)))


class _Panel:
    """Entities x periods target panel with prefix sums for window statistics."""

    def __init__(self, k: np.ndarray, p: np.ndarray, y: np.ndarray, n_keys: int, n_periods: int):
        Y = np.zeros((n_keys, n_periods))
        N = np.zeros((n_keys, n_periods))
        ok = (k >= 0) & np.isfinite(y)
        np.add.at(Y, (k[ok], p[ok]), y[ok])
        np.add.at(N, (k[ok], p[ok]), 1.0)
        has = N > 0
        Y = np.where(has, Y / np.maximum(N, 1), np.nan)
        self.Y, self.P = Y, n_periods
        z = np.zeros((n_keys, 1))
        f = np.isfinite(Y)
        self.S = np.concatenate([z, np.cumsum(np.where(f, Y, 0.0), 1)], 1)
        self.S2 = np.concatenate([z, np.cumsum(np.where(f, Y * Y, 0.0), 1)], 1)
        self.C = np.concatenate([z, np.cumsum(f, 1)], 1)
        nz = f & (Y > 0)
        self.Snz = np.concatenate([z, np.cumsum(np.where(nz, Y, 0.0), 1)], 1)
        self.Cnz = np.concatenate([z, np.cumsum(nz, 1)], 1)
        # First labelled period and last non-zero period at or before each period.
        first = np.where(f.any(1), f.argmax(1), n_periods).astype(float)
        self.first = first
        idx = np.where(nz, np.arange(n_periods)[None, :], -1)
        self.last_nz = np.maximum.accumulate(idx, axis=1)

    def _cut(self, A, k, a, b):
        """Sum of A's increments over periods (a, b], clipped to the panel."""
        a = np.clip(a + 1, 0, self.P)
        b = np.clip(b + 1, 0, self.P)
        b = np.maximum(a, b)
        return A[k, b] - A[k, a]

    def mean(self, k, a, b, nz=False):
        s = self._cut(self.Snz if nz else self.S, k, a, b)
        c = self._cut(self.Cnz if nz else self.C, k, a, b)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(c > 0, s / np.maximum(c, 1), np.nan), c

    def std(self, k, a, b):
        s, c = self._cut(self.S, k, a, b), self._cut(self.C, k, a, b)
        s2 = self._cut(self.S2, k, a, b)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = s / np.maximum(c, 1)
            return np.where(c > 1, np.sqrt(np.maximum(s2 / np.maximum(c, 1) - m * m, 0)), np.nan)

    def at(self, k, q):
        ok = (q >= 0) & (q < self.P)
        r = np.full(len(k), np.nan)
        r[ok] = self.Y[k[ok], q[ok]]
        return r


class ForecastFeatures:
    """Horizon-aware lag, window, seasonal and event features for forecasting contests.

    ``fit(X, y, X_unlabeled)`` detects the structure (``active_`` says whether it was
    found); ``transform(df)`` returns the new columns for any rows (training rows as of
    their block origin, later rows as of the last labelled period).

    Parameters
    ----------
    time_col, entity : "auto" or explicit column / list of columns.
    max_cells : panel size cap (entities x periods); older periods are dropped beyond it.
    log_target : "auto" (log1p for a non-negative, skewed target), True or False.
    """

    def __init__(self, time_col="auto", entity="auto", max_cells: int = 40_000_000, log_target="auto",
                 max_groups: int = 6, max_covariates: int = 8, origins: str = "random", align_week: bool = True,
                 random_state: int = 0, verbose: bool = True):
        self.origins = origins
        self.align_week = align_week
        self.random_state = random_state
        self.time_col = time_col
        self.entity = entity
        self.max_cells = max_cells
        self.log_target = log_target
        self.max_groups = max_groups
        self.max_covariates = max_covariates
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(f"[Forecast] {msg}", flush=True)

    # ------------------------------------------------------------ detection
    def _detect(self, X: pd.DataFrame, U: pd.DataFrame):
        cols = [c for c in X.columns if c in U.columns]
        tc = self.time_col
        if tc == "auto":
            tc = next((c for c in cols if _is_date(X[c])), None)
        if tc is None or tc not in cols:
            return "no date column"
        tx, tu = _to_days(X[tc]), _to_days(U[tc])
        if np.isfinite(tx).mean() < 0.99 or np.isfinite(tu).mean() < 0.99:
            return "dates missing"
        T = np.nanmax(tx)
        if np.mean(tu > T) < 0.95:
            return "unlabeled rows are not after the training period"
        ut = np.unique(np.concatenate([tx[np.isfinite(tx)], tu[np.isfinite(tu)]]))
        d = np.diff(ut)
        step = float(np.min(d[d > 0])) if np.any(d > 0) else 1.0
        if step < 1 - 1e-9:
            return "sub-daily timestamps"
        self.time_col_, self.step_ = tc, step
        A = pd.concat([X[cols], U[cols]], ignore_index=True)
        n = len(A)
        per = np.round((np.concatenate([tx, tu]) - np.nanmin(ut)) / step)
        cand = []
        for c in cols:
            if c == tc:
                continue
            s = A[c]
            nu = s.nunique()
            if nu < 2 or nu > n / 3 or not _integral(s):
                continue
            cand.append((nu, c))
        cand = [c for _, c in sorted(cand, reverse=True)][:8]
        ent = None
        if self.entity != "auto":
            ent = [self.entity] if isinstance(self.entity, str) else list(self.entity)
        else:
            for size in (1, 2, 3):
                for combo in combinations(cand, size):
                    k = _combo(A, combo)
                    ne = k.max() + 1
                    if ne > n / 5:
                        continue
                    u = len(pd.unique(k.astype(np.int64) * (int(per.max()) + 2) + per.astype(np.int64)))
                    if u / n >= 0.98:
                        ent = list(combo)
                        break
                if ent:
                    break
        if not ent:
            return "no entity key (rows per key and date are not unique)"
        self.entity_ = ent
        k = _combo(A, ent)
        ne = int(k.max()) + 1
        # Groups: columns constant within each entity, coarser than it.
        groups = [[c] for c in ent] if len(ent) > 1 else []
        for c in cols:
            if c in ent or c == tc or c in sum(groups, []):
                continue
            s = A[c]
            if pd.api.types.is_numeric_dtype(s) and not isinstance(s.dtype, pd.CategoricalDtype):
                continue
            nu = s.nunique()
            if nu < 2 or nu >= ne:
                continue
            per_ent = pd.Series(_codes(s)).groupby(k).nunique()
            if (per_ent <= 1).mean() >= 0.98:
                groups.append([c])
        groups = sorted(groups, key=lambda g: -A[g[0]].nunique())[: self.max_groups - 1]
        self.groups_ = groups + [[]]
        # Covariates known in advance: vary within entities, few levels, not a weekday.
        covs = []
        dow = np.floor(np.concatenate([tx, tu])).astype(np.int64) % 7
        for c in cols:
            if c in ent or c == tc or any(c in g for g in self.groups_):
                continue
            s = A[c]
            nu = s.nunique()
            if nu < 2 or nu > 60 or not _integral(s):
                continue
            x = _codes(s)
            if (pd.Series(x).groupby(k).nunique() > 1).mean() < 0.2:
                continue  # static attribute
            if step == 1 and (pd.Series(x).groupby(dow).nunique() <= 1).all():
                continue  # a weekday name
            vc = pd.Series(x).value_counts(normalize=True)
            mode = vc.index[0]
            rate = 1 - vc.iloc[0]
            if rate < 0.001 or rate > 0.6:
                continue
            covs.append((rate, c, _canon(s).value_counts().index[0]))
        self.covariates_ = [(c, m) for _, c, m in sorted(covs, key=lambda t: -t[0])][: self.max_covariates]
        return None

    # ------------------------------------------------------------ fitting
    def _periods(self, df):
        return np.round((_to_days(df[self.time_col_]) - self.t0_) / self.step_)

    def _keys(self, df, cols, vocab):
        if not cols:
            return np.zeros(len(df), dtype=np.int64)
        key = _keytext(df, cols)
        return vocab.get_indexer(key)

    def _vocab(self, df, cols):
        if not cols:
            return None
        key = _keytext(df, cols)
        return pd.Index(pd.unique(key))

    def fit(self, X: pd.DataFrame, y, X_unlabeled: Optional[pd.DataFrame] = None):
        self.active_ = False
        if X_unlabeled is None or len(X_unlabeled) < 20:
            self.reason_ = "no unlabeled rows"
            self._log(self.reason_)
            return self
        why = self._detect(X, X_unlabeled)
        if why:
            self.reason_ = why
            self._log(f"off: {why}")
            return self
        y = np.asarray(y, dtype=float)
        lt = self.log_target
        if lt == "auto":
            fy = y[np.isfinite(y)]
            pos = fy[fy > 0]
            lt = bool(len(pos) and fy.min() >= 0 and fy.max() > 3 * np.median(pos))
        self.log_ = bool(lt)
        yt = np.log1p(np.maximum(y, 0)) if self.log_ else y
        # A covariate level that forces a zero target (a closed store) says nothing about
        # demand: those periods are left out of the target panel instead of read as zeros.
        self.masked_ = []
        zero = np.isfinite(y) & (y == 0)
        for c, mode in self.covariates_:
            ev = (_canon(X[c]) != mode).to_numpy() & X[c].notna().to_numpy()
            if ev.sum() >= 20 and zero[ev].mean() >= 0.99 and ev[zero].mean() >= 0.3:
                self.masked_.append(c)
                yt = np.where(ev, np.nan, yt)
        tx = _to_days(X[self.time_col_])
        tu = _to_days(X_unlabeled[self.time_col_])
        self.t0_ = float(np.nanmin(np.concatenate([tx, tu])))
        px, pu = self._periods(X), self._periods(X_unlabeled)
        self.T_ = int(np.nanmax(px))
        h = pu - self.T_
        self.hmin_, self.hmax_ = int(max(1, np.nanmin(h))), int(np.nanmax(h))
        self.L_ = self.hmax_ - self.hmin_ + 1
        if self.step_ == 1:
            # Whole weeks, so every training origin falls on the test origin's weekday.
            self.L_ = int(np.ceil(self.L_ / 7) * 7)
        self.P_ = int(np.nanmax(np.concatenate([px, pu]))) + 1
        A = pd.concat([X, X_unlabeled[[c for c in X.columns if c in X_unlabeled.columns]]], ignore_index=True)
        self.ent_vocab_ = self._vocab(A, self.entity_)
        ne = len(self.ent_vocab_)
        # Cap the panel: drop the oldest periods beyond max_cells.
        keep = max(int(self.max_cells // max(ne, 1)), 4 * self.L_ + 400)
        self.p_lo_ = max(0, self.P_ - keep)
        if self.p_lo_:
            self._log(f"panel capped: periods before {self.p_lo_} dropped")
        Pn = self.P_ - self.p_lo_
        kx = self._keys(X, self.entity_, self.ent_vocab_)
        pp = (px - self.p_lo_)
        okx = np.isfinite(pp) & (pp >= 0)
        self.ent_ = _Panel(kx[okx], pp[okx].astype(np.int64), yt[okx], ne, Pn)
        self.intermittent_ = bool(np.mean(yt[np.isfinite(yt)] == 0) > 0.05)
        self.grp_ = []
        for g in self.groups_:
            vocab = self._vocab(A, g)
            kg = self._keys(X, g, vocab) if g else np.zeros(len(X), dtype=np.int64)
            ng = len(vocab) if g else 1
            self.grp_.append((g, vocab, _Panel(kg[okx], pp[okx].astype(np.int64), yt[okx], ng, Pn)))
        # Covariate event panels over every row (label-free), entity x period.
        ka = self._keys(A, self.entity_, self.ent_vocab_)
        pa = np.concatenate([px, pu]) - self.p_lo_
        oka = np.isfinite(pa) & (pa >= 0) & (ka >= 0)
        self.cov_ = []
        for c, mode in self.covariates_:
            ev = (_canon(A[c]) != mode).to_numpy() & A[c].notna().to_numpy()
            M = np.zeros((ne, Pn), dtype=np.int8)       # 0 unknown, 1 no event, 2 event
            M[ka[oka], pa[oka].astype(np.int64)] = np.where(ev[oka], 2, 1)
            pos = np.flatnonzero((M == 2).ravel()).astype(np.int64)   # entity * Pn + period, sorted
            evy = np.where(M == 2, 1.0, 0.0)
            # Event-conditional target sums for uplift.
            f = np.isfinite(self.ent_.Y)
            z = np.zeros((ne, 1))
            Se = np.concatenate([z, np.cumsum(np.where(f & (M == 2), self.ent_.Y, 0.0), 1)], 1)
            Ce = np.concatenate([z, np.cumsum(f & (M == 2), 1)], 1)
            self.cov_.append((c, mode, pos, Se, Ce))
            del evy
        self.n_ent_, self.Pn_ = ne, Pn
        if self.step_ == 1:
            self.windows_, self.season_, self.year_ = (1, 3, 7, 14, 28, 56, 112, 364), 7, 364
        elif abs(self.step_ - 7) < 1e-9:
            self.windows_, self.season_, self.year_ = (1, 2, 4, 8, 13, 26, 52), None, 52
        elif 28 <= self.step_ <= 31:
            self.windows_, self.season_, self.year_ = (1, 2, 3, 6, 12), None, 12
        else:
            self.windows_, self.season_, self.year_ = (1, 2, 4, 8, 16), None, None
        self.active_ = True
        self._log(f"on: time={self.time_col_} step={self.step_:g}d entity={self.entity_} ({ne}) "
                  f"horizon {self.hmin_}..{self.hmax_} groups={[g for g in self.groups_ if g]} "
                  f"covariates={[c for c, _ in self.covariates_]} masked={self.masked_} log={self.log_} intermittent={self.intermittent_}")
        return self

    # ------------------------------------------------------------ features
    def origin(self, p: np.ndarray, salt=None) -> np.ndarray:
        """Forecast origin of rows at periods ``p`` (absolute)."""
        if self.origins == "random" and salt is not None:
            # Each training row its own horizon, drawn from the test horizons, so rows of
            # one period do not share one origin (and one set of window values).
            # A hash of (entity, period), so a row's horizon does not depend on which rows
            # are transformed with it.
            z = (np.asarray(salt, dtype=np.int64).astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
                 + p.astype(np.int64).astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9)
                 + np.uint64(self.random_state + 1))
            z ^= z >> np.uint64(31)
            z *= np.uint64(0x94D049BB133111EB)
            z ^= z >> np.uint64(29)
            hr = self.hmin_ + (z % np.uint64(self.hmax_ - self.hmin_ + 1)).astype(np.int64)
            if self.step_ == 1 and self.align_week:
                # Same weekday as the test origin: h' = h + ((t - T) - h) mod 7 shift.
                hr = hr + np.mod((p - self.T_) - hr, 7)
            return np.where(p > self.T_, self.T_, p - hr)
        d = self.T_ - p
        k = np.ceil((self.hmin_ + d) / self.L_)
        k = np.maximum(k, 0)
        return self.T_ - self.L_ * k

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self, "active_", False):
            return pd.DataFrame(index=df.index)
        p_abs = self._periods(df)
        ok = np.isfinite(p_abs)
        p_abs = np.nan_to_num(p_abs, nan=self.T_ + 1)
        k = self._keys(df, self.entity_, self.ent_vocab_)
        kk = np.maximum(k, 0)
        o_abs = self.origin(p_abs, salt=kk)
        t = (p_abs - self.p_lo_).astype(np.int64)
        o = (o_abs - self.p_lo_).astype(np.int64)
        h = (p_abs - o_abs)
        F: Dict[str, np.ndarray] = {"fc_h": h}
        if self.step_ <= 7:
            # Calendar fields of the target date (lag features are read against them).
            dt = pd.to_datetime(self.t0_ + p_abs * self.step_, unit="D")
            if self.step_ == 1:
                F["fc_dom"] = dt.day.to_numpy()
                F["fc_doy"] = dt.dayofyear.to_numpy()
            F["fc_woy"] = dt.isocalendar().week.to_numpy().astype(float)
            F["fc_month"] = dt.month.to_numpy()
            F["fc_year"] = dt.year.to_numpy()
        E = self.ent_
        W = self.windows_
        for w in W:
            F[f"fc_mean{w}"] = E.mean(kk, o - w, o)[0]
        wm = W[4] if len(W) > 4 else W[-1]
        F[f"fc_std{wm}"] = E.std(kk, o - wm, o)
        if self.intermittent_:
            m, c = E.mean(kk, o - wm, o)
            nzm, cnz = E.mean(kk, o - wm, o, nz=True)
            F[f"fc_nzshare{wm}"] = np.where(c > 0, cnz / np.maximum(c, 1), np.nan)
            for w in W[2:7:2]:
                F[f"fc_nzmean{w}"] = E.mean(kk, o - w, o, nz=True)[0]
            q = np.clip(o, 0, E.P - 1)
            ln = E.last_nz[kk, q].astype(float)
            F["fc_since_nz"] = np.where((ln >= 0) & (o >= 0), t - ln, np.nan)
        if self.season_:
            s = self.season_
            j0 = np.ceil(h / s).astype(np.int64)
            vals = [E.at(kk, t - s * (j0 + i)) for i in range(8)]
            V = np.vstack(vals)
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                F["fc_same1"] = V[0]
                F["fc_same4"] = np.nanmean(V[:4], 0)
                F["fc_same8"] = np.nanmean(V, 0)
                if self.intermittent_:
                    Vn = np.where(V > 0, V, np.nan)
                    F["fc_same8_nz"] = np.nanmean(Vn, 0)
        if self.year_ and self.Pn_ > self.year_ + self.L_:
            yr = self.year_
            half = (self.season_ or 1) // 2
            ly, _ = E.mean(kk, np.minimum(t - yr - half - 1, o), np.minimum(t - yr + half, o), nz=self.intermittent_)
            base, _ = E.mean(kk, o - yr - 2 * half - 1, o - yr, nz=self.intermittent_)
            F["fc_ly"] = ly
            F["fc_ly_delta"] = ly - base
            rec = F[f"fc_nzmean{W[2]}"] if self.intermittent_ else F[f"fc_mean{W[2]}"]
            F["fc_ly_naive"] = rec + F["fc_ly_delta"]
        first = E.first[kk]
        F["fc_age"] = np.where(first <= o, o - first, np.nan)
        F["fc_trend_a"] = F[f"fc_mean{W[2]}"] - F[f"fc_mean{W[4]}"]
        F["fc_trend_b"] = F[f"fc_mean{W[4]}"] - F[f"fc_mean{W[6]}"] if len(W) > 6 else F[f"fc_mean{W[-1]}"]
        # Groups of entities.
        for g, vocab, G in self.grp_:
            tag = "+".join(g) if g else "all"
            kg = np.maximum(self._keys(df, g, vocab), 0) if g else np.zeros(len(df), dtype=np.int64)
            for w in (W[2], W[4], W[6] if len(W) > 6 else W[-1]):
                F[f"fc_g_{tag}_mean{w}"] = G.mean(kg, o - w, o, nz=self.intermittent_ and not g)[0]
            if self.season_:
                s = self.season_
                j0 = np.ceil(h / s).astype(np.int64)
                V = np.vstack([G.at(kg, t - s * (j0 + i)) for i in range(4)])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    F[f"fc_g_{tag}_same4"] = np.nanmean(V, 0)
            if self.year_ and self.Pn_ > self.year_ + self.L_:
                yr = self.year_
                half = (self.season_ or 1) // 2
                ly, _ = G.mean(kg, np.minimum(t - yr - half - 1, o), np.minimum(t - yr + half, o))
                base, _ = G.mean(kg, o - yr - 2 * half - 1, o - yr)
                F[f"fc_g_{tag}_ly_delta"] = ly - base
            if g:
                F[f"fc_g_{tag}_rel{W[4]}"] = F[f"fc_mean{W[4]}"] - F[f"fc_g_{tag}_mean{W[4]}"]
        # Known-in-advance covariates.
        Pn = self.Pn_
        vend = o + self.hmax_
        q = kk.astype(np.int64) * Pn + np.clip(t, 0, Pn - 1)
        blo = kk.astype(np.int64) * Pn
        wu = W[6] if len(W) > 6 else W[-1]
        for c, mode, pos, Se, Ce in self.cov_:
            ref = np.append(pos, np.iinfo(np.int64).max)
            lo = np.searchsorted(pos, q, side="left")
            prev = ref[np.maximum(lo - 1, 0)]
            since = np.where((lo > 0) & (prev >= blo), q - prev, np.nan)
            hi = np.searchsorted(pos, q, side="right")
            nxt = ref[np.minimum(hi, len(pos))]
            until = np.where(nxt < blo + Pn, nxt - q, np.nan)
            until = np.where(np.isfinite(until) & (t + until <= vend), until, np.nan)
            F[f"fc_ev_since__{c}"] = since
            F[f"fc_ev_until__{c}"] = until
            a, b = np.clip(o - wu + 1, 0, Pn), np.clip(o + 1, 0, Pn)
            b = np.maximum(a, b)
            se, ce = Se[kk, b] - Se[kk, a], Ce[kk, b] - Ce[kk, a]
            s_all, c_all = E._cut(E.S, kk, o - wu, o), E._cut(E.C, kk, o - wu, o)
            with np.errstate(invalid="ignore", divide="ignore"):
                m_ev = np.where(ce > 0, se / np.maximum(ce, 1), np.nan)
                m_no = np.where(c_all - ce > 0, (s_all - se) / np.maximum(c_all - ce, 1), np.nan)
            F[f"fc_ev_uplift__{c}"] = m_ev - m_no
        out = pd.DataFrame({n: np.asarray(v, dtype=np.float32) for n, v in F.items()}, index=df.index)
        bad = (k < 0) | ~ok
        if bad.any():
            out.loc[bad, [c for c in out.columns if c != "fc_h"]] = np.nan
        return out


def forecast_features(X: pd.DataFrame, y, X_unlabeled: pd.DataFrame, **kw):
    """Fit and return (train features, unlabeled features, fitted ForecastFeatures)."""
    f = ForecastFeatures(**kw).fit(X, y, X_unlabeled)
    return f.transform(X), f.transform(X_unlabeled), f
