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
import warnings
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


# Many candidates share the same key columns (every group statistic of key k,
# every count / encoding of k); factorising k once per frame instead of twice
# per candidate removes most of their materialisation time.
_CODE_CACHE: dict = {}


def _code_cache_get(df, cols):
    hit = _CODE_CACHE.get((id(df), tuple(cols)))
    if hit is None or hit[0]() is not df or hit[3] != len(df):
        return None
    return hit[1], hit[2]


def _code_cache_put(df, cols, vocabs, codes):
    import weakref
    if len(_CODE_CACHE) > 512:
        _CODE_CACHE.clear()
    try:
        _CODE_CACHE[(id(df), tuple(cols))] = (weakref.ref(df), vocabs, codes, len(df))
    except TypeError:
        pass


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
        hit = _code_cache_get(df, self._keys())
        if hit is not None:
            self.vocabs_ = hit[0]
            return hit[1].copy()
        self.vocabs_ = _fit_vocab(df, self._keys())
        codes = _codes(df, self._keys(), self.vocabs_)
        _code_cache_put(df, self._keys(), self.vocabs_, codes)
        return codes.copy()

    def _codes(self, df) -> np.ndarray:
        hit = _code_cache_get(df, self._keys())
        if hit is not None and hit[0] is self.vocabs_:
            return hit[1].copy()
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
        if op in ("add", "mul", "sub") and b < a:
            a, b = b, a  # symmetric (or sign-symmetric) ops: one canonical order
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
        k = self._codes(df)
        if getattr(self, "fold_stats_", None):
            # New rows get the average of the fold encoders, so their values carry
            # the same shrinkage as the out-of-fold values the model was trained on.
            return sum(self._apply(k, *st) for st in self.fold_stats_) / len(self.fold_stats_)
        return self._apply(k, self.sums_, self.counts_, self.prior_)

    def fit_transform_oof(self, df, y, ctx, folds):
        k = self._fit_codes(df)
        Y = self._targets(y)
        out = np.zeros((len(df), self.n_out))
        stats = []
        for tr, va in folds:
            st = self._stats(k[tr], Y[tr])
            stats.append(st)
            r = self._apply(k[va], *st)
            out[va] = r if r.ndim == 2 else r[:, None]
        self.fit(df, y, ctx)
        self.fold_stats_ = stats if getattr(ctx, "fold_avg_te", False) else None
        return out[:, 0] if self.n_out == 1 else out


def _nn_algo(M) -> str:
    """kd-trees degrade sharply past ~5 dimensions; blocked brute force does not."""
    return "brute" if M.shape[1] > 5 else "auto"


class KNNTarget(Spec):
    """Mean target of the k nearest training rows in a standardised numeric subspace.

    A classic contest meta-feature: it injects local, smooth structure that
    axis-aligned trees approximate poorly. Training rows are encoded out of
    fold (a row is never its own neighbour); new rows query all training rows.
    Multiclass targets give per-class neighbour frequencies.
    """
    target_dep = True

    def __init__(self, cols: Sequence[str], ks: Sequence[int], n_classes: int, label: str,
                 weights: Optional[Sequence[float]] = None, rank: bool = False):
        super().__init__(cols)
        self.weights = None if weights is None else np.asarray(weights, dtype=float)
        self.rank = rank
        self.ks = tuple(ks)
        self.n_classes = n_classes
        self.per = n_classes if n_classes > 2 else 1
        self.n_out = len(self.ks) * self.per
        self.name = f"knn__{label}"

    def _matrix(self, df, fit=False):
        M = df[self.parents].to_numpy(dtype=float)
        if getattr(self, "rank", False):
            # Rank-gauss: skewed / heavy-tailed axes get comparable spread.
            from scipy.special import ndtri
            if fit:
                self.qgrid_ = [np.unique(np.nanquantile(M[:, j], np.linspace(0, 1, 201)))
                               if np.isfinite(M[:, j]).any() else np.array([0.0])
                               for j in range(M.shape[1])]
            M = np.column_stack([
                ndtri(np.clip(np.interp(M[:, j], g, np.linspace(0, 1, len(g))) if len(g) > 1
                              else np.full(len(M), 0.5), 0.005, 0.995))
                for j, g in enumerate(self.qgrid_)])
            M = np.where(np.isfinite(df[self.parents].to_numpy(dtype=float)), M, np.nan)
        if fit:
            lo, hi = np.nanpercentile(M, 1, axis=0), np.nanpercentile(M, 99, axis=0)
            self.lo_, self.hi_ = lo, hi
            Mc = np.clip(M, lo, hi)
            self.med_ = np.nanmedian(Mc, axis=0)
            Mc = np.where(np.isnan(Mc), self.med_, Mc)
            self.mu_, self.sd_ = Mc.mean(0), Mc.std(0) + 1e-12
        Mc = np.clip(M, self.lo_, self.hi_)
        Mc = np.where(np.isnan(Mc), self.med_, Mc)
        Z = (Mc - self.mu_) / self.sd_
        if self.weights is not None:
            Z = Z * self.weights
        return Z.astype(np.float32)

    def _targets(self, y):
        y = np.asarray(y, dtype=float)
        return np.eye(self.n_classes)[y.astype(int)] if self.per > 1 else y[:, None]

    def _query(self, M_ref, Y_ref, M_q):
        from sklearn.neighbors import NearestNeighbors
        kmax = min(max(self.ks), len(M_ref))
        nn = NearestNeighbors(n_neighbors=kmax, algorithm=_nn_algo(M_ref)).fit(M_ref)
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


class KNNClassDist(KNNTarget):
    """Mean distance to the k nearest training rows of each class (k = 1, 2, 4, ...).

    The kNN distance features of top Otto / contest solutions: how close a row
    sits to each class's manifold, a margin-like signal trees cannot form
    from raw axes. Out of fold on training rows.
    """

    def __init__(self, cols, ks, n_classes, label, weights=None):
        super().__init__(cols, ks, n_classes, label, weights)
        self.n_out = len(self.ks) * n_classes
        self.name = f"knnd__{label}"

    def _query(self, M_ref, Y_ref, M_q):
        from sklearn.neighbors import NearestNeighbors
        yc = Y_ref.argmax(1) if Y_ref.shape[1] > 1 else Y_ref[:, 0].astype(int)
        out = np.zeros((len(M_q), self.n_out))
        kmax = max(self.ks)
        for c in range(self.n_classes):
            R = M_ref[yc == c]
            if len(R) == 0:
                out[:, c * len(self.ks):(c + 1) * len(self.ks)] = np.nan
                continue
            kk = min(kmax, len(R))
            d, _ = NearestNeighbors(n_neighbors=kk, algorithm=_nn_algo(R)).fit(R).kneighbors(M_q)
            cs = np.cumsum(d, axis=1)
            for i, k in enumerate(self.ks):
                k = min(k, kk)
                out[:, c * len(self.ks) + i] = cs[:, k - 1] / k
        return out

    def _targets(self, y):
        y = np.asarray(y, dtype=float)
        return np.eye(self.n_classes)[y.astype(int)] if self.n_classes > 2 else y[:, None]


class LinearOOF(KNNTarget):
    """Out-of-fold prediction of a spline-additive linear model (a ridge GAM).

    Smooth additive structure spread over many numerics (``sum_i f_i(x_i)``)
    costs a tree ensemble many splits; one stacked GAM margin hands it over
    in a single column. Training rows are encoded out of fold.
    """

    def __init__(self, cols: Sequence[str], n_classes: int, label: str):
        super().__init__(cols, (1,), n_classes, label)
        self.name = f"gam__{label}"

    def _model(self):
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import SplineTransformer, StandardScaler
        est = Ridge(alpha=10.0) if self.n_classes == 0 else LogisticRegression(C=0.3, max_iter=300)
        return make_pipeline(SplineTransformer(n_knots=6, degree=3), StandardScaler(), est)

    def _query(self, M_ref, Y_ref, M_q):
        m = self._model()
        if self.n_classes == 0:
            return m.fit(M_ref, Y_ref[:, 0]).predict(M_q)[:, None]
        yc = Y_ref.argmax(1) if self.per > 1 else Y_ref[:, 0].astype(int)
        m.fit(M_ref, yc)
        if self.per == 1:
            return m.decision_function(M_q)[:, None]
        P = np.full((len(M_q), self.per), 1e-6)
        P[:, m.classes_] = m.predict_proba(M_q)
        return np.log(P)


class Projection(KNNTarget):
    """Linear projections of standardised numerics: PCA (unsupervised) or PLS.

    Trees split on one axis at a time; a rotated basis turns oblique
    structure (``lat + lon``-like directions across many columns) into
    single splits. PLS directions are fitted out of fold on training rows.
    """

    def __init__(self, cols: Sequence[str], kind: str, n_comp: int, n_classes: int, label: str):
        super().__init__(cols, (1,), n_classes, label)
        self.kind = kind
        self.n_out = min(n_comp, len(cols))
        self.target_dep = kind == "pls"
        self.name = f"{kind}__{label}"

    def _proj(self, M_ref, Y_ref):
        if self.kind == "pca":
            from sklearn.decomposition import PCA
            return PCA(self.n_out, random_state=0).fit(M_ref)
        from sklearn.cross_decomposition import PLSRegression
        return PLSRegression(self.n_out, scale=False).fit(M_ref, Y_ref)

    def _query(self, M_ref, Y_ref, M_q):
        return self._proj(M_ref, Y_ref).transform(M_q)

    def fit(self, df, y, ctx):
        super().fit(df, y if y is not None else np.zeros(len(df)), ctx)
        self.model_ = self._proj(self.M_, self.Y_)
        return self

    def transform(self, df, ctx):
        return self.model_.transform(self._matrix(df))


class BinnedPairTE(TargetEnc):
    """Out-of-fold target map over a 2-D quantile grid of two numerics.

    Captures smooth two-way interactions (including rotated ones) that
    axis-aligned splits approximate with many leaves.
    """

    def __init__(self, a: str, b: str, n_classes: int = 0, n_bins: int = 16):
        super().__init__([a, b], n_classes)
        self.n_bins = n_bins
        self.name = f"te2d__{a}__{b}"

    def _bin(self, x, edges):
        finite = np.isfinite(x)
        return np.where(finite, np.searchsorted(edges, np.where(finite, x, 0.0), side="right"), len(edges) + 1)

    def _fit_codes(self, df):
        self.edges_ = []
        for c in self.parents:
            x = df[c].to_numpy(dtype=float)
            q = np.linspace(0, 1, self.n_bins + 1)[1:-1]
            self.edges_.append(np.unique(np.nanquantile(x[np.isfinite(x)], q)) if np.isfinite(x).any() else np.array([]))
        return self._codes(df)

    def _codes(self, df):
        a = self._bin(df[self.parents[0]].to_numpy(dtype=float), self.edges_[0])
        b = self._bin(df[self.parents[1]].to_numpy(dtype=float), self.edges_[1])
        return a.astype(np.int64) * (len(self.edges_[1]) + 2) + b


class _MixedCodes:
    """Joint code over keys and quantile-binned numerics (``binned`` names the latter)."""

    def _setup(self, binned: Sequence[str], n_bins: int):
        self.binned = list(binned)
        self.n_bins = n_bins

    def _fit_codes(self, df):
        self.edges_ = {}
        q = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        for c in self.binned:
            x = df[c].to_numpy(dtype=float)
            fin = np.isfinite(x)
            self.edges_[c] = np.unique(np.nanquantile(x[fin], q)) if fin.any() else np.array([])
        keys = [c for c in self.parents if c not in self.edges_]
        self.vocabs_ = _fit_vocab(df, keys)
        return self._codes(df)

    def _codes(self, df):
        out = np.zeros(len(df), dtype=np.int64)
        vocabs = iter(self.vocabs_)
        for c in self.parents:
            if c in self.edges_:
                e = self.edges_[c]
                x = df[c].to_numpy(dtype=float)
                fin = np.isfinite(x)
                idx = np.where(fin, np.searchsorted(e, np.where(fin, x, 0.0), side="right"), len(e) + 1)
                out = out * (len(e) + 2) + idx
            else:
                vocab = next(vocabs)
                idx = vocab.get_indexer(_key_values(df[c]))
                idx[idx < 0] = len(vocab)
                out = out * (len(vocab) + 1) + idx
        return out


class MixedTE(_MixedCodes, TargetEnc):
    """Out-of-fold target encoding of a key/numeric interaction cell.

    Numerics are quantile-binned, so ``(city, income_bin)`` or
    ``(age_bin, hours_bin, sex)`` become categorical cells whose target rate
    is learned directly instead of through many tree splits.
    """

    def __init__(self, cols: Sequence[str], binned: Sequence[str], n_classes: int = 0, n_bins: int = 8):
        TargetEnc.__init__(self, cols, n_classes)
        self._setup(binned, n_bins)
        self.name = "tex__" + "__".join(f"{c}~{n_bins}" if c in binned else c for c in cols)


class MixedCount(_MixedCodes, Count):
    """Frequency of a key/binned-numeric interaction cell."""

    def __init__(self, cols: Sequence[str], binned: Sequence[str], n_bins: int = 8):
        Count.__init__(self, cols)
        self._setup(binned, n_bins)
        self.name = "cntx__" + "__".join(f"{c}~{n_bins}" if c in binned else c for c in cols)


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
            # Ordered families (repeated measurements): trend over the running index.
            if self.stat == "slope":
                t = np.arange(M.shape[1], dtype=float)
                t -= t.mean()
                Mc = M - np.nanmean(M, axis=1, keepdims=True)
                return np.nansum(Mc * t, axis=1) / (t ** 2).sum()
            if self.stat == "delta":
                return M[:, 0] - M[:, -1]
            if self.stat == "npos":
                return (np.nan_to_num(M) > 0).sum(axis=1).astype(float)
        raise ValueError(self.stat)


def column_families(cols: Sequence[str], min_size: int = 3) -> Dict[str, List[str]]:
    """Group columns sharing a name stem that ends in a running index (``Soil_Type_12``, ``px3``)."""
    import re
    fam: Dict[str, List[str]] = {}
    idx: Dict[str, int] = {}
    for c in cols:
        m = re.match(r"^(.*?)[_.\-]?(\d+)$", str(c))
        if m and m.group(1):
            fam.setdefault(m.group(1), []).append(c)
            idx[c] = int(m.group(2))
    return {k: sorted(v, key=idx.get) for k, v in fam.items() if len(v) >= min_size}


def synthetic_rows(X: pd.DataFrame, U: pd.DataFrame, min_rate: float = 0.9, min_cols: int = 20) -> np.ndarray:
    """Unlabeled rows that look generated from other rows' values (Santander 2019's
    fake test rows): no value of theirs is unique among all known rows, while
    almost every labelled row has one. Label-free; only tables with many
    continuous columns qualify, and only when the unlabeled set has clearly more
    such rows than the labelled set. Returns a boolean mask over ``U``'s rows."""
    none = np.zeros(len(U), dtype=bool)
    cols = [c for c in X.columns if c in U.columns and pd.api.types.is_float_dtype(X[c])
            and pd.api.types.is_numeric_dtype(U[c]) and X[c].nunique() >= 0.05 * len(X)]
    if len(cols) < min_cols or len(U) < 1000:
        return none
    ux, uu = np.zeros(len(X), dtype=bool), np.zeros(len(U), dtype=bool)
    for c in cols:
        a, b = X[c].to_numpy(dtype=float), U[c].to_numpy(dtype=float)
        _, inv, cnt = np.unique(np.concatenate([a, b]), return_inverse=True, return_counts=True)
        once = cnt[inv] == 1
        ux |= once[:len(a)]
        uu |= once[len(a):]
    r_x, miss_u = ux.mean(), 1.0 - uu.mean()
    if r_x < min_rate or miss_u < 2 * (1.0 - r_x) + 0.05:
        return none
    return ~uu


class FamilyCount(Spec):
    """Value frequencies of every column of a homogeneous family, as one block.

    ``kind="count"``: how often each row's value occurs in its column (over every
    row whose features are known). ``kind="mask"``: the value itself where it
    repeats, NaN where it is unique. Santander-style anonymised tables hide
    signal in whether a value is shared; each column alone adds little, the
    whole block a lot, so the block is screened by summing per-column novel
    gains (``paired``).
    """
    paired = True

    def __init__(self, cols: Sequence[str], kind: str, label: str):
        super().__init__(cols)
        self.kind = kind
        self.n_out = len(cols)
        self.name = f"fam{kind}__{label}"

    def out_names(self):
        return [f"fam{self.kind}__{c}" for c in self.parents]

    def fit(self, df, y, ctx):
        self.vc_ = [df[c].value_counts() for c in self.parents]
        return self

    def transform(self, df, ctx):
        out = np.empty((len(df), self.n_out), dtype=np.float32)
        for j, (c, vc) in enumerate(zip(self.parents, self.vc_)):
            cnt = df[c].map(vc).to_numpy(dtype=float)
            if self.kind == "count":
                out[:, j] = np.nan_to_num(cnt, nan=0.0)
            else:
                x = df[c].to_numpy(dtype=float)
                out[:, j] = np.where(cnt > 1, x, np.nan)
        return out


class FamilyNB(Spec):
    """Per-column target maps over (value band, how often the value occurs), for a
    family of similar columns, plus their sum: a naive-Bayes score that treats
    the columns as independent evidence (the Santander winners' model, as
    features). Out-of-fold on training rows; frequencies use every row whose
    features are known (``ctx.extra_rows``)."""
    target_dep = True
    paired = True

    def __init__(self, cols: Sequence[str], label: str, n_bins: int = 32):
        super().__init__(cols)
        self.n_bins = n_bins
        self.n_out = len(cols) + 1
        self.name = f"famnb__{label}"

    def out_names(self):
        return [f"famnb__{c}" for c in self.parents] + [self.name + "__sum"]

    def _cells(self, df):
        cells = np.empty((len(df), len(self.parents)), dtype=np.int64)
        for j, c in enumerate(self.parents):
            x = df[c].to_numpy(dtype=float)
            cnt = np.nan_to_num(df[c].map(self.vc_[j]).to_numpy(dtype=float), nan=1.0)
            b = np.searchsorted(self.edges_[j], np.nan_to_num(x, nan=np.inf), side="right")
            cells[:, j] = b * 3 + (np.clip(cnt, 1, 3).astype(np.int64) - 1)
        return cells

    def _prep(self, df, ctx):
        extra = getattr(ctx, "extra_rows", None)
        ref = df[self.parents] if extra is None else pd.concat([df[self.parents], extra[self.parents]], ignore_index=True)
        self.vc_ = [ref[c].value_counts() for c in self.parents]
        self.edges_ = [np.unique(np.nanquantile(ref[c].to_numpy(dtype=float), np.linspace(0, 1, self.n_bins + 1)[1:-1]))
                       for c in self.parents]

    def _table(self, cells, y):
        size = (self.n_bins + 1) * 3
        prior = float(np.mean(y))
        tabs = []
        for j in range(cells.shape[1]):
            s = np.bincount(cells[:, j], y, minlength=size)
            n = np.bincount(cells[:, j], minlength=size).astype(float)
            m = (s + 20 * prior) / (n + 20)
            if self.binary_:
                m = np.clip(m, 1e-4, 1 - 1e-4)
                pr = np.clip(prior, 1e-4, 1 - 1e-4)
                tabs.append(np.log(m / (1 - m)) - np.log(pr / (1 - pr)))
            else:
                tabs.append(m - prior)
        return np.stack(tabs, 1)

    def _apply(self, cells, tab):
        v = np.take_along_axis(tab, cells, axis=0)
        return np.column_stack([v, v.sum(1)]).astype(np.float32)

    def fit(self, df, y, ctx):
        y = np.asarray(y, dtype=float)
        self.binary_ = ctx.task == "binary"
        self._prep(df, ctx)
        self.tab_ = self._table(self._cells(df), y)
        return self

    def transform(self, df, ctx):
        return self._apply(self._cells(df), self.tab_)

    def fit_transform_oof(self, df, y, ctx, folds):
        y = np.asarray(y, dtype=float)
        self.binary_ = ctx.task == "binary"
        self._prep(df, ctx)
        cells = self._cells(df)
        out = np.zeros((len(df), self.n_out), dtype=np.float32)
        for tr, va in folds:
            out[va] = self._apply(cells[va], self._table(cells[tr], y[tr]))
        self.tab_ = self._table(cells, y)
        return out


def slot_families(df: pd.DataFrame, cat_cols: Sequence[str]) -> List[tuple]:
    """Categorical column families that hold one multi-valued field in numbered slots
    (``component_id_1..8``, ``spec1..10``), each with its paired numeric family on the
    same slot numbers when there is one (``quantity_1..8``)."""
    import re
    cats = column_families([c for c in df.columns if c in cat_cols])
    nums = column_families([c for c in df.columns if c not in cat_cols])
    idx = lambda c: re.match(r"^.*?(\d+)$", str(c)).group(1)
    out = []
    for stem, cols in cats.items():
        # Slots of one field fill left to right (later slots mostly empty) and share their
        # values; a family of distinct anonymised variables (cat1..cat116) does neither.
        na = [df[c].isna().mean() + (_as_str(df[c]) == "__NA__").mean() for c in cols]
        if na[-1] < na[0] + 0.2:
            continue
        sets = [set(pd.unique(df[c].dropna().astype(str))) - {"__NA__"} for c in cols[:3]]
        if len(sets[0]) < 5 or len(sets[0] & sets[1]) < 0.2 * min(len(sets[0]), len(sets[1]) or 1):
            continue
        slots = [idx(c) for c in cols]
        pair = next((v for v in nums.values() if [idx(c) for c in v] == slots), None)
        out.append((stem, cols, pair))
    return out


def _slot_values(df, cols) -> np.ndarray:
    v = np.column_stack([_as_str(df[c]).to_numpy() for c in cols])
    return np.where(np.isin(v, ["__NA__", "nan", "None", "NA", ""]), None, v)


class SlotBag(Spec):
    """A multi-valued field stored in numbered slots, as a bag: number of filled slots,
    and how many times (or how much, with a paired quantity family) each frequent
    value occurs across the slots, whatever slot it sits in. Label-free."""

    def __init__(self, cols: Sequence[str], weights: Optional[Sequence[str]], label: str, top: int = 16):
        super().__init__(list(cols) + list(weights or []))
        self.cols, self.weights, self.top = list(cols), list(weights or []), top
        self.name = f"slotbag__{label}"
        self.n_out = 1 + top

    def out_names(self):
        return [f"{self.name}_n"] + [f"{self.name}_{k}" for k in range(self.n_out - 1)]

    def fit(self, df, y, ctx):
        extra = getattr(ctx, "extra_rows", None)
        ref = df if extra is None else pd.concat([df[self.parents], extra[self.parents]], ignore_index=True)
        v = _slot_values(ref, self.cols).ravel()
        vc = pd.Series(v[v != None]).value_counts()  # noqa: E711
        self.levels_ = list(vc.index[:self.top])  # fewer than ``top``: the rest stay zero
        return self

    def transform(self, df, ctx):
        V = _slot_values(df, self.cols)
        W = (np.column_stack([df[c].to_numpy(dtype=float) for c in self.weights]) if self.weights
             else np.ones(V.shape))
        W = np.nan_to_num(W, nan=1.0)
        out = [(V != None).sum(1).astype(float)]  # noqa: E711
        for lv in self.levels_:
            out.append(((V == lv) * W).sum(1))
        out += [np.zeros(len(V))] * (self.n_out - len(out))
        return np.column_stack(out).astype(np.float32)


class SlotTE(Spec):
    """Target maps of the values of a slot family, learned across all slots (each row's
    target counts once for every value it holds), then summarised per row: mean, max,
    min and (quantity-)weighted sum over its filled slots. Out-of-fold on training rows."""
    target_dep = True
    n_out = 4

    def __init__(self, cols: Sequence[str], weights: Optional[Sequence[str]], label: str):
        super().__init__(list(cols) + list(weights or []))
        self.cols, self.weights = list(cols), list(weights or [])
        self.name = f"slotte__{label}"

    def _table(self, V, y):
        rows = np.repeat(np.arange(len(V)), V.shape[1])
        v = V.ravel()
        ok = v != None  # noqa: E711
        f = pd.DataFrame({"v": v[ok], "y": np.asarray(y, dtype=float)[rows[ok]]})
        g = f.groupby("v")["y"].agg(["sum", "size"])
        prior = float(np.mean(y))
        return (g["sum"] + TE_SMOOTHING * prior) / (g["size"] + TE_SMOOTHING), prior

    def _apply(self, V, W, tab, prior):
        E = tab.reindex(V.ravel()).to_numpy(dtype=float).reshape(V.shape)
        E = np.where(V == None, np.nan, np.where(np.isnan(E), prior, E))  # noqa: E711
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.column_stack([np.nanmean(E, 1), np.nanmax(E, 1), np.nanmin(E, 1),
                                    np.nansum(np.nan_to_num(E - prior) * W, 1)])

    def _vw(self, df):
        V = _slot_values(df, self.cols)
        W = (np.nan_to_num(np.column_stack([df[c].to_numpy(dtype=float) for c in self.weights]), nan=1.0)
             if self.weights else np.ones(V.shape))
        return V, W

    def fit(self, df, y, ctx):
        V, _ = self._vw(df)
        self.tab_, self.prior_ = self._table(V, y)
        return self

    def transform(self, df, ctx):
        V, W = self._vw(df)
        return self._apply(V, W, self.tab_, self.prior_)

    def fit_transform_oof(self, df, y, ctx, folds):
        V, W = self._vw(df)
        y = np.asarray(y, dtype=float)
        out = np.full((len(df), self.n_out), np.nan)
        for tr, va in folds:
            out[va] = self._apply(V[va], W[va], *self._table(V[tr], y[tr]))
        self.fit(df, y, ctx)
        return out


class GroupStat(Spec):
    """Statistic of numeric ``num`` within groups of ``key``.

    ``key`` may be a tuple of columns: a composite group (``card`` x ``address``,
    the "user id" of fraud contests), whose statistics describe an entity no
    single column identifies. ``nunique`` counts distinct values of ``num``
    in the group.
    """

    def __init__(self, key, num: str, stat: str):
        keys = [key] if isinstance(key, str) else list(key)
        super().__init__(keys + [num])
        self.stat = stat
        self.name = f"grp_{stat}__{num}__by__{'+'.join(keys)}"

    def _keys(self):
        return self.parents[:-1]

    def fit(self, df, y, ctx):
        k = self._fit_codes(df)
        v = pd.Series(df[self.parents[-1]].to_numpy(dtype=float))
        g = v.groupby(k)
        if self.stat == "nunique":
            self.a_ = g.nunique()
        elif self.stat in ("mean", "dev"):
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
            return df[self.parents[-1]].to_numpy(dtype=float) - a
        if self.stat == "z":
            b = self.b_.reindex(k).to_numpy(dtype=float)
            with np.errstate(all="ignore"):
                r = (df[self.parents[-1]].to_numpy(dtype=float) - a) / np.where(b > 0, b, np.nan)
            return r
        return a


class EntityLag(Spec):
    """Where a row sits in its entity's history: time since the entity's previous
    and until its next row, and how many earlier rows it has. Label-free; with
    unlabeled rows the history spans train and test, as in a contest."""

    def __init__(self, key, time: str, kind: str):
        keys = [key] if isinstance(key, str) else list(key)
        super().__init__(keys + [time])
        self.kind = kind
        self.name = f"lag_{kind}__{'+'.join(keys)}__{time}"

    @property
    def novelty_parents(self):
        # As for events: the entity key scopes the feature; it is not what it re-expresses.
        return [self.parents[-1]]

    def _keys(self):
        return self.parents[:-1]

    def _span(self):
        # Room for times well outside the fitted range (later test periods).
        R = self.t1_ - self.t0_ + 1.0
        return 6.0 * R, 2.5 * R - self.t0_

    def _pos(self, k, t):
        span, off = self._span()
        return k.astype(np.float64) * span + np.clip(t + off, 0.0, span - 1.0)

    def fit(self, df, y, ctx):
        k = self._fit_codes(df)
        t = df[self.parents[-1]].to_numpy(dtype=float)
        ok = np.isfinite(t) & (k >= 0)
        self.t0_, self.t1_ = float(np.nanmin(t)), float(np.nanmax(t))
        self.ref_ = np.sort(self._pos(k[ok], t[ok]))
        return self

    def transform(self, df, ctx):
        k = self._codes(df)
        t = df[self.parents[-1]].to_numpy(dtype=float)
        q = self._pos(k, np.nan_to_num(t, nan=self.t0_))
        span = self._span()[0]
        lo = np.searchsorted(self.ref_, q, side="left")    # first ref row at or after t
        hi = np.searchsorted(self.ref_, q, side="right")   # first ref row after t
        start = np.searchsorted(self.ref_, np.floor(q / span) * span, side="left")
        end = np.searchsorted(self.ref_, (np.floor(q / span) + 1) * span, side="left")
        ref = np.append(self.ref_, np.inf)
        if self.kind == "prev":
            r = np.where(lo > start, q - ref[np.maximum(lo - 1, 0)], np.nan)
        elif self.kind == "next":
            r = np.where(hi < end, ref[np.minimum(hi, len(self.ref_))] - q, np.nan)
        elif self.kind.startswith(("win", "fwd")):
            # Rows of the entity within ``w`` time units before (win) / after (fwd).
            w = float(self.kind[3:])
            if self.kind.startswith("win"):
                r = (lo - np.maximum(np.searchsorted(self.ref_, q - w, side="left"), start)).astype(float)
            else:
                r = (np.minimum(np.searchsorted(self.ref_, q + w, side="right"), end) - hi).astype(float)
        else:  # "nth": earlier rows of the entity
            r = (lo - start).astype(float)
        bad = ~np.isfinite(t) | (k < 0)
        r = r.astype(float)
        r[bad] = np.nan
        return r


_DAY = 86400.0
_DATE_PARTS = ("dow", "dom", "month", "year", "doy", "woy", "to_month_end")


def _to_days(s: pd.Series) -> np.ndarray:
    """Datetimes as float days since 1970-01-01 (NaN where missing)."""
    d = pd.to_datetime(s, errors="coerce")
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    v = d.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float) / (_DAY * 1e9)
    v[d.isna().to_numpy()] = np.nan
    return v


def _is_date_like(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if not (s.dtype == object or pd.api.types.is_string_dtype(s)):
        return False
    sample = s.dropna().astype(str).head(500)
    if len(sample) < 20 or not sample.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").mean() > 0.95:
        return False
    return pd.to_datetime(sample, errors="coerce").notna().mean() > 0.95


class DatePart(Spec):
    """A calendar field of a date column (held internally as days since 1970)."""

    def __init__(self, col: str, part: str):
        super().__init__([col])
        self.part = part
        self.name = f"{part}__{col}"

    def transform(self, df, ctx):
        v = df[self.parents[0]].to_numpy(dtype=float)
        ok = np.isfinite(v)
        d = pd.to_datetime(np.where(ok, v, 0.0), unit="D")
        if self.part == "dow":
            r = d.dayofweek
        elif self.part == "dom":
            r = d.day
        elif self.part == "month":
            r = d.month
        elif self.part == "year":
            r = d.year
        elif self.part == "doy":
            r = d.dayofyear
        elif self.part == "woy":
            r = d.isocalendar().week.to_numpy()
        else:  # days to the end of the month
            r = d.days_in_month - d.day
        r = np.asarray(r, dtype=float)
        r[~ok] = np.nan
        return r


class EventRecency(Spec):
    """Time since the entity's last row where ``flag == level``, or until its next
    one (strictly before / after the row's own time). Rossmann-style calendar
    features: days since the store's last promotion, until the next holiday,
    since it last closed. Label-free; fitted over every row whose features are
    known, so test-period events count, as in the contest."""

    def __init__(self, keys: Sequence[str], time: str, flag: str, level, kind: str):
        super().__init__(list(keys) + [time, flag])
        self.level, self.kind = level, kind
        tag = "+".join(keys) if keys else "all"
        self.name = f"ev_{kind}__{flag}_{level}__{tag}"

    @property
    def novelty_parents(self):
        # The entity key only scopes the event: an entity's own level (a store's
        # usual sales) is no re-expression of when its promotions happen.
        return [self.parents[-1]]

    def _keys(self):
        return self.parents[:-2]

    def _key_codes(self, df, fit):
        if not self._keys():
            return np.zeros(len(df), dtype=np.int64)
        return self._fit_codes(df) if fit else self._codes(df)

    def _hit(self, df):
        f = df[self.parents[-1]]
        if isinstance(self.level, str):
            return (_as_str(f) == self.level).to_numpy()
        return (f.to_numpy(dtype=float) == self.level)

    def _pos(self, k, t):
        R = self.t1_ - self.t0_ + 1.0
        span, off = 6.0 * R, 2.5 * R - self.t0_
        return k.astype(np.float64) * span + np.clip(t + off, 0.0, span - 1.0), span

    def fit(self, df, y, ctx):
        k = self._key_codes(df, True)
        t = df[self.parents[-2]].to_numpy(dtype=float)
        self.t0_, self.t1_ = float(np.nanmin(t)), float(np.nanmax(t))
        ok = np.isfinite(t) & (k >= 0) & self._hit(df)
        self.ref_ = np.sort(self._pos(k[ok], t[ok])[0])
        return self

    def transform(self, df, ctx):
        k = self._key_codes(df, False)
        t = df[self.parents[-2]].to_numpy(dtype=float)
        q, span = self._pos(k, np.nan_to_num(t, nan=self.t0_))
        ref = np.append(self.ref_, np.inf)
        block = np.floor(q / span) * span
        if self.kind == "prev":
            lo = np.searchsorted(self.ref_, q, side="left")
            start = np.searchsorted(self.ref_, block, side="left")
            r = np.where(lo > start, q - ref[np.maximum(lo - 1, 0)], np.nan)
        else:
            hi = np.searchsorted(self.ref_, q, side="right")
            end = np.searchsorted(self.ref_, block + span, side="left")
            r = np.where(hi < end, ref[np.minimum(hi, len(self.ref_))] - q, np.nan)
        r = r.astype(float)
        r[~np.isfinite(t) | (k < 0)] = np.nan
        return r


class LaggedTargetMean(Spec):
    """Mean target of an entity's labelled rows in a trailing window that ends one
    test horizon before the row: ``[t - lag - window, t - lag)``. Training and test
    rows then see the same delay, so recent level shifts (a store's sales this
    quarter) carry into the test period. Uses only strictly earlier rows, so no
    out-of-fold scheme is needed."""
    target_dep = True
    time_safe = True

    def __init__(self, keys: Sequence[str], time: str, window: float, lag: float):
        super().__init__(list(keys) + [time])
        self.window, self.lag = float(window), float(lag)
        self.name = f"lagte__{'+'.join(keys)}__w{self.window:.6g}_l{self.lag:.6g}"

    @property
    def novelty_parents(self):
        return [self.parents[-1]]

    def _keys(self):
        return self.parents[:-1]

    def _pos(self, k, t):
        R = self.t1_ - self.t0_ + 1.0
        span, off = 6.0 * R, 2.5 * R - self.t0_
        return k.astype(np.float64) * span + np.clip(t + off, 0.0, span - 1.0), span

    def fit(self, df, y, ctx):
        k = self._fit_codes(df)
        t = df[self.parents[-1]].to_numpy(dtype=float)
        y = np.asarray(y, dtype=float)
        self.t0_, self.t1_ = float(np.nanmin(t)), float(np.nanmax(t))
        ok = np.isfinite(t) & (k >= 0) & np.isfinite(y)
        pos = self._pos(k[ok], t[ok])[0]
        o = np.argsort(pos, kind="stable")
        self.ref_ = pos[o]
        self.csum_ = np.concatenate([[0.0], np.cumsum(y[ok][o])])
        self.prior_ = float(np.mean(y[ok]))
        return self

    def transform(self, df, ctx):
        k = self._codes(df)
        t = df[self.parents[-1]].to_numpy(dtype=float)
        q, span = self._pos(k, np.nan_to_num(t, nan=self.t0_))
        block = np.floor(q / span) * span
        hi_q = np.maximum(q - self.lag, block)
        lo_q = np.maximum(q - self.lag - self.window, block)
        hi = np.searchsorted(self.ref_, hi_q, side="left")
        lo = np.searchsorted(self.ref_, lo_q, side="left")
        n = (hi - lo).astype(float)
        s = self.csum_[hi] - self.csum_[lo]
        r = (s + TE_SMOOTHING * self.prior_) / (n + TE_SMOOTHING)
        r[n == 0] = np.nan
        r[~np.isfinite(t) | (k < 0)] = np.nan
        return r

    def fit_transform_oof(self, df, y, ctx, folds):
        return self.fit(df, y, ctx).transform(df, ctx)


def _is_text(s: pd.Series, n_sample: int = 20_000) -> bool:
    """Free text: a string column of mostly distinct values written as several words
    (titles, descriptions, comments), as opposed to codes, ids or category labels."""
    if not _is_cat(s) or len(s) < 100:
        return False
    v = s.dropna().astype(str)
    if len(v) > n_sample:
        v = v.sample(n_sample, random_state=0)
    if len(v) < 50 or v.nunique() < max(50, 0.05 * len(v)):
        return False
    # Several space-separated words, not one code made of letters and digits.
    return (float(v.str.count(r"[^\W\d_]{2,}").mean()) >= 2.0
            and float(v.str.split().str.len().mean()) >= 2.5)


def _text_values(df, c, clip: Optional[int] = None) -> pd.Series:
    t = df[c].astype(str)
    t = t.where(t != "__NA__", "")
    # Long fields (review dumps) cost n-gram time quadratically in nothing but their tails.
    return t if clip is None else t.str.slice(0, clip)


TEXT_CLIP = 1500


class TextStats(Spec):
    """How a free-text field is written: characters, words, digits, capitals,
    punctuation, distinct-word share and emptiness. Label-free."""
    n_out = 7

    def __init__(self, col: str):
        super().__init__([col])
        self.name = f"txtstat__{col}"

    def transform(self, df, ctx):
        t = _text_values(df, self.parents[0])
        n_ch = t.str.len().to_numpy(dtype=float)
        words = t.str.lower().str.findall(r"\w+")
        n_w = words.str.len().to_numpy(dtype=float)
        n_u = words.map(lambda w: len(set(w))).to_numpy(dtype=float)
        dig = t.str.count(r"\d").to_numpy(dtype=float)
        up = t.str.count(r"[A-Z]").to_numpy(dtype=float)
        pun = t.str.count(r"[^\w\s]").to_numpy(dtype=float)
        den = np.maximum(n_ch, 1.0)
        return np.column_stack([n_ch, n_w, dig / den, up / den, pun / den,
                                n_u / np.maximum(n_w, 1.0), (n_ch == 0).astype(float)]).astype(np.float32)


def _tfidf(kind: str, max_features: int):
    from sklearn.feature_extraction.text import TfidfVectorizer
    if kind == "char":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=max_features,
                               sublinear_tf=True, dtype=np.float32)
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_features, sublinear_tf=True,
                           token_pattern=r"(?u)\b\w+\b", dtype=np.float32)


class TextSVD(Spec):
    """Latent semantic components of a free-text field: TF-IDF of words and word
    pairs reduced by truncated SVD, fitted label-free on every row whose text is
    known. Topic-like directions a tree can split on."""

    def __init__(self, col: str, n_comp: int = 16, max_fit_rows: int = 200_000):
        super().__init__([col])
        self.n_out = n_comp
        self.max_fit_rows = max_fit_rows
        self.name = f"txtsvd__{col}"

    def fit(self, df, y, ctx):
        from sklearn.decomposition import TruncatedSVD
        c = self.parents[0]
        extra = getattr(ctx, "extra_rows", None)
        t = _text_values(df, c, TEXT_CLIP)
        if extra is not None and c in extra:
            t = pd.concat([t, _text_values(extra, c, TEXT_CLIP)], ignore_index=True)
        if len(t) > self.max_fit_rows:
            t = t.sample(self.max_fit_rows, random_state=0)
        self.vec_ = _tfidf("word", 50_000).fit(t)
        T = self.vec_.transform(t)
        self.n_out = int(min(self.n_out, max(1, T.shape[1] - 1)))
        self.svd_ = TruncatedSVD(self.n_out, random_state=0, algorithm="randomized", n_iter=4).fit(T)
        return self

    def transform(self, df, ctx):
        return self.svd_.transform(self.vec_.transform(_text_values(df, self.parents[0], TEXT_CLIP))).astype(np.float32)


class TextLinearOOF(Spec):
    """Out-of-fold sparse linear model on the words (and character n-grams) of
    free-text fields, optionally with one-hot keys alongside: the ridge / logistic
    model behind most text-tabular contest wins (Mercari, Avito), as one column a
    tree ensemble can use. Vocabularies are fitted label-free on every row whose
    text is known; the model's training rows are encoded out of fold."""
    target_dep = True

    def __init__(self, cols: Sequence[str], n_classes: int, keys: Sequence[str] = (), label: str = "",
                 chars: bool = True):
        super().__init__(list(cols) + list(keys))
        self.text_cols = list(cols)
        self.keys = list(keys)
        self.n_classes = n_classes
        self.chars = chars
        self.n_out = n_classes if n_classes > 2 else 1
        self.name = f"txtlin__{label or '+'.join(cols)}"
        self.novelty_parents = list(cols)

    def _vocab(self, df, ctx):
        extra = getattr(ctx, "extra_rows", None)
        self.vecs_ = []
        for c in self.text_cols:
            t = _text_values(df, c, TEXT_CLIP)
            if extra is not None and c in extra:
                t = pd.concat([t, _text_values(extra, c, TEXT_CLIP)], ignore_index=True)
            if len(t) > 300_000:
                t = t.sample(300_000, random_state=0)
            vs = [_tfidf("word", 200_000).fit(t)]
            if self.chars and t.str.len().mean() <= 80:
                vs.append(_tfidf("char", 100_000).fit(t))
            self.vecs_.append(vs)
        if self.keys:
            from sklearn.preprocessing import OneHotEncoder
            self.ohe_ = OneHotEncoder(handle_unknown="ignore", min_frequency=2, dtype=np.float32).fit(
                df[self.keys].astype(str))

    def _design(self, df):
        from scipy import sparse
        blocks = [v.transform(_text_values(df, c, TEXT_CLIP)) for c, vs in zip(self.text_cols, self.vecs_) for v in vs]
        if self.keys:
            blocks.append(self.ohe_.transform(df[self.keys].astype(str)))
        return sparse.hstack(blocks, format="csr")

    def _query(self, A, y, B):
        from sklearn.linear_model import LogisticRegression, Ridge
        if self.n_classes == 0:
            mu = y.mean()
            return (Ridge(alpha=3.0, solver="sparse_cg").fit(A, y - mu).predict(B) + mu)[:, None]
        if self.n_classes == 2:
            m = LogisticRegression(C=1.0, solver="liblinear", tol=1e-3).fit(A, y.astype(int))
            return m.decision_function(B)[:, None]
        Y = np.eye(self.n_classes)[y.astype(int)]
        return Ridge(alpha=3.0, solver="sparse_cg").fit(A, Y - Y.mean(0)).predict(B) + Y.mean(0)

    def fit(self, df, y, ctx):
        self._vocab(df, ctx)
        self.A_ = self._design(df)
        self.y_ = np.asarray(y, dtype=float)
        return self

    def transform(self, df, ctx):
        r = self._query(self.A_, self.y_, self._design(df))
        return r[:, 0] if self.n_out == 1 else r

    def fit_transform_oof(self, df, y, ctx, folds):
        self.fit(df, y, ctx)
        out = np.empty((len(df), self.n_out))
        for tr, va in folds:
            out[va] = self._query(self.A_[tr], self.y_[tr], self.A_[va])
        return out[:, 0] if self.n_out == 1 else out


class CrossLinearOOF(Spec):
    """Out-of-fold sparse linear model on one-hot keys and all their pairwise crosses.

    The classic winning recipe on high-cardinality categorical contests (Amazon
    access, click-through, insurance): a regularised logistic / ridge model over
    one-hot features of every key and key pair learns thousands of interaction
    effects with shrinkage, which a tree ensemble can then use as one column.
    Training rows are encoded out of fold.
    """
    target_dep = True

    def __init__(self, keys: Sequence[str], n_classes: int, pairs: bool = True, label: str = "keys"):
        super().__init__(keys)
        self.n_classes = n_classes
        self.pairs = pairs
        self.n_out = n_classes if n_classes > 2 else 1
        self.name = f"xlin__{label}" + ("" if pairs else "_1")

    def _codes_matrix(self, df) -> np.ndarray:
        """int64 code per (row, block): each key, then each key pair."""
        cols = [_codes(df, [c], [v]) for c, v in zip(self.parents, self.vocabs_)]
        out = list(cols)
        if self.pairs:
            for i, j in combinations(range(len(cols)), 2):
                out.append(cols[i] * (len(self.vocabs_[j]) + 1) + cols[j])
        return np.stack(out, axis=1)

    @staticmethod
    def _design(C_ref, C_q):
        """Sparse one-hot of ``C_q`` over the levels seen at least twice per block of ``C_ref``."""
        from scipy import sparse
        rows, cols, off = [], [], 0
        n = len(C_q)
        for b in range(C_ref.shape[1]):
            lv, cnt = np.unique(C_ref[:, b], return_counts=True)
            lv = lv[cnt >= 2]
            pos = np.searchsorted(lv, C_q[:, b])
            pos = np.minimum(pos, max(len(lv) - 1, 0))
            hit = (len(lv) > 0) & (lv[pos] == C_q[:, b]) if len(lv) else np.zeros(n, bool)
            rows.append(np.nonzero(hit)[0])
            cols.append(pos[hit] + off)
            off += len(lv)
        r, c = np.concatenate(rows), np.concatenate(cols)
        return sparse.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, max(off, 1)))

    def _model(self, shape):
        from sklearn.linear_model import LogisticRegression, Ridge
        if self.n_classes == 0:
            return Ridge(alpha=3.0, solver="sparse_cg")
        if self.n_classes > 2:
            return LogisticRegression(C=0.5, max_iter=300)
        # Wide one-hot designs solve much faster in the dual.
        return LogisticRegression(C=0.5, solver="liblinear", tol=1e-2, dual=shape[1] > shape[0])

    def _query(self, S_ref, y_ref, S_q):
        A, B = self._design(S_ref, S_ref), self._design(S_ref, S_q)
        m = self._model(A.shape)
        if self.n_classes == 0:
            mu = y_ref.mean()
            m.fit(A, y_ref - mu)
            return (m.predict(B) + mu)[:, None]
        m.fit(A, y_ref.astype(int))
        if self.n_out == 1:
            return m.decision_function(B)[:, None]
        P = np.full((B.shape[0], self.n_out), 1e-6)
        P[:, m.classes_] = m.predict_proba(B)
        return np.log(P)

    def fit(self, df, y, ctx):
        self.vocabs_ = _fit_vocab(df, self.parents)
        self.S_ = self._codes_matrix(df)
        self.y_ = np.asarray(y, dtype=float)
        return self

    def transform(self, df, ctx):
        r = self._query(self.S_, self.y_, self._codes_matrix(df))
        return r[:, 0] if self.n_out == 1 else r

    def fit_transform_oof(self, df, y, ctx, folds):
        self.fit(df, y, ctx)
        out = np.empty((len(df), self.n_out))
        for tr, va in folds:
            out[va] = self._query(self.S_[tr], self.y_[tr], self.S_[va])
        return out[:, 0] if self.n_out == 1 else out


class KeyBinTE(BinnedPairTE):
    """Out-of-fold target map over (key level x quantile bin of a numeric).

    The categorical-by-numeric interaction ("price band within this model",
    "age band within this country") that a tree needs one split per level for.
    """

    def __init__(self, key: str, num: str, n_classes: int = 0, n_bins: int = 8):
        TargetEnc.__init__(self, [key, num], n_classes)
        self.n_bins = n_bins
        self.name = f"tekb__{key}__{num}"

    def _fit_codes(self, df):
        self.vocabs_ = _fit_vocab(df, self.parents[:1])
        x = df[self.parents[1]].to_numpy(dtype=float)
        q = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges_ = np.unique(np.nanquantile(x[np.isfinite(x)], q)) if np.isfinite(x).any() else np.array([])
        return self._codes(df)

    def _codes(self, df):
        k = _codes(df, self.parents[:1], self.vocabs_)
        b = self._bin(df[self.parents[1]].to_numpy(dtype=float), self.edges_)
        return k * (len(self.edges_) + 2) + b.astype(np.int64)


class Digits(Spec):
    """Value-representation features: fractional part, last digits, rounding residue.

    Contest tables (prices, synthetic Playground data, sensor readings) often carry
    signal in how a value is written rather than in its magnitude.
    """

    def __init__(self, col: str, kind: str):
        super().__init__([col])
        self.kind = kind
        self.name = f"dig_{kind}__{col}"

    def transform(self, df, ctx):
        x = df[self.parents[0]].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            if self.kind == "frac":
                return x - np.floor(x)
            if self.kind == "mod10":
                return np.mod(np.round(x), 10)
            if self.kind == "mod100":
                return np.mod(np.round(x), 100)
            if self.kind == "frac100":   # cents
                return np.mod(np.round(x * 100), 100)
        raise ValueError(self.kind)


def geo_points(cols: Sequence[str]) -> List[tuple]:
    """(lat, lon) column pairs found by name: ``*lat*`` matched with the same name using ``lon``/``lng``."""
    import re
    out = []
    low = {c.lower(): c for c in cols}
    for c in cols:
        lc = c.lower()
        if "lat" not in lc:
            continue
        for rep in ("lon", "lng", "long"):
            for pat in ("latitude", "lat"):
                if pat in lc:
                    cand = lc.replace(pat, "longitude" if rep == "lon" and pat == "latitude" else rep)
                    if cand in low and low[cand] != c:
                        out.append((c, low[cand]))
                        break
            else:
                continue
            break
    return list(dict.fromkeys(out))


class GeoPair(Spec):
    """Distance / bearing between two (lat, lon) points (haversine, km)."""

    def __init__(self, p: tuple, q: tuple, kind: str):
        super().__init__([p[0], p[1], q[0], q[1]])
        self.kind = kind
        self.name = f"geo_{kind}__{p[0]}__{q[0]}"

    def transform(self, df, ctx):
        la1, lo1, la2, lo2 = (np.radians(df[c].to_numpy(dtype=float)) for c in self.parents)
        dla, dlo = la2 - la1, lo2 - lo1
        with np.errstate(all="ignore"):
            if self.kind == "hav":
                a = np.sin(dla / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlo / 2) ** 2
                return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            if self.kind == "manh":
                return 6371.0 * (np.abs(dla) + np.abs(dlo) * np.cos((la1 + la2) / 2))
            if self.kind == "bearing":
                yb = np.sin(dlo) * np.cos(la2)
                xb = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(dlo)
                return np.degrees(np.arctan2(yb, xb))
        raise ValueError(self.kind)


class Rotate(Spec):
    """Coordinates rotated by a fixed angle: oblique boundaries become axis-aligned."""

    def __init__(self, a: str, b: str, deg: int):
        super().__init__([a, b])
        self.deg = deg
        self.name = f"rot{deg}__{a}__{b}"

    def transform(self, df, ctx):
        a = df[self.parents[0]].to_numpy(dtype=float)
        b = df[self.parents[1]].to_numpy(dtype=float)
        t = np.radians(self.deg)
        return a * np.cos(t) + b * np.sin(t)


# ============================================================================
# LightGBM helpers
# ============================================================================

def _lgb_objective(task: str, n_classes: int) -> dict:
    if task == "regression":
        return {"objective": "regression", "metric": "l2"}
    if task == "binary":
        return {"objective": "binary", "metric": "binary_logloss"}
    return {"objective": "multiclass", "metric": "multi_logloss", "num_class": n_classes}


def _row_loss(task, y, margin) -> np.ndarray:
    """Per-row loss of raw margins in the LightGBM objective's own units."""
    y = np.asarray(y)
    if task == "regression":
        return (y - margin) ** 2
    if task == "binary":
        p = 1 / (1 + np.exp(-margin))
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    m = margin - margin.max(axis=1, keepdims=True)
    logp = m - np.log(np.exp(m).sum(axis=1, keepdims=True))
    return -logp[np.arange(len(y)), y.astype(int)]


def _loss(task, y, margin) -> float:
    return float(np.mean(_row_loss(task, y, margin)))


# ============================================================================
# FeatureForge
# ============================================================================

class LinResid(Spec):
    """What the other numerics do not explain about one column (label-free).

    ``target`` is regressed (ridge) on the other ``cols`` and the residual is
    returned. Whole weight minus what the component weights predict, price
    minus what size and age predict: a linear combination over many columns
    that trees only approximate with many splits, and often the quantity that
    matters. ``log`` works on log1p of the non-negative columns
    (multiplicative structure: density, price per unit of size).
    """

    def __init__(self, cols: Sequence[str], target: str, log: bool = False):
        super().__init__([target] + [c for c in cols if c != target])
        self.log = log
        self.name = f"{'lresid' if log else 'resid'}__{target}"

    def _matrix(self, df, fit=False):
        M = df[self.parents].to_numpy(dtype=float)
        if fit:
            fin = np.where(np.isfinite(M), M, np.nan)
            self.log_ = (np.nanmin(fin, axis=0) >= 0) & self.log
        M = np.where(self.log_, np.log1p(np.where(self.log_, np.maximum(M, 0), 0)), M)
        if fit:
            self.lo_, self.hi_ = np.nanpercentile(M, 1, axis=0), np.nanpercentile(M, 99, axis=0)
        M = np.clip(M, self.lo_, self.hi_)
        if fit:
            self.med_ = np.nanmedian(M, axis=0)
        miss = ~np.isfinite(M[:, 0])
        M = np.where(np.isnan(M), self.med_, M)
        if fit:
            self.mu_, self.sd_ = M.mean(0), M.std(0) + 1e-12
        Z = (M - self.mu_) / self.sd_
        return Z, miss

    def fit(self, df, y, ctx):
        from sklearn.linear_model import Ridge
        Z, miss = self._matrix(df, fit=True)
        self.model_ = Ridge(alpha=1.0).fit(Z[~miss, 1:], Z[~miss, 0])
        return self

    def transform(self, df, ctx):
        Z, miss = self._matrix(df)
        r = Z[:, 0] - self.model_.predict(Z[:, 1:])
        r[miss] = np.nan
        return r


class Expr(Spec):
    """Arithmetic expression tree over numeric columns, found by the genetic search.

    ``tree`` is a column name or ``(op, left, right)`` with op in
    add / sub / mul / div. Depth-3 trees express ratios of sums, products of
    ratios and similar compound interactions that pairwise Arith cannot.
    """

    def __init__(self, tree):
        self.tree = _expr_canon(tree)
        super().__init__(sorted(set(_expr_leaves(self.tree))))
        self.name = "gp__" + _expr_str(self.tree)

    def transform(self, df, ctx):
        r = _expr_eval(self.tree, df)
        r = np.asarray(r, dtype=float).copy()
        r[~np.isfinite(r)] = np.nan
        return r


def _expr_leaves(t):
    return [t] if isinstance(t, str) else _expr_leaves(t[1]) + _expr_leaves(t[2])


def _expr_str(t):
    if isinstance(t, str):
        return t
    sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[t[0]]
    return f"({_expr_str(t[1])}{sym}{_expr_str(t[2])})"


def _expr_canon(t):
    if isinstance(t, str):
        return t
    a, b = _expr_canon(t[1]), _expr_canon(t[2])
    if t[0] in ("add", "mul") and _expr_str(b) < _expr_str(a):
        a, b = b, a
    return (t[0], a, b)


def _expr_depth(t):
    return 0 if isinstance(t, str) else 1 + max(_expr_depth(t[1]), _expr_depth(t[2]))


def _expr_eval(t, df):
    if isinstance(t, str):
        return df[t].to_numpy(dtype=float)
    a, b = _expr_eval(t[1], df), _expr_eval(t[2], df)
    with np.errstate(all="ignore"):
        if t[0] == "add":
            return a + b
        if t[0] == "sub":
            return a - b
        if t[0] == "mul":
            return a * b
        return a / np.where(b == 0, np.nan, b)


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
    nested_cv : bool | "auto"
        Recompute target encodings inside each CV fold when ranking feature sets,
        so search CV is not optimistic about sparse-key encodings. "auto": on with
        time-blocked search, where encodings fitted on random folds carry the
        validation block's labels (IEEE-CIS latest window 0.948 -> 0.953).
    gate_subsets : bool
        Also offer each round's label-free features alone to the gate.
    fold_avg_te : bool
        New rows get target encodings averaged over the out-of-fold encoders
        instead of one encoder fitted on all rows, matching the shrinkage of
        the training values (matters for sparse, many-level keys).
    entities : bool
        Look for hidden entity ids: a timestamp-like column minus a "days since
        X" column is constant per entity (account opening date, first visit);
        such anchors are added as columns and combined with ID-like columns
        into composite keys for counts, target maps and group statistics.
    entity_nums, entity_lags : int, bool
        Numerics aggregated per entity; whether to add each row's place in its
        entity's history (time since previous / until next row, rows before it)
        when a time column is known. Lags were neutral on IEEE-CIS, so off.
    group_col : str | "auto" | None
        Column whose test values are new groups (users, patients). "auto": a
        repeating integer / categorical column with under 20% of the unlabeled
        rows' values seen in training. The gate, search folds, screening split
        and out-of-fold encodings then split whole groups, and target maps over
        that column are not generated.
    lagged_te : bool
        With a time column and unlabeled rows later in time: an entity's (and
        entity x flag's) mean target over a trailing window ending one test
        horizon before each row. Off: on Rossmann the full-horizon lag made them
        stale (hand-made features + these: RMSPE 0.122 -> 0.146).
    time_cv : bool
        With a time column, search CV uses contiguous time blocks as folds and
        screening is measured on the latest rows, so features that only
        interpolate within a period are not selected.
    events : bool
        With a time column: time since / until each entity's last / next row
        with a given level of a low-cardinality column (days since the last
        promotion, until the next holiday). Date columns (datetime or ISO date
        strings) are always expanded into calendar fields as candidates.
    family_nb : bool
        Within a family of 8+ similar numeric columns (Santander's var_0..199),
        add per-column out-of-fold target maps over (value bin, value count)
        and their sum, a naive-Bayes score. Off: it lowered Santander's AUC
        from 0.918 to 0.914 next to the label-free family counts.
    text : bool
        Free-text columns (strings of mostly distinct values, several words each:
        titles, descriptions) get label-free writing statistics and TF-IDF SVD
        topics, and an out-of-fold sparse linear model on their words and character
        n-grams (alone, and all text plus one-hot keys) as candidates.
    time_col : str | "auto" | None
        Column that orders rows in time. When set, the gate holds out the most
        recent rows instead of a random sample, so features that only work
        within a period (neighbours of the same day, encodings of entities that
        stop appearing) are judged as they will be on a later test set. "auto"
        picks a numeric column whose unlabeled values (``X_unlabeled``) lie
        beyond the training range, as a competition's test period does.
    n_composite : int
        Key pairs (most interacting first) whose groups get numeric aggregations.
    gate_bags : int
        Models averaged per feature set on the gate (different seeds and
        early-stopping splits), to keep model noise out of the decision.
    evolve_time : float
        Seconds of genetic interaction search per round (0 disables): populations
        of key/binned-numeric cells (target-encoded) and arithmetic expression
        trees evolve under the novel residual gain, seeded from the mined pairs.
    n_interactions : int
        Number of feature pairs (and half as many triples) mined from the base
        model's tree paths and expanded into explicit interaction features.
    top_numeric, top_keys : int
        How many of the most important numeric / key columns seed candidates.
    gate_frac : float
        Fraction of rows held out (before search) to confirm the final gain.
        0 disables the gate.
    gate_z : float
        Required paired z-score of the (winsorised) per-row gate improvement
        (1.0 ~ 84% one-sided confidence; raised to 1.645 when the gate has fewer
        than 1000 rows). Protects small tables from noise-driven adds.
    """

    def __init__(self, task: Optional[str] = None, log_target: bool = False,
                 time_budget: float = 300, n_rounds: int = 3, max_new_features: int = 60,
                 top_numeric: int = 24, top_keys: int = 10, max_key_cardinality: int = 200,
                 arith_ops: Sequence[str] = ARITH_OPS, group_stats: Sequence[str] = GROUP_STATS,
                 gate_frac: float = 0.2, gate_z: float = 1.0, min_rel_gain: float = 0.001, cv: int = 3,
                 hc_threshold: int = 32, n_interactions: int = 40, novelty_slack: float = 1.0,
                 evolve_time: float = 0.0, gate_bags: int = 1, n_composite: int = 0, fold_avg_te: bool = False,
                 gate_subsets: bool = False, nested_cv="auto", entities: bool = True,
                 time_col: Optional[str] = "auto", entity_nums: int = 6, entity_lags: bool = False,
                 family_nb: bool = False, events: bool = True, time_cv: bool = True,
                 lagged_te: bool = False, group_col: Optional[str] = "auto", text: bool = True, drop_synthetic: bool = True,
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
        self.gate_z = gate_z
        self.min_rel_gain = min_rel_gain
        self.cv = cv
        self.hc_threshold = hc_threshold
        self.n_interactions = n_interactions
        self.novelty_slack = novelty_slack
        self.evolve_time = evolve_time
        self.gate_bags = gate_bags
        self.n_composite = n_composite
        self.fold_avg_te = fold_avg_te
        self.gate_subsets = gate_subsets
        self.nested_cv = nested_cv
        self.entities = entities
        self.time_col = time_col
        self.entity_nums = entity_nums
        self.entity_lags = entity_lags
        self.family_nb = family_nb
        self.events = events
        self.time_cv = time_cv
        self.lagged_te = lagged_te
        self.group_col = group_col
        self.text = text
        self.drop_synthetic = drop_synthetic
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
        g = getattr(self, "_groups", {}).get(n_rows)
        if g is not None:
            # Whole groups (users) per fold: test rows belong to unseen groups.
            levels, codes = np.unique(g, return_inverse=True)
            fold_of = np.random.default_rng(seed).permutation(len(levels)) % k
            f = fold_of[codes]
            return [(np.flatnonzero(f != i), np.flatnonzero(f == i)) for i in range(k)]
        if self.task_ == "regression":
            return list(KFold(k, shuffle=True, random_state=seed).split(np.zeros(n_rows)))
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(n_rows), y))

    def _model_frame(self, X: pd.DataFrame, recode: Optional[bool] = None) -> pd.DataFrame:
        """Model view of a frame: categoricals as ``category`` dtype, or, when the
        high-cardinality recode is active, those columns as frequency-rank numbers."""
        recode = self.recode_ if recode is None else recode
        X = X.copy()
        for c in self.cat_cols_:
            if c not in X.columns:
                continue
            if recode and c in self.rank_maps_:
                X[c] = self._rank(X[c], c)
            else:
                X[c] = pd.Categorical(_as_str(X[c]), categories=self.cat_levels_[c])
        return X

    def _fit_rank_maps(self, X: pd.DataFrame):
        """Frequency rank of each level of the high-cardinality categoricals (1 = most common)."""
        self.rank_maps_ = {}
        for c in self.hc_cols_:
            vc = _as_str(X[c]).value_counts()
            order = sorted(vc.index, key=lambda v: (-vc[v], v))
            self.rank_maps_[c] = pd.Series(np.arange(1, len(order) + 1, dtype=float), index=order)

    def _rank(self, s: pd.Series, c: str) -> np.ndarray:
        return self.rank_maps_[c].reindex(_as_str(s).to_numpy()).to_numpy(dtype=float)

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

    @staticmethod
    def _mine_paths(booster, names, pairs, triples, max_trees=300):
        """Accumulate split gain of parent->child (and grandparent chains) feature pairs.

        Features that a tree splits on in sequence along one path interact in
        the model; their gain is a cheap, model-guided interaction ranking.
        """
        dump = booster.dump_model(num_iteration=booster.best_iteration or None)
        for tree in dump["tree_info"][:max_trees]:
            stack = [(tree["tree_structure"], None, None)]
            while stack:
                node, par, gpar = stack.pop()
                if "split_feature" not in node:
                    continue
                f = names[node["split_feature"]]
                g = float(node.get("split_gain", 0.0))
                if par is not None and par != f:
                    key = tuple(sorted((par, f)))
                    pairs[key] = pairs.get(key, 0.0) + g
                    if gpar is not None and len({gpar, par, f}) == 3:
                        k3 = tuple(sorted((gpar, par, f)))
                        triples[k3] = triples.get(k3, 0.0) + g
                for side in ("left_child", "right_child"):
                    stack.append((node[side], f, par))

    def _nested_values(self, spec, W, y, key, tr, va):
        """Encodings of one CV fold computed from its training rows only.

        Out-of-fold target statistics over the whole search frame still let a
        training row's value depend on the validation rows' labels; for sparse
        keys that makes CV favour target encodings that do not generalise.
        Here the training rows get an inner out-of-fold encoding and the
        validation rows an encoding fitted on the training rows alone.
        """
        ck = (spec.name,) + key
        if ck not in self._nested_cache:
            import copy
            s = copy.copy(spec)
            inner = self._folds(len(tr), y[tr], 5, self.random_state + 7)
            vt = s.fit_transform_oof(W.iloc[tr].reset_index(drop=True), y[tr], self.ctx_, inner)
            vv = s.transform(W.iloc[va].reset_index(drop=True), self.ctx_)
            self._nested_cache[ck] = (np.asarray(vt, dtype=np.float32), np.asarray(vv, dtype=np.float32))
        return self._nested_cache[ck]

    def _cv(self, X: pd.DataFrame, y, repeats, lr=0.1, mine=False, nested=None):
        """Repeated K-fold: OOF raw margins (averaged over repeats), mean loss, summed gain importance.

        ``nested`` maps target-encoding specs to the frame they were built on;
        their columns are then recomputed per fold from the fold's training rows.
        """
        n = len(X)
        if mine:
            self.pair_gain_, self.triple_gain_ = {}, {}
        oof_mean = np.zeros((n, self.n_classes_)) if self.task_ == "multiclass" else np.zeros(n)
        imp = pd.Series(0.0, index=X.columns)
        losses = []
        for r_i, folds in enumerate(repeats):
            oof = np.zeros_like(oof_mean)
            for f_i, (tr, va) in enumerate(folds):
                Xtr, Xva = X.iloc[tr], X.iloc[va]
                specs = [sp for sp in (nested or {}).get("specs", []) if sp.out_names()[0] in X.columns]
                if specs:
                    Xtr, Xva = Xtr.copy(), Xva.copy()
                    for sp in specs:
                        vt, vv = self._nested_values(sp, nested["W"], y, (r_i, f_i), tr, va)
                        for j, col in enumerate(sp.out_names()):
                            Xtr[col] = vt if vt.ndim == 1 else vt[:, j]
                            Xva[col] = vv if vv.ndim == 1 else vv[:, j]
                b = self._fit_eval(Xtr, y[tr], Xva, y[va], lr=lr)
                oof[va] = b.predict(Xva, num_iteration=b.best_iteration, raw_score=True)
                imp += pd.Series(b.feature_importance("gain"), index=X.columns)
                if mine:
                    self._mine_paths(b, list(X.columns), self.pair_gain_, self.triple_gain_)
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
        existing = set(W.columns) | {sp.name for sp in selected}
        cands: List[Spec] = []

        time_keys = {self.time_col_, self.group_col_, *self.date_cols_, *getattr(self, "time_like_", [])} - {None}

        def add(spec):
            if spec.target_dep and time_keys & set(spec.parents) and not getattr(spec, "time_safe", False):
                return
            if spec.name not in existing:
                existing.add(spec.name)
                cands.append(spec)

        sel_names = [s.name for s in selected if s.n_out == 1 and s.name in num_cols]
        label_free = {s.name for s in selected if not s.target_dep}
        if round_idx == 0:
            raw_num = [c for c in self.base_cols_ if c not in self.cat_cols_]
            for stem, cols in column_families(raw_num).items():
                binary = all(set(pd.unique(W[c].dropna())) <= {0, 1} for c in cols)
                stats = ("argmax", "sum") if binary else ("sum", "mean", "std", "max", "min", "argmax",
                                                          "slope", "delta", "npos")
                for st in stats:
                    add(RowStat(cols, st, stem))
            for stem, cols in column_families(raw_num, min_size=8).items():
                cols = [c for c in cols if W[c].nunique() > 20]
                if len(cols) >= 8:
                    add(FamilyCount(cols, "count", stem))
                    add(FamilyCount(cols, "mask", stem))
                    if self.family_nb and self.n_classes_ <= 2:
                        add(FamilyNB(cols, stem))
            # Multi-valued fields stored in numbered slots (components, specs, items).
            for stem, cols, pair in slot_families(W[self.base_cols_], self.cat_cols_):
                add(SlotBag(cols, pair, stem))
                if self.n_classes_ <= 2:
                    add(SlotTE(cols, pair, stem))
            if len(raw_num) >= 3:
                add(RowStat(raw_num, "nonzero", "all"))
                if W[raw_num].isna().any().any():
                    add(RowStat(raw_num, "nan", "all"))
            dense_num = [c for c in top_num if W[c].nunique() > 10]
            if len(W) <= 300_000:
                sizes = sorted({d for d in (2, 4, 8, 16) if d <= len(dense_num)} | {min(len(dense_num), 32)})
                for d in sizes:
                    if d >= 2:
                        ks = (5, 20, 100) if self.n_classes_ <= 2 else (10, 50)
                        add(KNNTarget(dense_num[:d], ks, self.n_classes_, f"top{d}"))
                        if d >= 4:
                            # Importance-weighted metric: distance follows what the model uses.
                            w = np.sqrt(imp.reindex(dense_num[:d]).clip(lower=0).to_numpy() + 1e-12)
                            add(KNNTarget(dense_num[:d], ks, self.n_classes_, f"w{d}", w / w.mean()))
                            add(KNNTarget(dense_num[:d], ks, self.n_classes_, f"rw{d}", w / w.mean(), rank=True))
                            if 2 <= self.n_classes_ <= 10 and len(W) <= 200_000:
                                add(KNNClassDist(dense_num[:d], (1, 2, 4), self.n_classes_, f"w{d}", w / w.mean()))
                for d in (8, 24):
                    if len(dense_num) >= max(4, d // 2) and (d == 8 or len(dense_num) > 8):
                        add(Projection(dense_num[:d], "pca", 4, self.n_classes_, f"top{d}"))
                        add(Projection(dense_num[:d], "pls", 3, self.n_classes_, f"top{d}"))
                if len(dense_num) >= 4:
                    for c in dense_num[:8]:
                        add(LinResid(dense_num[:16], c))
                        add(LinResid(dense_num[:16], c, log=True))
                if len(dense_num) >= 3 and len(W) <= 200_000 and self.n_classes_ <= 2:
                    add(LinearOOF(dense_num[:16], self.n_classes_, "top"))
            # Same-scale sums of 3-4 columns (total area, total spend, ...).
            mag = {c: np.log10(np.nanmedian(np.abs(W[c].to_numpy(dtype=float))) + 1e-9) for c in dense_num[:10]}
            for m in (3, 4):
                for combo in combinations(dense_num[:10], m):
                    if max(mag[c] for c in combo) - min(mag[c] for c in combo) <= 1.0:
                        add(RowStat(list(combo), "sum", "+".join(combo)))
            for a, b in combinations(dense_num[:10], 2):
                add(BinnedPairTE(a, b, self.n_classes_))
            # Geography: distances/bearings between points, rotations, kNN on the map.
            pts = [p for p in geo_points(raw_num) if p[0] in W.columns]
            for p, q in combinations(pts, 2):
                for kind in ("hav", "manh", "bearing"):
                    add(GeoPair(p, q, kind))
            for la, lo in pts:
                for deg in (15, 30, 60, 75, 105, 120, 150, 165):
                    add(Rotate(la, lo, deg))
                if len(W) <= 300_000:
                    ks = (5, 20, 100) if self.n_classes_ <= 2 else (10, 50)
                    add(KNNTarget([la, lo], ks, self.n_classes_, f"geo_{la}"))
            # Free text: how it is written, its topics, and a sparse linear model on its words.
            for c in self.text_cols_:
                add(TextStats(c))
                add(TextSVD(c))
                add(TextLinearOOF([c], self.n_classes_))
            if self.text_cols_:
                tk = [k for k in keys_rank if k in self.base_cols_ and W[k].nunique() > 2][:12]
                add(TextLinearOOF(self.text_cols_, self.n_classes_, keys=tk, label="text+keys"))
            # Shrunken linear model over one-hot keys and all key pairs.
            lin_keys = [k for k in keys_rank if W[k].nunique() > 2]
            if len(lin_keys) >= 2 and len(W) <= 500_000:
                add(CrossLinearOOF(lin_keys[:8], self.n_classes_, pairs=True))
                add(CrossLinearOOF(lin_keys[:20], self.n_classes_, pairs=False))
            # Key x numeric-band target maps.
            for k in top_keys[:8]:
                for c in dense_num[:8]:
                    if c != k:
                        add(KeyBinTE(k, c, self.n_classes_))
            # Numerics as categories: target encoding of exact values.
            for c in top_num[:12]:
                nu = W[c].nunique()
                if c not in self.key_cols_ and 10 < nu <= len(W) // 4:
                    add(TargetEnc([c], self.n_classes_))
            # How the value is written.
            for c in top_num[:12]:
                x = W[c].to_numpy(dtype=float)
                fin = x[np.isfinite(x)]
                if len(fin) == 0 or W[c].nunique() <= 20:
                    continue
                if np.any(fin != np.round(fin)):
                    add(Digits(c, "frac"))
                    add(Digits(c, "frac100"))
                elif np.nanmax(np.abs(fin)) >= 100:
                    add(Digits(c, "mod10"))
                    add(Digits(c, "mod100"))
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
            # Composite-key aggregations: statistics of the strongest numerics within
            # the groups of the most interacting key pairs.
            kp = [p for p in sorted(getattr(self, "pair_gain_", {}), key=self.pair_gain_.get, reverse=True)
                  if all(c in self.key_cols_ for c in p)][:self.n_composite]
            for kk in kp:
                for c in [c for c in top_num if c not in kk and c not in self.cat_cols_][:6]:
                    for st in ("mean", "dev", "z", "nunique"):
                        add(GroupStat(kk, c, st))
            for c in top_num:
                if c not in self.key_cols_:
                    add(Count([c]))
            for k in top_keys:
                for c in top_num:
                    if c == k:
                        continue
                    for st in self.group_stats:
                        add(GroupStat(k, c, st))
            for c in self.date_cols_:
                for part in _DATE_PARTS:
                    add(DatePart(c, part))
            self._event_candidates(W, imp, add)
            self._entity_candidates(W, imp, top_num, add)
            self._interaction_candidates(W, add)
        else:
            # Compose selected features with the strongest raw columns.
            partners = top_num[: max(8, self.top_numeric // 2)]
            for s in sel_names:
                for c in partners:
                    if c == s:
                        continue
                    for op in self.arith_ops:
                        add(Arith(op, s, c))
                # Group statistics of an out-of-fold target feature would leak: other
                # rows' encodings were fitted on this row's label.
                if s in label_free:
                    for k in top_keys[:6]:
                        for st in ("mean", "dev", "z"):
                            add(GroupStat(k, s, st))
            for k1, k2, k3 in list(combinations(top_keys[:6], 3)):
                add(Count([k1, k2, k3]))
                add(TargetEnc([k1, k2, k3], self.n_classes_))
        return cands

    def _entity_candidates(self, W, imp, top_num, add):
        """Counts, target maps and group statistics over ID-like columns and over
        composite entities (ID x anchor, ID pair x anchor)."""
        ids = sorted(self.id_cols_, key=lambda c: -imp.get(c, 0.0))[:4]
        if not ids:
            return
        nums = [c for c in top_num if c not in ids and c not in self.cat_cols_
                and not c.startswith("anchor__")][:self.entity_nums]
        ents = [(k,) for k in ids]
        for _, _, _, a in self.anchors_:
            ents += [(k, a) for k in ids[:3]]
            ents += [(k1, k2, a) for k1, k2 in combinations(ids[:3], 2)]
        t = getattr(self, "time_col_", None)
        for e in ents:
            if self.entity_lags and t is not None and t not in e:
                tw = W[t].to_numpy(dtype=float)
                R = float(np.nanmax(tw) - np.nanmin(tw))
                wins = [f"{p}{R * f:.6g}" for f in (0.001, 0.01, 0.05) for p in ("win", "fwd")] if R > 0 else []
                for kind in ("prev", "next", "nth", *wins):
                    add(EntityLag(e if len(e) > 1 else e[0], t, kind))
            if len(e) > 1:
                add(Count(list(e)))
            add(TargetEnc(list(e), self.n_classes_))
            for c in nums:
                if c in e:
                    continue
                for st in ("mean", "dev", "std", "nunique"):
                    add(GroupStat(e if len(e) > 1 else e[0], c, st))

    def _lagged_target_candidates(self, W, ids, flags, add):
        """Trailing-window target means per entity (and entity x flag), lagged by the
        test horizon."""
        t = self.time_col_
        if self.horizon_ <= 0 or self.n_classes_ > 2 or not ids:
            return
        tw = W[t].to_numpy(dtype=float)
        R = float(np.nanmax(tw) - np.nanmin(tw))
        for f in (0.03, 0.1, 0.25):
            w = R * f
            for k in ids:
                add(LaggedTargetMean([k], t, w, self.horizon_))
                for fl in flags:
                    add(LaggedTargetMean([k, fl], t, w, self.horizon_))

    def _event_candidates(self, W, imp, add, max_levels=4):
        """With a time column: time since / until each entity's last / next row with
        a given level of a low-cardinality column (promotions, holidays, closures)."""
        t = getattr(self, "time_col_", None)
        if t is None or not self.events:
            return
        ids = sorted(self.id_cols_, key=lambda c: -imp.get(c, 0.0))[:2]
        keysets = [[k] for k in ids] or [[]]
        WU = W if self.U_search_ is None else pd.concat([W[self.raw_cols_], self.U_search_], ignore_index=True)
        if self.lagged_te and ids:
            flags = [f for f in self.base_cols_ if f != t and f not in ids and f not in self.date_cols_
                     and 2 <= W[f].nunique() <= 7
                     and W.groupby(ids[0], sort=False)[f].nunique().mean() >= 1.5][:3]
            self._lagged_target_candidates(W, ids[:2], flags, add)
        for f in self.base_cols_:
            if f == t or f in ids or f in self.date_cols_:
                continue
            vc = WU[f].value_counts(normalize=True)
            if not 2 <= len(vc) <= 6:
                continue
            for keys in keysets:
                # The flag must change within entities (not a fixed store attribute).
                if keys and WU.groupby(keys[0], sort=False)[f].nunique().mean() < 1.5:
                    continue
                for lv, share in list(vc.items())[:max_levels]:
                    if not 0.001 < share < 0.999:
                        continue
                    lv = lv if f in self.cat_cols_ else float(lv)
                    for kind in ("prev", "next"):
                        add(EventRecency(keys, t, f, lv, kind))

    def _cell_gain(self, cell, ncell, g, h, lam, fold, K, y, margin, base):
        """Cross-fitted Newton gain of a lookup table over integer cells."""
        idx = fold * ncell + cell
        def step_for(gk, hk):
            G = np.bincount(idx, gk, minlength=K * ncell).reshape(K, ncell)
            H = np.bincount(idx, hk, minlength=K * ncell).reshape(K, ncell)
            return (-(G.sum(0) - G) / (H.sum(0) - H + lam))[fold, cell]
        if g.ndim == 1:
            corr = step_for(g, h)
        else:
            corr = np.stack([step_for(g[:, k], h[:, k]) for k in range(g.shape[1])], 1)
        return base - _loss(self.task_, y, margin + corr)

    def _fast_interactions(self, W, y, margin, imp, n_cols=30, n_bins=8, n_levels=16):
        """FAST-style interaction detection on the current model's residuals.

        Every pair of the top columns is cut into a 2-D grid (quantile bins for
        numerics, frequent levels for keys) and scored by the cross-fitted
        Newton gain of the grid minus the better of its two 1-D gains: what a
        pairwise lookup adds that neither column explains alone, measured on
        rows the lookup did not see. Triples extend the best pairs the same way.
        """
        cols = [c for c in imp.sort_values(ascending=False).index if c in W.columns][:n_cols]
        n = len(y)
        K = 5
        fold = np.random.default_rng(self.random_state + 7).permutation(n) % K
        g, h = self._grad_hess(y, margin)
        lam = 10.0 * float(h.mean())
        base = _loss(self.task_, y, margin)
        codes, sizes = {}, {}
        for c in cols:
            if c in self.key_cols_ and (c in self.cat_cols_ or W[c].nunique() <= n_levels):
                v = pd.Series(_key_values(W[c]))
                top = v.value_counts().index[:n_levels]
                code = pd.Categorical(v, categories=top).codes.astype(np.int64)
                code[code < 0] = len(top)
                codes[c], sizes[c] = code, len(top) + 1
            elif c not in self.cat_cols_:
                x = W[c].to_numpy(dtype=float)
                fin = np.isfinite(x)
                if fin.sum() < 20:
                    continue
                e = np.unique(np.nanquantile(x[fin], np.linspace(0, 1, n_bins + 1)[1:-1]))
                codes[c] = np.where(fin, np.searchsorted(e, np.where(fin, x, 0.0), side="right"), len(e) + 1)
                sizes[c] = len(e) + 2
        args = (g, h, lam, fold, K, y, margin, base)
        one = {c: self._cell_gain(codes[c], sizes[c], *args) for c in codes}
        pairs = {}
        for a, b in combinations(list(codes), 2):
            pairs[(a, b)] = self._cell_gain(codes[a] * sizes[b] + codes[b], sizes[a] * sizes[b], *args) \
                - max(one[a], one[b], 0.0)
        best = sorted(pairs, key=pairs.get, reverse=True)
        triples = {}
        for a, b in best[:10]:
            if pairs[(a, b)] <= 0:
                break
            ab = codes[a] * sizes[b] + codes[b]
            for c in list(codes)[:15]:
                if c in (a, b):
                    continue
                t = tuple(sorted((a, b, c)))
                if t in triples:
                    continue
                triples[t] = self._cell_gain(ab * sizes[c] + codes[c], sizes[a] * sizes[b] * sizes[c], *args) \
                    - max(pairs[(a, b)] + max(one[a], one[b], 0.0), 0.0)
        return ({k: v for k, v in pairs.items() if v > 0}, {k: v for k, v in triples.items() if v > 0})

    def _interaction_candidates(self, W, add):
        """Turn tree-path interactions into explicit features of every applicable family."""
        if not self.n_interactions or not getattr(self, "pair_gain_", None):
            return
        is_key = lambda c: c in self.key_cols_
        dense = lambda c: c not in self.cat_cols_ and W[c].nunique() > 10
        half = self.n_interactions // 2
        fp, ft = getattr(self, "fast_pairs_", {}), getattr(self, "fast_triples_", {})
        pairs = list(dict.fromkeys(
            [tuple(sorted(p)) for p in sorted(fp, key=fp.get, reverse=True)[:half]]
            + sorted(self.pair_gain_, key=self.pair_gain_.get, reverse=True)))[:self.n_interactions]
        for a, b in pairs:
            if dense(a) and dense(b):
                for op in self.arith_ops:
                    add(Arith(op, a, b))
                add(BinnedPairTE(a, b, self.n_classes_))
                add(MixedCount([a, b], [a, b], 16))
            elif is_key(a) and is_key(b):
                add(Count([a, b]))
                add(TargetEnc([a, b], self.n_classes_))
            else:
                k, c = (a, b) if is_key(a) else (b, a)
                binned = [x for x in (a, b) if not is_key(x)]
                if not is_key(k) or any(x in self.cat_cols_ for x in binned):
                    continue
                add(MixedTE([k, c], binned, self.n_classes_, 8))
                add(MixedCount([k, c], binned, 8))
                if c not in self.cat_cols_:
                    for st in self.group_stats:
                        add(GroupStat(k, c, st))
        triples = list(dict.fromkeys(
            sorted(ft, key=ft.get, reverse=True)[:half // 2]
            + sorted(self.triple_gain_, key=self.triple_gain_.get, reverse=True)))[:half]
        for t in triples:
            binned = [c for c in t if not is_key(c) or (c not in self.cat_cols_ and W[c].nunique() > 16)]
            if any(c in self.cat_cols_ and c in binned for c in t):
                continue
            add(MixedTE(list(t), binned, self.n_classes_, 6))
            add(MixedCount(list(t), binned, 6))

    def _materialize(self, specs: List[Spec], W: pd.DataFrame, y, folds, keep_fn=None) -> Dict[str, np.ndarray]:
        out = {}
        # Label-free statistics (counts, group statistics, ...) over every row whose
        # features are known: search rows plus gate and unlabeled rows, as at the end.
        WU = None
        if getattr(self, "U_search_", None) is not None:
            WU = pd.concat([W[self.raw_cols_], self.U_search_], ignore_index=True)
            raw = set(self.raw_cols_)
        for s in specs:
            try:
                if s.target_dep:
                    v = s.fit_transform_oof(W, y, self.ctx_, folds)
                elif WU is not None and set(s.parents) <= raw:
                    v = s.fit(WU, None, self.ctx_).transform(W, self.ctx_)
                else:
                    v = s.fit(W, y, self.ctx_).transform(W, self.ctx_)
            except Exception:
                continue
            v = np.asarray(v, dtype=np.float32)
            v2 = v if v.ndim == 2 else v[:, None]
            col = v2[:, 0]
            finite = np.isfinite(col)
            if finite.mean() < 0.05 or np.nanstd(np.where(finite, col, np.nan)) == 0:
                continue
            if keep_fn is not None and not keep_fn(s, v, out):
                continue
            out[s.name] = v
        _CODE_CACHE.clear()  # factorised keys of large frames add up to gigabytes
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

    def _probe_setup(self, W, y, margin):
        """Cross-fitted residual probe for the current margin, plus each column's own gain."""
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
        return probe, parent_gain

    def _evolve(self, W, y, margin, imp, folds, budget, existing):
        """Genetic interaction search under the novel residual gain.

        Two genomes evolve side by side: column sets of 2-4 keys / binned
        numerics (expressed as an out-of-fold target-encoded cell) and
        arithmetic expression trees of depth <= 3 over numerics. Fitness is
        the cross-fitted residual gain beyond the best single parent, the same
        criterion screening uses, so the population climbs toward high-order
        interactions the current model has not found. The initial populations
        are seeded from the pairs and triples mined from the model's trees.
        Returns ``{name: (spec, values)}`` for the fittest distinct individuals.
        """
        t_end = time.time() + budget
        rng = np.random.default_rng(self.random_state + 17)
        probe, pg = self._probe_setup(W, y, margin)
        order = [c for c in imp.sort_values(ascending=False).index if c in W.columns]
        nums = [c for c in order if c not in self.cat_cols_ and W[c].nunique() > 10][:16]
        setcols = [c for c in order if c in self.key_cols_ or c not in self.cat_cols_][:20]
        nunq = {c: W[c].nunique() for c in setcols}
        out, fit = {}, {}

        def evaluate(spec, parents):
            if spec.name in fit or spec.name in existing:
                return fit.get(spec.name, -np.inf)
            try:
                v = spec.fit_transform_oof(W, y, self.ctx_, folds) if spec.target_dep \
                    else spec.fit(W, y, self.ctx_).transform(W, self.ctx_)
                v = np.asarray(v, dtype=np.float32)
                col = v if v.ndim == 1 else v[:, 0]
                if np.isfinite(col).mean() < 0.05 or len(np.unique(np.round(col[np.isfinite(col)][:2000], 6))) < 3:
                    raise ValueError
                f = probe(v) - max(pg.get(p, 0.0) for p in parents)
            except Exception:
                f, v = -np.inf, None
            fit[spec.name] = f
            if v is not None and f > 0:
                out[spec.name] = (spec, v)
            return f

        def set_spec(cols):
            cols = sorted(set(cols))
            binned = [c for c in cols if c not in self.cat_cols_ and (c not in self.key_cols_ or nunq[c] > 16)]
            return MixedTE(cols, binned, self.n_classes_, 8 if len(cols) <= 2 else (6 if len(cols) == 3 else 4))

        def mutate_set(cols):
            cols = list(cols)
            r = rng.random()
            if (r < 0.4 and len(cols) < 4) or len(cols) < 2:
                cols.append(setcols[int(rng.integers(len(setcols)))])
            elif r < 0.6 and len(cols) > 2:
                cols.pop(int(rng.integers(len(cols))))
            else:
                cols[int(rng.integers(len(cols)))] = setcols[int(rng.integers(len(setcols)))]
            return tuple(sorted(set(cols)))

        ops = ("add", "sub", "mul", "div")

        def rand_leaf():
            return nums[int(rng.integers(len(nums)))]

        def subtrees(t, path=()):
            yield path, t
            if not isinstance(t, str):
                yield from subtrees(t[1], path + (1,))
                yield from subtrees(t[2], path + (2,))

        def replace(t, path, new):
            if not path:
                return new
            t = list(t)
            t[path[0]] = replace(t[path[0]], path[1:], new)
            return tuple(t)

        def mutate_expr(t):
            subs = list(subtrees(t))
            path, node = subs[int(rng.integers(len(subs)))]
            r = rng.random()
            if r < 0.4:
                new = (ops[int(rng.integers(4))], node, rand_leaf())
                if rng.random() < 0.5:
                    new = (new[0], new[2], new[1])
            elif r < 0.7 or isinstance(node, str):
                new = rand_leaf() if isinstance(node, str) else (ops[int(rng.integers(4))], node[1], node[2])
            else:
                new = node[1] if rng.random() < 0.5 else node[2]
            return replace(t, path, new)

        def cross_expr(a, b):
            pa, _ = list(subtrees(a))[int(rng.integers(sum(1 for _ in subtrees(a))))]
            _, nb = list(subtrees(b))[int(rng.integers(sum(1 for _ in subtrees(b))))]
            return replace(a, pa, nb)

        # Seeds: mined pairs / triples, then random fill.
        mined = sorted(getattr(self, "pair_gain_", {}).items(), key=lambda kv: -kv[1])[:20]
        mined += sorted(getattr(self, "triple_gain_", {}).items(), key=lambda kv: -kv[1])[:10]
        mined += sorted(getattr(self, "fast_pairs_", {}).items(), key=lambda kv: -kv[1])[:10]
        P = 24
        sets, exprs = [], []
        for cols, _ in mined:
            cols = tuple(sorted(set(c for c in cols if c in nunq)))
            if len(cols) >= 2 and cols not in sets:
                sets.append(cols)
            nc = [c for c in cols if c in nums]
            if len(nc) >= 2:
                exprs.append((ops[int(rng.integers(4))], nc[0], nc[1]))
        while len(sets) < P and len(setcols) >= 2:
            sets.append(tuple(sorted(set(rng.choice(setcols, size=int(rng.integers(2, 4)), replace=False)))))
        while len(exprs) < P and len(nums) >= 2:
            a, b = rng.choice(nums, size=2, replace=False)
            exprs.append((ops[int(rng.integers(4))], str(a), str(b)))
        sets, exprs = sets[:P], exprs[:P]
        sfit = {c: evaluate(set_spec(c), c) for c in sets if time.time() < t_end}
        efit = {_expr_canon(e): evaluate(Expr(e), _expr_leaves(e)) for e in exprs if time.time() < t_end}

        def pick(pop):
            keys = list(pop)
            cand = [keys[int(rng.integers(len(keys)))] for _ in range(3)]
            return max(cand, key=pop.get)

        gens = 0
        while time.time() < t_end and (sfit or efit):
            gens += 1
            for _ in range(P):
                if time.time() >= t_end:
                    break
                if len(sfit) >= 2:
                    a = pick(sfit)
                    child = tuple(sorted(set(a) | set(pick(sfit)))) if rng.random() < 0.3 else a
                    if len(child) > 4:
                        child = tuple(sorted(rng.choice(child, size=4, replace=False)))
                    for _try in range(8):  # re-mutate until the child is new
                        if child != a and len(child) >= 2 and set_spec(child).name not in fit:
                            break
                        child = mutate_set(child)
                    if len(child) >= 2 and set_spec(child).name not in fit:
                        sfit[child] = evaluate(set_spec(child), child)
                if len(efit) >= 2:
                    a = pick(efit)
                    child = cross_expr(a, pick(efit)) if rng.random() < 0.4 else a
                    for _try in range(8):
                        child = _expr_canon(child)
                        if (child != a and not isinstance(child, str) and _expr_depth(child) <= 3
                                and Expr(child).name not in fit):
                            break
                        child = mutate_expr(child if not isinstance(child, str) else a)
                    child = _expr_canon(child)
                    if not isinstance(child, str) and _expr_depth(child) <= 3 and Expr(child).name not in fit:
                        efit[child] = evaluate(Expr(child), _expr_leaves(child))
            # Survivors: the fittest P of each population (elitist).
            sfit = dict(sorted(sfit.items(), key=lambda kv: -kv[1])[:P])
            efit = dict(sorted(efit.items(), key=lambda kv: -kv[1])[:P])
        # Populations converge on near-copies of one winner: keep distinct ones.
        best, kept = [], []
        samp = rng.permutation(len(y))[:5000]
        for nm in sorted(out, key=lambda nm: -fit[nm]):
            v = out[nm][1]
            x = (v if v.ndim == 1 else v[:, 0])[samp].astype(float)
            x = pd.Series(x).rank().to_numpy()
            x = np.where(np.isnan(x), np.nanmean(x), x)
            if x.std() == 0 or any(abs(np.corrcoef(x, k)[0, 1]) > 0.95 for k in kept):
                continue
            best.append(nm)
            kept.append(x)
            if len(best) >= 30:
                break
        self._log(f"  genetic search: {len(fit)} individuals over {gens} generations, "
                  f"{len(out)} with novel gain, top {len(best)} kept: "
                  + ", ".join(f"{nm}={fit[nm]:.2g}" for nm in best[:4]))
        return {nm: out[nm] for nm in best}

    def _spec_novelty(self, spec, v, probe, parent_gain):
        """(gain, novel gain) of a candidate under the residual probe."""
        if getattr(spec, "paired", False):
            # Block of per-column transforms: each output against its own parent, summed.
            gs = [probe(v[:, j]) for j in range(v.shape[1])]
            nov = sum(max(g - self.novelty_slack * parent_gain.get(p, 0.0), 0.0) for g, p in zip(gs, spec.parents))
            return sum(max(g, 0.0) for g in gs), nov
        g = probe(v)
        parents = getattr(spec, "novelty_parents", None) or spec.parents
        return g, g - self.novelty_slack * max(parent_gain.get(p, 0.0) for p in parents)

    def _screen(self, values: Dict[str, np.ndarray], specs: Dict[str, Spec], W: pd.DataFrame,
                y, margin, idx_a, idx_b, keep: int, scores=None) -> List[str]:
        """Rank candidates by *novel* residual gain.

        A candidate's raw residual gain is compared with the gain of its own
        parent columns under the same probe: an early-stopped model leaves some
        residual signal in existing columns, and any re-expression of such a
        column (``a + const``-like transforms, group statistics of it, ...) would
        otherwise look useful. Only gain beyond the best parent counts.
        """
        n = len(y)
        scores = dict(scores or {})
        missing = [nm for nm in values if nm not in scores]
        if missing:
            probe, parent_gain = self._probe_setup(W, y, margin)
            for name in missing:
                scores[name] = self._spec_novelty(specs[name], values[name], probe, parent_gain)
        gains = {nm: g for nm, (g, _) in scores.items()}
        novelty = {nm: v for nm, (_, v) in scores.items()}
        ranked = sorted([nm for nm in novelty if nm in values], key=novelty.get, reverse=True)
        alive = [nm for nm in ranked if novelty[nm] > 0][:keep]
        self._log(f"  screened {len(novelty)} candidates on {n} rows (5-fold cross-fitted) "
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
        self.n_synthetic_ = 0
        if X_unlabeled is not None and self.drop_synthetic:
            fake = synthetic_rows(X, X_unlabeled)
            if fake.any():
                self.n_synthetic_ = int(fake.sum())
                self._log(f"unlabeled rows: {self.n_synthetic_} of {len(fake)} look synthetic "
                          "(no value unique among all rows); left out of label-free statistics")
                X_unlabeled = X_unlabeled[~fake]
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
        self.date_cols_ = [c for c in X.columns if _is_date_like(X[c])]
        for c in self.date_cols_:
            X[c] = _to_days(X[c])
        self.cat_cols_ = [c for c in X.columns if _is_cat(X[c])]
        for c in self.cat_cols_:
            X[c] = _as_str(X[c])
        self.cat_levels_ = {c: pd.Index(sorted(pd.unique(X[c]))) for c in self.cat_cols_}
        self.text_cols_ = [c for c in self.cat_cols_ if _is_text(X[c])] if self.text else []
        # Native categorical splits on many-level columns overfit; frequency-rank codes
        # are the usual contest alternative. Which one wins is measured, not assumed.
        self.hc_cols_ = [c for c in self.cat_cols_ if X[c].nunique() > self.hc_threshold]
        self.recode_ = False
        self.rank_maps_ = {}
        self.base_cols_ = list(X.columns)
        self.id_cols_ = self._id_columns(X)
        self.anchors_ = self._find_anchors(X) if self.entities else []
        X = self._add_anchors(X)
        self.raw_cols_ = list(X.columns)
        self.key_cols_ = [c for c in X.columns
                          if 2 <= X[c].nunique() <= (0.5 * len(X) if c in self.cat_cols_
                                                     else self.max_key_cardinality)]
        self.key_cols_ += [a[3] for a in self.anchors_]
        self.key_cols_ = [c for c in self.key_cols_ if c not in self.text_cols_]
        if self.text_cols_:
            self._log(f"free text: {', '.join(self.text_cols_)}")
        self.ctx_ = Context(self.task_, self.n_classes_, self.random_state)
        self.ctx_.fold_avg_te = self.fold_avg_te

        # Gate split: these rows never influence the search.
        n = len(X)
        idx = np.arange(n)
        strat = y_np if self.task_ != "regression" else None
        self.time_col_ = self._detect_time(X, X_unlabeled)
        self.horizon_ = 0.0
        self._groups = {}
        self.group_col_ = None if self.time_col_ is not None else self._detect_group(X, X_unlabeled)
        if self.group_col_ is not None:
            self._log(f"test rows are new {self.group_col_} groups: grouped gate, folds and screening")
        if self.time_col_ is not None:
            # Levels of the time column never recur later: no target maps over it (label-
            # free counts and group statistics per date still carry over to test dates).
            self.id_cols_ = [c for c in self.id_cols_ if c != self.time_col_]
            self._log(f"time-ordered gate on {self.time_col_}")
        if self.gate_frac and n >= 200 and self.time_col_ is not None:
            tx = X[self.time_col_].to_numpy(dtype=float)
            order = np.argsort(tx, kind="stable")
            n_gate = int(round(self.gate_frac * n))
            if X_unlabeled is not None and self.time_col_ in X_unlabeled:
                # Hold out the latest rows covering the test period's horizon: a gate
                # that reaches much further back can hinge on one season (a holiday
                # peak just before it) that the real test period never sees.
                tu = self._prep(X_unlabeled)[self.time_col_].to_numpy(dtype=float)
                horizon = np.nanmax(tu) - np.nanmax(tx)
                self.horizon_ = float(horizon) if np.isfinite(horizon) else 0.0
                if np.isfinite(horizon) and horizon > 0:
                    n_h = int(np.sum(tx > np.nanmax(tx) - horizon))
                    n_gate = int(np.clip(n_h, 0.05 * n, n_gate))
            idx_sel, idx_gate = np.sort(order[:n - n_gate]), np.sort(order[n - n_gate:])
        elif self.gate_frac and n >= 200 and self.group_col_ is not None:
            gx = X[self.group_col_].to_numpy()
            levels, codes = np.unique(gx.astype(str) if gx.dtype == object else gx, return_inverse=True)
            in_gate = np.random.default_rng(self.random_state).random(len(levels)) < self.gate_frac
            idx_sel, idx_gate = np.flatnonzero(~in_gate[codes]), np.flatnonzero(in_gate[codes])
        elif self.gate_frac and n >= 200:
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
        if self.group_col_ is not None:
            gx = X[self.group_col_].astype(str).to_numpy()
            self._groups = {len(W): gx[idx_sel], n: gx}
        self.U_search_ = None
        if X_unlabeled is not None:
            self.U_search_ = pd.concat([X.iloc[idx_gate], self._prep(X_unlabeled)[self.raw_cols_]], ignore_index=True)
        self.ctx_.extra_rows = self.U_search_
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
        if self.group_col_ is not None:
            f0 = self._folds(len(W), yW, 10, self.random_state + 3)
            idx_b = np.sort(np.concatenate([va for _, va in f0[:3]]))
            idx_a = np.setdiff1d(np.arange(len(W)), idx_b)
        if self.time_col_ is not None and self.time_cv:
            # Rows close in time share their period's level, so random folds reward
            # features that interpolate between neighbouring days. Contiguous time
            # blocks as folds, and screening measured on the latest rows, reward only
            # what carries over to other periods, as the test period needs.
            order = np.argsort(W[self.time_col_].to_numpy(dtype=float), kind="stable")
            blocks = np.array_split(order, self.cv)
            folds = [[(np.sort(np.concatenate(blocks[:i] + blocks[i + 1:])), np.sort(b))
                      for i, b in enumerate(blocks)]]
            n_b = int(round(0.3 * len(W)))
            idx_a, idx_b = np.sort(order[:-n_b]), np.sort(order[-n_b:])

        selected: List[Spec] = []
        self.history_ = []
        self._nested_cache = {}
        Wm = self._model_frame(W)
        margin, cur_loss, imp = self._cv(Wm, yW, folds, mine=self.n_interactions > 0)
        if self.n_interactions:
            self.fast_pairs_, self.fast_triples_ = self._fast_interactions(W, yW, margin, imp)
        self.base_cv_loss_ = cur_loss
        if self.hc_cols_:
            self._fit_rank_maps(W)
            Wr = self._model_frame(W, recode=True)
            m_r, loss_r, imp_r = self._cv(Wr, yW, folds)
            self._log(f"high-cardinality recode of {len(self.hc_cols_)} columns: CV loss "
                      f"{cur_loss:.6f} -> {loss_r:.6f} ({100 * (cur_loss - loss_r) / cur_loss:+.2f}%)")
            if loss_r < cur_loss * (1 - self.min_rel_gain):
                self.recode_ = True
                Wm, margin, cur_loss, imp = Wr, m_r, loss_r, imp_r
        self._log(f"task={self.task_} rows={n} (search {len(W)}, gate {len(idx_gate)}) "
                  f"cols={X.shape[1]} base CV loss={cur_loss:.6f}")

        for r in range(self.n_rounds):
            if self._time_left() <= 0 or len(selected) >= self.max_new_features:
                break
            values = joint = None  # release the previous round's candidates first
            cands = self._generate(W, imp, selected, r)
            keep = min(80, 3 * self.max_new_features)
            spec_by_name = {s.name: s for s in cands}
            # Screen while materialising: only candidates with novel residual gain are
            # kept in memory (thousands of full-length columns otherwise).
            probe, parent_gain = self._probe_setup(W, yW, margin)
            scores = {}

            def keep_fn(spec, v, out):
                g, nov = self._spec_novelty(spec, v, probe, parent_gain)
                scores[spec.name] = (g, nov)
                if nov <= 0:
                    return False
                if len(out) >= 4 * keep:
                    worst = min(out, key=lambda nm: scores[nm][1])
                    if scores[worst][1] >= nov:
                        return False
                    del out[worst]
                return True
            values = self._materialize(cands, W, yW, te_folds, keep_fn=keep_fn)
            if self.evolve_time > 0 and r == 0:
                evolved = self._evolve(W, yW, margin, imp, te_folds, self.evolve_time,
                                       {s.name for s in cands} | {s.name for s in selected})
                for nm, (sp, v) in evolved.items():
                    cands.append(sp)
                    values[nm] = v
            self._log(f"round {r + 1}: {len(cands)} candidates generated, {len(scores)} valid")
            if not values:
                break
            spec_by_name = {s.name: s for s in cands}
            survivors = self._screen(values, spec_by_name, W, yW, margin, idx_a, idx_b,
                                     keep=keep, scores=scores)
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

            nested = None
            if self.nested_cv is True or (self.nested_cv == "auto" and self.time_col_ is not None and self.time_cv):
                te_specs = [sp for sp in list(selected) + [spec_by_name[nm] for nm in survivors]
                            if isinstance(sp, TargetEnc)]
                nested = {"specs": te_specs, "W": W}
            room = self.max_new_features - len(selected)
            # Two orderings: joint-model split gain, and novel residual gain from
            # screening. Wide candidate pools can push strong but narrow features
            # (a single pairwise interaction) down the split-gain order.
            novel_rank = sorted(survivors, key=lambda nm: -self._last_novelty.get(nm, 0.0))
            orders = [("gain", rank, (3, 6, 12, 25, 50, 80)), ("novelty", novel_rank, (3, 6, 12))]
            best_k, best_loss, best_fit, best_rank = 0, cur_loss, None, rank
            tried = set()
            for label, order, steps in orders:
                ladder = [k for k in steps if k < min(len(order), room)]
                if label == "gain":
                    ladder.append(min(len(order), room))
                for k in sorted(set(ladder)):
                    if k <= 0:
                        continue
                    key = frozenset(order[:k])
                    if key in tried:
                        continue
                    tried.add(key)
                    cols = list(Wm.columns) + [c for nm in order[:k] for c in spec_by_name[nm].out_names()]
                    oof_k, loss_k, imp_k = self._cv(joint[cols], yW, folds, nested=nested)
                    self._log(f"  {label} top-{k:<3d} CV loss={loss_k:.6f} ({100 * (cur_loss - loss_k) / cur_loss:+.2f}%)")
                    if loss_k < best_loss:
                        best_k, best_loss, best_fit, best_rank = k, loss_k, (oof_k, imp_k, cols), order
                    if self._time_left() < 0:
                        break
            rank = best_rank
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
        # Candidate values of the last round can be gigabytes on large tables.
        values = joint = cands = None
        import gc
        gc.collect()
        self.gate_passed_ = None
        if (selected or self.recode_ or self.anchors_) and len(idx_gate):
            best_set, self.gate_raw_loss_, self.gate_fe_loss_ = self._gate(X, y_np, idx_sel, idx_gate, W)
            self.gate_passed_ = best_set is not None
            if self.gate_passed_:
                self.selected_ = list(best_set)
            else:
                self.recode_ = False
            self._log(f"gate: raw={self.gate_raw_loss_:.6f} fe={self.gate_fe_loss_:.6f} "
                      f"({100 * (self.gate_raw_loss_ - self.gate_fe_loss_) / self.gate_raw_loss_:+.2f}%) "
                      f"-> {'PASS' if self.gate_passed_ else 'REJECT'}")
            if not self.gate_passed_:
                self.selected_ = []
                dropped = {a[3] for a in self.anchors_}
                self.raw_cols_ = [c for c in self.raw_cols_ if c not in dropped]
                self.anchors_ = []

        # Refit every spec's statistics on all training rows.
        self.U_search_ = None
        self._fit_full(X, y_np, None if X_unlabeled is None else self._prep(X_unlabeled))
        self.elapsed_ = time.time() - self._t0
        self._log(f"done: {len(self.selected_)} features added in {self.elapsed_:.1f}s")
        return self

    def _gate(self, X, y, idx_sel, idx_gate, W):
        """Score raw vs. engineered feature sets on the held-out gate rows.

        ``W`` holds the search rows with the selected features as computed during
        the search (target features out of fold); every selected spec is still
        fitted on exactly those rows, so the gate rows only need ``transform``.
        """
        Xg = X.iloc[idx_gate].reset_index(drop=True)
        ys, yg = y[idx_sel], y[idx_gate]
        Fs, Fg = W, Xg.copy()
        for s in self.selected_:
            vg = s.transform(Fg, self.ctx_)
            for j, col in enumerate(s.out_names()):
                Fg[col] = vg if np.ndim(vg) == 1 else vg[:, j]
        import lightgbm as lgb
        es_splits = [self._folds(len(Fs), ys, 5, self.random_state + 11 + k)[0] for k in range(self.gate_bags)]
        if self.time_col_ is not None:
            # Early-stop on the latest search rows too, so the gate's models are tuned
            # for later periods as the competition's are.
            order = np.argsort(Fs[self.time_col_].to_numpy(dtype=float), kind="stable")
            n_va = max(1, len(order) // 5)
            es_splits = [(np.sort(order[:-n_va]), np.sort(order[-n_va:]))] * self.gate_bags

        def gate_loss(cols, recode=None):
            # A bag of differently seeded / early-stopped models: one model's
            # randomness otherwise flips keep-or-drop decisions on small gates.
            A, G = self._model_frame(Fs[cols], recode), self._model_frame(Fg[cols], recode)
            margin = 0.0
            for k, (tr, va) in enumerate(es_splits):
                b = self._fit_eval(A.iloc[tr], ys[tr], A.iloc[va], ys[va], lr=0.05, es=100)
                # Refit on all search rows at the early-stopped size, then score the gate rows.
                full = lgb.train(self._lgb_params(0.05, seed=self.random_state + k), lgb.Dataset(A, ys),
                                 max(1, b.best_iteration))
                margin = margin + full.predict(G, raw_score=True) / len(es_splits)
            return _row_loss(self.task_, yg, margin)

        # The baseline is always the raw columns as given (native categoricals).
        raw_rows = gate_loss(self.base_cols_, recode=False)
        raw_l = float(raw_rows.mean())
        # Small gates are noisy: demand 95% one-sided confidence below 1000 rows.
        z_needed = self.gate_z if len(yg) >= 1000 or self.gate_z < 0 else max(self.gate_z, 1.645)
        # Candidate feature sets are the cumulative rounds; the best one on the gate is
        # kept only if its paired per-row improvement clears ``gate_z`` standard errors.
        best_set, best_l, best_z = None, raw_l, 0.0
        # Candidate sets: the cumulative rounds; round -1 is the high-cardinality recode
        # alone. Each round also offers its label-free subset: target statistics are the
        # features whose search CV can be optimistic (training rows' encodings carry the
        # validation rows' labels), so a gate can reject them without losing the rest.
        cands = [("anchors", [])] if self.anchors_ else []
        for r in ([-1] if self.recode_ else []) + sorted({s.round_ for s in self.selected_}):
            full = [s for s in self.selected_ if s.round_ <= r]
            cands.append((f"rounds<={r + 1}", full))
            if self.gate_subsets:
                free = [s for s in full if not s.target_dep]
                if free and len(free) < len(full):
                    cands.append((f"rounds<={r + 1} label-free", free))
        for label, specs in cands:
            cols = self.raw_cols_ + [c for s in specs for c in s.out_names()]
            rows = gate_loss(cols)
            d = raw_rows - rows
            # Winsorise so a handful of extreme rows cannot carry the decision.
            lo, hi = np.quantile(d, [0.01, 0.99])
            dw = np.clip(d, lo, hi)
            z = float(dw.mean() / (dw.std(ddof=1) / np.sqrt(len(dw)) + 1e-300))
            self._log(f"  gate {label}{' (recoded)' if self.recode_ else ''}: loss={rows.mean():.6f} vs raw {raw_l:.6f} (z={z:+.2f}, need {z_needed:.2f})")
            if rows.mean() < best_l and z >= z_needed:
                best_set, best_l, best_z = specs, float(rows.mean()), z
        return best_set, raw_l, best_l

    def _fit_full(self, X, y, U=None):
        if self.recode_:
            self._fit_rank_maps(X if U is None else pd.concat([X, U[self.raw_cols_]], ignore_index=True))
        folds = self._folds(len(X), y, 5, self.random_state + 1)
        self.ctx_.extra_rows = None if U is None else U[self.raw_cols_]
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
        self.new_columns_ = [a[3] for a in self.anchors_] + [c for s in self.selected_ for c in s.out_names()]
        self._train_frame = F[self.new_columns_].astype(np.float32)

    # ------------------------------------------------------------- transform
    def _prep(self, X):
        X = X.reset_index(drop=True).copy()
        for c in getattr(self, "date_cols_", []):
            X[c] = _to_days(X[c])
        for c in self.cat_cols_:
            X[c] = _as_str(X[c])
        return self._add_anchors(X)

    def _detect_group(self, X, U):
        """A column whose test values are (almost) all unseen in training while it
        repeats within training: users, sessions or patients, when the test set holds
        new ones. Validation must then split whole groups, and target maps over the
        column cannot help."""
        if self.group_col != "auto":
            return self.group_col if self.group_col in X.columns else None
        if U is None or len(U) < 50:
            return None
        Up = self._prep(U)
        n, best, best_ov = len(X), None, 0.2
        for c in X.columns:
            if c not in Up.columns or c in self.text_cols_:
                continue
            nu = X[c].nunique()
            if not (50 <= nu <= n / 2):
                continue
            if c not in self.cat_cols_:
                x = X[c].to_numpy(dtype=float)
                fin = x[np.isfinite(x)]
                if len(fin) < 0.9 * n or np.any(fin != np.round(fin)):
                    continue
            ov = float(Up[c].isin(set(X[c].unique())).mean())
            if ov < best_ov:
                best, best_ov = c, ov
        return best

    def _detect_time(self, X, U):
        if self.time_col != "auto":
            return self.time_col if self.time_col in X.columns else None
        if U is None or len(U) < 50:
            return None
        best, best_frac = None, 0.0
        Up = self._prep(U)
        self.time_like_ = []
        for c in X.columns:
            if c in self.cat_cols_ or c not in Up.columns:
                continue
            # Dates qualify at any granularity (many rows share a day); other numerics
            # need many levels, or test values entirely beyond the training range.
            nu = X[c].nunique()
            dense = c in self.date_cols_ or nu >= 0.05 * len(X)
            if not dense and nu < 20:
                continue
            x, u = X[c].to_numpy(dtype=float), Up[c].to_numpy(dtype=float)
            if np.isfinite(x).mean() < 0.99 or np.isfinite(u).mean() < 0.99:
                continue
            # Test rows later than (almost) every training row.
            frac = float(np.mean(u > np.nanquantile(x, 0.99)))
            if frac > 0.9:
                self.time_like_.append(c)  # week numbers, period ids: test levels are all new
            if frac > (0.9 if dense else 0.98) and (frac, c in self.date_cols_) > (best_frac, False):
                best, best_frac = c, frac
        return best

    # ------------------------------------------------------------- entities
    def _id_columns(self, X, cap=8):
        """Integer or categorical columns with many levels that repeat: card numbers,
        addresses, customer or store ids. Most levels first."""
        n, out = len(X), []
        for c in X.columns:
            if c in getattr(self, "date_cols_", []) or c in getattr(self, "text_cols_", []):
                continue
            nu = X[c].nunique()
            if not (100 <= nu <= n / 5):
                continue
            vc = X[c].value_counts(normalize=True)
            if vc.iloc[0] > 0.5:  # mostly one value: a count or flag, not an id
                continue
            if c not in self.cat_cols_:
                x = X[c].to_numpy(dtype=float)
                fin = x[np.isfinite(x)]
                if len(fin) < 0.5 * n or np.any(fin != np.round(fin)):
                    continue
                # Counts and amounts get rarer as they grow; codes do not.
                if pd.Series(vc.index.to_numpy(dtype=float)).corr(pd.Series(vc.to_numpy(dtype=float)),
                                                                  method="spearman") < -0.3:
                    continue
            out.append((nu, c))
        return [c for _, c in sorted(out, reverse=True)][:cap]

    def _find_anchors(self, X, max_anchors=3, max_ratio=0.8):
        """Find (time, scale, delta) triples where ``floor(time / scale) - delta`` is
        constant within entities, as an account-opening day is while "days since
        opening" keeps growing. Test: grouped by an ID-like column, ``t - delta``
        takes clearly fewer distinct values than the control ``t + delta``; for a
        delta unrelated to time the two counts match."""
        n = len(X)
        if n < 2000 or not self.id_cols_:
            return []
        self._t0 = getattr(self, "_t0", time.time())
        S = X.sample(min(n, 100_000), random_state=self.random_state) if n > 100_000 else X
        num = [c for c in X.columns if c not in self.cat_cols_]
        vals = {c: S[c].to_numpy(dtype=float) for c in num}
        def is_int(v):
            f = v[np.isfinite(v)]
            return len(f) > 0 and not np.any(f != np.round(f))
        times = [c for c in num if X[c].nunique() > 0.2 * n and np.nanmin(vals[c]) >= 0]
        deltas = sorted([c for c in num if is_int(vals[c]) and 50 <= X[c].nunique() <= 0.2 * n],
                        key=lambda c: -X[c].nunique())[:80]
        ids = []
        for c in self.id_cols_[:3]:
            v = S[c]
            ids.append(pd.factorize(v)[0].astype(np.int64) if c in self.cat_cols_
                       else np.nan_to_num(v.to_numpy(dtype=float), nan=-1).astype(np.int64))
        def n_pairs(idc, v):
            ok = np.isfinite(v)
            if ok.sum() < 0.3 * len(v):
                return 0
            code = idc[ok] * (1 << 32) + (v[ok].astype(np.int64) - int(v[ok].min()))
            return len(np.unique(code))
        found = []
        for t in times:
            tv = vals[t]
            for s in (1, 60, 3600, 86400, 604800):
                span = (np.nanmax(tv) - np.nanmin(tv)) / s
                if span < 10:
                    continue
                for d in deltas:
                    if d == t:
                        continue
                    dv = vals[d]
                    if np.nanmax(dv) - np.nanmin(dv) < 0.5 * span:
                        continue
                    a = np.floor(tv / s) - dv
                    ratios = []
                    for idc in ids:
                        nd = n_pairs(idc, a + 2 * dv)
                        if nd:
                            ratios.append(n_pairs(idc, a) / nd)
                    if ratios and min(ratios) < max_ratio:
                        found.append((min(ratios) / np.isfinite(dv).mean(), t, s, d))
        found.sort()
        out, used = [], set()
        for r, t, s, d in found:
            if d in used:
                continue
            used.add(d)
            out.append((t, s, d, f"anchor__{d}__{t}_{s}"))
            self._log(f"entity anchor: floor({t}/{s}) - {d} (score {r:.2f})")
            if len(out) >= max_anchors:
                break
        return out

    def _add_anchors(self, X):
        for t, s, d, name in getattr(self, "anchors_", []):
            X[name] = np.floor(X[t].to_numpy(dtype=float) / s) - X[d].to_numpy(dtype=float)
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add features to new rows using statistics from all training rows."""
        F = self._prep(X)
        for s in self.selected_:
            v = s.transform(F, self.ctx_)
            for j, col in enumerate(s.out_names()):
                F[col] = v if np.ndim(v) == 1 else v[:, j]
        out = self._recode_out(X)
        for c in self.new_columns_:
            out[c] = F[c].astype(np.float32).to_numpy()
        return out

    def _recode_out(self, X):
        out = X.reset_index(drop=True).copy()
        if self.recode_:
            for c in self.rank_maps_:
                out[c] = self._rank(out[c], c)
        return out

    def transform_train(self, X: pd.DataFrame) -> pd.DataFrame:
        """Training rows with features as fitted (target encodings out-of-fold)."""
        out = self._recode_out(X)
        if len(out) != len(self._train_frame):
            raise ValueError("transform_train expects the exact frame passed to fit")
        for c in self.new_columns_:
            out[c] = self._train_frame[c].to_numpy()
        return out

    def fit_transform(self, X, y, X_unlabeled=None):
        return self.fit(X, y, X_unlabeled).transform_train(X)

    def report(self) -> pd.DataFrame:
        return pd.DataFrame(self.history_)
