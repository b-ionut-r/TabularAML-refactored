"""Row statistics across groups of same-kind columns. Label-free, on from the data's structure.

Santander Value Prediction's 4,735 anonymised, mostly-zero columns are amounts of one kind; what predicts the target
is how a customer's row looks as a whole (how many non-zero cells, their typical size), which no single column or
pair carries and FeatureForge's search over columns never builds. Public kernels' row statistics over the non-zero
cells moved RMSLE 1.445 -> 1.356.

Structure: interchangeable columns share a name family with a running index (V1..V339, var_0..var_199) or all carry
opaque hashed names, and are sorted into kinds by what their values are (0/1 flags; mostly-zero amounts; other
amounts; signed or non-negative). A kind with at least ``min_cols`` such columns is a group of interchangeable cells, and
each row gets: the count of non-zero (and of missing) cells, and the mean, standard deviation, minimum, maximum,
median and sum of its cells (non-zero cells only for mostly-zero kinds; on log1p for non-negative kinds). Named
columns of different meaning (prices next to sizes) are never pooled; groups of a few columns are left to the search.

The statistics sit beside the file's columns, where the search's gate never judges them, so ``rows_help`` judges them
first: out-of-fold LightGBM on the training rows with and without them, kept when the paired per-row gain clears the
search gate's own bar (``FeatureForge.gate_z`` standard errors; 1.645 under 1,000 rows). Otto's 93 count columns: row
statistics cost 0.006-0.007 held-out log loss (z -4 to -6 here); Santander Customer Satisfaction -0.0005 to -0.0009
AUC (z +0.4 to +0.6); Santander Value Prediction RMSLE 1.405 -> 1.341 (z +9 to +10).
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from tabularaml.generate.forge import column_families


def _opaque(name) -> bool:
    """A hashed, meaningless column name (Santander Value's 48df886f9)."""
    return bool(re.fullmatch(r"[0-9a-f]{6,}", str(name))) and any(ch.isdigit() for ch in str(name))


def column_kinds(X: pd.DataFrame, sample: int = 200_000, min_cols: int = 10) -> Dict[str, List[str]]:
    """{kind: columns}. Interchangeable columns share a name family (V1..V339, var_0..var_199) or all carry
    opaque hashed names, and hold the same kind of value: binary | (sparse | dense) x (pos | signed)."""
    S = X.iloc[:sample]
    fam = {c: f for f, cols in column_families(list(S.columns), min_size=min_cols).items() for c in cols}
    out: Dict[str, List[str]] = {}
    for c in S.columns:
        name = "anon" if _opaque(c) else fam.get(c)
        if name is None:
            continue
        s = S[c]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        v = s.to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v) < 2 or np.all(v == v[0]):
            continue
        if np.isin(v, (0.0, 1.0)).all():
            k = "binary"
        else:
            k = ("sparse" if (v == 0).mean() >= 0.5 else "dense") + ("_pos" if v.min() >= 0 else "_signed")
        out.setdefault(f"{name}_{k}", []).append(c)
    return out


def _stats(M: np.ndarray, kind: str) -> Dict[str, np.ndarray]:
    kind = "binary" if kind.endswith("_binary") else "_".join(kind.split("_")[-2:])   # the value kind
    miss = ~np.isfinite(M)
    nz = np.isfinite(M) & (M != 0)
    out = {"nonzero": nz.sum(1).astype(float)}
    if miss.any():
        out["missing"] = miss.sum(1).astype(float)
    if kind == "binary":
        out["share"] = out["nonzero"] / np.maximum((~miss).sum(1), 1)
        return out
    V = np.where(miss, np.nan, M)
    if kind.startswith("sparse"):
        V = np.where(nz, V, np.nan)
    if kind.endswith("_pos"):
        V = np.log1p(V)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out["mean"] = np.nanmean(V, 1)
            out["std"] = np.nanstd(V, 1)
            out["min"] = np.nanmin(V, 1)
            out["max"] = np.nanmax(V, 1)
            out["median"] = np.nanmedian(V, 1)
            s = np.nansum(np.where(miss, np.nan, M), 1)
            out["sum"] = np.log1p(np.maximum(s, 0)) if kind.endswith("_pos") else s
    return out


def row_stat_features(Xtr: pd.DataFrame, Xte: pd.DataFrame, min_cols: int = 10, chunk: int = 100_000
                      ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Row statistics for every kind with ``min_cols`` or more columns (kinds read on the training rows)."""
    groups = {k: cols for k, cols in column_kinds(Xtr, min_cols=min_cols).items() if len(cols) >= min_cols}

    def build(X: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for k, cols in groups.items():
            cols = [c for c in cols if c in X.columns]
            blocks = []
            for i in range(0, len(X), chunk):
                M = X[cols].iloc[i:i + chunk].to_numpy(dtype=float)
                blocks.append(pd.DataFrame({f"rows_{k}__{n}": v for n, v in _stats(M, k).items()}))
            parts.append(pd.concat(blocks, ignore_index=True))
        return (pd.concat(parts, axis=1) if parts else pd.DataFrame(index=range(len(X)))).astype(np.float32)

    return build(Xtr), build(Xte), {k: len(v) for k, v in groups.items()}


def rows_help(X: pd.DataFrame, R: pd.DataFrame, y, task: str, z_needed: float = 1.0, folds: int = 5,
              max_rows: int = 50_000, seed: int = 0) -> Tuple[bool, float, float]:
    """(keep, gain %, z): 5-fold out-of-fold loss of a LightGBM on ``X`` against ``X`` + ``R``, paired per row
    (winsorised as the search's gate does). ``task``: regression (y already on the loss scale) | binary | multiclass."""
    import lightgbm as lgb
    from sklearn.model_selection import KFold, StratifiedKFold
    from tabularaml.generate.forge import _row_loss, _winsor
    y = np.asarray(y)
    rows = np.arange(len(y))
    if len(rows) > max_rows:
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_rows, replace=False))
    base = X.iloc[rows].reset_index(drop=True)
    base = pd.DataFrame({c: (base[c] if pd.api.types.is_numeric_dtype(base[c]) and not pd.api.types.is_bool_dtype(base[c])
                             else pd.Series(pd.factorize(base[c].astype(str))[0], dtype=float).where(base[c].notna().to_numpy()))
                         for c in base.columns})
    full = pd.concat([base, R.iloc[rows].reset_index(drop=True)], axis=1)
    y = y[rows]
    P = dict(objective=task, learning_rate=0.1, num_leaves=31, min_child_samples=20, feature_fraction=0.5,
             bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=4, seed=seed)
    if task == "multiclass":
        P["num_class"] = int(y.max()) + 1
    split = (KFold(folds, shuffle=True, random_state=seed).split(rows) if task == "regression"
             else StratifiedKFold(folds, shuffle=True, random_state=seed).split(rows, y))
    L = np.zeros((2, len(y)))
    for fi, vi in split:
        for j, F in enumerate((base, full)):
            b = lgb.train(P, lgb.Dataset(F.iloc[fi], y[fi]), 2000, valid_sets=[lgb.Dataset(F.iloc[vi], y[vi])],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            L[j, vi] = _row_loss(task, y[vi], b.predict(F.iloc[vi], raw_score=True))
    d = L[0] - L[1]
    dw = _winsor(d, None if task == "regression" else y)
    z = float(dw.mean() / (dw.std(ddof=1) / np.sqrt(len(dw)) + 1e-300))
    gain = float(100 * d.mean() / max(L[0].mean(), 1e-300))
    return z >= z_needed and d.mean() > 0, gain, z
