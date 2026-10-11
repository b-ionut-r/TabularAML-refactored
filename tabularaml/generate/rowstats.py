"""Row statistics across groups of same-kind columns. Label-free, on from the data's structure.

Santander Value Prediction's 4,735 anonymised, mostly-zero columns are amounts of one kind; what predicts the target
is how a customer's row looks as a whole (how many non-zero cells, their typical size), which no single column or
pair carries and FeatureForge's search over columns never builds. Public kernels' row statistics over the non-zero
cells moved RMSLE 1.445 -> 1.356.

Structure: numeric columns are sorted into kinds by what their values are (0/1 flags; mostly-zero amounts; other
amounts; signed or non-negative). A kind with at least ``min_cols`` columns is a group of interchangeable cells, and
each row gets: the count of non-zero (and of missing) cells, and the mean, standard deviation, minimum, maximum,
median and sum of its cells (non-zero cells only for mostly-zero kinds; on log1p for non-negative kinds). Groups of a
few columns are left to the column search.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def column_kinds(X: pd.DataFrame, sample: int = 200_000) -> Dict[str, List[str]]:
    """{kind: columns}; kind = binary | (sparse | dense) x (pos | signed)."""
    S = X.iloc[:sample]
    out: Dict[str, List[str]] = {}
    for c in S.columns:
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
        out.setdefault(k, []).append(c)
    return out


def _stats(M: np.ndarray, kind: str) -> Dict[str, np.ndarray]:
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
    groups = {k: cols for k, cols in column_kinds(Xtr).items() if len(cols) >= min_cols}

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
