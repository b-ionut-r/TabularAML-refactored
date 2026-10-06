"""Automatic aggregation of related tables onto the main table.

Many competitions ship one row per entity (a loan application, a customer) plus
child tables with many rows per entity (past loans, payments, card balances).
Most of the winning features in such contests are aggregations of the children:
counts, means and extremes of every column, shares of each category, and the
same over the most recent records, often of *within-row* comparisons first
(paid minus owed, days late = paid day minus due day).

``RelatedTables`` does this generically:

1. Within each child row, differences and ratios of columns that share a unit
   stem (``AMT_PAYMENT`` vs ``AMT_INSTALMENT``, ``DAYS_ENTRY_PAYMENT`` vs
   ``DAYS_INSTALMENT``).
2. Grandchildren are aggregated onto their child first (``bureau_balance`` onto
   ``bureau``), so chains of any depth collapse onto the main table.
3. Per main-table key: row count; mean / max / min / sum / std of numerics;
   share of each frequent level and number of distinct levels of categoricals;
   optionally the same over each entity's most recent rows when a time column
   is named.

The output is a label-free frame keyed like the main table; FeatureForge then
selects among the columns and searches interactions on top of them.

    rt = RelatedTables([
        Child("bureau", bureau, key="SK_ID_CURR", time="DAYS_CREDIT",
              children=[Child("bb", bureau_balance, key="SK_ID_BUREAU")]),
        Child("prev", previous_application, key="SK_ID_CURR", time="DAYS_DECISION"),
    ])
    X = rt.join(app, key="SK_ID_CURR")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional, Sequence

import re

import numpy as np
import pandas as pd


def _safe(name: str) -> str:
    """Column names that every model library accepts."""
    for a, b in ((":", "__"), ("/", "_div_"), ("-", "_sub_"), ("=", "_is_")):
        name = name.replace(a, b)
    return re.sub(r"[^0-9A-Za-z_]+", "_", name)


@dataclass
class Child:
    name: str
    df: pd.DataFrame
    key: str                       # column linking a row to its parent
    time: Optional[str] = None     # larger = more recent
    children: List["Child"] = field(default_factory=list)
    drop: Sequence[str] = ()       # ids that are not features (other foreign keys)


def _stem(c: str) -> str:
    return c.split("_")[0].upper()


def _row_pairs(df: pd.DataFrame, num: Sequence[str], max_pairs: int) -> pd.DataFrame:
    """Differences and ratios of same-unit numeric columns within each row."""
    out = {}
    by_stem = {}
    for c in num:
        by_stem.setdefault(_stem(c), []).append(c)
    pairs = []
    for stem, cols in by_stem.items():
        if len(cols) < 2 or len(stem) < 2:
            continue
        # Prefer the most complete columns of each stem.
        cols = sorted(cols, key=lambda c: -df[c].notna().mean())[:6]
        pairs += list(combinations(cols, 2))
    for a, b in pairs[:max_pairs]:
        x, y = df[a].to_numpy(dtype=float), df[b].to_numpy(dtype=float)
        out[f"{a}-{b}"] = x - y
        with np.errstate(all="ignore"):
            r = x / np.where(y == 0, np.nan, y)
        r[~np.isfinite(r)] = np.nan
        out[f"{a}/{b}"] = r
    return pd.DataFrame(out, index=df.index)


class RelatedTables:
    def __init__(self, children: Sequence[Child], stats=("mean", "max", "min", "sum", "std"),
                 recent: int = 3, top_levels: int = 8, max_pairs: int = 12, row_pairs: bool = True):
        self.children = list(children)
        self.stats = tuple(stats)
        self.recent = recent
        self.top_levels = top_levels
        self.max_pairs = max_pairs
        self.row_pairs = row_pairs

    def _flatten(self, ch: Child) -> pd.DataFrame:
        """The child table with its own children already aggregated onto it."""
        df = ch.df
        for g in ch.children:
            parent_key = g.key
            agg = self._aggregate(g, parent_key)
            df = df.merge(agg, left_on=parent_key, right_index=True, how="left")
        return df

    def _aggregate(self, ch: Child, key: str) -> pd.DataFrame:
        df = self._flatten(ch)
        ids = {key, *ch.drop} | {g.key for g in ch.children if g.key != key}
        feats = [c for c in df.columns if c not in ids]
        cats = [c for c in feats if df[c].dtype == object or isinstance(df[c].dtype, pd.CategoricalDtype)
                or pd.api.types.is_string_dtype(df[c]) or df[c].dtype == bool]
        num = [c for c in feats if c not in cats and pd.api.types.is_numeric_dtype(df[c])]
        parts = [df[[key]]]
        if num:
            parts.append(df[num].astype(np.float32))
        if self.row_pairs and len(num) >= 2:
            parts.append(_row_pairs(df, num, self.max_pairs).astype(np.float32))
        for c in cats:
            s = df[c].astype(str)
            for lv in s.value_counts().index[:self.top_levels]:
                parts.append(pd.DataFrame({f"{c}={lv}": (s == lv).astype(np.float32)}, index=df.index))
        W = pd.concat(parts, axis=1)
        vals = [c for c in W.columns if c != key]
        dense = [c for c in vals if "=" not in c]
        onehot = [c for c in vals if "=" in c]
        g = W.groupby(key, sort=False)
        out = [g.size().rename("count").to_frame()]
        if dense:
            a = g[dense].agg(list(self.stats))
            a.columns = [f"{c}_{s}" for c, s in a.columns]
            out.append(a)
        if onehot:
            out.append(g[onehot].mean())
        for c in cats:
            out.append(df.groupby(key, sort=False)[c].nunique().rename(f"{c}_nunique").to_frame())
        if ch.time is not None and self.recent and dense:
            # Rank within the entity: rows sorted by time, most recent first.
            r = W.assign(_t=df[ch.time].to_numpy()).sort_values([key, "_t"], ascending=[True, False])
            r = r[r.groupby(key, sort=False).cumcount() < self.recent]
            a = r.groupby(key, sort=False)[dense].mean()
            a.columns = [f"{c}_last{self.recent}" for c in a.columns]
            out.append(a)
        res = pd.concat(out, axis=1)
        res.columns = [_safe(f"{ch.name}:{c}") for c in res.columns]
        res = res.loc[:, ~res.columns.duplicated()]
        # Drop constant and almost-empty columns.
        keep = [c for c in res.columns if res[c].notna().mean() > 0.01 and res[c].nunique() > 1]
        return res[keep].astype(np.float32)

    def features(self) -> pd.DataFrame:
        """Aggregations of every child, indexed by the main table's key."""
        frames = [self._aggregate(ch, ch.key) for ch in self.children]
        return pd.concat(frames, axis=1)

    def join(self, main: pd.DataFrame, key: str) -> pd.DataFrame:
        F = self.features()
        out = main.merge(F, left_on=key, right_index=True, how="left")
        for ch in self.children:
            c = _safe(f"{ch.name}:count")
            if c in out:
                out[c] = out[c].fillna(0)
        return out
