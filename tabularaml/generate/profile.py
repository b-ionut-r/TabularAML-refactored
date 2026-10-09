"""Client profiles: wide label-free statistics of every row's discovered client.

Fraud, credit and churn tables log events (transactions, applications) of clients whose id is
never given. FeatureForge's hidden-entity search recovers one: an ID-like column pair plus an
*anchor*, a timestamp minus a "days since X" column that stays constant per client (IEEE-CIS:
card1 + addr1 + transaction day - D1, the winners' client id). Its other families aggregate only
a handful of columns per entity. The winners described each client by many columns at once:
the mean and spread of the amount, the counters, the match flags, the other "days since"
columns re-based to the dates they point to, and how many distinct emails or devices the client
used. A fraudster's transaction differs from the rest of its client's history, and a client
whose "account opening day" wanders is not one client.

``ClientProfile`` is one candidate block: for one composite client key, the per-client mean and
standard deviation of up to ``max_nums`` numeric columns (importance order) and of every re-based
delta column, the number of distinct values of up to ``max_cats`` categorical columns, and the
client's row count. Statistics are computed over every row whose features are known (training,
gate and unlabeled rows), as the winners computed them over train + test; labels are never used.
Rows of a client never seen get NaN. The block is screened like a column family: the novel gains
of its columns, each against its own source column, are summed.
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _key_strings(df: pd.DataFrame, keys: Sequence[str]) -> np.ndarray:
    """One string per row naming its key tuple; numbers are written from float64 so that frames
    storing a column as int, float32 or float64 agree."""
    parts = []
    for k in keys:
        v = df[k]
        v = v.astype(np.float64).astype(str) if pd.api.types.is_numeric_dtype(v) else v.astype(str)
        parts.append(v.fillna("nan"))  # pandas' string dtype keeps missing values missing
    s = parts[0]
    for p in parts[1:]:
        s = s + "|" + p
    return s.to_numpy()


class ClientProfile:
    """Per-client means / spreads / distinct counts of many columns, as one block (label-free)."""
    target_dep = False
    paired = True

    def __init__(self, keys: Sequence[str], nums: Sequence[str], cats: Sequence[str],
                 rebase: Sequence[Tuple[str, float, str]] = (), label: str = ""):
        self.keys, self.nums, self.cats, self.rebase = list(keys), list(nums), list(cats), list(rebase)
        self.parents = self.keys + self.nums + self.cats + sorted({t for t, _, _ in self.rebase} | {d for _, _, d in self.rebase})
        self.name = f"profile__{label or '+'.join(self.keys)}"
        names, src = [], []
        for c in self.nums:
            names += [f"{self.name}__{c}__mean", f"{self.name}__{c}__std"]
            src += [c, c]
        for t, s, d in self.rebase:
            names += [f"{self.name}__{d}@{t}__mean", f"{self.name}__{d}@{t}__std"]
            src += [d, d]
        for c in self.cats:
            names.append(f"{self.name}__{c}__nunique")
            src.append(c)
        names.append(f"{self.name}__rows")
        src.append(self.keys[-1])
        self._names, self.paired_parents = names, src
        self.n_out = len(names)

    def out_names(self) -> List[str]:
        return self._names

    def __repr__(self):
        return self.name

    def _values(self, df: pd.DataFrame) -> pd.DataFrame:
        V = {c: df[c].to_numpy(dtype=np.float32) for c in self.nums}
        for t, s, d in self.rebase:
            V[f"{d}@{t}"] = (np.floor(df[t].to_numpy(dtype=float) / s) - df[d].to_numpy(dtype=float)).astype(np.float32)
        return pd.DataFrame(V, index=df.index)

    def fit(self, df: pd.DataFrame, y=None, ctx=None):
        code, uniq = pd.factorize(_key_strings(df, self.keys))
        V = self._values(df)
        g = V.groupby(code, sort=True)
        mean, std = g.mean(), g.std()
        cols = []
        for c in V.columns:
            cols += [mean[c].to_numpy(dtype=np.float32), std[c].to_numpy(dtype=np.float32)]
        for c in self.cats:
            cols.append(pd.Series(df[c].to_numpy()).groupby(code, sort=True).nunique().to_numpy(dtype=np.float32))
        cols.append(np.bincount(code).astype(np.float32))
        self.table_ = np.column_stack(cols)  # one row per client, in code order
        self.index_ = pd.Index(uniq)
        return self

    def transform(self, df: pd.DataFrame, ctx=None) -> np.ndarray:
        pos = self.index_.get_indexer(_key_strings(df, self.keys))
        out = np.full((len(df), self.n_out), np.nan, dtype=np.float32)
        hit = pos >= 0
        out[hit] = self.table_[pos[hit]]
        return out

    def fit_transform_oof(self, df, y, ctx, folds):
        return self.fit(df).transform(df)


def profile_candidates(W: pd.DataFrame, ids: Sequence[str], anchors: Sequence[tuple],
                       rebase: Sequence[Tuple[str, float, str]], num_rank: Sequence[str],
                       cat_rank: Sequence[str], max_nums: int = 120, max_cats: int = 16,
                       exclude: Sequence[str] = ()) -> List[ClientProfile]:
    """Profile of the discovered client: (two strongest ID columns, best anchor). Nothing without an anchor: plain column combinations
    are not clients (composite-key aggregations over them were neutral)."""
    if not anchors or not ids:
        return []
    a = anchors[0][3]
    skip = set(ids) | {x[3] for x in anchors} | set(exclude)
    nums = [c for c in num_rank if c not in skip and W[c].nunique() > 2][:max_nums]
    cats = [c for c in cat_rank if c not in skip and 2 < W[c].nunique() < 0.5 * len(W)][:max_cats]
    # One key: the two strongest ID columns plus the best anchor (on IEEE-CIS a second block
    # keyed on one ID column and the anchor lowered AUC 0.9409 -> 0.9398).
    keys = [tuple(ids[:2]) + (a,)] if len(ids) >= 2 else [(ids[0], a)]
    return [ClientProfile(list(k), nums, cats, rebase, label="+".join(k)) for k in keys]
