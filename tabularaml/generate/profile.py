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

A client mean over all rows sees the client's later rows. That is harmless for raw event columns
but leaks when a column is built from earlier labels (a running mean of a user's past answers
on Riiid): its later values carry the current row's label. ``label_history_cols`` finds such
columns, whose change across a row moves with that row's label, and they are left
out of the block.
"""
from itertools import combinations
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


def label_history_cols(W: pd.DataFrame, y: np.ndarray, keys: Sequence[str], cols: Sequence[str],
                       time_col: Optional[str] = None, max_rows: int = 200_000,
                       thresh: float = 0.2, seed: int = 0) -> List[str]:
    """Columns that carry earlier rows' labels: within a client (rows in time order), the rank
    correlation between the current row's label and the column's change from the previous row
    to the next one. A label-free column relates to the label the same way before and after it
    (through the client's traits), so the change cancels; a column built from labels moves with
    the label it took in. On IEEE-CIS a running mean of a client's earlier labels scores over 0.3 and
    the strongest raw columns tested (D2, V258) between 0.1 and 0.2 (a fraud changes what the card
    does next), hence ``thresh=0.2``."""
    y = np.asarray(y, dtype=float)
    if len(W) != len(y) or not cols:
        return []
    order = np.arange(len(W))
    if time_col is not None and time_col in W.columns:
        order = np.argsort(W[time_col].to_numpy(dtype=float), kind="stable")
    code = pd.factorize(_key_strings(W, keys))[0][order]
    order = order[np.argsort(code, kind="stable")]
    code = np.sort(code, kind="stable")
    i = np.flatnonzero((code[1:-1] == code[:-2]) & (code[1:-1] == code[2:])) + 1  # previous and next row of the same client
    if len(i) < 100:
        return []
    if len(i) > max_rows:
        i = np.sort(np.random.default_rng(seed).choice(i, max_rows, replace=False))
    prv, cur, nxt = order[i - 1], order[i], order[i + 1]
    yc = pd.Series(y[cur]).rank().to_numpy()
    yc = yc - yc.mean()
    out = []
    for c in cols:
        v = W[c].to_numpy(dtype=float)
        d = v[nxt] - v[prv]
        ok = np.isfinite(d)
        if ok.sum() < 100 or np.nanstd(d[ok]) == 0:
            continue
        r = pd.Series(d[ok]).rank().to_numpy()
        r, a = r - r.mean(), yc[ok] - yc[ok].mean()
        den = np.sqrt((r * r).sum() * (a * a).sum())
        if den > 0 and abs((r * a).sum() / den) > thresh:
            out.append(c)
    return out


def profile_candidates(W: pd.DataFrame, ids: Sequence[str], anchors: Sequence[tuple],
                       rebase: Sequence[Tuple[str, float, str]], num_rank: Sequence[str],
                       cat_rank: Sequence[str], max_nums: int = 120, max_cats: int = 16,
                       exclude: Sequence[str] = (), y: Optional[np.ndarray] = None,
                       time_col: Optional[str] = None) -> List[ClientProfile]:
    """Profile of the discovered client: (two strongest ID columns, best anchor). Nothing without an anchor: plain column combinations
    are not clients (composite-key aggregations over them were neutral)."""
    if not anchors or not ids:
        return []
    a = anchors[0][3]
    skip = set(ids) | {x[3] for x in anchors} | set(exclude)
    nums = [c for c in num_rank if c not in skip and W[c].nunique() > 2][:max_nums]
    cats = [c for c in cat_rank if c not in skip and 2 < W[c].nunique() < 0.5 * len(W)][:max_cats]
    # One key: the best anchor plus the pair of strong ID columns that splits the rows into the
    # most clients. A pair where one column nearly determines the other (IEEE-CIS: card2 follows
    # card1) merges different clients: card1 + card2 gives 152k clients and holds out at 0.9288
    # on a later window, card1 + addr1 gives 218k and 0.9332, the winners' key. A second block
    # keyed on one ID column and the anchor lowered AUC (0.9409 -> 0.9398).
    if len(ids) >= 2:
        S = W.sample(min(len(W), 200_000), random_state=0) if len(W) > 200_000 else W
        ak = _key_strings(S, [a])
        best = max(combinations(ids[:4], 2),
                   key=lambda p: len(pd.unique(_key_strings(S, list(p)) + "|" + ak)))
        keys = [tuple(best) + (a,)]
    else:
        keys = [(ids[0], a)]
    if y is not None:  # columns built from earlier labels never enter a block over all rows
        bad = set(label_history_cols(W, y, list(keys[0]), nums + [d for _, _, d in rebase], time_col))
        nums = [c for c in nums if c not in bad]
        rebase = [r for r in rebase if r[2] not in bad]
    return [ClientProfile(list(k), nums, cats, rebase, label="+".join(k)) for k in keys]
