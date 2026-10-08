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
                 recent: int = 3, top_levels: int = 8, max_pairs: int = 12, row_pairs: bool = True,
                 n_split: int = 0):
        self.children = list(children)
        self.stats = tuple(stats)
        self.recent = recent
        self.top_levels = top_levels
        self.max_pairs = max_pairs
        self.row_pairs = row_pairs
        self.n_split = n_split

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
        # Conditional aggregates: the same statistics within the main levels of a
        # low-cardinality status column (active vs closed credits, approved vs
        # refused applications).
        splits = []
        for c in cats:
            vc = df[c].astype(str).value_counts(normalize=True)
            if 2 <= len(vc) <= 6 and vc.iloc[0] < 0.9:
                splits.append((vc.iloc[0], c, list(vc.index[:2])))
        for _, c, levels in sorted(splits)[:self.n_split]:
            s = df[c].astype(str).to_numpy()
            for lv in levels:
                sub = W.loc[s == lv, [key] + dense]
                a = sub.groupby(key, sort=False)[dense].agg(["mean", "max", "sum"])
                a.columns = [f"{c}={lv}:{v}_{st}" for v, st in a.columns]
                cnt = sub.groupby(key, sort=False).size().rename(f"{c}={lv}:count")
                out += [a, cnt.to_frame()]
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


def child_model_features(ch: Child, y_by_key: pd.Series, test_keys: Sequence, n_folds: int = 5,
                         seed: int = 0, max_rows: int = 3_000_000, rounds: int = 300,
                         task: str = "binary") -> pd.DataFrame:
    """Out-of-fold child-row models: every child row is labelled with its parent's
    target, a LightGBM learns which past loans / payments / transactions look like
    a positive parent, and its predictions are aggregated per parent (mean, max,
    min, most recent). Folds split *parents*, so a training parent's features come
    from a model that never saw its label; test parents use a model fitted on all
    labelled parents.

    ``y_by_key``: labels indexed by the parent key (training parents only).
    Returns features indexed by parent key for training and ``test_keys`` parents.
    """
    import lightgbm as lgb

    df = ch.df
    key = ch.key
    ids = {key, *ch.drop} | {g.key for g in ch.children}
    feats = [c for c in df.columns if c not in ids]
    X = df[feats].copy()
    num = [c for c in feats if pd.api.types.is_numeric_dtype(X[c])]
    if len(num) >= 2:
        X = pd.concat([X, _row_pairs(df, num, 12).astype(np.float32)], axis=1)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category")
    X.columns = [_safe(c) for c in X.columns]
    X = X.loc[:, ~X.columns.duplicated()]
    k = df[key].to_numpy()
    train_keys = y_by_key.index.to_numpy()
    rng = np.random.default_rng(seed)
    fold_of = pd.Series(rng.permutation(len(train_keys)) % n_folds, index=train_keys)
    row_fold = fold_of.reindex(k).to_numpy()            # NaN for test / unlabelled parents
    row_y = y_by_key.reindex(k).to_numpy(dtype=float)
    is_test = pd.Index(test_keys).get_indexer(k) >= 0
    pred = np.full(len(df), np.nan)
    P = dict(objective=task, learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
             bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=seed, num_threads=4)

    def fit(rows):
        if len(rows) > max_rows:
            rows = rng.choice(rows, max_rows, replace=False)
        return lgb.train(P, lgb.Dataset(X.iloc[rows], row_y[rows]), rounds)

    labelled = np.flatnonzero(np.isfinite(row_fold))
    for f in range(n_folds):
        tr = labelled[row_fold[labelled] != f]
        va = labelled[row_fold[labelled] == f]
        if len(va):
            pred[va] = fit(tr).predict(X.iloc[va])
    te = np.flatnonzero(is_test)
    if len(te):
        pred[te] = fit(labelled).predict(X.iloc[te])
    out = pd.DataFrame({key: k, "p": pred})
    if ch.time is not None:
        out["t"] = df[ch.time].to_numpy()
    out = out[np.isfinite(out["p"])]
    g = out.groupby(key, sort=False)["p"]
    res = pd.DataFrame({"mean": g.mean(), "max": g.max(), "min": g.min(), "std": g.std()})
    if ch.time is not None:
        last = out.sort_values([key, "t"]).groupby(key, sort=False)["p"].last()
        res["last"] = last
    res.columns = [_safe(f"{ch.name}:model_{c}") for c in res.columns]
    return res.astype(np.float32)


def asof_features(main: pd.DataFrame, key: str, time: str, ch: Child, recent: Sequence[int] = (5, 20),
                  top_levels: int = 12, stats: Sequence[str] = ("mean", "sum", "max", "min", "std")) -> pd.DataFrame:
    """Aggregations of a child event table as of each main row's time: only child rows of
    the same key strictly before ``main[time]`` count, so a main row with several labelled
    moments per entity (assessments of a player, orders of a user) sees exactly the
    history available then, as a test row will.

    Per main row: number of earlier events, time since the first and the last; mean /
    sum / max / min / std of each numeric child column over all earlier events and the
    mean over the last ``recent`` ones; share of earlier events at each frequent level of
    each categorical column, their count over the last ``recent[0]``, and the number of
    distinct levels seen. Computed with prefix sums over the child sorted by (key, time),
    so the cost is O((rows + events) log) whatever the number of main rows per key.
    Returns a frame aligned with ``main``'s rows. Label-free.
    """
    df = ch.df
    keys = pd.Index(pd.unique(main[key]))
    ck = keys.get_indexer(df[key])
    keep = ck >= 0
    df, ck = df.loc[keep], ck[keep]
    ct = df[ch.time].to_numpy(dtype=float)
    mk = keys.get_indexer(main[key])
    mt = main[time].to_numpy(dtype=float)
    # Positions in the child sorted by (key, time): a time rank over child and main times
    # makes the pair one sortable integer.
    allt = np.unique(np.concatenate([ct[np.isfinite(ct)], mt[np.isfinite(mt)]]))
    big = len(allt) + 2
    crank = np.searchsorted(allt, ct) + 1
    order = np.lexsort((crank, ck))
    cs = ck[order].astype(np.int64) * big + crank[order]
    q = mk.astype(np.int64) * big + (np.searchsorted(allt, mt) + 1)
    p = np.searchsorted(cs, q, side="left")                   # first event at or after the row's time
    lo = np.searchsorted(cs, mk.astype(np.int64) * big, side="left")
    n = (p - lo).astype(np.float64)
    has = n > 0
    out = {}
    pre = _safe(ch.name)
    out[f"{pre}__n"] = n
    ts = ct[order]
    with np.errstate(all="ignore"):
        out[f"{pre}__since_last"] = np.where(has, mt - ts[np.maximum(p - 1, 0)], np.nan)
        out[f"{pre}__since_first"] = np.where(has, mt - ts[np.minimum(lo, len(ts) - 1)], np.nan)

    def prefix(v):
        return np.concatenate([[0.0], np.cumsum(v, dtype=np.float64)])

    def window(S, a, b):
        return S[b] - S[a]

    ids = {key, ch.time, *ch.drop}
    feats = [c for c in df.columns if c not in ids]
    cats = [c for c in feats if df[c].dtype == object or isinstance(df[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(df[c]) or df[c].dtype == bool]
    num = [c for c in feats if c not in cats and pd.api.types.is_numeric_dtype(df[c])]
    blk = ck[order]
    for c in num:
        x = df[c].to_numpy(dtype=float)[order]
        ok = np.isfinite(x)
        xz = np.where(ok, x, 0.0)
        S, S2, N = prefix(xz), prefix(xz ** 2), prefix(ok.astype(float))
        m = window(N, lo, p)
        with np.errstate(all="ignore"):
            mean = window(S, lo, p) / m
            if "mean" in stats:
                out[f"{pre}__{c}_mean"] = mean
            if "sum" in stats:
                out[f"{pre}__{c}_sum"] = window(S, lo, p)
            if "std" in stats:
                out[f"{pre}__{c}_std"] = np.sqrt(np.maximum(window(S2, lo, p) / m - mean ** 2, 0))
            for k in recent:
                a = np.maximum(lo, p - k)
                out[f"{pre}__{c}_last{k}"] = window(S, a, p) / window(N, a, p)
        for st, fn in (("max", np.fmax), ("min", np.fmin)):
            if st in stats:
                s = pd.Series(np.where(ok, x, np.nan))
                cum = (s.groupby(blk).cummax() if st == "max" else s.groupby(blk).cummin()).to_numpy()
                out[f"{pre}__{c}_{st}"] = np.where(has, cum[np.maximum(p - 1, 0)], np.nan)
    for c in cats:
        s = df[c].astype(str).to_numpy()[order]
        for lv in pd.Series(s).value_counts().index[:top_levels]:
            S = prefix((s == lv).astype(float))
            with np.errstate(all="ignore"):
                out[f"{pre}__{c}={lv}_share"] = window(S, lo, p) / n
            a = np.maximum(lo, p - recent[0])
            out[f"{pre}__{c}={lv}_last{recent[0]}"] = window(S, a, p)
        first = ~pd.DataFrame({"k": blk, "v": s}).duplicated().to_numpy()
        out[f"{pre}__{c}_nunique"] = window(prefix(first.astype(float)), lo, p)
    res = pd.DataFrame(out, index=main.index)
    res.columns = [_safe(c) for c in res.columns]
    res = res.loc[:, ~res.columns.duplicated()]
    keep = [c for c in res.columns if res[c].notna().mean() > 0.01 and res[c].nunique() > 1]
    return res[keep].astype(np.float32)


def lookup_features(main: pd.DataFrame, table: pd.DataFrame, key: str, name: str,
                    min_overlap: float = 0.5) -> pd.DataFrame:
    """Attributes of a lookup table (one row per ``key`` value: components, products,
    stores) attached to every main column whose values are keys of the table. A single
    column gets the attributes as they are; a family of numbered slot columns
    (``component_id_1..8``) gets, per numeric attribute, the sum (weighted by the paired
    quantity family when there is one), max and mean over the filled slots, and the
    count of each frequent categorical attribute level. Label-free.
    """
    from tabularaml.generate.forge import column_families
    tab = table.drop_duplicates(key).set_index(key)
    keys = set(tab.index.astype(str))
    num = [c for c in tab.columns if pd.api.types.is_numeric_dtype(tab[c])]
    cat = [c for c in tab.columns if c not in num and 2 <= tab[c].nunique() <= 30]

    def hits(cols):
        # The columns hold the table's keys: most of their values are keys, or (a table
        # covering one kind of component among many) most of the table's keys occur in them.
        v = pd.concat([main[c].dropna().astype(str) for c in cols])
        if len(v) == 0:
            return False
        seen = set(pd.unique(v))
        return v.isin(keys).mean() >= min_overlap or len(keys & seen) >= max(3, min_overlap * len(keys))

    str_cols = [c for c in main.columns if not pd.api.types.is_numeric_dtype(main[c])]
    # A slot family (component_id_1..8) is tested and used as a whole.
    cand = []
    fam_all = column_families(str_cols)
    for cols in fam_all.values():
        if hits(cols):
            cand += cols
    in_any = {c for cols in fam_all.values() for c in cols}
    cand += [c for c in str_cols if c not in in_any and hits([c])]
    if not cand:
        return pd.DataFrame(index=main.index)
    fams = column_families(cand)
    in_fam = {c for cols in fams.values() for c in cols}
    nfam = column_families([c for c in main.columns if pd.api.types.is_numeric_dtype(main[c])])
    out = {}
    T = tab.copy()
    T.index = T.index.astype(str)
    for c in [c for c in cand if c not in in_fam]:
        A = T.reindex(main[c].astype(str).to_numpy())
        for a in num:
            out[f"{name}__{c}__{a}"] = A[a].to_numpy(dtype=float)
        for a in cat:
            out[f"{name}__{c}__{a}_code"] = pd.factorize(A[a])[0].astype(float)
    import re
    idx = lambda c: re.match(r"^.*?(\d+)$", str(c)).group(1)
    for stem, cols in fams.items():
        pair = next((v for v in nfam.values() if [idx(x) for x in v] == [idx(x) for x in cols]), None)
        V = np.column_stack([main[c].astype(str).where(main[c].notna(), None).to_numpy() for c in cols])
        filled = np.column_stack([main[c].notna().to_numpy() for c in cols])
        Q = (np.nan_to_num(np.column_stack([main[c].to_numpy(dtype=float) for c in pair]), nan=0.0)
             if pair else filled.astype(float))
        for a in num:
            M = T[a].reindex(V.ravel()).to_numpy(dtype=float).reshape(V.shape)
            M = np.where(filled, M, np.nan)
            with np.errstate(all="ignore"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out[f"{name}__{stem}__{a}_sum"] = np.nansum(M * Q, 1)
                    out[f"{name}__{stem}__{a}_max"] = np.nanmax(M, 1)
                    out[f"{name}__{stem}__{a}_mean"] = np.nanmean(M, 1)
        for a in cat:
            L = T[a].reindex(V.ravel()).to_numpy().reshape(V.shape)
            for lv in pd.Series(L[filled]).value_counts().index[:8]:
                out[f"{name}__{stem}__{a}={lv}"] = ((L == lv) & filled).astype(float).__mul__(Q).sum(1)
    res = pd.DataFrame(out, index=main.index)
    res.columns = [_safe(c) for c in res.columns]
    keep = [c for c in res.columns if res[c].notna().mean() > 0.01 and res[c].nunique() > 1]
    return res[keep].astype(np.float32)


def match_features(main: pd.DataFrame, key: str, ch: Child, cols: Sequence[str],
                   main_time: Optional[str] = None, windows: Sequence[int] = (30, 90, 180)) -> pd.DataFrame:
    """Child rows of the same key that also share a main row's value in other columns.

    A customer's purchases in the offer's category, from its company, of its brand (Acquire
    Valued Shoppers); a user's past orders of the row's product. For every column the child
    shares with the main table besides the key: the number of matching child rows and the sum
    of each numeric child column over them, all-time and, when the main row has a date and the
    child a time, over the last ``windows`` days before that date (child rows dated on or after
    it are left out). One output row per main row; label-free.
    """
    out = pd.DataFrame(index=main.index)
    df = ch.df
    nums = [c for c in df.columns if c not in (key, ch.time) and c not in cols and c not in ch.drop
            and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2][:4]
    t_child = t_main = None
    if main_time is not None and ch.time is not None:
        t_child = pd.to_datetime(df[ch.time].astype(str), errors="coerce").to_numpy("datetime64[D]").astype(np.int64)
        t_main = pd.to_datetime(main[main_time].astype(str), errors="coerce").to_numpy("datetime64[D]").astype(np.int64)
    rows = pd.DataFrame({"__r": np.arange(len(main)), key: main[key].to_numpy()})
    for c in cols:
        rows[c] = main[c].to_numpy()
        j = df[[key, c] + nums].copy()
        if t_child is not None:
            j["__t"] = t_child
        m = rows[["__r", key, c]].merge(j, on=[key, c], how="inner")
        if t_child is not None:
            m["__age"] = t_main[m["__r"].to_numpy()] - m["__t"].to_numpy()
            m = m[m["__age"] > 0]
        spans = [("all", None)] + ([(f"{w}d", w) for w in windows] if t_child is not None else [])
        for lab, w in spans:
            mm = m if w is None else m[m["__age"] <= w]
            g = mm.groupby("__r")
            base = f"{ch.name}__same_{_safe(c)}__{lab}"
            out[base + "__n"] = g.size().reindex(np.arange(len(main))).fillna(0).to_numpy(dtype=np.float32)
            for v in nums:
                out[f"{base}__{_safe(v)}_sum"] = g[v].sum().reindex(np.arange(len(main))).fillna(0).to_numpy(dtype=np.float32)
        if t_child is not None:
            last = m.groupby("__r")["__age"].min().reindex(np.arange(len(main)))
            out[f"{ch.name}__same_{_safe(c)}__days_since"] = last.to_numpy(dtype=np.float32)
        del m, j
    return out
