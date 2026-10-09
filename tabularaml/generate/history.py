"""Latest-state features of a keyed history table (statements, transactions, visits per customer).

``RelatedTables`` summarises a child table over all of a key's rows (mean, min, max, sum, std, mean of the
last three). Winning solutions on per-customer histories (Amex Default Prediction) also used where the
customer stands now: the latest value, how far it is from the customer's own average, and how it moved since
the previous row. ``history_features`` builds those for every numeric column of a child table that has a
time column and several rows per key. Label-free: it reads the child table only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .relational import Child, _safe


def repeats(ch: Child, min_share: float = 0.3) -> bool:
    """A time column and several rows for a good share of the keys."""
    if ch.time is None or ch.time not in ch.df.columns:
        return False
    n = ch.df.groupby(ch.key, sort=False).size()
    return bool((n >= 2).mean() >= min_share)


def history_features(ch: Child) -> pd.DataFrame:
    """Per key: last, last - mean and last - previous of each numeric column, rows ordered by time."""
    df = ch.df
    ids = {ch.key, ch.time, *ch.drop}
    num = [c for c in df.columns if c not in ids and pd.api.types.is_numeric_dtype(df[c])
           and not pd.api.types.is_bool_dtype(df[c])]
    if not num:
        return pd.DataFrame(index=pd.Index(pd.unique(df[ch.key]), name=ch.key))
    order = np.lexsort((df[ch.time].to_numpy(), df[ch.key].astype(str).to_numpy()))
    d = df.iloc[order][[ch.key] + num].reset_index(drop=True)
    for c in num:
        d[c] = d[c].astype(np.float32)
    g = d.groupby(ch.key, sort=False)
    last, mean = g[num].last(), g[num].mean()
    end = g.tail(1).index
    prev = d[num].groupby(d[ch.key], sort=False).shift(1).loc[end]
    prev.index = d[ch.key].loc[end].to_numpy()
    prev = prev.reindex(last.index)
    out = pd.concat([last.add_suffix("_last"), (last - mean).add_suffix("_last_mean"),
                     (last - prev).add_suffix("_last_prev")], axis=1)
    out.columns = [_safe(f"{ch.name}:{c}") for c in out.columns]
    keep = [c for c in out.columns if out[c].notna().mean() > 0.01 and out[c].nunique() > 1]
    return out[keep].astype(np.float32)


def _sorted_index(keys_ev, t_ev, keys_q, t_q):
    """Events sorted by (key, time); for each query the slice [lo, p) of the same key's events
    strictly before its time."""
    allt = np.unique(np.concatenate([t_ev[np.isfinite(t_ev)], t_q[np.isfinite(t_q)]]))
    big = len(allt) + 2
    crank = np.searchsorted(allt, t_ev) + 1
    order = np.lexsort((crank, keys_ev))
    cs = keys_ev[order].astype(np.int64) * big + crank[order]
    q = keys_q.astype(np.int64) * big + (np.searchsorted(allt, t_q) + 1)
    p = np.searchsorted(cs, q, side="left")
    lo = np.searchsorted(cs, keys_q.astype(np.int64) * big, side="left")
    return order, lo, p


def event_log_features(main: pd.DataFrame, ch: Child, outcome: str, time: str, item: str | None = None,
                       lags: int = 5, windows=(5, 20, 100), outcome_values=None) -> pd.DataFrame:
    """As-of outcome features of an event log whose rows carry the outcome being predicted (a student's
    earlier answers, a player's earlier results). Only the same key's events strictly before each main row's
    time count, so a row never sees its own outcome or later ones, and rows that share a time (one bundle)
    do not see each other's.

    Per main row: earlier outcomes (count, mean, mean of the last 5 / 20 / 100, the last ``lags`` values one by
    one), time since the latest earlier event and since the 3rd and 10th latest; and with ``item`` (the question,
    the game) the key's earlier outcomes on that same item (attempts, mean, last) and the time since its latest.
    Events whose outcome is not one of ``outcome_values`` (lectures) count for time features only.
    """
    df = ch.df
    keys = pd.Index(pd.unique(main[ch.key]))
    ek = keys.get_indexer(df[ch.key]); keep = ek >= 0
    df, ek = df.loc[keep], ek[keep]
    et = df[ch.time].to_numpy(dtype=float)
    mk = keys.get_indexer(main[ch.key]); mt = main[time].to_numpy(dtype=float)
    yv = pd.to_numeric(df[outcome], errors="coerce").to_numpy(dtype=float)
    if outcome_values is not None:
        yv = np.where(np.isin(yv, list(outcome_values)), yv, np.nan)
    pre = _safe(f"{ch.name}:{outcome}:hist")   # apart from asof_features' own column names
    out = {}
    order, lo, p = _sorted_index(ek, et, mk, mt)
    ys, ts = yv[order], et[order]
    f = np.isfinite(ys)
    S = np.concatenate([[0.0], np.cumsum(np.where(f, ys, 0.0))])
    C = np.concatenate([[0.0], np.cumsum(f)])
    n = C[p] - C[lo]
    with np.errstate(all="ignore"):
        out[f"{pre}_n"] = n
        out[f"{pre}_mean"] = np.where(n > 0, (S[p] - S[lo]) / n, np.nan)
        # Positions of outcome events only, for "last k outcomes".
        fpos = np.flatnonzero(f)
        cp, clo = C[p].astype(np.int64), C[lo].astype(np.int64)   # outcome events before the row, in fpos
        Sf = np.concatenate([[0.0], np.cumsum(ys[fpos])])
        for w in windows:
            a = np.maximum(cp - w, clo)
            out[f"{pre}_last{w}"] = np.where(cp > a, (Sf[cp] - Sf[a]) / np.maximum(cp - a, 1), np.nan)
        for j in range(1, lags + 1):
            i = cp - j
            out[f"{pre}_lag{j}"] = np.where(i >= clo, ys[fpos[np.clip(i, 0, max(len(fpos) - 1, 0))]] if len(fpos) else np.nan, np.nan)
        for j in (1, 3, 10):
            i = p - j
            out[f"{pre}_since{j}"] = np.where(i >= lo, mt - ts[np.clip(i, 0, len(ts) - 1)], np.nan)
    if item is not None and item in df.columns and item in main.columns:
        iv = pd.Index(pd.unique(pd.concat([df[item], main[item]]).astype(str)))
        ki = ek.astype(np.int64) * (len(iv) + 1) + iv.get_indexer(df[item].astype(str))
        qi = mk.astype(np.int64) * (len(iv) + 1) + iv.get_indexer(main[item].astype(str))
        _, inv_e = np.unique(np.concatenate([ki, qi]), return_inverse=True)
        o2, lo2, p2 = _sorted_index(inv_e[:len(ki)], et, inv_e[len(ki):], mt)
        y2, t2 = yv[o2], et[o2]
        f2 = np.isfinite(y2)
        S2 = np.concatenate([[0.0], np.cumsum(np.where(f2, y2, 0.0))]); C2 = np.concatenate([[0.0], np.cumsum(f2)])
        n2 = C2[p2] - C2[lo2]
        with np.errstate(all="ignore"):
            out[f"{pre}_item_n"] = n2
            out[f"{pre}_item_mean"] = np.where(n2 > 0, (S2[p2] - S2[lo2]) / n2, np.nan)
            out[f"{pre}_item_since"] = np.where(p2 > lo2, mt - t2[np.clip(p2 - 1, 0, len(t2) - 1)], np.nan)
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in out.items()}, index=main.index)
