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
