"""Price paths inside a child table: realized volatility and moves per parent, label-free, on from structure.

An order book or trade log (Optiver's book and trade rows, a crypto tick file, a sensor trace) holds, per parent
row, a sequence of prices in time order. Contest winners described each sequence by how much it moved: the realized
volatility sqrt(sum of squared log returns), over the whole window and over its later part, and the net move.
RelatedTables aggregates each column's level (mean, max, std) but never a return between consecutive rows, which is
what volatility is made of.

A price column is told apart from sizes and counts by how it moves: positive, continuous, and from one row of a
parent to the next it changes by a small fraction of itself (median |log return| under 1%). Rows are put in order by
the table's time column, or a numeric column that never decreases within a parent in file order.
Reads only the child table, never the labels.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


def _order_col(df: pd.DataFrame, key: str, time: Optional[str], cand: Sequence[str]) -> Optional[str]:
    if time is not None and time in df.columns:
        return time
    same = (df[key].to_numpy()[1:] == df[key].to_numpy()[:-1])
    if same.mean() < 0.5:
        return None   # parents' rows are not stored together
    for c in cand:
        v = df[c].to_numpy(dtype=float)
        d = np.diff(v)[same]
        if np.mean(d >= 0) > 0.99 and np.mean(d > 0) > 0.5 and len(np.unique(v[:100_000])) >= 20:
            return c
    return None


def price_cols(df: pd.DataFrame, key: str, num: Sequence[str], sample: int = 500_000) -> List[str]:
    S = df.iloc[:sample]
    same = (S[key].to_numpy()[1:] == S[key].to_numpy()[:-1])
    out = []
    for c in num:
        v = S[c].to_numpy(dtype=float)
        if not np.isfinite(v).all() or v.min() <= 0 or np.mean(v == np.round(v)) > 0.5:
            continue   # prices are positive and not whole numbers (sizes, counts)
        r = np.abs(np.diff(np.log(v)))[same]
        if len(r) >= 100 and np.median(r) < 0.01 and np.mean(r > 0) > 0.05:
            out.append(c)
    return out


SIDES = (("bid", "ask"), ("buy", "sell"))
PRICE_WORDS = ("price", "px")
SIZE_WORDS = ("size", "qty", "quantity", "volume", "vol", "amount")


def _swap(name: str, a: str, b: str) -> str:
    """``name`` with the word ``a`` replaced by ``b`` in the same case (bid -> ask, Bid -> Ask, BID -> ASK)."""
    import re
    return re.sub(a, lambda m: b.upper() if m.group().isupper() else b.capitalize() if m.group()[0].isupper() else b,
                  name, flags=re.IGNORECASE)


def book_levels(cols: Sequence[str], prices: Sequence[str]) -> dict:
    """Levels of an order book found by name: two price columns that differ only by side (bid / ask, buy / sell),
    each with its size column (the price word swapped for size / qty / volume ...). Returns
    {"wap_<level>": (price_a, size_a, price_b, size_b)}."""
    cols = list(cols)
    out = {}
    for pa in prices:
        low = pa.lower()
        for a, b in SIDES:
            pw = next((w for w in PRICE_WORDS if w in low), None)
            if a not in low or pw is None:
                continue
            pb = _swap(pa, a, b)
            if pb not in prices:
                continue
            for sw in SIZE_WORDS:
                sa, sb = _swap(pa, pw, sw), _swap(pb, pw, sw)
                if sa in cols and sb in cols:
                    lvl = _swap(_swap(pa, a, ""), pw, "").strip("_- ").lower() or "1"
                    out[f"wap_{lvl}"] = (pa, sa, pb, sb)
                    break
    return out


def return_features(df: pd.DataFrame, key: str, name: str, time: Optional[str] = None,
                    plan: Optional[dict] = None) -> pd.DataFrame:
    """Per parent key: realized volatility of each price column (and of their row mean, a mid price) over the
    whole window and its later half, the net log move, and the share of rows where it moved. Empty when the table
    has no price path. A book with bid / ask prices and sizes also gets each level's size-weighted price.
    ``plan``: a dict shared by the chunks of one table; the first chunk records which columns are prices."""
    num = [c for c in df.columns if c != key and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    if len(df) < 1000 or not df[key].duplicated().any():
        return pd.DataFrame()
    if time is not None and time in df.columns:
        df = df.sort_values([key, time], kind="stable")
    elif not (df[key].to_numpy()[1:] >= df[key].to_numpy()[:-1]).all():
        df = df.sort_values(key, kind="stable")   # a parent's rows together, file order kept within it
    if plan is not None and "prices" in plan:
        prices, order = plan["prices"], plan["order"]   # decided on an earlier chunk of the same table
    else:
        prices = price_cols(df, key, num)
        order = _order_col(df, key, time, [c for c in num if c not in prices]) if prices else None
        if plan is not None:
            plan.update(prices=prices, order=order)
    if not prices or order is None:
        return pd.DataFrame()
    d = df[[key, order] + prices]
    k = d[key].to_numpy()
    first = np.r_[True, k[1:] != k[:-1]]
    late = (d[order].to_numpy(dtype=float) >= np.nanmedian(d[order].to_numpy(dtype=float)))
    series = {c: d[c].to_numpy(dtype=float) for c in prices}
    if len(prices) >= 2:
        series["mid"] = d[prices].to_numpy(dtype=float).mean(axis=1)
    for name_, (pa, sa, pb, sb) in book_levels(df.columns, prices).items():
        # The size-weighted price of a book level (Optiver's WAP): each side's price weighted by the
        # other side's size, so it leans toward the side about to be taken.
        A, B = df.loc[d.index, sa].to_numpy(dtype=float), df.loc[d.index, sb].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            w = (df.loc[d.index, pa].to_numpy(dtype=float) * B + df.loc[d.index, pb].to_numpy(dtype=float) * A) / (A + B)
        series[name_] = np.where(np.isfinite(w) & (w > 0), w, np.nan)
    out = {}
    g = pd.Series(k)
    for c, v in series.items():
        lv = np.log(v)
        r = np.diff(lv, prepend=np.nan)
        r[first] = np.nan
        r2 = r ** 2
        base = f"{name}__{c}"
        out[f"{base}__rv"] = np.sqrt(pd.Series(r2).groupby(g, sort=False).sum(min_count=1))
        out[f"{base}__rv_late"] = np.sqrt(pd.Series(np.where(late, r2, np.nan)).groupby(g, sort=False).sum(min_count=1))
        gl = pd.Series(lv).groupby(g, sort=False)
        out[f"{base}__move"] = gl.last() - gl.first()
        out[f"{base}__moved_share"] = pd.Series(np.abs(r) > 0).groupby(g, sort=False).mean()
    F = pd.DataFrame(out).astype(np.float32)
    F.index.name = key
    return F
