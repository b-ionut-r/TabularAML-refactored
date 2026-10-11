"""Intraday panels: each entity against the market at the same moment, and its own last steps. Label-free, on from
the data's structure.

Optiver's Trading at the Close holds one row per stock and 10-second step of each day's closing auction. Winners
described a row by where the stock stands among all stocks at that moment (its value minus the moment's mean, its
rank) and by how it moved over its last few steps that day. FeatureForge's group statistics see a stock's rows or a
moment's rows but not the stock's earlier steps on the same day, and its target encodings of the moment cannot
reach test days.

Structure: an entity column (10-5,000 levels, recurring in the test rows), a moment (one column, or two such as day and second) at which every
entity has at most one row and at least 10 entities report, and a day that holds several moments of each entity
(the test rows' days are new ones).
Daily panels (a store's sales by date) have one moment per day and stay off. Only earlier steps are used: the test
rows of a real auction arrive one moment at a time.
"""
from __future__ import annotations

from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _intlike(s: pd.Series) -> bool:
    if isinstance(s.dtype, pd.CategoricalDtype):
        return True
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    v = s.dropna().to_numpy()[:100_000]
    return len(v) > 0 and np.all(np.mod(v, 1) == 0)


def find_panel(X: pd.DataFrame, U: Optional[pd.DataFrame] = None, sample: int = 500_000) -> Optional[dict]:
    """{"entity", "moment" (list of columns), "day"} or None. An entity recurs: the test rows' levels were seen in
    training (stocks do; days do not)."""
    S = X.iloc[:sample] if len(X) > sample else X
    ints = [c for c in X.columns if _intlike(X[c]) and X[c].notna().mean() > 0.99]
    nu = {c: S[c].nunique() for c in ints}
    # A day the test rows have not seen (time-split data); without one there is nothing to look for.
    days = [c for c in ints if U is not None and c in U.columns and U[c].isin(set(X[c].unique())).mean() <= 0.1]
    if U is not None and not days:
        return None
    ents = [c for c in ints if 10 <= nu[c] <= 5000]
    if U is not None:
        ents = [c for c in ents if c in U.columns and U[c].isin(set(X[c].unique())).mean() >= 0.9]
    for e in sorted(ents, key=lambda c: nu[c]):
        others = [c for c in ints if c != e and nu[c] >= 2]
        cands = [[m] for m in others if nu[m] >= 20 and len(S) / nu[m] >= 10] + \
                [list(p) for p in combinations(others[:8], 2) if nu[p[0]] * nu[p[1]] >= 20]
        for m in cands:
            g = S.groupby(m, sort=False, observed=True)[e]
            per = g.size()
            if len(per) < 20 or per.mean() < 10 or S.duplicated(m + [e]).mean() > 0.001:
                continue
            # Recorded a moment at a time: a moment's rows sit together in the file (not a stock's days).
            k = S[m].to_numpy()
            if (k[1:] == k[:-1]).all(axis=1).mean() < 0.5:
                continue
            # A day: a column fixed within each moment, holding several moments of an entity.
            for d in [c for c in m + others if c != e]:
                if len(m) == 1 and d == m[0]:
                    continue
                if (S.groupby(m, sort=False, observed=True)[d].nunique() > 1).any():
                    continue
                if U is not None and d in U.columns and not U[d].isin(set(X[d].unique())).mean() <= 0.1:
                    continue   # the day is time: the test rows' days are new
                steps = S.groupby([e, d], sort=False, observed=True).size()
                if steps.mean() >= 5:
                    return {"entity": e, "moment": m, "day": d}
    return None


UNIT_WORDS = (("price", "px", "wap"), ("size", "qty", "quantity", "volume", "amount"))


def panel_features(Xtr: pd.DataFrame, Xte: pd.DataFrame, lags: Tuple[int, ...] = (1, 2, 3), max_cols: int = 12
                   ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[dict]]:
    """Imbalances between same-unit columns (named prices, named sizes) and their rank within the moment; per
    numeric column: value minus the moment's mean over entities, rank within the moment, change over the
    entity's last ``lags`` steps of the day, and that 1-step change against the moment's mean change. Train and
    test rows are computed on their own files (moments do not span them)."""
    P = find_panel(Xtr, Xte)
    n = len(Xtr)
    if P is None or any(c not in Xte.columns for c in P["moment"] + [P["entity"], P["day"]]):
        return pd.DataFrame(index=range(n)), pd.DataFrame(index=range(len(Xte))), None
    e, m, d = P["entity"], P["moment"], P["day"]
    keys = {e, d, *m}
    num = [c for c in Xtr.columns if c not in keys and pd.api.types.is_numeric_dtype(Xtr[c])
           and not pd.api.types.is_bool_dtype(Xtr[c]) and Xtr[c].iloc[:200_000].nunique() > 20][:max_cols]
    if not num:
        return pd.DataFrame(index=range(n)), pd.DataFrame(index=range(len(Xte))), None

    def build(X: pd.DataFrame) -> pd.DataFrame:
        X = X.reset_index(drop=True)
        out = {}
        mk = [X[c] for c in m]
        order = np.lexsort([X[c].to_numpy() for c in reversed(m)] + [X[d].to_numpy(), X[e].to_numpy()])
        # rows sorted by entity, day, moment: shifts within (entity, day) are earlier steps
        Xs = X.iloc[order]
        gs = Xs.groupby([Xs[e], Xs[d]], sort=False, observed=True)
        # Imbalances between same-unit columns (bid vs ask size, reference vs far price): (a - b) / (a + b),
        # and where the middle of three sits between the other two. Ranked within the moment below.
        imb = {}
        for words in UNIT_WORDS:
            cols = [c for c in num if any(w in c.lower() for w in words) and np.nanmin(X[c].to_numpy(dtype=float)) >= 0]
            for a, b in combinations(cols[:6], 2):
                xa, xb = X[a].to_numpy(dtype=float), X[b].to_numpy(dtype=float)
                with np.errstate(all="ignore"):
                    imb[f"{a}__imb__{b}"] = (xa - xb) / (xa + xb)
            for t in combinations(cols[:4], 3):
                v = X[list(t)].to_numpy(dtype=float)
                mx, mn = np.nanmax(v, 1), np.nanmin(v, 1)
                md = v.sum(1) - mx - mn
                with np.errstate(all="ignore"):
                    imb["__triplet__".join(t)] = np.where(md - mn > 0, (mx - md) / (md - mn), np.nan)
        out.update(imb)
        for c in imb:
            out[f"{c}__moment_rank"] = pd.Series(imb[c]).groupby(mk, observed=True).rank(pct=True).to_numpy()
        for c in num:
            x = X[c].astype(float)
            out[f"{c}__vs_moment"] = (x - x.groupby(mk, observed=True).transform("mean")).to_numpy()
            out[f"{c}__moment_rank"] = x.groupby(mk, observed=True).rank(pct=True).to_numpy()
            xs = Xs[c].astype(float)
            pos = bool(np.nanmin(xs.to_numpy()) > 0)
            for k in lags:
                prev = gs[c].shift(k).astype(float)
                ch = (xs / prev - 1) if pos else (xs - prev)
                v = np.empty(len(X)); v[order] = ch.to_numpy()
                out[f"{c}__step_change{k}"] = v
            r1 = pd.Series(out[f"{c}__step_change1"])
            out[f"{c}__step_change1_vs_moment"] = (r1 - r1.groupby(mk, observed=True).transform("mean")).to_numpy()
        F = pd.DataFrame(out).replace([np.inf, -np.inf], np.nan)
        return F.astype(np.float32)

    return build(Xtr), build(Xte), P
