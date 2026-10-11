"""List-valued columns and per-unit amounts: two label-free families that switch on from the data's structure.

**List columns.** A cell that holds a list (a listing's amenities, its photo URLs, a product's tags) is either a
Python list / array or a string of short items joined by a delimiter (" ; ", "|", ","). Winners turned such columns
into the number of items and one indicator per common item (Two Sigma Connect's "Doorman", "Elevator", ...).
FeatureForge otherwise reads them as free text, where an item of several words is split up and counts of URLs are
counts of their path pieces.

**Per-unit amounts.** A skewed amount (a rent, a price, a loan) divided by small counts on the same row (bedrooms,
bathrooms, rooms in total, children): what a contest kernel calls price per room. FeatureForge's search builds
a / b of two columns but not an amount over a sum of counts.

Both read only the feature columns of train and test rows, never the labels.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tabularaml.generate.strs import as_text

DELIMS = (" ; ", ";", "|", ",")


def _safe(s: str, n: int = 40) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(s)).strip("_")[:n] or "x"


def _fixed_places(head: pd.Series, d: str, top: int = 20) -> float:
    """How often the most common parts sit at one place (counted from the end) in their cells."""
    pos: Dict[str, Counter] = {}
    for x in head:
        parts = [i.strip() for i in x.split(d) if i.strip()]
        for k, i in enumerate(parts):
            pos.setdefault(i, Counter())[len(parts) - k] += 1
    common = [i for i in sorted(pos, key=lambda i: -sum(pos[i].values()))[:top] if sum(pos[i].values()) >= 5]
    if not common:
        return 0.0
    return float(np.mean([max(pos[i].values()) / sum(pos[i].values()) for i in common]))


def as_lists(s: pd.Series, sample: int = 5000) -> Optional[pd.Series]:
    """The column as lists of stripped items, or None when it does not hold lists."""
    v = s.dropna()
    if len(v) == 0:
        return None
    head = v.iloc[:sample]
    if head.map(lambda x: isinstance(x, (list, tuple, np.ndarray))).mean() > 0.9:
        return s.map(lambda x: [str(i).strip() for i in x] if isinstance(x, (list, tuple, np.ndarray)) else [])
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)):
        return None
    s = s.astype(object).where(s.notna(), None)
    head = as_text(head)
    if head.nunique() <= 50:
        return None   # a category whose names hold a comma ("Stone, brick"), or a few fixed item sets
    for d in DELIMS:
        if head.str.contains(d, regex=False).mean() < 0.3:
            continue
        items = [i.strip() for x in head for i in x.split(d) if i.strip()]
        if not items:
            continue
        lens = np.array([len(i) for i in items])
        # Short items that recur across rows (an amenity, a tag), not clauses of free text.
        # " ; " is how list cells arrive (contest_features.read joins them): photo URLs are a list too.
        if d != " ; " and (np.median(lens) > 30 or len(set(items)) > 0.5 * len(items)):
            continue
        # Parts that keep their place are fields of one value, not list items: an address's
        # "Chicago, IL 60634, USA" puts the same city, state and country last on every row.
        if d != " ; " and _fixed_places(head, d) > 0.9:
            continue
        return s.map(lambda x: [i.strip() for i in str(x).split(d) if i.strip()] if isinstance(x, str) else [])
    return None


def list_features(Xtr: pd.DataFrame, Xte: pd.DataFrame, top: int = 100, min_share: float = 0.005
                  ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Per list column: the number of items, the number of distinct ones, and an indicator per item held by at
    least ``min_share`` of the rows (the ``top`` most common). Returns the train / test features and the list
    columns found."""
    n = len(Xtr)
    out: Dict[str, np.ndarray] = {}
    found = []
    for c in Xtr.columns:
        both = pd.concat([Xtr[c], Xte[c]], ignore_index=True)
        L = as_lists(both)
        if L is None:
            continue
        found.append(c)
        base = _safe(c)
        out[f"{base}__n_items"] = L.map(len).to_numpy(dtype=np.float32)
        out[f"{base}__n_distinct"] = L.map(lambda v: len(set(v))).to_numpy(dtype=np.float32)
        low = L.map(lambda v: {i.lower() for i in v})
        cnt = Counter(i for v in low for i in v)
        for item, k in cnt.most_common(top):
            if k < max(min_share * len(L), 20) or k == len(L):
                break
            name = f"{base}__has__{_safe(item)}"
            if name not in out:
                out[name] = low.map(lambda v, item=item: item in v).to_numpy(dtype=np.float32)
    F = pd.DataFrame(out)
    return F.iloc[:n].reset_index(drop=True), F.iloc[n:].reset_index(drop=True), found


def _count_like(s: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    v = s.dropna().to_numpy(dtype=float)
    if len(v) == 0 or v.min() not in (0, 0.5, 1) or v.max() > 20:   # counts start at zero or one
        return False
    # Whole or half units, mostly non-zero (rooms on a listing; not rare-event tallies such as late payments).
    return np.mean(v == np.round(v * 2) / 2) > 0.99 and 3 <= len(np.unique(v)) <= 25 and np.mean(v == 0) <= 0.5


def _amount_like(s: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    v = s.dropna().to_numpy(dtype=float)
    if len(v) < 100 or v.min() < 0 or len(np.unique(v[:20000])) < 100:
        return False
    med = np.median(v)
    return med > 0 and np.quantile(v, 0.99) / med > 3   # skewed, like money


def unit_ratio_features(Xtr: pd.DataFrame, Xte: pd.DataFrame, max_amounts: int = 3, max_counts: int = 4,
                        min_rho: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Each skewed amount over each small count it grows with (Spearman >= ``min_rho``) and over their total (zero
    counts give NaN). On only for an amount with at least two such counts."""
    both = pd.concat([Xtr, Xte[Xtr.columns]], ignore_index=True)
    counts = [c for c in both.columns if _count_like(both[c])]
    amounts = [c for c in both.columns if c not in counts and _amount_like(both[c])][:max_amounts]
    out: Dict[str, np.ndarray] = {}
    S = both.sample(min(len(both), 50_000), random_state=0)
    for a in amounts:
        # Units of the amount: small counts it grows with (rent with bedrooms), not codes that share their
        # value range (a weekday, a region rating).
        rho = {c: S[a].corr(S[c], method="spearman") for c in counts}
        units = sorted([c for c in counts if rho[c] >= min_rho], key=lambda c: -rho[c])[:max_counts]
        if len(units) < 2:
            continue
        C = both[units].to_numpy(dtype=float)
        total = np.nansum(C, axis=1)
        x = both[a].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            for j, c in enumerate(units):
                out[f"{_safe(a)}__per__{_safe(c)}"] = np.where(C[:, j] > 0, x / C[:, j], np.nan)
            out[f"{_safe(a)}__per__{'_'.join(_safe(c) for c in units)}"] = np.where(total > 0, x / total, np.nan)
    F = pd.DataFrame(out, dtype=np.float32)
    n = len(Xtr)
    return F.iloc[:n].reset_index(drop=True), F.iloc[n:].reset_index(drop=True)
