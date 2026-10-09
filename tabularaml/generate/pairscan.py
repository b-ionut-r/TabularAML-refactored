"""Wide scan of numeric column pairs for differences and ratios that carry residual signal.

FeatureForge composes arithmetic only among its top-ranked numeric columns and the pairs its
trees already use together. Contest-winning pairs often sit outside both: Loan Default's
``f528 - f527`` joins two near-duplicate columns that the model barely uses on their own, and the
winners found it by brute force over every pair. This module does that brute force cheaply.

Every pair among the strongest numerics, plus every pair of closely related columns (rank
correlation >= ``min_corr``: the same quantity measured twice, or two amounts on one scale), is
turned into ``a - b`` and ``a / b`` on a row subsample. Each candidate is scored by the same
cross-fitted histogram Newton gain on the current model's gradients that FeatureForge screens
with, minus the better of its two parents' gains, all candidates of a chunk in one vectorised
pass. Only the top few pairs become ordinary candidates; screening, CV and the gate then decide.
"""
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _bin_ranks(M: np.ndarray, n_bins: int) -> Tuple[np.ndarray, int]:
    """Equal-frequency bins per column (by rank); missing values get bin ``n_bins``."""
    n = M.shape[0]
    fin = np.isfinite(M)
    order = np.argsort(np.where(fin, M, np.inf), axis=0, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(n)[:, None], axis=0)
    n_fin = fin.sum(0)
    bins = (ranks * n_bins) // np.maximum(n_fin, 1)[None, :]
    bins = np.where(fin, np.minimum(bins, n_bins - 1), n_bins)
    return bins, n_bins + 1


def _gains(M: np.ndarray, g: np.ndarray, h: np.ndarray, fold: np.ndarray, K: int, lam: float,
           n_bins: int, loss_fn, base: float, margin: np.ndarray) -> np.ndarray:
    """Cross-fitted loss reduction of a one-feature histogram Newton step, per column of ``M``."""
    n, m = M.shape
    bins, nb = _bin_ranks(M, n_bins)
    cell = (fold[:, None] * nb + bins) + (np.arange(m) * K * nb)[None, :]
    G = np.bincount(cell.ravel(), np.repeat(g[:, None], m, 1).ravel(), minlength=m * K * nb).reshape(m, K, nb)
    H = np.bincount(cell.ravel(), np.repeat(h[:, None], m, 1).ravel(), minlength=m * K * nb).reshape(m, K, nb)
    step = -(G.sum(1, keepdims=True) - G) / (H.sum(1, keepdims=True) - H + lam)  # (m, K, nb)
    corr = step[np.arange(m)[None, :], fold[:, None], bins]  # (n, m)
    out = np.array([base - loss_fn(margin + corr[:, j]) for j in range(m)])
    # A column that is mostly one value cannot be binned into informative cells.
    out[np.isfinite(M).sum(0) < 20] = -np.inf
    return out


def scan_pairs(W: pd.DataFrame, num_cols: Sequence[str], strong: Sequence[str], g: np.ndarray,
               h: np.ndarray, loss_fn, margin: np.ndarray, *, n_rows: int = 20_000, min_corr: float = 0.9,
               max_pairs: int = 20_000, top: int = 12, chunk: int = 512, time_budget: float = 120.0,
               min_rel_gain: float = 1e-3, seed: int = 0) -> List[Tuple[str, str, str, float]]:
    """Rank ``(op, a, b)`` difference / ratio pairs by novel residual gain.

    ``g``, ``h`` and ``margin`` are the current model's gradients, hessians and margin on the
    rows of ``W`` (binary / regression: 1-D). ``loss_fn(margin)`` returns the mean loss on a
    subsample's labels. Returns at most ``top`` tuples ``(op, a, b, novel_gain)`` whose novel gain
    exceeds ``min_rel_gain`` times the current loss, best first; ``op`` is ``"sub"`` or ``"div"`` (``a / b``).
    """
    t0 = time.time()
    rng = np.random.default_rng(seed + 11)
    n = len(W)
    rows = np.sort(rng.choice(n, n_rows, replace=False)) if n > n_rows else np.arange(n)
    cols = [c for c in num_cols if W[c].nunique() > 10]
    if len(cols) < 2:
        return []
    X = np.array(W[cols].iloc[rows].to_numpy(dtype=float), dtype=float, copy=True)
    X[~np.isfinite(X)] = np.nan
    g, h, margin = g[rows], h[rows], margin[rows]
    loss = lambda m: loss_fn(m, rows)
    K = 5
    fold = rng.permutation(len(rows)) % K
    lam = 10.0 * float(h.mean())
    n_bins = int(np.clip(len(rows) // 400, 16, 64))
    base = loss(margin)
    parent = np.maximum(_gains(X, g, h, fold, K, lam, n_bins, loss, base, margin), 0.0)

    idx = {c: i for i, c in enumerate(cols)}
    pairs = {}
    s = [idx[c] for c in strong if c in idx]
    for a in range(len(s)):
        for b in range(a + 1, len(s)):
            pairs[(min(s[a], s[b]), max(s[a], s[b]))] = 2.0
    # Related columns: rank correlation over the subsample (missing values at the median).
    R = pd.DataFrame(X).rank(pct=True).fillna(0.5).to_numpy()
    R = R - R.mean(0)
    sd = np.sqrt((R ** 2).sum(0))
    sd[sd == 0] = np.inf
    C = np.abs((R.T @ R) / np.outer(sd, sd))
    iu, ju = np.triu_indices(len(cols), 1)
    cc = C[iu, ju]
    for k in np.argsort(-cc):
        if cc[k] < min_corr or len(pairs) >= max_pairs:
            break
        pairs.setdefault((int(iu[k]), int(ju[k])), float(cc[k]))
    plist = list(pairs)

    scores: Dict[Tuple[str, str, str], float] = {}
    with np.errstate(all="ignore"):
        for start in range(0, len(plist), chunk):
            if time.time() - t0 > time_budget:
                break
            P = np.array(plist[start:start + chunk])
            A, B = X[:, P[:, 0]], X[:, P[:, 1]]
            pmax = np.maximum(parent[P[:, 0]], parent[P[:, 1]])
            for op, V in (("sub", A - B), ("div", A / np.where(B == 0, np.nan, B))):
                V[~np.isfinite(V)] = np.nan
                gain = _gains(V, g, h, fold, K, lam, n_bins, loss, base, margin) - pmax
                for (i, j), v in zip(P, gain):
                    scores[(op, cols[i], cols[j])] = float(v)
    # A pair must cut the loss by at least ``min_rel_gain`` of it: on Home Credit the best pair
    # gained 1e-5 and only took the place of search time.
    best = sorted((k for k in scores if scores[k] > min_rel_gain * base), key=scores.get, reverse=True)
    out, used = [], {}
    for op, a, b in best:
        # One op per pair, and no column in more than three kept pairs (near-copies of one
        # column would otherwise fill the list with the same signal).
        if (a, b) in {(x[1], x[2]) for x in out} or used.get(a, 0) >= 3 or used.get(b, 0) >= 3:
            continue
        out.append((op, a, b, scores[(op, a, b)]))
        used[a] = used.get(a, 0) + 1
        used[b] = used.get(b, 0) + 1
        if len(out) >= top:
            break
    return out
