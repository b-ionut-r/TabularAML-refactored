"""Out-of-fold ensembling.

``hill_climb`` is Caruana-style forward selection with replacement: start from
the best single model and repeatedly add (with repetition) the model whose
inclusion most improves the OOF metric. The resulting integer counts give
non-negative weights that sum to one, which is robust with few models and
works for any metric, including non-differentiable ones such as AUC or QWK.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .metrics import Metric


def blend(preds: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    out = np.zeros_like(np.asarray(preds[0], dtype=float))
    for p, w in zip(preds, weights):
        if w:
            out += w * np.asarray(p, dtype=float)
    return out


def hill_climb(oof: Dict[str, np.ndarray], y: np.ndarray, metric: Metric,
               max_iter: int = 100, tol: float = 1e-7) -> Dict[str, float]:
    """Return ``{model_name: weight}`` maximising ``metric`` on OOF predictions."""
    names = list(oof)
    preds = [np.asarray(oof[n], dtype=float) for n in names]
    loss = lambda p: metric.sign() * metric(y, p)

    single = [loss(p) for p in preds]
    best_idx = int(np.argmin(single))
    counts = np.zeros(len(names), dtype=int)
    counts[best_idx] = 1
    current = preds[best_idx].copy()
    best_loss = single[best_idx]

    for _ in range(max_iter):
        n = counts.sum()
        trial = [loss((current * n + p) / (n + 1)) for p in preds]
        j = int(np.argmin(trial))
        if trial[j] < best_loss - tol:
            counts[j] += 1
            current = (current * n + preds[j]) / (n + 1)
            best_loss = trial[j]
        else:
            break

    w = counts / counts.sum()
    return {name: float(wi) for name, wi in zip(names, w) if wi > 0}
