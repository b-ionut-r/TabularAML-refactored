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


def _stack_design(preds: List[np.ndarray], task: str) -> np.ndarray:
    cols = []
    for p in preds:
        p = np.asarray(p, dtype=float)
        if task == "regression":
            cols.append(p[:, None] if p.ndim == 1 else p)
        elif p.ndim == 1:
            q = np.clip(p, 1e-6, 1 - 1e-6)
            cols.append(np.log(q / (1 - q))[:, None])
        else:
            cols.append(np.log(np.clip(p, 1e-6, 1)))
    return np.hstack(cols)


class Stacker:
    """Level-2 linear model on OOF predictions (logits for classification).

    Learns weights that need not be non-negative or sum to one, plus a
    calibration intercept/temperature, which hill climbing cannot express.
    """

    def __init__(self, task: str, C: float = 1.0):
        self.task, self.C = task, C

    def fit(self, preds: List[np.ndarray], y):
        from sklearn.linear_model import LogisticRegression, Ridge
        Z = _stack_design(preds, self.task)
        self.model_ = (Ridge(alpha=1.0 / self.C) if self.task == "regression"
                       else LogisticRegression(C=self.C, max_iter=1000))
        self.model_.fit(Z, y)
        return self

    def predict(self, preds: List[np.ndarray]) -> np.ndarray:
        Z = _stack_design(preds, self.task)
        if self.task == "regression":
            return self.model_.predict(Z)
        p = self.model_.predict_proba(Z)
        return p[:, 1] if self.task == "binary" else p


def choose_ensemble(oof: Dict[str, np.ndarray], y: np.ndarray, metric: Metric, task: str,
                    n_folds: int = 5, seed: int = 0):
    """Pick hill climbing or a linear stacker by cross-validation on the OOF rows.

    Returns ``(kind, cv_scores)`` with kind in {"hill", "stack"}. Stacking is
    only considered for metrics scored on probabilities / values.
    """
    from sklearn.model_selection import KFold, StratifiedKFold
    names = list(oof)
    y = np.asarray(y)
    splitter = (KFold(n_folds, shuffle=True, random_state=seed) if task == "regression"
                else StratifiedKFold(n_folds, shuffle=True, random_state=seed))
    folds = list(splitter.split(np.zeros(len(y)), y))
    out = {"hill": np.zeros_like(np.asarray(oof[names[0]], dtype=float)),
           "stack": np.zeros_like(np.asarray(oof[names[0]], dtype=float))}
    for tr, va in folds:
        w = hill_climb({k: np.asarray(v)[tr] for k, v in oof.items()}, y[tr], metric)
        out["hill"][va] = blend([np.asarray(oof[k])[va] for k in w], np.array(list(w.values())))
        st = Stacker(task).fit([np.asarray(oof[k])[tr] for k in names], y[tr])
        out["stack"][va] = st.predict([np.asarray(oof[k])[va] for k in names])
    scores = {k: metric(y, v) for k, v in out.items()}
    kind = "stack" if metric.better(scores["stack"], scores["hill"]) else "hill"
    return kind, scores
