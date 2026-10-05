"""Contest metrics with a single calling convention.

Every metric takes ``(y_true, y_pred)`` where ``y_pred`` is what the solver
produces: a 1-D score/probability for regression and binary tasks, a 2-D
probability matrix for multiclass. Label-style metrics (accuracy, f1, ...)
threshold or argmax internally so blends can stay in probability space.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn import metrics as skm


@dataclass(frozen=True)
class Metric:
    name: str
    fn: Callable[[np.ndarray, np.ndarray], float]
    greater_is_better: bool
    # Tasks the metric is defined for: subset of {"regression", "binary", "multiclass"}.
    tasks: tuple

    def __call__(self, y_true, y_pred) -> float:
        return float(self.fn(np.asarray(y_true), np.asarray(y_pred)))

    def better(self, a: float, b: float) -> bool:
        """True when score ``a`` is strictly better than ``b``."""
        return a > b if self.greater_is_better else a < b

    def sign(self) -> float:
        """Multiplier that turns the metric into a loss (lower is better)."""
        return -1.0 if self.greater_is_better else 1.0


def _clip(p, eps=1e-15):
    return np.clip(p, eps, 1 - eps)


def _logloss(y, p):
    if p.ndim == 2:
        p = _clip(p)
        p = p / p.sum(axis=1, keepdims=True)
        return skm.log_loss(y, p, labels=np.arange(p.shape[1]))
    return skm.log_loss(y, _clip(p), labels=[0, 1])


def _auc(y, p):
    if p.ndim == 2:
        return skm.roc_auc_score(y, p, multi_class="ovr", average="macro",
                                 labels=np.arange(p.shape[1]))
    return skm.roc_auc_score(y, p)


def _labels(p):
    return p.argmax(axis=1) if p.ndim == 2 else (p >= 0.5).astype(int)


def _rmsle(y, p):
    return float(np.sqrt(np.mean((np.log1p(np.clip(p, 0, None)) - np.log1p(y)) ** 2)))


_ALL = ("regression", "binary", "multiclass")
_CLS = ("binary", "multiclass")

METRICS = {
    m.name: m for m in [
        Metric("rmse", lambda y, p: float(np.sqrt(np.mean((y - p) ** 2))), False, ("regression",)),
        Metric("mse", lambda y, p: float(np.mean((y - p) ** 2)), False, ("regression",)),
        Metric("mae", lambda y, p: float(np.mean(np.abs(y - p))), False, ("regression",)),
        Metric("rmsle", _rmsle, False, ("regression",)),
        Metric("r2", skm.r2_score, True, ("regression",)),
        Metric("logloss", _logloss, False, _CLS),
        Metric("auc", _auc, True, _CLS),
        Metric("accuracy", lambda y, p: skm.accuracy_score(y, _labels(p)), True, _CLS),
        Metric("balanced_accuracy", lambda y, p: skm.balanced_accuracy_score(y, _labels(p)), True, _CLS),
        Metric("f1", lambda y, p: skm.f1_score(y, _labels(p)), True, ("binary",)),
        Metric("f1_macro", lambda y, p: skm.f1_score(y, _labels(p), average="macro"), True, _CLS),
        Metric("mcc", lambda y, p: skm.matthews_corrcoef(y, _labels(p)), True, _CLS),
        Metric("qwk", lambda y, p: skm.cohen_kappa_score(y, _labels(p), weights="quadratic"), True, _CLS),
    ]
}


def get_metric(metric) -> Metric:
    if isinstance(metric, Metric):
        return metric
    try:
        return METRICS[metric]
    except KeyError:
        raise ValueError(f"Unknown metric {metric!r}. Known: {sorted(METRICS)}") from None


def default_metric(task: str) -> Metric:
    return METRICS["rmse"] if task == "regression" else METRICS["logloss"]
