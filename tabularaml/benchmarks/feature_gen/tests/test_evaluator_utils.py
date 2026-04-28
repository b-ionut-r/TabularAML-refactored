"""Verify benchmark evaluator utilities (sanitization and splitting)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from tabularaml.benchmarks.feature_gen.evaluator import (
    compute_holdout_metrics,
    compute_metric_gains,
    sanitize_features,
    split_early_stopping_validation,
    pct_improvement,
)
from tabularaml.eval.cv import cross_val_score
from tabularaml.eval.scorers import accuracy
from tabularaml.eval.scorers import binary_crossentropy, rmse

def test_sanitize_features_handles_inf_and_extremes():
    df = pd.DataFrame({
        "a": [1.0, np.inf, -np.inf, np.nan],
        "b": [1e10, -1e10, 0.0, 5.0],
        "c": pd.Series(["x", "y", "x", "y"], dtype="category")
    })
    
    sanitized = sanitize_features(df)
    
    # Inf should be replaced with NaN
    assert sanitized["a"].isna().sum() == 3
    assert np.isfinite(sanitized["a"].dropna()).all()
    assert not np.isinf(sanitized["b"].to_numpy()).any()
    
    # Categories should be preserved
    assert isinstance(sanitized["c"].dtype, pd.CategoricalDtype)


class _RareClassEvalGuardModel:
    fit_calls = 0

    def fit(self, X, y, eval_set=None, **kwargs):
        type(self).fit_calls += 1
        train_labels = set(np.unique(np.asarray(y)))
        if eval_set is not None:
            _, y_val = eval_set[0]
            unseen = set(np.unique(np.asarray(y_val))) - train_labels
            assert not unseen
        self._default_pred = min(train_labels)
        return self

    def predict(self, X):
        return np.full(len(X), self._default_pred)

    def get_params(self, deep=True):
        return {}


def test_cross_val_score_reduces_classification_folds_for_rare_classes():
    _RareClassEvalGuardModel.fit_calls = 0
    X = pd.DataFrame({"feat": np.arange(42)})
    y = pd.Series([0] * 20 + [1] * 20 + [2] * 2)

    score = cross_val_score(
        _RareClassEvalGuardModel(),
        X,
        y,
        scorer=accuracy,
        cv=5,
    )

    assert np.isfinite(score)
    assert _RareClassEvalGuardModel.fit_calls == 2


def test_cross_val_score_skips_eval_set_when_splitter_hides_a_class():
    _RareClassEvalGuardModel.fit_calls = 0
    X = pd.DataFrame({"feat": np.arange(42)})
    y = pd.Series([0] * 20 + [1] * 20 + [2] * 2)

    score = cross_val_score(
        _RareClassEvalGuardModel(),
        X,
        y,
        scorer=accuracy,
        cv=KFold(n_splits=5, shuffle=False),
    )

    assert np.isfinite(score)
    assert _RareClassEvalGuardModel.fit_calls == 5

def test_split_early_stopping_validation_stratifies():
    X = pd.DataFrame({"feat": np.arange(100)})
    y = np.array([0] * 50 + [1] * 50)
    
    X_tr, X_val, y_tr, y_val = split_early_stopping_validation(
        X, y, task="classification", seed=42, validation_fraction=0.2
    )
    
    assert len(X_val) == 20
    assert len(X_tr) == 80
    # Check stratification
    assert (y_val == 0).sum() == 10
    assert (y_val == 1).sum() == 10


def test_split_early_stopping_validation_stratifies_multiclass():
    X = pd.DataFrame({"feat": np.arange(90)})
    y = np.array([0] * 30 + [1] * 30 + [2] * 30)

    X_tr, X_val, y_tr, y_val = split_early_stopping_validation(
        X, y, task="multiclass", seed=42, validation_fraction=0.2
    )

    assert len(X_val) == 18
    assert len(X_tr) == 72
    assert set(np.unique(y_tr)) == {0, 1, 2}
    assert set(np.unique(y_val)) == {0, 1, 2}
    assert (y_val == 0).sum() == 6
    assert (y_val == 1).sum() == 6
    assert (y_val == 2).sum() == 6

def test_split_early_stopping_validation_falls_back_on_tiny_data():
    X = pd.DataFrame({"feat": np.arange(5)})
    y = np.array([0, 0, 0, 0, 1])  # Only one member of class 1

    # Should not raise even if stratification is impossible
    X_tr, X_val, y_tr, y_val = split_early_stopping_validation(
        X, y, task="classification", seed=42, validation_fraction=0.4
    )
    assert len(X_val) >= 1
    assert len(X_tr) >= 1


def test_pct_improvement_sign_normalization():
    """positive = framework beats no-FE, for both greater-is-better and lower-is-better scorers."""
    # Binary crossentropy (greater_is_better=False): lower is better.
    assert pct_improvement(0.6, 0.7, binary_crossentropy) > 0
    # Binary crossentropy: higher is worse.
    assert pct_improvement(0.8, 0.7, binary_crossentropy) < 0

    # RMSE (greater_is_better=False): framework 0.5 < nofe 0.8 → positive (lower RMSE is better)
    assert pct_improvement(0.5, 0.8, rmse) > 0
    # RMSE: framework 0.9 > nofe 0.8 → negative (higher RMSE is worse)
    assert pct_improvement(0.9, 0.8, rmse) < 0

    # Zero denominator should not raise
    assert pct_improvement(0.5, 0.0, binary_crossentropy) == 0.0


def test_split_regression_is_unstratified():
    """Regression splits must not attempt stratification (would crash on floats)."""
    X = pd.DataFrame({"feat": np.arange(100)})
    y = np.random.default_rng(0).normal(size=100)  # continuous target

    X_tr, X_val, y_tr, y_val = split_early_stopping_validation(
        X, y, task="regression", seed=0, validation_fraction=0.2
    )
    assert len(X_val) == 20
    assert len(X_tr) == 80


def test_compute_holdout_metrics_returns_multiple_metrics():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0.1, 0.8, 0.3, 0.9, 0.7, 0.2])

    metrics = compute_holdout_metrics(y_true, y_pred, task="classification", n_classes=2)

    assert metrics["binary_crossentropy"] is not None
    assert metrics["binary_roc_auc"] is not None
    assert metrics["accuracy"] is not None
    assert metrics["precision"] is not None
    assert metrics["recall"] is not None
    assert metrics["f1"] is not None


def test_compute_holdout_metrics_swallows_aux_metric_failures(monkeypatch):
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    def _boom(y_true, y_pred):
        raise ValueError("boom")

    monkeypatch.setattr(
        "tabularaml.benchmarks.feature_gen.evaluator.rmsle.score",
        _boom,
    )

    metrics = compute_holdout_metrics(y_true, y_pred, task="regression", n_classes=0)

    assert metrics["rmse"] is not None
    assert metrics["rmsle"] is None


def test_compute_metric_gains_uses_metric_direction():
    gains = compute_metric_gains(
        metric_scores={"rmse": 0.5, "accuracy": 0.8, "f1": None},
        baseline_metric_scores={"rmse": 0.8, "accuracy": 0.7, "f1": 0.6},
    )

    assert gains["rmse"] > 0
    assert gains["accuracy"] > 0
    assert gains["f1"] is None
