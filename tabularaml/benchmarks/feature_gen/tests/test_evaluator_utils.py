"""Verify benchmark evaluator utilities (sanitization and splitting)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from tabularaml.benchmarks.feature_gen.evaluator import (
    sanitize_features,
    split_early_stopping_validation,
    pct_improvement,
)
from tabularaml.eval.scorers import binary_roc_auc, rmse

def test_sanitize_features_handles_inf_and_extremes():
    df = pd.DataFrame({
        "a": [1.0, np.inf, -np.inf, 2.0],
        "b": [1e10, -1e10, 0.0, 5.0],
        "c": pd.Series(["x", "y", "x", "y"], dtype="category")
    })
    
    sanitized = sanitize_features(df)
    
    # Inf should be replaced with NaN
    assert sanitized["a"].isna().sum() == 2
    assert np.isfinite(sanitized["a"].dropna()).all()
    
    # Large values should be clipped
    assert sanitized["b"].max() <= 1e6
    assert sanitized["b"].min() >= -1e6
    
    # Categories should be preserved
    assert isinstance(sanitized["c"].dtype, pd.CategoricalDtype)

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
    # ROC-AUC (greater_is_better=True): framework 0.8 > nofe 0.7 → positive
    assert pct_improvement(0.8, 0.7, binary_roc_auc) > 0
    # ROC-AUC: framework 0.6 < nofe 0.7 → negative
    assert pct_improvement(0.6, 0.7, binary_roc_auc) < 0

    # RMSE (greater_is_better=False): framework 0.5 < nofe 0.8 → positive (lower RMSE is better)
    assert pct_improvement(0.5, 0.8, rmse) > 0
    # RMSE: framework 0.9 > nofe 0.8 → negative (higher RMSE is worse)
    assert pct_improvement(0.9, 0.8, rmse) < 0

    # Zero denominator should not raise
    assert pct_improvement(0.5, 0.0, binary_roc_auc) == 0.0


def test_split_regression_is_unstratified():
    """Regression splits must not attempt stratification (would crash on floats)."""
    X = pd.DataFrame({"feat": np.arange(100)})
    y = np.random.default_rng(0).normal(size=100)  # continuous target

    X_tr, X_val, y_tr, y_val = split_early_stopping_validation(
        X, y, task="regression", seed=0, validation_fraction=0.2
    )
    assert len(X_val) == 20
    assert len(X_tr) == 80
