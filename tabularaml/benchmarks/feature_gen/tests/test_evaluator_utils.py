"""Verify benchmark evaluator utilities (sanitization and splitting)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from tabularaml.benchmarks.feature_gen.evaluator import (
    sanitize_features,
    split_early_stopping_validation,
)

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
