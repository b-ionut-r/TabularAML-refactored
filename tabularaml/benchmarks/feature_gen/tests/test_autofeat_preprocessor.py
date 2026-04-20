"""Verify AutoFeat's internal _TrainOnlyPreprocessor doesn't leak test data.

The preprocessor must be fitted exclusively on training data (medians, modes,
category sets) and then applied deterministically to test data — no information
from the test fold should influence the transformations.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from tabularaml.benchmarks.feature_gen.adapters.autofeat_adapter import (
    _TrainOnlyPreprocessor,
)


def test_numeric_medians_come_from_train_only():
    """Median impute values must be computed from train data exclusively."""
    train = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, np.nan],
        "b": [10.0, 20.0, np.nan, np.nan],
    })
    test = pd.DataFrame({
        "a": [np.nan, 100.0],
        "b": [np.nan, 200.0],
    })

    pre = _TrainOnlyPreprocessor()
    pre.fit_transform(train)
    test_out = pre.transform(test)

    # Medians: a=2.0, b=15.0 (from train only)
    assert test_out["a"].iloc[0] == pytest.approx(2.0)
    assert test_out["b"].iloc[0] == pytest.approx(15.0)
    # Test values should not affect fill — second row keeps its original value.
    assert test_out["a"].iloc[1] == pytest.approx(100.0)


def test_categorical_encoding_from_train_only():
    """Ordinal codes and fill values must come from training data exclusively."""
    train = pd.DataFrame({
        "cat": ["a", "b", "c", np.nan],
        "num": [1.0, 2.0, 3.0, 4.0],
    })
    test = pd.DataFrame({
        "cat": ["a", "d", np.nan],   # "d" is unseen
        "num": [5.0, 6.0, 7.0],
    })

    pre = _TrainOnlyPreprocessor()
    pre.fit_transform(train)
    test_out = pre.transform(test)

    # Known category "a" → its ordinal code
    a_code = test_out["cat"].iloc[0]
    assert a_code >= 0, "Known category must map to a non-negative code"

    # Unseen category "d" → -1 sentinel
    d_code = test_out["cat"].iloc[1]
    assert d_code == pytest.approx(-1.0), "Unseen category must map to -1 sentinel"


def test_fit_transform_idempotent_schema():
    """Running fit_transform twice on the same data must produce identical schemas."""
    rng = np.random.default_rng(42)
    train = pd.DataFrame({
        "num": rng.normal(size=50),
        "cat": pd.Categorical(rng.choice(["x", "y"], size=50)),
    })

    pre1 = _TrainOnlyPreprocessor()
    out1 = pre1.fit_transform(train)

    pre2 = _TrainOnlyPreprocessor()
    out2 = pre2.fit_transform(train.copy())

    assert list(out1.columns) == list(out2.columns)
    pd.testing.assert_frame_equal(out1, out2)


def test_all_nan_column_gets_zero_fill():
    """If a numeric column is entirely NaN, median is NaN → fallback to 0.0."""
    train = pd.DataFrame({
        "a": [np.nan, np.nan, np.nan],
        "b": [1.0, 2.0, 3.0],
    })

    pre = _TrainOnlyPreprocessor()
    out = pre.fit_transform(train)

    assert out["a"].iloc[0] == pytest.approx(0.0)
    assert np.isfinite(out["a"].values).all()


def test_pandas_categorical_dtype_accepted():
    """_TrainOnlyPreprocessor must handle pd.Categorical dtype without error.

    The worker calls _preprocess() which casts object→category, so the adapter
    receives pandas Categorical columns, not plain object columns. The preprocessor
    must treat them identically to object columns: ordinal-encode from train cats,
    sentinel -1 for unseen categories on test.
    """
    train = pd.DataFrame({
        "cat": pd.Categorical(["a", "b", "c", "a"]),
        "num": [1.0, 2.0, 3.0, 4.0],
    })
    test = pd.DataFrame({
        "cat": pd.Categorical(["a", "d"]),  # "d" is unseen
        "num": [5.0, 6.0],
    })

    pre = _TrainOnlyPreprocessor()
    train_out = pre.fit_transform(train)
    test_out = pre.transform(test)

    # All output values must be finite floats
    assert np.isfinite(train_out.to_numpy(dtype=float)).all()
    assert train_out["cat"].dtype == float

    # Known category "a" → non-negative code; unseen "d" → -1
    a_code = test_out["cat"].iloc[0]
    d_code = test_out["cat"].iloc[1]
    assert a_code >= 0, "Known category must produce a non-negative code"
    assert d_code == pytest.approx(-1.0), "Unseen category must produce sentinel -1"
