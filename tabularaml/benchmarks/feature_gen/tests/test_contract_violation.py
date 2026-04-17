"""A misbehaving adapter must be caught by _check_contract, not crash the runner."""
from __future__ import annotations
import pandas as pd
import pytest

from tabularaml.benchmarks.feature_gen.adapters.base import (
    FEFrameworkAdapter, _check_contract, ContractViolationError,
)


class _BadColumnsAdapter(FEFrameworkAdapter):
    name = "bad_columns"

    def fit_transform(self, X_train, y_train):
        return X_train.copy()

    def transform(self, X_test):
        out = X_test.copy()
        out["extra_only_on_test"] = 0.0
        return out


class _AllNaNAdapter(FEFrameworkAdapter):
    name = "all_nan"

    def fit_transform(self, X_train, y_train):
        out = X_train.copy().astype(float)
        out["all_nan_col"] = float("nan")
        return out

    def transform(self, X_test):
        out = X_test.copy().astype(float)
        out["all_nan_col"] = float("nan")
        return out


def test_mismatched_columns_raises():
    X_tr = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    X_te = pd.DataFrame({"a": [5.0], "b": [6.0]})
    ad = _BadColumnsAdapter(task="regression", time_budget_s=1, random_state=0)
    tr = ad.fit_transform(X_tr, pd.Series([0.0, 1.0]))
    te = ad.transform(X_te)
    with pytest.raises(ContractViolationError):
        _check_contract(tr, te, len(X_tr), len(X_te))


def test_all_nan_column_raises():
    X_tr = pd.DataFrame({"a": [1.0, 2.0]})
    X_te = pd.DataFrame({"a": [5.0]})
    ad = _AllNaNAdapter(task="regression", time_budget_s=1, random_state=0)
    tr = ad.fit_transform(X_tr, pd.Series([0.0, 1.0]))
    te = ad.transform(X_te)
    with pytest.raises(ContractViolationError):
        _check_contract(tr, te, len(X_tr), len(X_te))


def test_partial_nan_is_allowed():
    """XGBoost handles NaN natively, so partial NaN columns must pass the check."""
    X_tr = pd.DataFrame({"a": [1.0, float("nan"), 3.0]})
    X_te = pd.DataFrame({"a": [5.0, float("nan")]})
    _check_contract(X_tr, X_te, len(X_tr), len(X_te))  # must not raise
