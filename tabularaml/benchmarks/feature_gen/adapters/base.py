"""Abstract adapter contract for the cross-framework FE benchmark.

The runner works with instances of FEFrameworkAdapter exclusively. Concrete
subclasses encapsulate framework-specific quirks (NaN handling, dataframe ↔
array conversion, time-budget plumbing) so that the runner stays generic.

Contract (enforced via _check_contract):
    * X_train_fe.columns == X_test_fe.columns (order-preserving)
    * All output columns are numeric or pandas `category` dtype
    * Row counts equal their inputs
    * NaN is allowed in both train and test — the base learner (XGBoost
      with tree_method=hist) handles missing values natively, and some
      frameworks (TabularAML nested divisions, AutoFeat reciprocal ops)
      legitimately emit NaN for extreme inputs. We only reject entirely-NaN
      columns (no useful signal).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Literal, Optional
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas import CategoricalDtype


def _is_categorical(series: pd.Series) -> bool:
    return isinstance(series.dtype, CategoricalDtype)


class ContractViolationError(RuntimeError):
    """Raised by _check_contract when an adapter's output is malformed."""


class FEFrameworkAdapter(ABC):
    name: str = "base"
    supports_regression: bool = True
    supports_classification: bool = True
    supports_multiclass: bool = True
    requires_nan_free: bool = False
    supports_categorical: bool = True
    gpu: bool = False
    version: str = "0.1.0"

    def __init__(
        self,
        task: Literal["regression", "classification"],
        time_budget_s: int,
        random_state: int,
        n_jobs: int = -1,
        **framework_kwargs,
    ):
        self.task = task
        self.time_budget_s = int(time_budget_s)
        self.random_state = int(random_state)
        self.n_jobs = n_jobs
        self.framework_kwargs = framework_kwargs
        self._n_features_before: Optional[int] = None
        self._n_features_after: Optional[int] = None

    @abstractmethod
    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        """Fit the framework on (X_train, y_train) and return engineered training features."""

    @abstractmethod
    def transform(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transformation to held-out data."""

    def get_feature_count_added(self) -> int:
        if self._n_features_before is None or self._n_features_after is None:
            return 0
        return max(0, self._n_features_after - self._n_features_before)

    def get_internal_log(self) -> dict:
        """Optional diagnostics surfaced per-adapter (e.g. TabularAML's self.state)."""
        return {}


def _check_contract(
    X_train_fe: pd.DataFrame,
    X_test_fe: pd.DataFrame,
    n_train_expected: int,
    n_test_expected: int,
) -> None:
    if not isinstance(X_train_fe, pd.DataFrame) or not isinstance(X_test_fe, pd.DataFrame):
        raise ContractViolationError(
            f"Adapter must return DataFrames, got {type(X_train_fe)} and {type(X_test_fe)}"
        )
    if len(X_train_fe) != n_train_expected:
        raise ContractViolationError(
            f"Train row count changed: got {len(X_train_fe)}, expected {n_train_expected}"
        )
    if len(X_test_fe) != n_test_expected:
        raise ContractViolationError(
            f"Test row count changed: got {len(X_test_fe)}, expected {n_test_expected}"
        )
    if list(X_train_fe.columns) != list(X_test_fe.columns):
        only_train = set(X_train_fe.columns) - set(X_test_fe.columns)
        only_test = set(X_test_fe.columns) - set(X_train_fe.columns)
        raise ContractViolationError(
            f"Train/test columns differ. Only in train: {sorted(only_train)[:5]}... "
            f"Only in test: {sorted(only_test)[:5]}..."
        )
    all_nan = [c for c in X_train_fe.columns if X_train_fe[c].isna().all()]
    if all_nan:
        raise ContractViolationError(
            f"Train output has all-NaN columns (no signal): {all_nan[:10]}"
        )
    bad_dtype = [
        c for c in X_train_fe.columns
        if not (is_numeric_dtype(X_train_fe[c]) or _is_categorical(X_train_fe[c]))
    ]
    if bad_dtype:
        raise ContractViolationError(
            f"Non-numeric / non-category columns present: {bad_dtype[:10]}"
        )
