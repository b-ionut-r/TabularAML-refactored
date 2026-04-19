"""Adapter wrapping AutoFeat's Regressor/Classifier.

AutoFeat cannot tolerate NaN in training data and has no native support for
raw pandas `category` / object dtypes, so the adapter owns an internal
SimpleImputer + OrdinalEncoder that are fitted strictly on the training fold.
"""
from __future__ import annotations
from typing import Literal, Optional
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .base import FEFrameworkAdapter


class _AutofeatInternalNaNError(RuntimeError):
    """Raised when AutoFeat's internal LassoLarsCV/LogisticRegressionCV hits NaN."""


class _AutofeatUpstreamBugError(RuntimeError):
    """Raised for known AutoFeat upstream bugs (inhomogeneous shape, SymPy codegen)."""


class _TrainOnlyPreprocessor:
    """Median-impute numeric cols, ordinal-encode (+ mode-impute) non-numeric cols.

    Everything fitted on train only; transform uses the stored artefacts.
    """

    def __init__(self):
        self.numeric_cols: list = []
        self.cat_cols: list = []
        self.num_fill: dict = {}
        self.cat_fill: dict = {}
        self.cat_categories: dict = {}

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        self.numeric_cols = [c for c in X.columns if is_numeric_dtype(X[c])]
        self.cat_cols = [c for c in X.columns if c not in self.numeric_cols]
        for c in self.numeric_cols:
            med = X[c].median()
            self.num_fill[c] = 0.0 if pd.isna(med) else float(med)
            X[c] = X[c].fillna(self.num_fill[c]).astype(float)
        for c in self.cat_cols:
            mode_vals = X[c].mode(dropna=True)
            fill = mode_vals.iloc[0] if len(mode_vals) else "__missing__"
            self.cat_fill[c] = fill
            X[c] = X[c].fillna(fill).astype(str)
            cats = pd.Index(sorted(X[c].unique()))
            self.cat_categories[c] = cats
            X[c] = pd.Categorical(X[c], categories=cats).codes.astype(float)
        return X[self.numeric_cols + self.cat_cols]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.numeric_cols:
            X[c] = X[c].fillna(self.num_fill[c]).astype(float)
        for c in self.cat_cols:
            X[c] = X[c].fillna(self.cat_fill[c]).astype(str)
            codes = pd.Categorical(X[c], categories=self.cat_categories[c]).codes.astype(float)
            # Unseen categories come back as -1; map to NaN → fill with -1.0 (stable sentinel).
            X[c] = np.where(codes < 0, -1.0, codes)
        return X[self.numeric_cols + self.cat_cols]


class AutoFeatAdapter(FEFrameworkAdapter):
    name = "autofeat"
    version = "upstream-2.1.x"
    requires_nan_free = True
    supports_categorical = False  # raw object/category not supported; adapter pre-encodes

    def __init__(
        self,
        task: Literal["regression", "classification"],
        time_budget_s: int,
        random_state: int,
        n_jobs: int = -1,
        feateng_steps: int = 2,
        featsel_runs: int = 5,
        **framework_kwargs,
    ):
        super().__init__(task, time_budget_s, random_state, n_jobs, **framework_kwargs)
        self.feateng_steps = feateng_steps
        self.featsel_runs = featsel_runs
        self._pre = _TrainOnlyPreprocessor()
        self._af = None
        self._train_columns_fe: Optional[list] = None

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        from autofeat import AutoFeatRegressor, AutoFeatClassifier

        self._n_features_before = X_train.shape[1]
        X_pre = self._pre.fit_transform(X_train)

        if self.task == "regression":
            self._af = AutoFeatRegressor(
                feateng_steps=self.feateng_steps,
                featsel_runs=self.featsel_runs,
                n_jobs=self.n_jobs,
                verbose=0,
            )
        else:
            self._af = AutoFeatClassifier(
                feateng_steps=self.feateng_steps,
                featsel_runs=self.featsel_runs,
                n_jobs=self.n_jobs,
                verbose=0,
            )

        try:
            X_train_fe = self._af.fit_transform(X_pre, pd.Series(y_train).values)
        except ValueError as e:
            msg = str(e)
            if "missing values" in msg or "does not accept missing" in msg:
                raise _AutofeatInternalNaNError(msg) from e
            if "inhomogeneous" in msg or "could not broadcast" in msg:
                raise _AutofeatUpstreamBugError(msg) from e
            raise
        except Exception as e:
            msg = str(e)
            if "duplicate argument" in msg or "DuplicateArgumentError" in type(e).__name__:
                raise _AutofeatUpstreamBugError(msg) from e
            raise
        if not isinstance(X_train_fe, pd.DataFrame):
            X_train_fe = pd.DataFrame(X_train_fe, index=X_pre.index)
        X_train_fe = X_train_fe.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self._train_columns_fe = list(X_train_fe.columns)
        self._n_features_after = X_train_fe.shape[1]
        return X_train_fe

    def transform(self, X_test: pd.DataFrame) -> pd.DataFrame:
        if self._af is None:
            raise RuntimeError("AutoFeatAdapter.transform called before fit_transform")
        X_pre = self._pre.transform(X_test)
        X_test_fe = self._af.transform(X_pre)
        if not isinstance(X_test_fe, pd.DataFrame):
            X_test_fe = pd.DataFrame(X_test_fe, index=X_pre.index)
        X_test_fe = X_test_fe.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return X_test_fe[self._train_columns_fe]
