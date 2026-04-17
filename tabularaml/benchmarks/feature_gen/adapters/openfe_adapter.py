"""Adapter wrapping OpenFE (ICML 2023).

Note on known leakage: upstream OpenFE's `transform()` concatenates train and
test before computing aggregates. We deliberately use the upstream package as
distributed because the benchmark's purpose is to measure what a grant reader
would reproduce with `pip install openfe`. A companion unit test
(test_openfe_leakage_probe.py) flags the leakage so the final report can
annotate it honestly.
"""
from __future__ import annotations
import os
import tempfile
from typing import Literal, Optional
import pandas as pd

from .base import FEFrameworkAdapter


class OpenFEAdapter(FEFrameworkAdapter):
    name = "openfe"
    version = "upstream-0.0.12"

    def __init__(
        self,
        task: Literal["regression", "classification"],
        time_budget_s: int,
        random_state: int,
        n_jobs: int = -1,
        n_data_blocks: int = 8,
        feature_boosting: bool = False,
        **framework_kwargs,
    ):
        super().__init__(task, time_budget_s, random_state, n_jobs, **framework_kwargs)
        self.n_data_blocks = n_data_blocks
        self.feature_boosting = feature_boosting
        self._ofe = None
        self._features = None
        self._x_train_cache: Optional[pd.DataFrame] = None
        self._train_columns_fe: Optional[list] = None

    def _task_for_openfe(self, y: pd.Series) -> str:
        if self.task == "regression":
            return "regression"
        n_classes = int(pd.Series(y).nunique())
        return "classification"  # OpenFE handles binary + multiclass via label dtype

    @staticmethod
    def _patch_dependencies_for_openfe() -> None:
        """OpenFE 0.0.12 calls mean_squared_error(..., squared=False) which was
        removed in sklearn 1.4+. Patch before openfe is imported so its
        module-level `from sklearn.metrics import mean_squared_error` gets the
        compatible version (fork-safe on Linux).
        Also patch LightGBM's Dataset.set_feature_name to strip characters that
        crash LightGBM >= 4.0.0 (like commas from OpenFE's generated names)."""
        import numpy as np
        import sklearn.metrics as sm
        import lightgbm as lgb
        import re

        if getattr(sm.mean_squared_error, "_openfe_patched", False):
            return

        _orig = sm.mean_squared_error
        def _compat(y_true, y_pred, *, sample_weight=None,
                    multioutput="uniform_average", squared=True):
            result = _orig(y_true, y_pred, sample_weight=sample_weight,
                           multioutput=multioutput)
            return result if squared else np.sqrt(result)
        _compat._openfe_patched = True
        sm.mean_squared_error = _compat

        _orig_set_feature_name = lgb.Dataset.set_feature_name
        def _patched_set_feature_name(self, feature_name):
            if feature_name is not None:
                feature_name = [
                    re.sub(r'[^A-Za-z0-9_]', '_', col)
                    for col in feature_name
                ]
            return _orig_set_feature_name(self, feature_name)
        lgb.Dataset.set_feature_name = _patched_set_feature_name

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        self._patch_dependencies_for_openfe()
        from openfe import OpenFE, transform  # imported lazily to keep startup light

        self._n_features_before = X_train.shape[1]

        # Reset to 0-based RangeIndex: train_test_split leaves X_train with a
        # scattered subset index; OpenFE's internal .loc[train_idx + val_idx]
        # assumes contiguous 0-based labels and raises KeyError otherwise.
        # Also clean column names to prevent lightgbm crashes on special characters.
        import re
        X_train = X_train.rename(columns=lambda col: re.sub(r'[^A-Za-z0-9_]', '_', str(col)))
        X_train = X_train.reset_index(drop=True)
        y_train = pd.Series(y_train).reset_index(drop=True)
        self._x_train_cache = X_train.copy()

        # OpenFE expects label as a DataFrame with a single column.
        y_df = pd.DataFrame({"_label": y_train.values}, index=X_train.index)

        # Use fewer blocks for small datasets so each block has at least 100 rows.
        effective_blocks = min(self.n_data_blocks, max(2, len(X_train) // 100))

        self._ofe = OpenFE()
        # OpenFE writes ./openfe_tmp_data_xx.feather to CWD with a predictable
        # name; chdir to a fresh temp dir so parallel workers don't collide.
        os.chdir(tempfile.mkdtemp())
        self._features = self._ofe.fit(
            data=X_train,
            label=y_df,
            task=self._task_for_openfe(y_train),
            n_jobs=max(1, self.n_jobs if self.n_jobs > 0 else 1),
            n_data_blocks=effective_blocks,
            feature_boosting=self.feature_boosting,
            seed=self.random_state,
            verbose=False,
        )

        # Use a copy of X_train as the "test" input so the returned train frame
        # is generated through the same `transform()` code path that will later
        # produce X_test_fe. This guarantees column identity without actually
        # leaking any extra rows (train == the "test" argument).
        X_train_fe, _ = transform(
            X_train, X_train.iloc[:1].copy(),
            self._features, n_jobs=max(1, self.n_jobs if self.n_jobs > 0 else 1),
        )
        self._train_columns_fe = list(X_train_fe.columns)
        self._n_features_after = X_train_fe.shape[1]
        return X_train_fe

    def transform(self, X_test: pd.DataFrame) -> pd.DataFrame:
        from openfe import transform
        import re
        if self._features is None or self._x_train_cache is None:
            raise RuntimeError("OpenFEAdapter.transform called before fit_transform")
        X_test = X_test.rename(columns=lambda col: re.sub(r'[^A-Za-z0-9_]', '_', str(col)))
        _, X_test_fe = transform(
            self._x_train_cache, X_test.reset_index(drop=True), self._features,
            n_jobs=max(1, self.n_jobs if self.n_jobs > 0 else 1),
        )
        # Enforce the same column order as train.
        return X_test_fe[self._train_columns_fe]
