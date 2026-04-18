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
import warnings
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
        self._col_mapping: dict = {}

    def _task_for_openfe(self, y: pd.Series) -> str:
        if self.task == "regression":
            return "regression"
        # OpenFE handles binary + multiclass via label dtype
        return "classification"

    @staticmethod
    def _patch_init_score_flatten() -> None:
        """Patch LightGBM.fit to correctly reshape init_score for multiclass tasks.
        Handles both 2D arrays and list-of-arrays inputs from OpenFE."""
        import numpy as np
        import lightgbm as lgb

        if getattr(lgb.LGBMModel.fit, "_openfe_init_patched", False):
            return

        _orig_fit = lgb.LGBMModel.fit

        def _fit_with_correct_init_score(self_lgb, X, y, **kwargs):
            init_score = kwargs.get('init_score', None)
            if init_score is not None:
                # If init_score is a list of arrays (e.g., per-class scores), stack them
                if isinstance(init_score, (list, tuple)):
                    try:
                        init_score = np.column_stack(init_score)
                    except ValueError:
                        # If they are already 2D-like, concatenate along axis 1
                        init_score = np.concatenate(
                            [np.asarray(arr).reshape(-1, 1) for arr in init_score],
                            axis=1
                        )
                else:
                    init_score = np.asarray(init_score)

                # Now init_score is 2D (n_samples, n_classes) for multiclass,
                # or 1D (n_samples,) for binary/regression.
                if init_score.ndim == 2:
                    # LightGBM expects a 1D array in column-major (Fortran) order
                    kwargs['init_score'] = init_score.ravel(order='F')
                elif init_score.ndim == 1:
                    # Binary or regression – leave as is
                    pass
                else:
                    # Unexpected dimensionality – remove init_score to be safe
                    kwargs.pop('init_score', None)

            return _orig_fit(self_lgb, X, y, **kwargs)

        _fit_with_correct_init_score._openfe_init_patched = True
        lgb.LGBMModel.fit = _fit_with_correct_init_score

    @staticmethod
    def _patch_sklearn_mse() -> None:
        """Patch sklearn mean_squared_error for squared=False removal (sklearn>=1.4)."""
        import sklearn.metrics as sm
        import numpy as np

        if getattr(sm.mean_squared_error, "_openfe_patched", False):
            return

        _orig_mse = sm.mean_squared_error

        def _compat_mse(
            y_true,
            y_pred,
            *,
            sample_weight=None,
            multioutput="uniform_average",
            squared=True,
        ):
            result = _orig_mse(
                y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput
            )
            return result if squared else np.sqrt(result)

        _compat_mse._openfe_patched = True
        sm.mean_squared_error = _compat_mse

    def _safe_column_names(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Replace column names with safe alphanumeric identifiers (col_0, col_1, ...)."""
        safe_names = [f"col_{i}" for i in range(df.shape[1])]
        mapping = dict(zip(df.columns, safe_names))
        return df.rename(columns=mapping), mapping

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        # Apply patches (init_score flattening and sklearn MSE)
        self._patch_init_score_flatten()
        self._patch_sklearn_mse()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from openfe import OpenFE, transform

            self._n_features_before = X_train.shape[1]

            # Reset to contiguous 0-based index (required by OpenFE)
            X_train = X_train.reset_index(drop=True)
            y_train = pd.Series(y_train).reset_index(drop=True)
            self._x_train_cache = X_train.copy()

            # Replace column names with safe identifiers to avoid Feather/LightGBM crashes
            X_train_safe, self._col_mapping = self._safe_column_names(X_train)
            reverse_mapping = {v: k for k, v in self._col_mapping.items()}

            y_df = pd.DataFrame({"_label": y_train.values}, index=X_train_safe.index)

            effective_blocks = min(self.n_data_blocks, max(2, len(X_train_safe) // 100))

            self._ofe = OpenFE()
            # Change to a fresh temp directory so parallel workers don't collide
            os.chdir(tempfile.mkdtemp())
            self._features = self._ofe.fit(
                data=X_train_safe,
                label=y_df,
                task=self._task_for_openfe(y_train),
                n_jobs=max(1, self.n_jobs if self.n_jobs > 0 else 1),
                n_data_blocks=effective_blocks,
                feature_boosting=self.feature_boosting,
                seed=self.random_state,
                verbose=False,
            )
                
            X_train_fe_safe, _ = transform(
                X_train_safe,
                X_train_safe,
                self._features,
                n_jobs=max(1, self.n_jobs if self.n_jobs > 0 else 1),
            )

            # Restore original column names where possible (base features)
            # Generated features keep their safe names (they never had special chars anyway)
            new_cols = []
            for col in X_train_fe_safe.columns:
                if col in reverse_mapping:
                    new_cols.append(reverse_mapping[col])
                else:
                    new_cols.append(col)
            X_train_fe_safe.columns = new_cols

            self._train_columns_fe = list(X_train_fe_safe.columns)
            self._n_features_after = X_train_fe_safe.shape[1]
            return X_train_fe_safe

    def transform(self, X_test: pd.DataFrame) -> pd.DataFrame:
        if self._features is None or self._x_train_cache is None:
            raise RuntimeError("OpenFEAdapter.transform called before fit_transform")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from openfe import transform

            X_test = X_test.reset_index(drop=True)
            # Apply the same column renaming to test data
            X_test_safe = X_test.rename(columns=self._col_mapping)

            _, X_test_fe_safe = transform(
                self._x_train_cache.rename(columns=self._col_mapping),
                X_test_safe,
                self._features,
                n_jobs=max(1, self.n_jobs if self.n_jobs > 0 else 1),
            )

            reverse_mapping = {v: k for k, v in self._col_mapping.items()}
            new_cols = []
            for col in X_test_fe_safe.columns:
                if col in reverse_mapping:
                    new_cols.append(reverse_mapping[col])
                else:
                    new_cols.append(col)
            X_test_fe_safe.columns = new_cols

        # Enforce same column order as train
        return X_test_fe_safe[self._train_columns_fe]