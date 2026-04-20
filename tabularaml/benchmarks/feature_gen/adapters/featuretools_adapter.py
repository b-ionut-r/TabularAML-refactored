"""Adapter wrapping Featuretools in single-table mode.

Uses the modern (1.x) API: ft.EntitySet().add_dataframe + ft.dfs +
ft.calculate_feature_matrix.
"""
from __future__ import annotations
from typing import Literal, Optional
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_datetime64_any_dtype

from .base import FEFrameworkAdapter


class _FeaturetoolsUpstreamBugError(RuntimeError):
    """Raised for known Featuretools upstream bugs (e.g. TypeConversionError Int64)."""


class FeaturetoolsAdapter(FEFrameworkAdapter):
    name = "featuretools"
    version = "upstream-1.x"

    def __init__(
        self,
        task: Literal["regression", "classification"],
        time_budget_s: int,
        random_state: int,
        n_jobs: int = -1,
        max_depth: int = 1,
        **framework_kwargs,
    ):
        super().__init__(task, time_budget_s, random_state, n_jobs, **framework_kwargs)
        self.max_depth = int(max_depth)
        self._feature_defs = None
        self._train_columns_fe: Optional[list] = None
        self._num_medians: dict = {}

    @staticmethod
    def _to_index_frame(X: pd.DataFrame) -> pd.DataFrame:
        """Attach a stable integer index column for ft's EntitySet."""
        frame = X.reset_index(drop=True).copy()
        frame.insert(0, "_bench_idx", np.arange(len(frame)))
        # Convert categoricals back to object to prevent Pandas comparison TypeErrors
        # ('Categoricals can only be compared if categories are the same')
        # Featuretools (Woodwork) still automatically infers them as Categorical.
        for c in frame.columns:
            if isinstance(frame[c].dtype, pd.CategoricalDtype):
                frame[c] = frame[c].astype(object)
        return frame

    def _dfs(self, frame: pd.DataFrame):
        import featuretools as ft
        es = ft.EntitySet(id="bench")
        es = es.add_dataframe(
            dataframe_name="df",
            dataframe=frame,
            index="_bench_idx",
        )
        return ft, es

    def _postprocess(self, matrix: pd.DataFrame, fit: bool) -> pd.DataFrame:
        matrix = matrix.copy()
        # Replace inf with NaN — XGBoost handles NaN natively, so do NOT fill.
        for c in matrix.columns:
            if is_numeric_dtype(matrix[c]):
                matrix[c] = matrix[c].replace([np.inf, -np.inf], np.nan)
        # Drop constant / all-NaN columns (train time only — decide which to keep).
        if fit:
            keep = [
                c for c in matrix.columns
                if not matrix[c].isna().all()
                and (matrix[c].dropna().nunique() > 1 if is_numeric_dtype(matrix[c]) else True)
            ]
            matrix = matrix[keep]
        # Cast object → category.
        for c in matrix.columns:
            if is_object_dtype(matrix[c]):
                matrix[c] = matrix[c].astype("category")
        return matrix

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        self._n_features_before = X_train.shape[1]
        keep = [c for c in X_train.columns if not X_train[c].isna().all()]
        self._valid_cols = keep
        frame = self._to_index_frame(X_train[keep])
        ft, es = self._dfs(frame)
        
        t_primitives = [
            "add_numeric", "subtract_numeric", "multiply_numeric", "divide_numeric",
            "absolute", "modulo_numeric", "square_root", "natural_logarithm",
            "greater_than", "equal", "and", "not"
        ]
        if any(is_datetime64_any_dtype(frame[c]) for c in frame.columns):
            t_primitives.extend(["year", "month", "weekday", "day", "hour", "is_weekend"])

        try:
            matrix, feature_defs = ft.dfs(
                entityset=es,
                target_dataframe_name="df",
                ignore_columns={"df": ["_bench_idx"]},
                trans_primitives=t_primitives,
                groupby_trans_primitives=["cum_sum", "cum_mean", "cum_min", "cum_max", "cum_count"],
                max_depth=self.max_depth,
                verbose=False,
                n_jobs=self.n_jobs,
            )
        except Exception as e:
            if "TypeConversionError" in type(e).__name__ or "TypeConversionError" in str(type(e).__mro__):
                raise _FeaturetoolsUpstreamBugError(str(e)) from e
            raise
        self._feature_defs = feature_defs
        matrix = self._postprocess(matrix, fit=True)
        if "_bench_idx" in matrix.columns:
            matrix = matrix.drop(columns=["_bench_idx"])
        self._train_columns_fe = list(matrix.columns)
        self._n_features_after = matrix.shape[1]
        matrix.index = X_train.index
        return matrix

    def transform(self, X_test: pd.DataFrame) -> pd.DataFrame:
        import featuretools as ft
        if self._feature_defs is None:
            raise RuntimeError("FeaturetoolsAdapter.transform called before fit_transform")
        valid_cols = getattr(self, '_valid_cols', X_test.columns.tolist())
        frame = self._to_index_frame(X_test[[c for c in valid_cols if c in X_test.columns]])
        _, es = self._dfs(frame)
        matrix = ft.calculate_feature_matrix(
            features=self._feature_defs,
            entityset=es,
            verbose=False,
            n_jobs=self.n_jobs,
        )
        matrix = self._postprocess(matrix, fit=False)
        if "_bench_idx" in matrix.columns:
            matrix = matrix.drop(columns=["_bench_idx"])
        # Align columns to training schema; missing columns filled with NaN
        # (XGBoost handles NaN natively via learned split directions).
        for c in self._train_columns_fe:
            if c not in matrix.columns:
                matrix[c] = np.nan
        matrix = matrix[self._train_columns_fe]
        matrix.index = X_test.index
        return matrix
