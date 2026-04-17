"""Adapter wrapping Featuretools in single-table mode.

Uses the modern (1.x) API: ft.EntitySet().add_dataframe + ft.dfs +
ft.calculate_feature_matrix. Restricted to arithmetic transform primitives to
keep the operator set comparable to TabularAML's unary/binary numeric ops.
"""
from __future__ import annotations
from typing import Literal, Optional, Sequence
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype

from .base import FEFrameworkAdapter


DEFAULT_PRIMITIVES = (
    "add_numeric",
    "subtract_numeric",
    "multiply_numeric",
    "divide_numeric",
    "absolute",
)


class FeaturetoolsAdapter(FEFrameworkAdapter):
    name = "featuretools"
    version = "upstream-1.x"

    def __init__(
        self,
        task: Literal["regression", "classification"],
        time_budget_s: int,
        random_state: int,
        n_jobs: int = -1,
        trans_primitives: Sequence[str] = DEFAULT_PRIMITIVES,
        max_depth: int = 1,
        **framework_kwargs,
    ):
        super().__init__(task, time_budget_s, random_state, n_jobs, **framework_kwargs)
        self.trans_primitives = list(trans_primitives)
        self.max_depth = int(max_depth)
        self._feature_defs = None
        self._train_columns_fe: Optional[list] = None
        self._num_medians: dict = {}

    @staticmethod
    def _to_index_frame(X: pd.DataFrame) -> pd.DataFrame:
        """Attach a stable integer index column for ft's EntitySet."""
        frame = X.reset_index(drop=True).copy()
        frame.insert(0, "_bench_idx", np.arange(len(frame)))
        # Featuretools needs no-NaN index and numeric/object-consistent dtypes.
        for c in frame.columns:
            if is_object_dtype(frame[c]):
                frame[c] = frame[c].astype("category")
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
        # Drop constant / all-NaN columns (train time only — decide which to keep).
        if fit:
            keep = [
                c for c in matrix.columns
                if not matrix[c].isna().all()
                and (matrix[c].dropna().nunique() > 1 if is_numeric_dtype(matrix[c]) else True)
            ]
            matrix = matrix[keep]
            # Compute medians for later NaN filling + remember schema.
            self._num_medians = {
                c: float(np.nanmedian(matrix[c].astype(float).values))
                for c in matrix.columns if is_numeric_dtype(matrix[c])
            }
        # Ensure numeric NaN filled; cast object → category.
        for c in matrix.columns:
            if is_numeric_dtype(matrix[c]):
                fill = self._num_medians.get(c, 0.0)
                matrix[c] = matrix[c].astype(float).fillna(fill if not np.isnan(fill) else 0.0)
            elif is_object_dtype(matrix[c]):
                matrix[c] = matrix[c].astype("category")
        return matrix

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        self._n_features_before = X_train.shape[1]
        keep = [c for c in X_train.columns if not X_train[c].isna().all()]
        self._valid_cols = keep
        frame = self._to_index_frame(X_train[keep])
        ft, es = self._dfs(frame)
        matrix, feature_defs = ft.dfs(
            entityset=es,
            target_dataframe_name="df",
            trans_primitives=self.trans_primitives,
            max_depth=self.max_depth,
            verbose=False,
        )
        self._feature_defs = feature_defs
        matrix = self._postprocess(matrix, fit=True)
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
        )
        matrix = self._postprocess(matrix, fit=False)
        # Align columns to training schema; missing columns filled with medians.
        for c in self._train_columns_fe:
            if c not in matrix.columns:
                matrix[c] = self._num_medians.get(c, 0.0)
        matrix = matrix[self._train_columns_fe]
        matrix.index = X_test.index
        return matrix
