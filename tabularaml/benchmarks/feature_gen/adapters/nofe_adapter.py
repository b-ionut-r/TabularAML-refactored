"""Identity adapter. Serves as the denominator in pct_improvement_over_nofe."""
from __future__ import annotations
import pandas as pd
from .base import FEFrameworkAdapter


class NoFEAdapter(FEFrameworkAdapter):
    name = "nofe"
    version = "1.0.0"

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        self._n_features_before = X_train.shape[1]
        self._n_features_after = X_train.shape[1]
        return X_train.copy()

    def transform(self, X_test: pd.DataFrame) -> pd.DataFrame:
        return X_test.copy()
