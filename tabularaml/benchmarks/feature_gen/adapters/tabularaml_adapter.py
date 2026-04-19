"""Adapter wrapping TabularAML's FeatureGenerator.

Uses mode="medium" (15 min preset) by default; per-run time budget from the
runner overrides the preset's internal time_budget so global wall-clock caps
stay authoritative.
"""
from __future__ import annotations
from typing import Literal, Optional
import pandas as pd

from tabularaml.generate.features import FeatureGenerator
from tabularaml.eval.scorers import rmse, binary_roc_auc, categorical_crossentropy

from .base import FEFrameworkAdapter


class TabularAMLAdapter(FEFrameworkAdapter):
    name = "tabularaml"
    version = "0.2.0"
    gpu = False

    def __init__(
        self,
        task: Literal["regression", "classification"],
        time_budget_s: int,
        random_state: int,
        n_jobs: int = -1,
        mode: str = "medium",
        use_gpu: bool = False,  # CPU-only for fair comparison; pass use_gpu=True for a separate GPU arm
        **framework_kwargs,
    ):
        super().__init__(task, time_budget_s, random_state, n_jobs, **framework_kwargs)
        self.mode = mode
        self.use_gpu = use_gpu
        self._gen: Optional[FeatureGenerator] = None
        self._internal_log: dict = {}

    def _pick_scorer(self, y: pd.Series):
        if self.task == "regression":
            return rmse
        n_classes = int(pd.Series(y).nunique())
        return binary_roc_auc if n_classes == 2 else categorical_crossentropy

    @staticmethod
    def _decategorise(X: pd.DataFrame) -> pd.DataFrame:
        """TabularAML's SHAP interaction path chokes on pandas Categorical dtypes
        (setitem with a new category raises). Convert cat columns to object so
        the internal CategoricalEncoder sees canonical inputs.
        """
        X = X.copy()
        for c in X.columns:
            if isinstance(X[c].dtype, pd.CategoricalDtype):
                X[c] = X[c].astype(object)
        return X

    @staticmethod
    def _recategorise(X: pd.DataFrame) -> pd.DataFrame:
        """Cast any lingering object-dtype column to pandas `category` so the
        benchmark's numeric-or-category contract holds.
        """
        X = X.copy()
        for c in X.columns:
            if X[c].dtype == object:
                X[c] = X[c].astype("category")
        return X

    def fit_transform(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        self._n_features_before = X_train.shape[1]
        X_train = self._decategorise(X_train)

        scorer = self._pick_scorer(y_train)
        self._gen = FeatureGenerator(
            task=self.task,
            scorer=scorer,
            mode=self.mode,
            # time_budget=self.time_budget_s,  # Intentionally omitted: mode preset owns the per-run budget
            use_gpu=self.use_gpu,
            log_file=None,
        )
        self._gen.generate(X_train, y_train)
        # Refit the pipeline on the full training frame so transform() is canonical.
        self._gen.fit(X_train, y_train)
        X_train_fe = self._recategorise(self._gen.transform(X_train))

        # Capture a small internal-log snapshot (sign-normalised where easy).
        self._internal_log = {
            "mode": self.mode,
            "elapsed_time": getattr(self._gen, "elapsed_time", None),
            "n_init_feats": getattr(self._gen, "n_init_feats", None),
            "n_added_feats": getattr(self._gen, "n_added_feats", None),
            "initial_metric": getattr(self._gen, "initial_metric", None),
            "final_metric": getattr(self._gen, "final_metric", None),
            "pct_gain": getattr(self._gen, "pct_gain", None),
        }

        self._n_features_after = X_train_fe.shape[1]
        return X_train_fe

    def transform(self, X_test: pd.DataFrame) -> pd.DataFrame:
        if self._gen is None:
            raise RuntimeError("TabularAMLAdapter.transform called before fit_transform")
        return self._recategorise(self._gen.transform(self._decategorise(X_test)))

    def get_internal_log(self) -> dict:
        return dict(self._internal_log)
