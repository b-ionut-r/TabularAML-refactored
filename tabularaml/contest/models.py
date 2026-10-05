"""Gradient-boosting model zoo with contest-grade defaults.

Each model is a thin wrapper exposing ``fit(X_tr, y_tr, X_va, y_va)`` with
early stopping on the validation fold and ``predict(X)`` returning a 1-D array
(regression / binary positive-class probability) or a 2-D probability matrix
(multiclass). Categorical columns arrive as pandas ``category`` dtype and are
handled natively by every backend.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


EARLY_STOPPING_ROUNDS = 200
MAX_ROUNDS = 20_000


class BaseModel:
    name = "base"

    def __init__(self, task: str, n_classes: int = 0, seed: int = 0,
                 n_jobs: int = -1, params: Optional[Dict] = None):
        self.task = task
        self.n_classes = n_classes
        self.seed = seed
        self.n_jobs = n_jobs
        self.params = dict(params or {})
        self.model = None
        self.best_iteration_: Optional[int] = None

    def fit(self, X_tr, y_tr, X_va, y_va):
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        raise NotImplementedError

    def _finish_proba(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p)
        if self.task == "binary" and p.ndim == 2:
            return p[:, 1]
        return p


class LGBMModel(BaseModel):
    name = "lgbm"

    DEFAULTS = dict(
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        colsample_bytree=0.7,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=1.0,
        max_bin=255,
        cat_smooth=10,
        min_data_per_group=50,
        verbosity=-1,
    )

    def fit(self, X_tr, y_tr, X_va, y_va):
        import lightgbm as lgb
        params = {**self.DEFAULTS, **self.params}
        params.update(n_estimators=params.pop("n_estimators", MAX_ROUNDS),
                      random_state=self.seed, n_jobs=self.n_jobs)
        if self.task == "regression":
            self.model = lgb.LGBMRegressor(**params)
        else:
            self.model = lgb.LGBMClassifier(**params)
        self.model.fit(
            X_tr, y_tr, eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        self.best_iteration_ = self.model.best_iteration_
        return self

    def predict(self, X):
        if self.task == "regression":
            return self.model.predict(X)
        return self._finish_proba(self.model.predict_proba(X))


class XGBModel(BaseModel):
    name = "xgb"

    DEFAULTS = dict(
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        tree_method="hist",
        max_cat_to_onehot=8,
        verbosity=0,
    )

    def fit(self, X_tr, y_tr, X_va, y_va):
        import xgboost as xgb
        params = {**self.DEFAULTS, **self.params}
        params.update(n_estimators=params.pop("n_estimators", MAX_ROUNDS),
                      random_state=self.seed, n_jobs=self.n_jobs,
                      enable_categorical=True,
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        if self.task == "regression":
            self.model = xgb.XGBRegressor(**params)
        elif self.task == "binary":
            self.model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **params)
        else:
            self.model = xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", **params)
        self.model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        self.best_iteration_ = int(self.model.best_iteration)
        return self

    def predict(self, X):
        if self.task == "regression":
            return self.model.predict(X)
        return self._finish_proba(self.model.predict_proba(X))


class CatBoostModel(BaseModel):
    name = "catboost"

    DEFAULTS = dict(
        learning_rate=0.06,
        depth=6,
        l2_leaf_reg=3.0,
        border_count=254,
        verbose=0,
        allow_writing_files=False,
    )

    @staticmethod
    def _prep(X: pd.DataFrame, cat_cols):
        if not cat_cols:
            return X
        X = X.copy()
        for c in cat_cols:
            X[c] = X[c].astype(str)
        return X

    def fit(self, X_tr, y_tr, X_va, y_va):
        from catboost import CatBoostClassifier, CatBoostRegressor
        self.cat_cols_ = [c for c in X_tr.columns if isinstance(X_tr[c].dtype, pd.CategoricalDtype)]
        params = {**self.DEFAULTS, **self.params}
        params.update(iterations=params.pop("iterations", MAX_ROUNDS),
                      random_seed=self.seed,
                      thread_count=self.n_jobs if self.n_jobs > 0 else -1,
                      od_type="Iter", od_wait=EARLY_STOPPING_ROUNDS,
                      use_best_model=True)
        if self.task == "regression":
            self.model = CatBoostRegressor(loss_function="RMSE", **params)
        elif self.task == "binary":
            self.model = CatBoostClassifier(loss_function="Logloss", **params)
        else:
            self.model = CatBoostClassifier(loss_function="MultiClass", **params)
        self.model.fit(self._prep(X_tr, self.cat_cols_), y_tr,
                       eval_set=(self._prep(X_va, self.cat_cols_), y_va),
                       cat_features=self.cat_cols_ or None)
        self.best_iteration_ = int(self.model.get_best_iteration())
        return self

    def predict(self, X):
        X = self._prep(X, self.cat_cols_)
        if self.task == "regression":
            return self.model.predict(X)
        return self._finish_proba(self.model.predict_proba(X))


MODEL_REGISTRY = {
    "lgbm": LGBMModel,
    "xgb": XGBModel,
    "catboost": CatBoostModel,
}


def make_model(spec, task: str, n_classes: int, seed: int, n_jobs: int) -> BaseModel:
    """``spec`` is a registry name or ``(name, params_dict)``."""
    if isinstance(spec, str):
        name, params = spec, {}
    else:
        name, params = spec
    cls = MODEL_REGISTRY[name.split(":")[0]]
    m = cls(task=task, n_classes=n_classes, seed=seed, n_jobs=n_jobs, params=params)
    m.name = name
    return m
