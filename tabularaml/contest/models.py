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


class MLPModel(BaseModel):
    """Small scikit-learn MLP on quantile-normalised numerics and one-hot categories.

    Weaker than the boosters on its own but makes different errors, which is
    what the hill-climbed blend needs. Early-stopped on the validation fold.
    """
    name = "mlp"

    DEFAULTS = dict(hidden_layer_sizes=(256, 128), alpha=1e-4, learning_rate_init=1e-3,
                    batch_size=256, max_epochs=200, patience=12, max_onehot=50)

    def _design(self, X: pd.DataFrame, fit: bool):
        from sklearn.preprocessing import QuantileTransformer
        if fit:
            self.num_cols_ = [c for c in X.columns if not isinstance(X[c].dtype, pd.CategoricalDtype)]
            self.cat_cols_ = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
            self.levels_ = {}
            for c in self.cat_cols_:
                vc = X[c].value_counts()
                self.levels_[c] = list(vc.index[:self.p["max_onehot"]])
        parts = []
        if self.num_cols_:
            M = X[self.num_cols_].to_numpy(dtype=float)
            nan = np.isnan(M)
            if fit:
                self.med_ = np.nanmedian(M, axis=0)
                self.med_ = np.where(np.isnan(self.med_), 0.0, self.med_)
                self.qt_ = QuantileTransformer(n_quantiles=min(1000, len(X)), output_distribution="normal",
                                               subsample=100_000, random_state=self.seed)
                self.qt_.fit(np.where(nan, self.med_, M))
                self.nan_cols_ = np.where(nan.any(0))[0]
            parts.append(self.qt_.transform(np.where(nan, self.med_, M)))
            parts.append(nan[:, self.nan_cols_].astype(float))
        for c in self.cat_cols_:
            v = X[c].astype(object).to_numpy()
            parts.append(np.stack([v == lv for lv in self.levels_[c]], 1).astype(float)
                         if self.levels_[c] else np.zeros((len(X), 0)))
        return np.hstack(parts).astype(np.float32) if parts else np.zeros((len(X), 1), np.float32)

    def fit(self, X_tr, y_tr, X_va, y_va):
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        self.p = {**self.DEFAULTS, **self.params}
        A, B = self._design(X_tr, True), self._design(X_va, False)
        y_tr, y_va = np.asarray(y_tr), np.asarray(y_va)
        if self.task == "regression":
            self.mu_, self.sd_ = float(y_tr.mean()), float(y_tr.std() + 1e-12)
            est = MLPRegressor
        else:
            est = MLPClassifier
        kw = dict(hidden_layer_sizes=self.p["hidden_layer_sizes"], alpha=self.p["alpha"],
                  learning_rate_init=self.p["learning_rate_init"],
                  batch_size=min(self.p["batch_size"], len(A)), random_state=self.seed)
        self.model = est(max_iter=1, warm_start=True, **kw)
        classes = np.arange(self.n_classes if self.task == "multiclass" else 2)
        best, best_state, bad = np.inf, None, 0
        import copy
        import warnings
        for epoch in range(self.p["max_epochs"]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.task == "regression":
                    self.model.partial_fit(A, (y_tr - self.mu_) / self.sd_)
                else:
                    self.model.partial_fit(A, y_tr, classes=classes)
            loss = self._val_loss(B, y_va)
            if loss < best - 1e-6:
                best, best_state, bad = loss, copy.deepcopy(self.model), 0
                self.best_iteration_ = epoch + 1
            else:
                bad += 1
                if bad >= self.p["patience"]:
                    break
        self.model = best_state or self.model
        return self

    def _raw_predict(self, A):
        if self.task == "regression":
            return self.model.predict(A) * self.sd_ + self.mu_
        return np.clip(self.model.predict_proba(A), 1e-7, 1 - 1e-7)

    def _val_loss(self, B, y):
        p = self._raw_predict(B)
        if self.task == "regression":
            return float(np.mean((p - y) ** 2))
        return float(-np.mean(np.log(p[np.arange(len(y)), y.astype(int)])))

    def predict(self, X):
        return self._finish_proba(self._raw_predict(self._design(X, False)))


MODEL_REGISTRY = {
    "lgbm": LGBMModel,
    "xgb": XGBModel,
    "catboost": CatBoostModel,
    "mlp": MLPModel,
}

# Diverse variants for a wider blend (``ContestSolver(models=ZOO)``).
ZOO = [
    "lgbm",
    ("lgbm:deep", dict(num_leaves=127, min_child_samples=10, learning_rate=0.02, colsample_bytree=0.5)),
    ("lgbm:shallow", dict(num_leaves=7, min_child_samples=40, learning_rate=0.05, reg_lambda=5.0)),
    "xgb",
    ("xgb:deep", dict(max_depth=0, grow_policy="lossguide", max_leaves=63, min_child_weight=3.0,
                      colsample_bytree=0.5)),
    "catboost",
    ("catboost:deep", dict(depth=8, l2_leaf_reg=6.0)),
]


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
