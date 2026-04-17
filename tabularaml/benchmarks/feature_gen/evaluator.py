"""Fixed base learner + held-out scoring for the cross-framework benchmark.

Keeps the base learner identical across all frameworks so the only variable
in the experiment is the FE pipeline applied to X_train.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from tabularaml.eval.scorers import (
    Scorer, rmse, binary_roc_auc, categorical_crossentropy,
)


BASE_LEARNER_PARAMS = dict(
    tree_method="hist",
    enable_categorical=True,
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    early_stopping_rounds=50,
    n_jobs=-1,
    verbosity=0,
)


def select_scorer(task: str, n_classes: int) -> Scorer:
    if task == "regression":
        return rmse
    return binary_roc_auc if n_classes == 2 else categorical_crossentropy


def build_base_learner(task: str, n_classes: int, seed: int):
    params = dict(BASE_LEARNER_PARAMS)
    params["random_state"] = int(seed)
    params["seed"] = int(seed)
    if task == "regression":
        return xgb.XGBRegressor(**params)
    if n_classes == 2:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"
        return xgb.XGBClassifier(**params)
    params["objective"] = "multi:softprob"
    params["eval_metric"] = "mlogloss"
    params["num_class"] = int(n_classes)
    return xgb.XGBClassifier(**params)


def _predict_for_scorer(model, X, task: str, n_classes: int) -> np.ndarray:
    if task == "regression":
        return np.asarray(model.predict(X))
    if n_classes == 2:
        return np.asarray(model.predict_proba(X))[:, 1]
    return np.asarray(model.predict_proba(X))


def score_on_holdout(
    X_train_fe: pd.DataFrame,
    y_train: np.ndarray,
    X_test_fe: pd.DataFrame,
    y_test: np.ndarray,
    task: str,
    n_classes: int,
    seed: int,
) -> Tuple[float, int]:
    """Train the fixed base learner on (X_train_fe, y_train) with a 10% internal
    val split used for early stopping, then score on (X_test_fe, y_test).

    Returns (score, n_boost_rounds_used).
    """
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    stratify = y_train if task == "classification" else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_fe, y_train,
        test_size=0.1, random_state=int(seed),
        stratify=stratify if (stratify is not None and len(np.unique(stratify)) > 1) else None,
    )

    model = build_base_learner(task, n_classes, seed)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = _predict_for_scorer(model, X_test_fe, task, n_classes)
    scorer = select_scorer(task, n_classes)
    score = float(scorer.score(y_test, y_pred))
    n_rounds = int(getattr(model, "best_iteration", model.n_estimators) or model.n_estimators)
    return score, n_rounds


def pct_improvement(score_framework: float, score_nofe: float, scorer: Scorer) -> float:
    """Sign-normalised so positive means 'framework beats no-FE'."""
    denom = abs(score_nofe)
    if denom == 0:
        return 0.0
    raw = (score_framework - score_nofe) / denom
    return raw if scorer.greater_is_better else -raw
