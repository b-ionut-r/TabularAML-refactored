"""Fixed base learner + held-out scoring for the cross-framework benchmark.

Keeps the base learner identical across all frameworks so the only variable
in the experiment is the FE pipeline applied to X_train.
"""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd
# XGBoost <=2.0.x references np.NaN which was removed in NumPy 2.0.
if not hasattr(np, "NaN"):
    np.NaN = np.nan  # type: ignore[attr-defined]
import xgboost as xgb
from sklearn.model_selection import train_test_split

from tabularaml.eval.cv import sanitize_model_features
from tabularaml.eval.scorers import (
    PREDEFINED_SCORERS,
    Scorer,
    accuracy,
    binary_crossentropy,
    binary_roc_auc,
    categorical_crossentropy,
    categorical_roc_auc,
    f1,
    mae,
    mse,
    pearson,
    precision,
    r2,
    recall,
    rmse,
    rmsle,
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
    verbosity=0,
)


EARLY_STOP_VALIDATION_FRACTION = 0.1


def select_scorer(task: str, n_classes: int) -> Scorer:
    if task == "regression":
        return rmse
    return binary_roc_auc if n_classes == 2 else categorical_crossentropy


def select_logging_scorers(task: str, n_classes: int) -> Tuple[Scorer, ...]:
    if task == "regression":
        return (rmse, mae, mse, r2, pearson, rmsle)
    if n_classes == 2:
        return (binary_roc_auc, binary_crossentropy, accuracy, precision, recall, f1)
    return (categorical_crossentropy, categorical_roc_auc, accuracy, precision, recall, f1)


def _safe_metric_value(scorer: Scorer, y_true, y_pred):
    try:
        value = float(scorer.score(y_true, y_pred))
    except Exception:
        return None
    if not np.isfinite(value):
        return None
    return value


def compute_holdout_metrics(
    y_true,
    y_pred,
    task: str,
    n_classes: int,
) -> Dict[str, float | None]:
    metrics: Dict[str, float | None] = {}
    for scorer in select_logging_scorers(task, n_classes):
        metrics[scorer.name] = _safe_metric_value(scorer, y_true, y_pred)
    return metrics


def compute_metric_gains(
    metric_scores: Dict[str, float | None],
    baseline_metric_scores: Dict[str, float | None],
) -> Dict[str, float | None]:
    gains: Dict[str, float | None] = {}
    for metric_name, framework_score in metric_scores.items():
        scorer = PREDEFINED_SCORERS.get(metric_name)
        baseline_score = baseline_metric_scores.get(metric_name)
        if scorer is None or framework_score is None or baseline_score is None:
            gains[metric_name] = None
            continue
        gains[metric_name] = float(pct_improvement(framework_score, baseline_score, scorer))
    return gains


def build_base_learner(task: str, n_classes: int, seed: int, n_jobs: int = 1):
    params = dict(BASE_LEARNER_PARAMS)
    params["random_state"] = int(seed)
    params["seed"] = int(seed)
    params["n_jobs"] = int(n_jobs)
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


def _safe_stratify_for_classification(y: np.ndarray, task: str):
    """Return stratification labels only when it is safe to use them."""
    if task not in ("classification", "multiclass"):
        return None
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) <= 1 or counts.min() < 2:
        return None
    return y


def split_early_stopping_validation(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    task: str,
    seed: int,
    validation_fraction: float = EARLY_STOP_VALIDATION_FRACTION,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Split training rows into fit/validation partitions for early stopping.

    This split is intentionally performed on raw training data, before feature
    engineering, so the validation fold remains unseen by adapter.fit_transform.
    """
    y_train = np.asarray(y_train)
    split_kwargs = dict(
        test_size=float(validation_fraction),
        random_state=int(seed),
    )
    stratify = _safe_stratify_for_classification(y_train, task)
    if stratify is not None:
        split_kwargs["stratify"] = stratify
    try:
        return train_test_split(X_train, y_train, **split_kwargs)
    except ValueError:
        # Fall back to an unstratified split for tiny/degenerate label layouts.
        split_kwargs.pop("stratify", None)
        return train_test_split(X_train, y_train, **split_kwargs)


def sanitize_features(X: pd.DataFrame) -> pd.DataFrame:
    """Replace infinities before model handoff while preserving NaNs."""
    return sanitize_model_features(X)


def score_on_holdout(
    X_fit_fe: pd.DataFrame,
    y_fit: np.ndarray,
    X_val_fe: pd.DataFrame,
    y_val: np.ndarray,
    X_test_fe: pd.DataFrame,
    y_test: np.ndarray,
    task: str,
    n_classes: int,
    seed: int,
    n_jobs: int = 1,
    return_all_metrics: bool = False,
) -> Tuple[float, int] | Tuple[float, int, Dict[str, float | None]]:
    """Train the fixed base learner with an externally isolated early-stop fold.

    `X_val_fe` / `y_val` must be generated by transforming rows that were held
    out before feature engineering fit.

    Returns (score, n_boost_rounds_used).
    """
    X_fit_fe = sanitize_features(X_fit_fe)
    X_val_fe = sanitize_features(X_val_fe)
    X_test_fe = sanitize_features(X_test_fe)

    y_fit = np.asarray(y_fit)
    y_val = np.asarray(y_val)
    y_test = np.asarray(y_test)

    model = build_base_learner(task, n_classes, seed, n_jobs=n_jobs)

    for c in X_fit_fe.columns:
        if isinstance(X_fit_fe[c].dtype, pd.CategoricalDtype):
            cats = X_fit_fe[c].cat.categories
            X_val_fe[c] = pd.Categorical(X_val_fe[c], categories=cats)
            X_test_fe[c] = pd.Categorical(X_test_fe[c], categories=cats)

    model.fit(X_fit_fe, y_fit, eval_set=[(X_val_fe, y_val)], verbose=False)
    y_pred = _predict_for_scorer(model, X_test_fe, task, n_classes)
    scorer = select_scorer(task, n_classes)
    all_metrics = compute_holdout_metrics(y_test, y_pred, task, n_classes)
    score = all_metrics.get(scorer.name)
    if score is None:
        score = float(scorer.score(y_test, y_pred))
    n_rounds = int(getattr(model, "best_iteration", model.n_estimators) or model.n_estimators)
    if return_all_metrics:
        return score, n_rounds, all_metrics
    return score, n_rounds


def pct_improvement(score_framework: float, score_nofe: float, scorer: Scorer) -> float:
    """Sign-normalised so positive means 'framework beats no-FE'."""
    denom = abs(score_nofe)
    if denom == 0:
        return 0.0
    raw = (score_framework - score_nofe) / denom
    return raw if scorer.greater_is_better else -raw
