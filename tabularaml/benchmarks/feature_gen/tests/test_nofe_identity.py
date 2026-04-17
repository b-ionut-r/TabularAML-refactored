"""NoFE adapter output must match a direct XGBoost run on raw features.

Guards against any silent preprocessing creeping into the evaluator pipeline:
if this test fails, the NoFE denominator is no longer a clean 'no-FE baseline'
and every pct_improvement number loses meaning.
"""
from __future__ import annotations
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

from tabularaml.benchmarks.feature_gen.adapters import get_adapter_cls
from tabularaml.benchmarks.feature_gen.evaluator import (
    BASE_LEARNER_PARAMS,
    score_on_holdout,
    select_scorer,
    split_early_stopping_validation,
)


def test_nofe_matches_direct_xgb(toy_cls):
    X, y = toy_cls
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y,
    )
    y_tr = y_tr.values
    y_te = y_te.values

    adapter = get_adapter_cls("nofe")(
        task="classification", time_budget_s=60, random_state=7, n_jobs=1,
    )
    X_tr_fit, X_tr_val, y_tr_fit, y_tr_val = split_early_stopping_validation(
        X_tr,
        y_tr,
        task="classification",
        seed=7,
    )
    X_tr_fe = adapter.fit_transform(X_tr_fit, y_tr_fit)
    X_val_fe = adapter.transform(X_tr_val)
    X_te_fe = adapter.transform(X_te)

    score, _ = score_on_holdout(
        X_tr_fe,
        y_tr_fit,
        X_val_fe,
        y_tr_val,
        X_te_fe,
        y_te,
        task="classification", n_classes=2, seed=7,
    )

    # Direct XGBoost call mirroring build_base_learner + same early-stop split.
    params = dict(BASE_LEARNER_PARAMS)
    params["random_state"] = 7
    params["seed"] = 7
    params["objective"] = "binary:logistic"
    params["eval_metric"] = "logloss"
    direct = xgb.XGBClassifier(**params)
    direct.fit(X_tr_fit, y_tr_fit, eval_set=[(X_tr_val, y_tr_val)], verbose=False)
    y_pred = direct.predict_proba(X_te)[:, 1]
    scorer = select_scorer("classification", 2)
    direct_score = float(scorer.score(y_te, y_pred))

    assert np.isclose(score, direct_score, atol=1e-10), (
        f"NoFE adapter diverges from direct XGBoost: {score} vs {direct_score}"
    )
