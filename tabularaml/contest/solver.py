"""End-to-end contest solver: K-fold GBDT zoo + out-of-fold ensembling.

Typical use::

    solver = ContestSolver(metric="auc").fit(train.drop(columns="target"), train["target"])
    submission["target"] = solver.predict(test)
    print(solver.leaderboard_)

Every model is trained on the same folds, so the out-of-fold (OOF) predictions
are directly comparable and the hill-climbing ensemble is fitted on them. The
OOF ensemble score is mildly optimistic (weights are chosen on the same OOF
rows); the benchmark in ``scripts/bench_contest.py`` measures on an untouched
outer holdout instead.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.preprocessing import LabelEncoder

from .ensemble import Stacker, blend, choose_ensemble, hill_climb
from .metrics import Metric, default_metric, get_metric
from .models import make_model

DEFAULT_MODELS = ("lgbm", "xgb", "catboost")


def infer_task(y: pd.Series) -> str:
    y = pd.Series(y)
    if y.dtype == object or isinstance(y.dtype, pd.CategoricalDtype) or y.dtype == bool:
        return "binary" if y.nunique() == 2 else "multiclass"
    n_unique = y.nunique()
    if n_unique == 2:
        return "binary"
    if pd.api.types.is_integer_dtype(y) and n_unique <= 20:
        return "multiclass"
    return "regression"


def _as_str(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), "__NA__").astype(str)


def prepare_frames(X: pd.DataFrame, X_test: Optional[pd.DataFrame] = None,
                   cat_cols: Optional[Sequence[str]] = None):
    """Cast object/bool/declared columns to a shared ``category`` dtype.

    Categories are the union of train and test values so that every backend
    sees identical codes on both sides.
    """
    X = X.copy()
    X_test = X_test.copy() if X_test is not None else None
    declared = set(cat_cols or [])
    for c in X.columns:
        is_cat = (c in declared or X[c].dtype == object or X[c].dtype == bool
                  or isinstance(X[c].dtype, pd.CategoricalDtype)
                  or pd.api.types.is_string_dtype(X[c]))
        if not is_cat:
            continue
        vals = _as_str(X[c])
        if X_test is not None:
            test_vals = _as_str(X_test[c])
            cats = pd.Index(pd.unique(pd.concat([vals, test_vals]))).sort_values()
            X_test[c] = pd.Categorical(test_vals, categories=cats)
        else:
            cats = pd.Index(pd.unique(vals)).sort_values()
        X[c] = pd.Categorical(vals, categories=cats)
    return X, X_test


class ContestSolver:
    """K-fold GBDT ensemble tuned for leaderboard metrics.

    Parameters
    ----------
    task : "regression" | "binary" | "multiclass" | None
        Inferred from ``y`` when None.
    metric : str | Metric | None
        Leaderboard metric; drives the ensemble weights. Defaults to rmse /
        logloss.
    models : sequence of model specs
        Registry names (``"lgbm"``, ``"xgb"``, ``"catboost"``) or
        ``("lgbm:deep", {"num_leaves": 127})`` tuples for variants.
    n_folds : int
    seeds : sequence of int
        One full K-fold run per seed, each with a different fold split; OOF
        and test predictions are averaged over seeds.
    cv : sklearn splitter, optional
        Overrides the default (Stratified)KFold, e.g. ``GroupKFold`` or
        ``TimeSeriesSplit``. With a single seed only.
    groups : array-like, optional
        Group labels; switches the default splitter to ``GroupKFold``.
    cat_cols : list of str, optional
        Extra columns to treat as categorical (object/bool columns always are).
    log_target : bool | None
        Train regression models on ``log1p(y)``. Defaults to True for rmsle.
    ensemble : "auto" | "hill" | "stack"
        Blend by Caruana hill climbing, by a linear stacker on OOF logits, or
        pick whichever scores better under cross-validation on the OOF rows.
    """

    def __init__(self, task: Optional[str] = None, metric=None,
                 models: Sequence = DEFAULT_MODELS, n_folds: int = 5,
                 seeds: Sequence[int] = (0,), cv=None, groups=None,
                 cat_cols: Optional[Sequence[str]] = None,
                 log_target: Optional[bool] = None, n_jobs: int = -1,
                 ensemble: str = "auto", verbose: bool = True):
        self.task = task
        self.metric = metric
        self.models = list(models)
        self.n_folds = n_folds
        self.seeds = list(seeds)
        self.cv = cv
        self.groups = groups
        self.cat_cols = cat_cols
        self.log_target = log_target
        self.n_jobs = n_jobs
        self.ensemble = ensemble
        self.verbose = verbose

    # ------------------------------------------------------------------ utils
    def _log(self, msg: str):
        if self.verbose:
            print(f"[ContestSolver] {msg}", flush=True)

    def _splits(self, X, y, seed):
        if self.cv is not None:
            return list(self.cv.split(X, y, self.groups))
        if self.groups is not None:
            return list(GroupKFold(n_splits=self.n_folds).split(X, y, self.groups))
        if self.task_ == "regression":
            return list(KFold(self.n_folds, shuffle=True, random_state=seed).split(X))
        return list(StratifiedKFold(self.n_folds, shuffle=True, random_state=seed).split(X, y))

    def _empty_pred(self, n):
        return np.zeros((n, self.n_classes_)) if self.task_ == "multiclass" else np.zeros(n)

    def _to_output(self, p):
        return np.expm1(p) if self.log_target_ else p

    # -------------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame, y, X_test: Optional[pd.DataFrame] = None):
        t0 = time.time()
        y = pd.Series(np.asarray(y))
        self.task_ = self.task or infer_task(y)
        self.metric_: Metric = get_metric(self.metric) if self.metric else default_metric(self.task_)
        if self.task_ not in self.metric_.tasks:
            raise ValueError(f"metric {self.metric_.name} is not defined for task {self.task_}")
        if self.cv is not None and len(self.seeds) > 1:
            raise ValueError("a custom cv splitter supports a single seed")

        if self.task_ == "regression":
            self.label_encoder_ = None
            self.n_classes_ = 0
            y_fit = y.astype(float).to_numpy()
            self.log_target_ = (self.metric_.name == "rmsle") if self.log_target is None else self.log_target
            y_model = np.log1p(y_fit) if self.log_target_ else y_fit
        else:
            self.label_encoder_ = LabelEncoder().fit(y)
            y_fit = self.label_encoder_.transform(y)
            self.n_classes_ = len(self.label_encoder_.classes_)
            self.log_target_ = False
            y_model = y_fit
        self.y_ = y_fit

        X, X_test = prepare_frames(X.reset_index(drop=True),
                                   None if X_test is None else X_test.reset_index(drop=True),
                                   self.cat_cols)
        self.columns_ = list(X.columns)
        self.categories_ = {c: X[c].cat.categories for c in X.columns
                            if isinstance(X[c].dtype, pd.CategoricalDtype)}

        n = len(X)
        self.oof_: Dict[str, np.ndarray] = {}
        self.test_pred_: Dict[str, np.ndarray] = {}
        self.fold_models_: Dict[str, List] = {}
        self.oof_scores_: Dict[str, float] = {}
        rows = []

        for spec in self.models:
            name = spec if isinstance(spec, str) else spec[0]
            oof_sum = self._empty_pred(n)
            test_sum = None if X_test is None else self._empty_pred(len(X_test))
            fitted, iters, t_model = [], [], time.time()
            for seed in self.seeds:
                for tr, va in self._splits(X, y_fit, seed):
                    m = make_model(spec, self.task_, self.n_classes_, seed, self.n_jobs)
                    m.fit(X.iloc[tr], y_model[tr], X.iloc[va], y_model[va])
                    oof_sum[va] += m.predict(X.iloc[va])
                    if X_test is not None:
                        test_sum += m.predict(X_test)
                    fitted.append(m)
                    iters.append(m.best_iteration_ or 0)
            n_folds_run = len(fitted) // len(self.seeds)
            oof = self._to_output(oof_sum / len(self.seeds))
            self.oof_[name] = oof
            self.fold_models_[name] = fitted
            if X_test is not None:
                self.test_pred_[name] = self._to_output(test_sum / (len(self.seeds) * n_folds_run))
            score = self.metric_(y_fit, oof)
            self.oof_scores_[name] = score
            rows.append(dict(model=name, oof_score=score, mean_best_iter=float(np.mean(iters)),
                             fit_seconds=time.time() - t_model))
            self._log(f"{name:<16} OOF {self.metric_.name}={score:.6f}  "
                      f"iters~{np.mean(iters):.0f}  {time.time() - t_model:.1f}s")

        self.weights_ = hill_climb(self.oof_, y_fit, self.metric_)
        self.oof_ensemble_ = blend([self.oof_[k] for k in self.weights_], np.array(list(self.weights_.values())))
        self.stacker_ = None
        self.ensemble_kind_ = "hill"
        if self.ensemble in ("auto", "stack") and len(self.oof_) > 1:
            y_st = y_model if self.task_ == "regression" else y_fit
            oof_st = {k: (np.log1p(v) if self.log_target_ else v) for k, v in self.oof_.items()}
            kind, cv_scores = choose_ensemble(oof_st, y_st, self.metric_ if not self.log_target_
                                              else get_metric("rmse"), self.task_)
            self._log(f"ensemble CV on OOF rows: {cv_scores}")
            if self.ensemble == "stack" or kind == "stack":
                self.ensemble_kind_ = "stack"
                self.stacker_ = Stacker(self.task_).fit([oof_st[k] for k in self.oof_], y_st)
                self.oof_ensemble_ = self._to_output(self.stacker_.predict([oof_st[k] for k in self.oof_]))
        self.oof_ensemble_score_ = self.metric_(y_fit, self.oof_ensemble_)
        rows.append(dict(model="ensemble", oof_score=self.oof_ensemble_score_,
                         mean_best_iter=np.nan, fit_seconds=time.time() - t0))
        self.leaderboard_ = pd.DataFrame(rows).sort_values(
            "oof_score", ascending=not self.metric_.greater_is_better).reset_index(drop=True)
        self._log(f"ensemble {self.ensemble_kind_} weights {self.weights_}  OOF {self.metric_.name}="
                  f"{self.oof_ensemble_score_:.6f}  total {time.time() - t0:.1f}s")
        if X_test is not None:
            self.test_ensemble_ = self._blend_test(self.test_pred_)
        return self

    # ---------------------------------------------------------------- predict
    def _blend_test(self, preds: Dict[str, np.ndarray]) -> np.ndarray:
        if self.stacker_ is not None:
            z = [(np.log1p(preds[k]) if self.log_target_ else preds[k]) for k in self.oof_]
            return self._to_output(self.stacker_.predict(z))
        return blend([preds[k] for k in self.weights_], np.array(list(self.weights_.values())))

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.reset_index(drop=True)[self.columns_].copy()
        for c, cats in self.categories_.items():
            X[c] = pd.Categorical(_as_str(X[c]), categories=cats)
        return X

    def predict_models(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Per-model test predictions, averaged over all fold models."""
        X = self._align(X)
        out = {}
        for name, models in self.fold_models_.items():
            out[name] = self._to_output(np.mean([m.predict(X) for m in models], axis=0))
        return out

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Ensemble prediction in the metric's space (probabilities for classification)."""
        return self._blend_test(self.predict_models(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Regression values, or class labels for classification."""
        p = self.predict_proba(X)
        if self.task_ == "regression":
            return p
        idx = p.argmax(axis=1) if p.ndim == 2 else (p >= 0.5).astype(int)
        return self.label_encoder_.inverse_transform(idx)
