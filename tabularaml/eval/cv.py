import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import OneHotEncoder
from tabularaml.eval.scorers import Scorer
from tabularaml.preprocessing.pipeline import PipelineWrapper
from copy import deepcopy
from typing import Optional, Union
import inspect

# --- FIX: Add numpy() method to pandas Series if it doesn't exist ---
if not hasattr(pd.Series, 'numpy'):
    def _numpy(self):
        return self.values
    pd.Series.numpy = _numpy


def sanitize_model_features(X):
    """Replace +/-inf with NaN without dropping or filling missing values."""
    if isinstance(X, np.ndarray):
        if not np.issubdtype(X.dtype, np.number):
            return X
        X = X.astype(float, copy=True)
        X[np.isinf(X)] = np.nan
        return X

    if not isinstance(X, pd.DataFrame):
        return X

    X = X.copy()

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'string':
            X[col] = pd.Categorical(X[col])

    num_cols = X.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        numeric_block = X[num_cols].apply(pd.to_numeric, errors="coerce")
        numeric_block = numeric_block.replace([np.inf, -np.inf], np.nan)
        X.loc[:, num_cols] = numeric_block

    for col in X.select_dtypes(include=['category']).columns:
        cat_dtype = X[col].cat.categories.dtype
        if getattr(cat_dtype, "kind", None) in ("f", "i", "u", "c"):
            X[col] = X[col].cat.rename_categories(X[col].cat.categories.astype(str))

    return X


def make_cv_splitter(cv, y, shuffle=True, random_state=42, groups=None):
    """Build a safer default splitter for classification with rare classes."""
    if not isinstance(cv, int):
        return cv

    if groups is not None:
        from sklearn.model_selection import GroupKFold
        return GroupKFold(n_splits=cv)

    y_arr = np.asarray(y)
    is_regression = type_of_target(y_arr) in ("continuous", "continuous-multioutput")
    n_samples = len(y_arr)
    if n_samples < 2:
        raise ValueError("cross_val_score requires at least 2 samples")

    requested_splits = max(2, min(int(cv), n_samples))
    if is_regression:
        return KFold(n_splits=requested_splits, shuffle=shuffle, random_state=random_state)

    class_counts = pd.Series(y_arr).value_counts(dropna=False)
    safe_splits = min(requested_splits, int(class_counts.min())) if not class_counts.empty else requested_splits
    if safe_splits >= 2:
        return StratifiedKFold(n_splits=safe_splits, shuffle=shuffle, random_state=random_state)

    return KFold(n_splits=requested_splits, shuffle=shuffle, random_state=random_state)


@dataclass
class FoldScores:
    """Lightweight per-fold CV result used by the feature search hot loop."""
    mean_val: float
    fold_scores: np.ndarray
    per_group: Optional[dict] = None  # group/era id -> score, when scorer is group-aware


def _early_stopping_requested(model, model_fit_kwargs) -> bool:
    """Detect whether the model/fit-kwargs configure early stopping on an eval_set.

    Covers XGBoost / LightGBM / CatBoost sklearn wrappers. False for models whose
    early stopping is internal (e.g. sklearn HistGB's n_iter_no_change, which does
    not consume eval_set).
    """
    if getattr(model, "early_stopping_rounds", None):
        return True
    try:
        params = model.get_params()
    except Exception:
        params = {}
    for key in ("early_stopping_rounds", "early_stopping_round", "early_stopping",
                "od_wait", "od_type"):
        if params.get(key):
            return True
    if model_fit_kwargs:
        for key in ("early_stopping_rounds", "early_stopping_round"):
            if model_fit_kwargs.get(key):
                return True
        for cb in model_fit_kwargs.get("callbacks", []) or []:
            if hasattr(cb, "stopping_rounds") or "earlystop" in type(cb).__name__.lower().replace("_", ""):
                return True
    return False


def _carve_es_split(train_idx, y, groups, es_split_frac, random_state):
    """Split positional train indices into (reduced_train_idx, es_idx) for early stopping.

    The early-stopping eval set is carved out of the TRAIN fold so the validation
    fold is never seen during fit. Stratifies for classification when possible and
    respects groups when provided.
    """
    from sklearn.model_selection import train_test_split, GroupShuffleSplit

    n_es = max(1, int(round(len(train_idx) * es_split_frac)))
    if len(train_idx) - n_es < 2:
        return train_idx, None

    if groups is not None:
        groups_train = np.asarray(groups)[train_idx]
        if len(np.unique(groups_train)) >= 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=es_split_frac, random_state=random_state)
            tr_pos, es_pos = next(gss.split(train_idx.reshape(-1, 1), groups=groups_train))
            return train_idx[tr_pos], train_idx[es_pos]

    y_train = np.asarray(y)[train_idx]
    stratify = None
    if type_of_target(y_train) not in ("continuous", "continuous-multioutput"):
        counts = pd.Series(y_train).value_counts(dropna=False)
        if counts.min() >= 2 and len(counts) <= n_es:
            stratify = y_train
    try:
        tr, es = train_test_split(train_idx, test_size=es_split_frac,
                                  random_state=random_state, stratify=stratify)
    except ValueError:
        tr, es = train_test_split(train_idx, test_size=es_split_frac,
                                  random_state=random_state, stratify=None)
    return tr, es


def _slice_xy(X, y, idx, is_dataframe):
    if is_dataframe:
        return X.iloc[idx], y.iloc[idx] if hasattr(y, "iloc") else np.asarray(y)[idx]
    return X[idx], y.iloc[idx] if hasattr(y, "iloc") else np.asarray(y)[idx]


def _run_single_fold(fold_idx, train_idx, val_idx, model, X, y, scorer, pipeline,
                     model_fit_kwargs, groups, compute_train, keep_model,
                     eval_set_policy, es_split_frac, random_state, model_threads,
                     y_is_classification=True):
    """Fit and score a single CV fold. Thread-safe: clones model/pipeline internally."""
    is_dataframe = isinstance(X, pd.DataFrame)

    try:
        model_clone = deepcopy(model)
    except Exception:
        model_clone = type(model)(**model.get_params())

    if model_threads is not None and hasattr(model_clone, "set_params"):
        try:
            if "n_jobs" in model_clone.get_params():
                model_clone.set_params(n_jobs=model_threads)
        except Exception:
            pass

    fit_signature = inspect.signature(model_clone.fit)
    supports_eval_set = 'eval_set' in fit_signature.parameters
    es_requested = (eval_set_policy != "none" and supports_eval_set and
                    (eval_set_policy == "legacy" or
                     _early_stopping_requested(model_clone, model_fit_kwargs)))

    es_idx = None
    if es_requested and eval_set_policy == "auto":
        train_idx, es_idx = _carve_es_split(np.asarray(train_idx), y, groups,
                                            es_split_frac, random_state)
        if es_idx is None:
            es_requested = False

    X_train_raw, y_train = _slice_xy(X, y, train_idx, is_dataframe)
    X_val_raw, y_val = _slice_xy(X, y, val_idx, is_dataframe)
    if is_dataframe:
        X_train_raw, X_val_raw = X_train_raw.copy(), X_val_raw.copy()

    pipeline_clone = None
    if pipeline is not None:
        try:
            pipeline_clone = deepcopy(pipeline)
        except Exception:
            if hasattr(pipeline, "get_params"):
                pipeline_clone = type(pipeline)(**pipeline.get_params())
            else:
                raise ValueError("Cannot clone pipeline. Make sure it has a get_params method.")
        X_train = pipeline_clone.fit_transform(X_train_raw, y_train)
        X_val = pipeline_clone.transform(X_val_raw)
        X_train = sanitize_model_features(X_train)
        X_val = sanitize_model_features(X_val)
    else:
        # X was sanitized once at entry; fold slices need no re-sanitization
        X_train, X_val = X_train_raw, X_val_raw

    # Build eval_set according to policy
    fit_kwargs = model_fit_kwargs.copy() if model_fit_kwargs else {}
    eval_set = None
    if es_requested:
        if eval_set_policy == "legacy":
            # Bit-for-bit legacy guard (it also suppresses eval_set for
            # continuous targets, since unique float values always differ).
            train_labels = np.unique(np.asarray(y_train))
            val_labels = np.unique(np.asarray(y_val))
            if np.setdiff1d(val_labels, train_labels).size == 0:
                eval_set = [(X_val, y_val)]
        else:  # auto: carved from train
            X_es_raw, y_es = _slice_xy(X, y, es_idx, is_dataframe)
            if is_dataframe:
                X_es_raw = X_es_raw.copy()
            X_es = pipeline_clone.transform(X_es_raw) if pipeline_clone is not None else X_es_raw
            if pipeline_clone is not None:
                X_es = sanitize_model_features(X_es)
            if y_is_classification:
                train_labels = np.unique(np.asarray(y_train))
                es_labels = np.unique(np.asarray(y_es))
                if np.setdiff1d(es_labels, train_labels).size == 0:
                    eval_set = [(X_es, y_es)]
            else:
                eval_set = [(X_es, y_es)]

    if supports_eval_set and eval_set is not None:
        if 'verbose' in fit_signature.parameters:
            fit_kwargs.setdefault('verbose', False)
        elif 'callbacks' in fit_signature.parameters and 'callbacks' not in fit_kwargs:
            try:
                import lightgbm as lgb
                fit_kwargs['callbacks'] = [lgb.log_evaluation(period=0)]
            except ImportError:
                pass
        try:
            model_clone.fit(X_train, y_train, eval_set=eval_set, **fit_kwargs)
        except (ValueError, TypeError):
            model_clone.fit(X_train, y_train, **fit_kwargs)
    else:
        try:
            model_clone.fit(X_train, y_train, **fit_kwargs)
        except (ValueError, TypeError) as e:
            # Model demands an eval_set for its configured early stopping but
            # detection missed it: carve an ES split now and retrain on the rest.
            if supports_eval_set and "stopping" in str(e).lower():
                train_idx2, es_idx2 = _carve_es_split(np.asarray(train_idx), y, groups,
                                                      es_split_frac, random_state)
                if es_idx2 is not None:
                    X_tr_raw, y_train = _slice_xy(X, y, train_idx2, is_dataframe)
                    X_es_raw, y_es = _slice_xy(X, y, es_idx2, is_dataframe)
                    if pipeline_clone is not None:
                        X_train = sanitize_model_features(pipeline_clone.transform(
                            X_tr_raw.copy() if is_dataframe else X_tr_raw))
                        X_es = sanitize_model_features(pipeline_clone.transform(
                            X_es_raw.copy() if is_dataframe else X_es_raw))
                    else:
                        X_train, X_es = X_tr_raw, X_es_raw
                    model_clone.fit(X_train, y_train, eval_set=[(X_es, y_es)], **fit_kwargs)
                else:
                    raise
            else:
                raise

    val_preds = model_clone.predict_proba(X_val) if scorer.from_probs else model_clone.predict(X_val)

    requires_onehot = scorer.name == "categorical_crossentropy"
    one_hot = None
    if requires_onehot:
        one_hot = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        one_hot.fit(np.asarray(y_train).reshape(-1, 1))

    groups_val = np.asarray(groups)[val_idx] if groups is not None else None
    scorer_kwargs = {}
    if getattr(scorer, "needs_groups", False) and groups_val is not None:
        scorer_kwargs["groups"] = groups_val

    y_val_for_score = one_hot.transform(np.asarray(y_val).reshape(-1, 1)) if requires_onehot else y_val
    val_score = scorer.score(y_true=y_val_for_score, y_pred=val_preds, **scorer_kwargs)

    result = {"val_score": val_score}

    if getattr(scorer, "needs_groups", False) and groups_val is not None and hasattr(scorer, "score_per_group"):
        try:
            result["val_group_scores"] = scorer.score_per_group(y_val, val_preds, groups_val)
        except Exception:
            pass

    if compute_train:
        train_preds = model_clone.predict_proba(X_train) if scorer.from_probs else model_clone.predict(X_train)
        y_train_for_score = one_hot.transform(np.asarray(y_train).reshape(-1, 1)) if requires_onehot else y_train
        train_kwargs = {}
        if getattr(scorer, "needs_groups", False) and groups is not None:
            train_kwargs["groups"] = np.asarray(groups)[train_idx]
        result["train_score"] = scorer.score(y_true=y_train_for_score, y_pred=train_preds, **train_kwargs)

    if keep_model:
        result["model"] = model_clone
        if pipeline_clone is not None:
            result["pipeline"] = pipeline_clone
        if hasattr(model_clone, "feature_importances_"):
            if is_dataframe and hasattr(X_train, "columns"):
                result["feature_importance"] = dict(zip(X_train.columns, model_clone.feature_importances_))
            elif hasattr(model_clone, "feature_names_in_"):
                result["feature_importance"] = dict(zip(model_clone.feature_names_in_, model_clone.feature_importances_))

    return result


def cross_val_score(model, X, y, scorer: Scorer, cv = 5, shuffle = True, random_state = 42,
                    pipeline: Union[Pipeline, PipelineWrapper] = None, return_dict = False,
                    model_fit_kwargs = {}, folds_weights = None, groups = None,
                    compute_train_scores: Optional[bool] = None,
                    keep_fold_models: Optional[bool] = None,
                    eval_set_policy: str = "auto",
                    es_split_frac: float = 0.10,
                    n_jobs_folds: int = 1):
    """
    Perform cross-validation evaluation of a model.

    Parameters
    ----------
    model : object
        The model to evaluate. Must implement fit() and predict() methods.
    X : array-like or DataFrame
        Feature dataset.
    y : array-like
        Target values.
    scorer : Scorer
        Object that implements a score() method for model evaluation.
    cv : int or cross-validation generator, default=5
        Cross-validation strategy.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.
    random_state : int, default=42
        Random seed for reproducibility.
    pipeline: sklearn Pipeline or custom PipelineWrapper, default=None
        Preprocessing pipeline applied per fold (fit on train, transform on val).
    return_dict : bool, default=False
        If True, returns a dictionary with detailed results for each fold.
    model_fit_kwargs: dict, default={}
        Extra params used when fitting the model.
    folds_weights : array-like, default=None
        Weighted average of fold scores instead of simple mean.
    groups : array-like, default=None
        Group labels (passed to the splitter and, for group-aware scorers, sliced
        per fold and forwarded to scorer.score).
    compute_train_scores : bool, default=None
        Whether to score the model on training folds. None preserves legacy
        behavior (computed when return_dict=True). Pass False to skip the
        train-side predict/score entirely (large speedup in hot loops).
    keep_fold_models : bool, default=None
        Whether to keep fitted fold models/pipelines/importances in the result
        dict. None preserves legacy behavior (kept when return_dict=True).
    eval_set_policy : {"auto", "legacy", "none"}, default="auto"
        "auto": pass an eval_set only when the model has early stopping
        configured, carved out of the TRAIN fold (the validation fold is never
        seen during fit — no optimistic bias). "legacy": pass the validation
        fold as eval_set whenever the model supports it (pre-existing behavior).
        "none": never pass an eval_set.
    es_split_frac : float, default=0.10
        Fraction of the train fold carved out for early stopping under "auto".
    n_jobs_folds : int, default=1
        Number of folds fitted in parallel (threads). Model n_jobs is clamped
        per fold to avoid oversubscription.

    Returns
    -------
    float or dict
        Mean validation score, or a detailed dict when return_dict=True
        (includes "fold_val_scores": list of per-fold validation scores).
    """
    y = y.copy()
    X = sanitize_model_features(X)

    assert hasattr(model, "fit"), "Model must have a .fit() method."
    assert hasattr(model, "predict"), "Model must have a .predict() method."
    if scorer.from_probs:
        assert hasattr(model, "predict_proba"), "Model must have a .predict_proba() method."

    if compute_train_scores is None:
        compute_train_scores = return_dict
    if keep_fold_models is None:
        keep_fold_models = return_dict

    if isinstance(cv, int):
        cv = make_cv_splitter(cv, y, shuffle=shuffle, random_state=random_state, groups=groups)

    splits = list(cv.split(X, y, groups))
    n_splits = len(splits)

    if folds_weights is not None:
        folds_weights = np.array(folds_weights)
        if len(folds_weights) != n_splits:
            raise ValueError(f"folds_weights length ({len(folds_weights)}) must match number of folds ({n_splits})")
        folds_weights = folds_weights / np.sum(folds_weights)

    model_threads = None
    if n_jobs_folds and n_jobs_folds > 1:
        model_threads = max(1, (os.cpu_count() or 4) // n_jobs_folds)

    y_is_classification = type_of_target(np.asarray(y)) not in (
        "continuous", "continuous-multioutput")

    def _run(fold_idx, tr, va):
        return _run_single_fold(fold_idx, tr, va, model, X, y, scorer, pipeline,
                                model_fit_kwargs, groups, compute_train_scores,
                                keep_fold_models, eval_set_policy, es_split_frac,
                                random_state, model_threads, y_is_classification)

    if n_jobs_folds and n_jobs_folds > 1 and n_splits > 1:
        from joblib import Parallel, delayed
        fold_results = Parallel(n_jobs=min(n_jobs_folds, n_splits), prefer="threads")(
            delayed(_run)(i, tr, va) for i, (tr, va) in enumerate(splits))
    else:
        fold_results = [_run(i, tr, va) for i, (tr, va) in enumerate(splits)]

    val_results = [r["val_score"] for r in fold_results]
    train_results = [r["train_score"] for r in fold_results if "train_score" in r]

    if folds_weights is not None:
        val = np.sum(np.array(val_results) * folds_weights)
    else:
        val = np.mean(val_results)

    if not return_dict:
        return val

    all_results = {}
    for idx, r in enumerate(fold_results):
        fold_result = {"val_score": r["val_score"]}
        if "train_score" in r:
            fold_result["train_score"] = r["train_score"]
        if "model" in r:
            fold_result["model"] = r["model"]
        if "pipeline" in r:
            fold_result["pipeline"] = r["pipeline"]
        if "feature_importance" in r:
            fold_result["feature_importance"] = r["feature_importance"]
        all_results[f"fold_{idx}"] = fold_result

    all_results["fold_val_scores"] = list(val_results)
    group_dicts = [r["val_group_scores"] for r in fold_results if "val_group_scores" in r]
    if group_dicts:
        merged = {}
        for d in group_dicts:
            merged.update(d)
        all_results["fold_val_groups_scores"] = group_dicts
        all_results["val_group_scores"] = merged

    if train_results:
        if folds_weights is not None and len(train_results) == n_splits:
            all_results["mean_train_score"] = np.sum(np.array(train_results) * folds_weights)
        else:
            all_results["mean_train_score"] = np.mean(train_results)
    else:
        all_results["mean_train_score"] = None
    all_results["mean_val_score"] = val
    return all_results


def cross_val_fold_scores(model, X, y, scorer: Scorer, cv=5, *, pipeline=None,
                          model_fit_kwargs={}, groups=None, shuffle=True,
                          random_state=42, eval_set_policy="auto",
                          n_jobs_folds: int = 1) -> FoldScores:
    """Light CV path for hot loops: per-fold validation scores only.

    Skips train-side predictions and fold-model retention. Fold order is
    deterministic for a fixed splitter, so two calls under the same splitter
    state produce pairable per-fold vectors.
    """
    res = cross_val_score(model, X, y, scorer, cv=cv, shuffle=shuffle,
                          random_state=random_state, pipeline=pipeline,
                          return_dict=True, model_fit_kwargs=model_fit_kwargs,
                          groups=groups, compute_train_scores=False,
                          keep_fold_models=False, eval_set_policy=eval_set_policy,
                          n_jobs_folds=n_jobs_folds)
    return FoldScores(mean_val=res["mean_val_score"],
                      fold_scores=np.asarray(res["fold_val_scores"], dtype=float),
                      per_group=res.get("val_group_scores"))
