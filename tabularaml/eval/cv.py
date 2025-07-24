import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import OneHotEncoder
from tabularaml.eval.scorers import Scorer
from tabularaml.preprocessing.pipeline import PipelineWrapper
from copy import deepcopy
from typing import Union
import inspect


def cross_val_score(model, X, y, scorer: Scorer, cv = 5, shuffle = True, random_state = 42,
                    pipeline: Union[Pipeline, PipelineWrapper] = None, return_dict = False,
                    model_fit_kwargs = {}, folds_weights = None):
    """
    Perform cross-validation evaluation of a model.
    This function evaluates a machine learning model using cross-validation,
    providing scores for each fold and optionally returning detailed results.
    Parameters
    ----------
    model : object
        The model to evaluate. Must implement fit() and predict() methods.
    X : array-like or DataFrame
        Feature dataset. Can be a pandas DataFrame or a numpy array.
    y : array-like
        Target values.
    scorer : Scorer
        Object that implements a score() method for model evaluation.
    cv : int or cross-validation generator, defaault=5
        Cross-validation strategy. If an integer is provided, KFold or StratifiedKFold 
        (for continuous or categorical targets respectively) will be used.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.
    random_state : int, default=42
        Random seed for reproducibility.
    pipeline: sklearn Pipeline or custom PipelineWrapper, default=None
        Preprocessing pipeline to be applied before training the model during cv loop.
        If provided, fit_transform will be called on training data and transform on validation data.
    return_dict : bool, default=False
        If True, returns a dictionary with detailed results for each fold.
        If False, returns only the mean validation score.
    model_fit_kwargs: dict, default={}
        If provided, these extra params will be used when fitting the model.
    folds_weights : array-like, default=None
        If provided, these weights will be used to compute a weighted average instead of
        a simple mean. The weights will be normalized by their sum automatically.
        Must have the same length as the number of folds.
    Returns
    -------
    float or dict
        If return_dict=False, returns the mean validation score across folds.
        If return_dict=True, returns a dictionary containing:
        - Model, train score, and validation score for each fold
        - Feature importances (if the model has them)
        - Mean train and validation scores
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from tabularaml.eval.metrics import accuracy_scorer
    >>> X, y = load_data()
    >>> model = RandomForestClassifier()
    >>> score = cross_val_score(model, X, y, scorer=accuracy_scorer)
    >>> print(f"Cross-validation score: {score:.4f}")
    """

    X = X.copy()
    y = y.copy()
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'string':
            X[col] = pd.Categorical(X[col])

    assert hasattr(model, "fit"), "Model must have a .fit() method."
    assert hasattr(model, "predict"), "Model must have a .predict() method."
    if scorer.from_probs:
        assert hasattr(model, "predict_proba"), "Model must have a .predict_proba() method."
    
    # Convert X to DataFrame if it's not already one
    is_dataframe = isinstance(X, pd.DataFrame)
    
    if isinstance(cv, int):
        cv = KFold(n_splits = cv, 
                   shuffle = shuffle, 
                   random_state = random_state) if type_of_target(y) == "continuous" \
        else StratifiedKFold(n_splits = cv, 
                            shuffle = shuffle, 
                            random_state = random_state)
    
    # Validate folds_weights if provided
    if folds_weights is not None:
        folds_weights = np.array(folds_weights)
        # FIXED: Don't exhaust the CV generator by calling split()
        if hasattr(cv, 'get_n_splits'):
            n_splits = cv.get_n_splits(X, y)
        elif hasattr(cv, 'n_splits'):
            n_splits = cv.n_splits
        else:
            # Skip validation if we can't determine n_splits without exhausting generator
            n_splits = len(folds_weights)
            
        if len(folds_weights) != n_splits:
            raise ValueError(f"folds_weights length ({len(folds_weights)}) must match number of folds ({n_splits})")
        # Normalize weights by sum
        folds_weights = folds_weights / np.sum(folds_weights)
                                  
    all_results = {}
    train_results = []
    val_results = []
    
    for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Always create a fresh copy of the model for each fold
        try:
            model_clone = deepcopy(model)
        except:
            # Fallback if deepcopy fails
            model_clone = type(model)(**model.get_params())
            
        # Handle both DataFrame and numpy array inputs
        if is_dataframe:
            X_train_raw, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy() if hasattr(y, 'copy') else y.iloc[train_idx]
            X_val_raw, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx].copy() if hasattr(y, 'copy') else y.iloc[val_idx]
        else:
            X_train_raw, y_train = X[train_idx].copy(), y[train_idx].copy() if hasattr(y, 'copy') else y[train_idx]
            X_val_raw, y_val = X[val_idx].copy(), y[val_idx].copy() if hasattr(y, 'copy') else y[val_idx]
          # Apply pipeline if provided
        if pipeline is not None:
            try:
                pipeline_clone = deepcopy(pipeline)
            except:
                # Fallback if deepcopy fails
                if hasattr(pipeline, "get_params"):
                    pipeline_clone = type(pipeline)(**pipeline.get_params())
                else:
                    raise ValueError("Cannot clone pipeline. Make sure it has a get_params method.")
            
            X_train = pipeline_clone.fit_transform(X_train_raw, y_train)
            X_val = pipeline_clone.transform(X_val_raw)
        else:
            X_train, X_val = X_train_raw, X_val_raw

        fit_signature = inspect.signature(model_clone.fit)
        if 'eval_set' in fit_signature.parameters:
            # Suppress eval_set verbose output for XGBoost
            fit_kwargs = model_fit_kwargs.copy()
            if 'verbose' not in fit_kwargs:
                fit_kwargs['verbose'] = False
            model_clone.fit(X_train, y_train, eval_set=[(X_val, y_val)], **fit_kwargs)
        else:
            model_clone.fit(X_train, y_train, **model_fit_kwargs)

        val_preds = model_clone.predict_proba(X_val) if scorer.from_probs else model_clone.predict(X_val)

        requires_onehot = False
        if scorer.name == "categorical_crossentropy":
            requires_onehot = True
            one_hot = OneHotEncoder(sparse_output=False, handle_unknown = "ignore")
            one_hot.fit(y_train.reshape(-1, 1))
        val_score = scorer.score(y_true = y_val if not requires_onehot else one_hot.transform(y_val.reshape(-1, 1)), 
                                 y_pred = val_preds)        
        val_results.append(val_score)
        
        if return_dict:
            train_preds = model_clone.predict_proba(X_train) if scorer.from_probs else model_clone.predict(X_train)
            train_score = scorer.score(y_true = y_train if not requires_onehot else one_hot.transform(y_train.reshape(-1, 1)), 
                                       y_pred = train_preds)
            train_results.append(train_score)
            fold_result = {
                "model": model_clone,
                "train_score": train_score,
                "val_score": val_score,
            }
            
            if pipeline is not None:
                fold_result["pipeline"] = pipeline_clone
                
            all_results[f"fold_{idx}"] = fold_result
            
            if hasattr(model_clone, "feature_importances_"):
                # Handle feature importances with proper column names
                if is_dataframe and hasattr(X_train, "columns"):
                    all_results[f"fold_{idx}"]["feature_importance"] = dict(zip(X_train.columns, model_clone.feature_importances_))
                elif hasattr(model_clone, "feature_names_in_"):
                    all_results[f"fold_{idx}"]["feature_importance"] = dict(zip(model_clone.feature_names_in_, model_clone.feature_importances_))
            
    # Compute weighted or simple average
    if folds_weights is not None:
        val = np.sum(np.array(val_results) * folds_weights)
        if return_dict:
            train_weighted_avg = np.sum(np.array(train_results) * folds_weights)
    else:
        val = np.mean(val_results)
        if return_dict:
            train_weighted_avg = np.mean(train_results)
    
    if return_dict:
        all_results["mean_train_score"] = train_weighted_avg
        all_results["mean_val_score"] = val
        return all_results
        
    return val