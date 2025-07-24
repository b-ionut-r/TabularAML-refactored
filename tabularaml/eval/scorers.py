"""
Scorers module providing compatibility layers for various gradient boosting frameworks.

This module implements scoring functionality that can be used with LightGBM, XGBoost,
and CatBoost models, handling the differences in their APIs. It provides:

1. A generic Scorer class with model-specific callback signatures
2. A specialized CatScorer class for CatBoost's evaluation API
3. Pre-defined scorers for common regression and classification metrics

The module adapts metric functions from metrics.regression and metrics.classification
to work with any of the supported gradient boosting frameworks' early stopping implementation.
"""

import numpy as np
import pandas as pd
from .metrics.regression import root_mean_squared_error, mean_absolute_error, mean_squared_error, r2_score
from .metrics.classification import accuracy_score, precision_score, recall_score, f1_score, log_loss, multi_log_loss
from sklearn.metrics import roc_auc_score
from typing import Optional, Dict, Union, List
import xgboost as xgb


class CatScorer:

    def __init__(self,
                 name: str,
                 scorer: callable,
                 greater_is_better: bool,
                 extra_params: Dict[str, any], 
                 type: Optional[str] = None,
                 from_probs: Optional[bool] = False):

        """
        Custom eval metric class especially designed for CatBoost API.

        This class implements CatBoost's evaluation API interface, allowing
        custom metrics to be used with CatBoost models.

        Parameters:
            name (str): Scorer name.
            scorer (obj): Sklearn loss / scoring function from sklearn.metrics or 
                         similar callable object.
            greater_is_better (callable): Whether higher values mean better performance.
                                     Usually, True for scorers used in binary / multiclass tasks 
                                     and False for losses used in regression.
            extra_params (dict): A dictionary of extra kwargs to be passed to Sklearn scorer,
                                along with y_true and y_pred.
            type (str): Model type to be used with. (CAT)
            from_probs (bool): Whether the metric should be computed directly from probabilities.
        """

        self.name = name
        self.__name__ = name
        self.scorer = scorer
        self.greater_is_better = greater_is_better
        self.type = type
        self.extra_params = extra_params
        self.from_probs = from_probs


    def score(self, 
              y_true: Union[np.ndarray, list],
              y_pred: Union[np.ndarray, list]) -> float:
        """
        Calculate the score using the provided metric function.
        
        Parameters:
            y_true (Union[np.ndarray, list]): Ground truth (correct) target values.
            y_pred (Union[np.ndarray, list]): Estimated target values.
            
        Returns:
            float: The calculated score.
        """
        return self.scorer(y_true, y_pred, **self.extra_params)
    

    def is_max_optimal(self) -> bool:
        """
        Returns whether higher values of the metric are better.
        
        Required by CatBoost API to determine optimization direction.
        
        Returns:
            bool: True if higher values are better, False otherwise.
        """
        return self.greater_is_better
    

    def evaluate(self, approxes: list, target: list, _) -> tuple[float, int]:
        """
        Evaluate the metric following CatBoost's custom metric API.
        
        This method handles CatBoost's specific format for predictions and targets,
        converts probabilities to discrete predictions if needed, and returns
        the score in the format expected by CatBoost.
        
        Parameters:
            approxes (list): Model predictions in CatBoost format.
                For regression and binary classification: a list with one element - predictions
                For multiclass: a list of prediction arrays, one per class
            target (list): True target values.
            _ (any): Unused parameter required by CatBoost API.
            
        Returns:
            tuple[float, int]: A tuple containing:
                - The calculated metric value
                - An integer flag (1 if greater is better, 0 otherwise)
        """
        if len(approxes) == 1:  # Regression or binary classification
            y_pred = approxes[0]
        else:  # Multiclass classification
            y_pred = np.vstack(approxes).T

        if not self.from_probs:
            if len(approxes) == 1 and ((target == 0) | (target == 1)).all():  # Binary classification
                y_pred = (y_pred > 0.5).astype(int)
            elif len(approxes) > 1:  # Multiclass classification
                y_pred = y_pred.argmax(axis=-1)

        score = self.score(y_true = target, y_pred = y_pred)

        return score, int(self.greater_is_better)
    

    def get_final_error(self, error: float, _):
        """
        Return the final error value as expected by CatBoost API.
        
        Parameters:
            error (float): The error value.
            _ (any): Unused parameter required by CatBoost API.
            
        Returns:
            float: The final error value.
        """
        return error






class Scorer:

    def __new__(cls, *args, **kwargs):
        """
        Factory method to create appropriate scorer instance based on model type.
        
        Returns a CatScorer instance if model type is "cat", otherwise returns
        a regular Scorer instance for LightGBM and XGBoost.
        
        Parameters:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
            
        Returns:
            Union[CatScorer, Scorer]: The appropriate scorer instance.
        """
        # If model is CatBoost, instantiate custom-signature implementation (CatScorer instance)
        if kwargs.get("type") == "cat":
            return CatScorer(*args, **kwargs)
        

        # Else, for LightGBM and XGBoost, use regular Scorer
        else:
            instance = super().__new__(cls)
            return instance
        

    def __init__(self, 
                 name: str, 
                 scorer: callable, 
                 greater_is_better: bool, 
                 extra_params: Dict[str, any],
                 type: Optional[str] = None,
                 from_probs: Optional[bool] = False):

        """
        Custom eval metric class designed to work with all GBM models.
        
        This class provides a unified interface for evaluation metrics that can be
        used with different gradient boosting frameworks (LightGBM, XGBoost, and CatBoost).
        Each framework expects a different API for custom metrics, which this class handles.

        Parameters:
            name (str): Scorer name.
            scorer (callable): Sklearn loss / scoring function from sklearn.metrics.
            greater_is_better (bool): Whether higher values mean better performance.
                                     Usually, True for scorers used in binary / multiclass tasks 
                                     and False for losses used in regression.
            extra_params (dict): A dictionary of extra kwargs to be passed to Sklearn scorer,
                                along with y_true and y_pred.
            type (str): Model type to be used with. (lgb, xgb, cat)
            from_probs (bool): Whether the metric should be computed directly from probabilities.
        """

        self.name = name
        self.__name__ = name
        self.scorer = scorer
        self.greater_is_better = greater_is_better
        self.extra_params = extra_params
        self.type = type
        self.from_probs = from_probs




    def score(self, 
              y_true: Union[np.ndarray, list],
              y_pred: Union[np.ndarray, list]) -> float:
        """
        Calculate the score using the provided metric function.
        
        This method handles conversion from probability outputs to discrete
        predictions when necessary, based on the metric requirements.
        
        Parameters:
            y_true (Union[np.ndarray, list]): Ground truth (correct) target values.
            y_pred (Union[np.ndarray, list]): Estimated target values.
            
        Returns:
            float: The calculated score.
        """
        if not self.from_probs:
            if len(y_pred.shape) == 2: # check for multiclass
                y_pred = y_pred.argmax(axis=-1)
                
            elif ((y_true==0) | (y_true==1)).all(): # check for binary
                y_pred = np.round(y_pred)
                

        return self.scorer(y_true, y_pred, **self.extra_params)
        
    
    def __call__(self, y1, y2):
        """
        Make the Scorer instance callable for use with gradient boosting frameworks.
        
        This method implements the framework-specific call signatures:
        - For LightGBM: returns (name, score, greater_is_better)
        - For XGBoost: returns (name, score)
        
        Parameters:
            y1: For LightGBM: y_true, For XGBoost: y_pred
            y2: For LightGBM: y_pred, For XGBoost: y_true as DMatrix
            
        Returns:
            Union[tuple[str, float, bool], tuple[str, float]]: 
                Framework-specific return format for the score.
        """
        # Handle model-specific return signature.

        if self.type == "lgb":

            # order y_true, y_pred
            y_true = y1
            y_pred = y2
            
            score = self.score(y_true, y_pred)
            return self.name, score, self.greater_is_better
        

        elif self.type == "xgb":
            
           
            if isinstance(y2, xgb.DMatrix):
                 # order y_pred, y_true DMatrix
                y_pred = y1
                y_true = y2.get_label()
            else:
                # order y_true, y_pred
                y_true = y1
                y_pred = y2

            score = self.score(y_true, y_pred)

            return self.name, score



# Predefined scorers (regression)
rmse = Scorer(name = "rmse",
              scorer = root_mean_squared_error,
              greater_is_better = False,
              extra_params = {},
              type = None)  # Set type when using with specific model
mae = Scorer(name = "mae",
             scorer = mean_absolute_error,
             greater_is_better = False,
             extra_params = {},
             type = None)  # Set type when using with specific model
mse = Scorer(name = "mse",
             scorer = mean_squared_error,
             greater_is_better = False,
             extra_params = {},
             type = None)  # Set type when using with specific model
r2 = Scorer(name = "r2",
            scorer = r2_score,
            greater_is_better = True,
            extra_params = {},
            type = None)  # Set type when using with specific model
PREDEFINED_REG_SCORERS = {
    "rmse": rmse,
    "mae": mae,
    "mse": mse,
    "r2": r2
}




# Predefined scorers (classification)
accuracy = Scorer(name = "accuracy",
                  scorer = accuracy_score,
                  greater_is_better = True,
                  extra_params = {},
                  from_probs = False,
                  type = None)  # Set type when using with specific model
precision = Scorer(name = "precision",
                   scorer = precision_score,
                   greater_is_better = True,
                   extra_params = {},
                   from_probs = False,
                   type = None)  # Set type when using with specific model
recall = Scorer(name = "recall",
                scorer = recall_score,
                greater_is_better = True,
                extra_params = {},
                from_probs = False,
                type = None)  # Set type when using with specific model
f1 = Scorer(name = "f1",
            scorer = f1_score,
            greater_is_better = True,
            extra_params = {},
            from_probs = False,
            type = None)  # Set type when using with specific model
binary_crossentropy = Scorer(name = "binary_crossentropy",
                            scorer = log_loss,
                            greater_is_better = False,
                            extra_params = {},
                            from_probs = True,
                            type = None)  # Set type when using with specific model
categorical_crossentropy = Scorer(name = "categorical_crossentropy",
                                  scorer = multi_log_loss,
                                  greater_is_better = False,
                                  extra_params = {},
                                  from_probs = True,
                                  type = None)  # Set type when using with specific model

binary_roc_auc = Scorer(
    name="binary_roc_auc",
    scorer=roc_auc_score,
    greater_is_better=True,
    extra_params={},  # No need for 'multi_class' or 'average'
    from_probs=True
)

categorical_roc_auc = Scorer(
    name="categorical_roc_auc",
    scorer=roc_auc_score,
    greater_is_better=True,
    extra_params={"multi_class": "ovr", "average": "macro"},
    from_probs=True
)

PREDEFINED_CLS_SCORERS = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "binary_crossentropy": binary_crossentropy,
    "categorical_crossentropy": categorical_crossentropy,
    "binary_roc_auc": binary_roc_auc,
    "categorical_roc_auc": categorical_roc_auc
}


PREDEFINED_SCORERS = PREDEFINED_REG_SCORERS | PREDEFINED_CLS_SCORERS