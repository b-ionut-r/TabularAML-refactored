from .regression import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    regression_score
)

from .classification import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_score,
    log_loss,
    multi_log_loss
)

# what to import when calling "from mlkit.metrics import *"
__all__ = ["mean_absolute_error", "mean_squared_error", "root_mean_squared_error", "r2_score", "regression_score",
           "confusion_matrix", "accuracy_score", "precision_score", "recall_score", 
           "f1_score", "classification_score", "log_loss", "multi_log_loss"]
