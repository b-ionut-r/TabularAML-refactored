import numpy as np
import pandas as pd

def confusion_matrix(y_true, y_pred, positive_label=None):
    """
    Computes the confusion matrix for binary classification.

    Parameters:
    y_true (numpy.ndarray): Array of true labels.
    y_pred (numpy.ndarray): Array of predicted labels (discrete values).
    positive_label (int or str): The label representing the positive class.

    Returns:
    numpy.ndarray: Confusion matrix as a 2x2 array:
                    [[TP, FN],
                     [FP, TN]]
    """
    # assert np.issubdtype(y_pred.dtype, np.integer), "Predictions should be discrete labels, not continuous probabilities."
    if positive_label is None:
        if len(np.unique(y_true)) == 2:
            positive_label = 1
        else:
            raise Exception("For multiclass, please provide the positive label.")
    else:
         assert positive_label in y_true, "Chosen positive label is not valid."

    # Vectorized implementation
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Calculate true positives, true negatives, false positives, false negatives
    tp = np.sum((y_pred == positive_label) & (y_true == positive_label))
    fp = np.sum((y_pred == positive_label) & (y_true != positive_label))
    fn = np.sum((y_pred != positive_label) & (y_true == positive_label))
    tn = np.sum((y_pred != positive_label) & (y_true != positive_label))
    
    return np.array([[tp, fn], [fp, tn]])


def compute_classification_metrics(y_true, y_pred):
    """
    Efficiently computes all classification metrics in a single pass.
    
    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    
    Returns:
    tuple: (accuracy, precision, recall, f1) metrics for binary classification
           or averaged across all classes for multiclass.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    unique_labels = np.unique(y_true)
    
    if len(unique_labels) == 2:
        # Binary classification - compute metrics in one pass
        cm = confusion_matrix(y_true, y_pred, positive_label=1)
        tp, fn, fp, tn = cm.ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
    else:
        # Multiclass - compute metrics for each class and average
        accuracies, precisions, recalls, f1s = [], [], [], []
        
        for label in unique_labels:
            cm = confusion_matrix(y_true, y_pred, positive_label=label)
            tp, fn, fp, tn = cm.ravel()
            
            # Calculate metrics for this class
            label_accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
            label_precision = tp / (tp + fp + 1e-8)
            label_recall = tp / (tp + fn + 1e-8)
            label_f1 = 2 * label_precision * label_recall / (label_precision + label_recall + 1e-8)
            
            accuracies.append(label_accuracy)
            precisions.append(label_precision)
            recalls.append(label_recall)
            f1s.append(label_f1)
            
        # Average metrics across all classes
        accuracy = np.mean(accuracies)
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)
        
    return accuracy, precision, recall, f1


def accuracy_score(y_true, y_pred):
    """
    Computes the accuracy score for binary or multi-class classification.

    In binary classification, accuracy is calculated as:
        (TP + TN) / (TP + TN + FP + FN)
    For multi-class classification, accuracy is computed for each class 
    and then averaged across all classes.

    Parameters:
    y_true (numpy.ndarray): Array of true labels.
    y_pred (numpy.ndarray): Array of predicted labels.

    Returns:
    float: Accuracy score, either as a single value for binary classification 
           or the average for multi-class classification.
    """
    accuracy, _, _, _ = compute_classification_metrics(y_true, y_pred)
    return accuracy


def precision_score(y_true, y_pred):
    """
    Calculate the precision score for binary or multiclass classification.
    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True labels of the data.
    y_pred : array-like of shape (n_samples,)
        Predicted labels of the data.
    Returns:
    --------
    precision : float
        Precision score, which is the ratio of true positives to the sum of true positives and false positives.
        For binary classification, it returns the precision for the positive class.
        For multiclass classification, it returns the average precision across all classes.
    """
    _, precision, _, _ = compute_classification_metrics(y_true, y_pred)
    return precision


def recall_score(y_true, y_pred):
    """
    Calculate the recall score for binary or multiclass classification.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True labels of the data.
    y_pred : array-like of shape (n_samples,)
        Predicted labels of the data.

    Returns:
    --------
    recall : float
        Recall score, which is the ratio of true positives to the sum of true positives and false negatives.
        For binary classification, it returns the recall for the positive class.
        For multiclass classification, it returns the average recall across all classes.
    """
    _, _, recall, _ = compute_classification_metrics(y_true, y_pred)
    return recall


def f1_score(y_true, y_pred):
    """
    Calculate the F1 score, which is the harmonic mean of precision and recall.
    Parameters:
    y_true (list or array-like): True labels.
    y_pred (list or array-like): Predicted labels.
    Returns:
    float: F1 score.
    """
    _, _, _, f1 = compute_classification_metrics(y_true, y_pred)
    return f1


def classification_score(y_true, y_pred):
    """
    Calculate various classification metrics for given true and predicted labels.
    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    Returns:
    dict: A dictionary containing the following classification metrics:
        - "accuracy": Accuracy score.
        - "precision": Precision score.
        - "recall": Recall score.
        - "f1": F1 score.
    """
    accuracy, precision, recall, f1 = compute_classification_metrics(y_true, y_pred)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def log_loss(y_true, y_pred):
    """
    Compute the logistic loss (log loss) between true labels and predicted probabilities.

    Log loss, also known as logistic regression loss or cross-entropy loss, measures the performance of a classification model
    where the prediction is a probability value between 0 and 1. The loss increases as the predicted probability diverges from
    the actual label.

    Parameters:
    y_true (array-like): True binary labels in range {0, 1}.
    y_pred (array-like): Predicted probabilities, as returned by a classifier's predict_proba method.
                         Can be either a 1D array of positive class probabilities or a 2D array of shape
                         (n_samples, 2) as returned by sklearn's predict_proba().

    Returns:
    float: The computed log loss value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Handle 2D probability array from predict_proba() which returns probabilities for both classes
    if y_pred.ndim == 2 and y_pred.shape[1] == 2:
        # Take only the positive class probabilities (second column)
        y_pred = y_pred[:, 1]
    
    # Clip probabilities to avoid log(0)
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)
    
    # Vectorized implementation
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def multi_log_loss(y_true, y_pred):
    """
    Computes the multi-class logarithmic loss (cross-entropy loss).

    Parameters:
    y_true (array-like): True labels (either one-hot encoded or class indices).
    y_pred (numpy.ndarray): Predicted probabilities, shape (n_samples, n_classes).

    Returns:
    float: The computed multi-class log loss.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)


    # Avoid log(0) by clipping y_pred values
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Compute the multi-class log loss
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

