import numpy as np


def mean_absolute_error(y_true, y_pred):
    """
    Calculate the mean absolute error (MAE) between the true and predicted values.

    The mean absolute error is a measure of errors between paired observations 
    expressing the same phenomenon. It is calculated as the average of the absolute 
    differences between the predicted values and the actual values.

    Parameters:
    y_true (array-like): Array of true values.
    y_pred (array-like): Array of predicted values.

    Returns:
    float: The mean absolute error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # Vectorized implementation
    return np.mean(np.abs(y_pred - y_true))


def mean_squared_error(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) between the true and predicted values.

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.

    Returns:
    float: The mean squared error.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # Using np.square is faster than ** 2 for large arrays
    return np.mean(np.square(y_pred - y_true))


def root_mean_squared_error(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between the true and predicted values.

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.

    Returns:
    float: The RMSE value.
    """
    # Calling mean_squared_error to avoid duplicate calculation
    return np.sqrt(mean_squared_error(y_true, y_pred))


def r2_score(y_true, y_pred):
    """
    Calculate the R-squared (coefficient of determination) regression score.

    R-squared is a statistical measure that represents the proportion of the variance 
    for a dependent variable that's explained by an independent variable or variables 
    in a regression model. It provides an indication of goodness of fit and therefore 
    a measure of how well unseen samples are likely to be predicted by the model.

    Parameters:
    y_true (array-like): True values of the target variable.
    y_pred (array-like): Predicted values of the target variable.

    Returns:
    float: The R-squared score, which ranges from 0 to 1. A score of 1 indicates 
           perfect prediction, while a score of 0 indicates that the model does not 
           explain any of the variability of the response data around its mean.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Handle edge case
    if len(y_true) <= 1:
        return 1.0

    # Calculate mean once to avoid recalculation
    y_true_mean = np.mean(y_true)
    
    # Compute numerator and denominator
    numerator = np.sum(np.square(y_true - y_pred))
    denominator = np.sum(np.square(y_true - y_true_mean))
    
    # Handle the case where the denominator is zero
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else 0.0
    
    return 1.0 - numerator / denominator


def regression_score(y_true, y_pred):
    """
    Calculate various regression metrics between true and predicted values.

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.

    Returns:
    dict: A dictionary containing the following regression metrics:
        - "mae": Mean Absolute Error
        - "mse": Mean Squared Error
        - "rmse": Root Mean Squared Error
        - "r2": R-squared Score
    """
    # Convert to numpy arrays once
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Calculate MSE once and reuse for RMSE
    mse = mean_squared_error(y_true, y_pred)
    
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_true, y_pred)
    }