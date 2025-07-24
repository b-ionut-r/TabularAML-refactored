import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.utils.validation import check_array, check_X_y
from sklearn.utils import indexable


class StratifiedKFoldReg(BaseCrossValidator):
    """
    Stratified K-Fold cross-validator for regression tasks.
    
    This class stratifies continuous target values by binning them into equal-frequency bins, 
    and ensures that the target distribution is maintained across train/test splits.

    Parameters:
    ----------
    n_splits : int, default=5
        Number of folds for cross-validation.

    n_bins : int, default=10
        Number of bins to create for stratification of the target variable.
        
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.
        
    random_state : int or RandomState, default=42
        Controls the randomness of the training and testing indices produced.
        Pass an int for reproducible output across multiple function calls.

    Examples:
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>> X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    >>> y = np.array([1.2, 2.3, 3.4, 4.5, 5.6, 6.7])
    >>> cv = StratifiedKFoldReg(n_splits=3, n_bins=3, random_state=42)
    >>> for train_idx, test_idx in cv.split(X, y):
    ...     print(f"TRAIN: {train_idx}, TEST: {test_idx}")
    ...     print(f"y_train: {y[train_idx]}, y_test: {y[test_idx]}")
    """

    def __init__(self, 
                 n_splits=5, 
                 n_bins=10, 
                 shuffle=True,
                 random_state=42):
        self.n_splits = n_splits
        self.n_bins = n_bins
        self.shuffle = shuffle
        self.random_state = random_state

    def _get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits

    def split(self, X, y, groups=None):
        """
        Generate indices to split data into training and test set.
        
        Parameters:
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
            
        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
            
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset.
            This parameter should be left to None for this stratified regressor.
            
        Yields:
        -------
        train : ndarray
            The training set indices for that split.
            
        test : ndarray
            The testing set indices for that split.
        """
        X, y = indexable(X, y)
        
        if groups is not None:
            raise ValueError("The 'groups' parameter should be None for this stratified regressor")
            
        # Check if y is numeric
        y = np.asarray(y)
        if not np.issubdtype(y.dtype, np.number):
            raise ValueError("Target variable must be numeric for regression stratification")
        
        # Ensure n_bins doesn't exceed sample size
        effective_n_bins = min(self.n_bins, len(np.unique(y)))
        if effective_n_bins < self.n_bins:
            import warnings
            warnings.warn(
                f"n_bins={self.n_bins} is greater than the number of unique values in y. "
                f"Using n_bins={effective_n_bins} instead.", UserWarning
            )
        
        # Create bins using quantiles for equal-frequency binning
        if effective_n_bins == 1:
            # If only one bin, assign all samples to the same bin
            y_binned = np.zeros(len(y), dtype=int)
        else:
            # Compute bin edges ensuring they are unique
            quantiles = np.linspace(0, 1, effective_n_bins + 1)[1:-1]
            bin_edges = np.unique(np.quantile(y, quantiles))
            
            # If unique quantiles are fewer than requested, adjust bin strategy
            if len(bin_edges) < effective_n_bins - 1:
                # Fall back to equal-width binning if equal-frequency doesn't work well
                bin_edges = np.linspace(np.min(y), np.max(y), effective_n_bins + 1)[1:-1]
                
            y_binned = np.digitize(y, bins=bin_edges)
            
        # Create array of indices
        indices = np.arange(len(y))
        
        # Shuffle if requested
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            indices = rng.permutation(len(y))
            y_binned = y_binned[indices]
        
        # Use StratifiedKFold from sklearn if available, otherwise fallback to manual implementation
        try:
            from sklearn.model_selection import StratifiedKFold
            stratified_kfold = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=False,  # Already shuffled above if needed
                random_state=None  # Already used random state above if needed
            )
            
            for train_index, test_index in stratified_kfold.split(X, y_binned):
                yield indices[train_index], indices[test_index]
                
        except (ImportError, AttributeError):
            # Manual implementation of stratified k-fold
            # This ensures compatibility if sklearn's StratifiedKFold is not available
            unique_binned = np.unique(y_binned)
            fold_sizes = np.full(self.n_splits, len(y_binned) // self.n_splits, dtype=int)
            fold_sizes[:len(y_binned) % self.n_splits] += 1
            
            # Distribute samples from each bin across folds
            current_fold = 0
            for bin_value in unique_binned:
                bin_indices = np.where(y_binned == bin_value)[0]
                
                # Distribute bin samples across folds
                for idx in bin_indices:
                    fold_indices = np.zeros(self.n_splits, dtype=bool)
                    fold_indices[current_fold] = True
                    current_fold = (current_fold + 1) % self.n_splits
                    
                    test_fold = np.where(fold_indices)[0][0]
                    
                    # Create train-test split
                    test_mask = np.zeros(len(y_binned), dtype=bool)
                    test_mask[idx] = True
                    
                    train_indices = indices[~test_mask]
                    test_indices = indices[test_mask]
                    
                    yield train_indices, test_indices
                    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        return self._get_n_splits(X, y, groups)