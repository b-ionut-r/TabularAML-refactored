import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer, SimpleImputer as SklearnSimpleImputer
from sklearn.ensemble import RandomForestRegressor
import warnings

# SimpleImputer without validations
class SimpleImputer(BaseEstimator, TransformerMixin):
    DEFAULT_MISSING_VALUES = ["-", "", " ", np.nan, "NaN", "nan", "Nan", "NAN", "Unknown", "unknown", None]

    def __init__(self, numerical_missing_values=None, categorical_missing_values=None):
        self.numerical_missing_values = numerical_missing_values
        self.categorical_missing_values = categorical_missing_values

    def _get_missing_value_list(self, user_list):
        base_missing = {np.nan, None}
        return list(set(user_list or self.DEFAULT_MISSING_VALUES) | base_missing)

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.numerical_columns_ = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_columns_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Numerical imputation values
        self.numerical_impute_values_ = {}
        for col in self.numerical_columns_:
            self.numerical_impute_values_[col] = X[col].replace(
                self._get_missing_value_list(self.numerical_missing_values), np.nan
            ).mean() or 0.0

        # Categorical imputation values
        self.categorical_impute_values_ = {}
        for col in self.categorical_columns_:
            self.categorical_impute_values_[col] = X[col].replace(
                self._get_missing_value_list(self.categorical_missing_values), np.nan
            ).mode()[0] if X[col].notna().any() else "missing"
        
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X = X.copy()
        
        # Numerical imputation
        for col in self.numerical_impute_values_:
            X[col] = X[col].replace(
                self._get_missing_value_list(self.numerical_missing_values), np.nan
            ).fillna(self.numerical_impute_values_[col])

        # Categorical imputation
        for col in self.categorical_impute_values_:
            # Ensure the fill value is in categories to avoid 'Cannot setitem on a Categorical with a new category'
            fill_val = self.categorical_impute_values_[col]
            if pd.api.types.is_categorical_dtype(X[col]) or isinstance(X[col].dtype, pd.CategoricalDtype):
                if fill_val not in X[col].cat.categories:
                    X[col] = X[col].cat.add_categories([fill_val])
                # We should also ensure missing values we replace are handled without crashing.
                # Converting to object safely handles replace/fillna, then convert back
                cat_type = X[col].dtype
                X[col] = X[col].astype('object')
                X[col] = X[col].replace(
                    self._get_missing_value_list(self.categorical_missing_values), np.nan
                ).fillna(fill_val)
                X[col] = X[col].astype(cat_type)
            else:
                X[col] = X[col].replace(
                    self._get_missing_value_list(self.categorical_missing_values), np.nan
                ).fillna(fill_val)
        
        return X

# AdvancedImputer without validations
class AdvancedImputer(BaseEstimator, TransformerMixin):
    DEFAULT_MISSING_VALUES = [None, np.nan, "NaN", "nan", "Nan", "NAN", "-", "", " ", "Unknown", "unknown"]

    def __init__(self, strategy='auto', knn_neighbors=5, max_iter=10, 
                 iterative_estimator=None, random_state=None, missing_values=None,
                 n_jobs=None, verbose=0):
        self.strategy = strategy
        self.knn_neighbors = knn_neighbors
        self.max_iter = max_iter
        self.iterative_estimator = iterative_estimator or RandomForestRegressor()
        self.random_state = random_state
        self.missing_values = missing_values or self.DEFAULT_MISSING_VALUES
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.numerical_columns_ = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_columns_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Handle missing values
        X_imputed = X.replace(self.missing_values, np.nan)
        
        # Determine strategy
        self.strategy_ = self.strategy
        if self.strategy == 'auto':
            self.strategy_ = 'simple'  # Simplified auto strategy
        
        # Fit numerical imputer
        self.imputer_ = {}
        if self.numerical_columns_:
            if self.strategy_ == 'knn':
                self.imputer_['numerical'] = KNNImputer(n_neighbors=self.knn_neighbors).fit(X_imputed[self.numerical_columns_])
            else:
                self.imputer_['numerical'] = SklearnSimpleImputer(strategy='mean').fit(X_imputed[self.numerical_columns_])
        
        # Fit categorical imputer
        if self.categorical_columns_:
            self.imputer_['categorical'] = SklearnSimpleImputer(strategy='most_frequent').fit(X_imputed[self.categorical_columns_])
        
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X = X.replace(self.missing_values, np.nan)
        
        # Transform numerical columns
        if self.numerical_columns_ and 'numerical' in self.imputer_:
            X[self.numerical_columns_] = self.imputer_['numerical'].transform(X[self.numerical_columns_])
        
        # Transform categorical columns
        if self.categorical_columns_ and 'categorical' in self.imputer_:
            imputed_cats = self.imputer_['categorical'].transform(X[self.categorical_columns_])
            for i, col in enumerate(self.categorical_columns_):
                imputed_col = imputed_cats[:, i]
                if isinstance(X[col].dtype, pd.CategoricalDtype):
                    new_cats = set(pd.Series(imputed_col).unique()) - set(X[col].cat.categories)
                    if new_cats:
                        X[col] = X[col].cat.add_categories(list(new_cats))
                    cat_type = X[col].dtype
                    X[col] = X[col].astype('object')
                    X[col] = imputed_col
                    X[col] = X[col].astype(cat_type)
                else:
                    X[col] = imputed_col
        
        return X