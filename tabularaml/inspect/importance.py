import numpy as np
import pandas as pd
import warnings
import os
import sys
import math
from time import time
from typing import Union

# For feature importance calculation
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from tabularaml.preprocessing.pipeline import PipelineWrapper
from tabularaml.preprocessing.imputers import SimpleImputer


# For plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import cm
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# For tree-based models
try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor, LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# For SHAP values
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

class FeatureImportanceAnalyzer:
    """
    Analyzes and aggregated feature importance using multiple metrics: tree-based importance,
    statistical correlations, permutation importance, and SHAP values.
    """
    
    def __init__(self, task_type="regression", weights=None, max_features_for_permutation=100,
                 max_features_for_shap=200, cv=3, random_state=42, pipeline=None,
                 use_gpu=False, verbose=False, suppress_warnings=True, n_jobs=-1,
                 preferred_gbm='xgboost'):
        """Initialize the analyzer with configuration for feature importance methods."""
        self.task_type = task_type
        self.weights = weights or {"tree": 0.4, "correlation": 0.2, "permutation": 0.25, "shap": 0.15}
        self.max_features_for_permutation = max_features_for_permutation
        self.max_features_for_shap = max_features_for_shap
        self.n_jobs = n_jobs
        self.preferred_gbm = preferred_gbm.lower()
        
        # Handle CV parameter - support custom CV splitters like TimeSeriesSplit
        if isinstance(cv, tuple):
            self.cv_splitter, self.n_cv_folds = cv
        elif hasattr(cv, 'split'):  # It's a CV splitter object like TimeSeriesSplit
            self.cv_splitter = cv
            self.n_cv_folds = getattr(cv, 'n_splits', 5)  # Get n_splits from the splitter
        else:
            self.n_cv_folds = cv
            self.cv_splitter = None
            
        self.random_state = random_state
        self.verbose = verbose
        self.pipeline = pipeline if pipeline is not None else PipelineWrapper(imputer = SimpleImputer(),
                                                                              scaler = None,
                                                                              encoder = None)        
        self.use_gpu = use_gpu
        self.suppress_warnings = suppress_warnings
        # Initialize results with empty dicts for each requested importance type
        self.results = {method: {} for method in ["correlation", "tree", "permutation", "shap"] 
                       if method in self.weights and self.weights[method] > 0}
        self.cv_results = {}
        
        # Fix typo in weights dictionary if present
        if "permuation" in self.weights and "permutation" not in self.weights:
            self.weights["permutation"] = self.weights.pop("permuation")
            
        if self.suppress_warnings:
            self._setup_warning_suppression()            
        if self.use_gpu and self.verbose:
            self._check_gpu_availability()

    def _check_gpu_availability(self):
        """Check if GPU is available for tree-based models using lightweight methods."""
        print("GPU acceleration enabled for compatible models")
        
        # Check for XGBoost GPU support
        if XGBOOST_AVAILABLE:
            try:
                import xgboost as xgb
                # Create a small test data to check if GPU works
                try:
                    # Try to build a small model with GPU
                    X = np.random.rand(10, 2)
                    y = np.random.rand(10)
                    model = xgb.XGBRegressor(tree_method='gpu_hist', verbosity=0, n_estimators=1)
                    model.fit(X, y)
                    print("XGBoost GPU support confirmed")
                except Exception as e:
                    if 'device' in str(e).lower() or 'cuda' in str(e).lower() or 'gpu' in str(e).lower():
                        print("XGBoost GPU support not available:", str(e))
                    else:
                        print("XGBoost GPU test failed:", str(e))
            except ImportError:
                print("XGBoost not available")
                
        # Check for LightGBM GPU support
        if LIGHTGBM_AVAILABLE:
            try:
                import lightgbm as lgb
                print("Could not determine LightGBM GPU support status")
            except ImportError:
                print("LightGBM not available")

    def _is_gpu_available_for_xgboost(self):
        """Quick check if GPU is available for XGBoost without verbose output."""
        if not XGBOOST_AVAILABLE:
            return False
        try:
            import xgboost as xgb
            # Quick test with minimal data
            X = np.random.rand(5, 2)
            y = np.random.rand(5)
            model = xgb.XGBRegressor(tree_method='gpu_hist', verbosity=0, n_estimators=1)
            model.fit(X, y)
            return True
        except Exception:
            return False

    def _is_gpu_available_for_lightgbm(self):
        """Quick check if GPU is available for LightGBM without verbose output."""
        if not LIGHTGBM_AVAILABLE:
            return False
        try:
            import lightgbm as lgb            # Quick test with minimal data
            X = np.random.rand(5, 2)
            y = np.random.rand(5)
            model = lgb.LGBMRegressor(device='gpu', verbosity=-1, n_estimators=1)
            model.fit(X, y)
            return True
        except Exception:
            return False

    def _setup_warning_suppression(self):
        """Set up warning suppression for tree-based models."""
        # Suppress LightGBM verbose output
        try:
            import lightgbm as lgb
            lgb.set_verbosity(0)
            if self.verbose:
                print("LightGBM verbosity suppressed")
        except (ImportError, AttributeError):
            pass
            
        # Suppress XGBoost verbose output
        try:
            import xgboost as xgb
            try:
                xgb.set_config(verbosity=0)
                if self.verbose:
                    print("XGBoost verbosity suppressed")
            except AttributeError:
                # Handle older versions of XGBoost that don't have set_config
                pass
        except ImportError:
            pass
            
        # Suppress general Python warnings
        if self.suppress_warnings:
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)

    def fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series]) -> 'FeatureImportanceAnalyzer':
        """Calculate feature importance using multiple methods."""
        start_time = time()
        
        if self.verbose:
            print(f"Starting feature importance analysis on {X.shape[1]} features")
            
        # Capture original stdout if we need to suppress output
        original_stdout = None
        if self.suppress_warnings:
            from contextlib import redirect_stdout
            null_output = open(os.devnull, 'w') 
            original_stdout = sys.stdout
            sys.stdout = null_output
            
        try:
            if self.n_cv_folds is not None:
                self._fit_with_cv(X, y)
            else:
                self._fit_direct(X, y)
        finally:
            # Restore stdout if we redirected it
            if self.suppress_warnings and original_stdout:
                sys.stdout = original_stdout
                null_output.close()
            
        if self.verbose:
            print(f"Feature importance analysis completed in {time() - start_time:.2f} seconds")
            
        return self

    def _fit_direct(self, X, y):
        """Calculate feature importance without CV."""
        # Create copies of the data to ensure original data remains unchanged
        X_copy = X.copy()
        y_copy = y.copy() if hasattr(y, 'copy') else y
        
        # Prepare the pipeline and transform data
        pipeline = self.pipeline.get_pipeline(X_copy)
        pipeline.fit(X_copy, y_copy)
        X_transformed = pipeline.transform(X_copy)
        
        # Ensure we have dataframe output
        if not isinstance(X_transformed, pd.DataFrame):
            features = X_copy.columns
            X_transformed = pd.DataFrame(X_transformed, index=X_copy.index)
            X_transformed.columns = features
        
        if self.verbose:
            print(f"Data transformed with pipeline: {X_transformed.shape[1]} features after transformation")
        
        # Initialize null_output outside the try block to avoid NameError
        null_output = None
        
        try:
            # Only compute importance methods that are specified in weights dictionary with non-zero weights
            if 'correlation' in self.weights and self.weights['correlation'] > 0:
                self._calculate_correlation_importance(X_transformed, y_copy)
            
            if 'tree' in self.weights and self.weights['tree'] > 0:
                self._calculate_tree_importance(X_transformed, y_copy)
              # Calculate all requested importance methods, regardless of dimensionality
            if 'permutation' in self.weights and self.weights['permutation'] > 0:
                # Never skip permutation importance regardless of feature count
                self._calculate_permutation_importance(X_transformed, y_copy)
                if self.verbose and X_transformed.shape[1] > self.max_features_for_permutation:
                    print(f"Computing permutation importance despite high dimensionality ({X_transformed.shape[1]} features)")
            
            if 'shap' in self.weights and self.weights['shap'] > 0:
                if SHAP_AVAILABLE:
                    # Never skip SHAP importance regardless of feature count
                    self._calculate_shap_importance(X_transformed, y_copy)
                    if self.verbose and X_transformed.shape[1] > self.max_features_for_shap:
                        print(f"Computing SHAP importance despite high dimensionality ({X_transformed.shape[1]} features)")
                elif self.verbose:
                    print("SHAP not available. Install with 'pip install shap'")
        finally:
            # No cleanup needed for direct fit
            pass
                
    def _fit_with_cv(self, X, y):
        """Calculate feature importance with cross-validation."""
        if self.verbose:
            print(f"Running {self.n_cv_folds}-fold cross-validation")
            
        # Initialize CV results dictionary for only the methods specified in weights with non-zero values
        self.cv_results = {method: {} for method in ["correlation", "tree", "permutation", "shap"] 
                          if method in self.weights and self.weights[method] > 0}
        
        # Use provided CV splitter if available, otherwise create default
        if self.cv_splitter is not None:
            cv = self.cv_splitter
            if self.verbose:
                print(f"Using provided CV splitter: {type(cv).__name__}")
        else:
            # Create appropriate CV splitter
            if self.task_type == 'classification':
                cv = StratifiedKFold(n_splits=self.n_cv_folds, shuffle=True, random_state=self.random_state)
            else:
                cv = KFold(n_splits=self.n_cv_folds, shuffle=True, random_state=self.random_state)
            if self.verbose:
                print(f"Using default CV splitter: {type(cv).__name__} with {self.n_cv_folds} folds")
            
        # Run calculations for each fold
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            if self.verbose:
                print(f"Processing fold {fold_idx + 1}/{self.n_cv_folds}")
                
            # Get train/validation split - create copies to ensure original data remains unchanged
            X_train = X.iloc[train_idx].copy()
            X_val = X.iloc[val_idx].copy()
            y_train = y.iloc[train_idx].copy() if hasattr(y, 'iloc') else y[train_idx].copy() if hasattr(y, 'copy') else y[train_idx]
            y_val = y.iloc[val_idx].copy() if hasattr(y, 'iloc') else y[val_idx].copy() if hasattr(y, 'copy') else y[val_idx]
            
            # Prepare the pipeline and transform data for this fold
            pipeline = self.pipeline.get_pipeline(X_train)
            pipeline.fit(X_train, y_train)
            X_train_transformed = pipeline.transform(X_train)
            X_val_transformed = pipeline.transform(X_val)
            
            # Ensure we have dataframe output
            if not isinstance(X_train_transformed, pd.DataFrame):
                features = X_train.columns
                X_train_transformed = pd.DataFrame(X_train_transformed, index=X_train.index)
                X_train_transformed.columns = features
                X_val_transformed = pd.DataFrame(X_val_transformed, index=X_val.index)
                X_val_transformed.columns = features
            
            # Reset results for this fold
            fold_results = {}
            
            # Calculate importances on validation data - only for methods specified in weights with non-zero values
            if 'correlation' in self.weights and self.weights['correlation'] > 0:
                self._calculate_correlation_importance(X_val_transformed, y_val, fold_results)
            
            if 'tree' in self.weights and self.weights['tree'] > 0:
                self._calculate_tree_importance(X_train_transformed, y_train, fold_results, X_val_transformed, y_val)
              # Calculate all requested importance methods, regardless of dimensionality
            if 'permutation' in self.weights and self.weights['permutation'] > 0:
                # Never skip permutation importance regardless of feature count
                self._calculate_permutation_importance(X_train_transformed, y_train, fold_results, X_val_transformed, y_val)
                if self.verbose and X_train_transformed.shape[1] > self.max_features_for_permutation:
                    print(f"Computing permutation importance despite high dimensionality ({X_train_transformed.shape[1]} features)")
            
            if 'shap' in self.weights and self.weights['shap'] > 0:
                if SHAP_AVAILABLE:
                    # Never skip SHAP importance regardless of feature count
                    self._calculate_shap_importance(X_train_transformed, y_train, fold_results, X_val_transformed, y_val)
                    if self.verbose and X_train_transformed.shape[1] > self.max_features_for_shap:
                        print(f"Computing SHAP importance despite high dimensionality ({X_train_transformed.shape[1]} features)")
                elif self.verbose:
                    print("SHAP not available. Install with 'pip install shap'")
            
            # Accumulate results for each method
            for method, features in fold_results.items():
                for feature, importance in features.items():
                    if feature not in self.cv_results[method]:
                        self.cv_results[method][feature] = []
                    self.cv_results[method][feature].append(importance)
        
        # Average results across folds
        self.results = {method: {feature: np.mean(values) for feature, values in method_results.items() if values}
                        for method, method_results in self.cv_results.items()}

    def _calculate_correlation_importance(self, X, y, results_dict=None):
        """Calculate correlation-based feature importance with balanced metrics for different feature types."""
        corr_results = {}
        
        # Identify column types
        num_columns = X.select_dtypes(include=['number']).columns
        cat_columns = X.select_dtypes(exclude=['number']).columns
        
        ###
        # Check for and handle NaN values - mutual information doesn't handle NaN
        if X.isnull().any().any():
            # Use simple imputation for correlation calculation
            X_clean = X.copy()
            for col in num_columns:
                X_clean[col] = X_clean[col].fillna(X_clean[col].mean() if X_clean[col].notna().any() else 0)
            for col in cat_columns:
                X_clean[col] = X_clean[col].fillna(X_clean[col].mode()[0] if X_clean[col].notna().any() else 'missing')
            X = X_clean
        
        # Use mutual information for all features (both numeric and categorical)
        mi_func = mutual_info_classif if self.task_type == 'classification' else mutual_info_regression
        
        # Calculate mutual information for all features
        all_mi_scores = {}
        
        # Process numerical columns
        if len(num_columns) > 0:
            num_mi_scores = mi_func(X[num_columns], y, n_jobs=self.n_jobs)
            all_mi_scores.update({col: score for col, score in zip(num_columns, num_mi_scores)})
        
        # Process categorical columns
        for col in cat_columns:
            try:
                # Handle each categorical column individually 
                cat_vals = pd.get_dummies(X[col], drop_first=False)
                if cat_vals.shape[1] > 0:
                    cat_mi = mi_func(cat_vals, y, n_jobs=self.n_jobs)
                    # Changed from np.sum to np.mean to avoid bias toward categorical features with many levels
                    all_mi_scores[col] = np.mean(cat_mi) if len(cat_mi) > 0 else 0.001
                else:
                    all_mi_scores[col] = 0.001
            except Exception:
                all_mi_scores[col] = 0.001
                
        # Use mutual information as the base importance for ALL features
        for col in X.columns:
            corr_results[col] = all_mi_scores.get(col, 0.001)
        
        # Add supplementary correlation metrics for numerical features only
        # but don't let them completely dominate the importance
        if len(num_columns) > 0:
            pearson_corr = {}
            spearman_corr = {}
            
            # Calculate Pearson and Spearman for numerical features
            for col in num_columns:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        pearson_corr[col] = abs(np.corrcoef(X[col], y)[0, 1])
                    except (ValueError, TypeError):
                        pearson_corr[col] = 0
                    
                    try:
                        spearman_corr[col] = abs(pd.Series(X[col]).corr(pd.Series(y), method='spearman'))
                    except (ValueError, TypeError):
                        spearman_corr[col] = 0
            
            # Normalize correlation values to avoid bias toward numerical features
            max_pearson = max(pearson_corr.values()) if pearson_corr else 1
            max_spearman = max(spearman_corr.values()) if spearman_corr else 1
            max_mi_num = max(all_mi_scores[col] for col in num_columns) if num_columns.size > 0 else 1
            max_mi_cat = max((all_mi_scores[col] for col in cat_columns), default=1) if cat_columns.size > 0 else 1
            
            # Ensure fair weighting for all features (both numerical and categorical)
            for col in X.columns:
                if col in num_columns:
                    # For numerical features: average normalized MI, Pearson, and Spearman 
                    metrics = [
                        all_mi_scores.get(col, 0) / max(max_mi_num, 0.001), 
                        pearson_corr.get(col, 0) / max(max_pearson, 0.001), 
                        spearman_corr.get(col, 0) / max(max_spearman, 0.001)
                    ]
                    corr_results[col] = np.mean([m for m in metrics if not np.isnan(m)])
                else:
                    # For categorical features: normalize by maximum MI among categorical features
                    corr_results[col] = all_mi_scores.get(col, 0) / max(max_mi_cat, 0.001)        
        # Store feature correlations for numeric columns only
        if not hasattr(self, '_feature_correlations') and len(num_columns) > 0:
            self._feature_correlations = X[num_columns].corr()
        
        # Store results
        target = results_dict if results_dict is not None else self.results
        target['correlation'] = corr_results
        
    def _convert_to_categorical(self, X):
        """
        Convert categorical columns to proper categorical dtype and ensure numerical columns are properly handled.
        This helps models that can natively handle categorical features.
        """
        # Force output to console by temporarily restoring stdout
        original_stdout = sys.stdout
        sys.stdout = sys.__stdout__  # Use the original stdout to bypass any redirection
        
        try:
            X_result = X.copy()
            
            # Identify categorical columns
            categorical_columns = X.select_dtypes(exclude=['number']).columns

            # Process categorical columns
            for col in categorical_columns:
                try:
                    # First ensure no NaNs/None
                    X_result[col] = X_result[col].fillna('missing')
                    
                    # Then convert to categorical
                    X_result[col] = X_result[col].astype('category')
                    print(f"Converted {col} to category with {len(X_result[col].cat.categories)} categories")
                        
                except Exception as e:
                    print(f"Error converting column {col} to category: {str(e)}")
                    # Fallback - use the column as-is
                    pass
                    
            # Process numeric columns with special handling for inf/NaN values
            numeric_columns = X.select_dtypes(include=['number']).columns
            for col in numeric_columns:
                # Replace inf with large values and NaN with 0
                X_result[col] = pd.to_numeric(X_result[col], errors='coerce')  # Force numeric
                X_result[col] = X_result[col].replace([np.inf, -np.inf], [1e9, -1e9])
                X_result[col] = X_result[col].fillna(0)
                
                if X_result[col].isna().any():
                    print(f"WARNING: Column {col} still has NaNs after conversion")
                
            # Restore original stdout
            sys.stdout = original_stdout
            return X_result
        except Exception as e:
            print(f"Error in _convert_to_categorical: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Restore original stdout
            sys.stdout = original_stdout
            # Return original if conversion fails
            return X

    def _calculate_tree_importance(self, X, y, results_dict=None, X_val=None, y_val=None):
        """Calculate tree-based feature importance using the preferred GBM model."""
        # Check if there are any features to process
        if X.shape[1] == 0:
            if self.verbose:
                print("No features available for tree importance calculation")
            target = results_dict if results_dict is not None else self.results
            target["tree"] = {}
            return
            
        # Use validation data if provided, otherwise use training data
        eval_X = X_val if X_val is not None else X
        eval_y = y_val if y_val is not None else y

        # Create copies with categorical conversion for training
        # Original X and X_val remain unchanged
        X_train = self._convert_to_categorical(X)
        eval_X_train = self._convert_to_categorical(eval_X) if X_val is not None else X_train

        # Identify column types
        categorical_columns = X_train.select_dtypes(exclude=['number']).columns
        numerical_columns = X_train.select_dtypes(include=['number']).columns
        
        # Edge case: Check if data contains any features of the required types
        has_categorical = len(categorical_columns) > 0
        has_numerical = len(numerical_columns) > 0
        
        # Log the data composition if verbose
        if self.verbose:
            print(f"Data composition: {len(numerical_columns)} numerical features, {len(categorical_columns)} categorical features")
            
        # Common parameters
        n_estimators = 100
        importances = []
        
        # Handle different scenarios based on data composition
        try:            # LightGBM importance - can handle both categorical and numerical features natively
            if LIGHTGBM_AVAILABLE and self.preferred_gbm == 'lightgbm':
                if self.verbose:
                    print("Using LightGBM for tree importance")
                    
                lgb_params = {'n_estimators': n_estimators, 'random_state': self.random_state, 'n_jobs': self.n_jobs}
                
                # Add GPU parameters if enabled and available
                if self.use_gpu:
                    if self._is_gpu_available_for_lightgbm():
                        lgb_params.update({'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0})
                    elif self.verbose:
                        print("LightGBM GPU not available, using CPU")
                        
                model_class = LGBMClassifier if self.task_type == "classification" else LGBMRegressor
                lgb_model = model_class(**lgb_params)
                lgb_model.fit(X_train, y)
                importances.append({feature: imp for feature, imp in zip(X.columns, lgb_model.feature_importances_)})
            
            # XGBoost importance
            elif XGBOOST_AVAILABLE and self.preferred_gbm == 'xgboost':
                if self.verbose:
                    print("Using XGBoost for tree importance")
                    
                xgb_params = {'n_estimators': n_estimators, 'random_state': self.random_state, 'n_jobs': self.n_jobs}
                
                # Control verbosity - suppress training output unless explicitly verbose
                if not self.verbose:
                    xgb_params['verbosity'] = 0
                
                # Only use enable_categorical when categorical features are present
                if has_categorical:
                    xgb_params['enable_categorical'] = True
                    
                # Add GPU parameters if enabled and available
                if self.use_gpu:
                    if self._is_gpu_available_for_xgboost():
                        xgb_params.update({'tree_method': 'gpu_hist', 'gpu_id': 0})
                    elif self.verbose:
                        print("XGBoost GPU not available, using CPU")
                    
                # For XGBoost, we might need to handle pure categorical data differently
                # as it may not handle them well in some versions
                if has_categorical and not has_numerical:
                    # Convert the categorical values to numeric to ensure XGBoost can process them
                    from sklearn.preprocessing import OrdinalEncoder
                    X_encoded = X_train.copy()
                    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    
                    for col in categorical_columns:
                        X_encoded[col] = encoder.fit_transform(X_train[[col]])
                        
                    model_class = XGBClassifier if self.task_type == "classification" else XGBRegressor
                    xgb_model = model_class(**xgb_params)
                    xgb_model.fit(X_encoded, y)
                else:
                    # Mix of numerical and categorical or only numerical
                    model_class = XGBClassifier if self.task_type == "classification" else XGBRegressor
                    xgb_model = model_class(**xgb_params)
                    xgb_model.fit(X_train, y)
                    
                importances.append({feature: imp for feature, imp in zip(X.columns, xgb_model.feature_importances_)})
            
            # Fallback to available model if preferred one is not available
            elif LIGHTGBM_AVAILABLE:
                if self.verbose:
                    print(f"Preferred GBM '{self.preferred_gbm}' not available, falling back to LightGBM")                    
                lgb_params = {'n_estimators': n_estimators, 'random_state': self.random_state, 'n_jobs': self.n_jobs}
                
                # Add GPU parameters if enabled and available
                if self.use_gpu:
                    if self._is_gpu_available_for_lightgbm():
                        lgb_params.update({'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0})
                    elif self.verbose:
                        print("LightGBM GPU not available, using CPU")
                    
                model_class = LGBMClassifier if self.task_type == "classification" else LGBMRegressor
                lgb_model = model_class(**lgb_params)
                lgb_model.fit(X_train, y)
                importances.append({feature: imp for feature, imp in zip(X.columns, lgb_model.feature_importances_)})
                
            elif XGBOOST_AVAILABLE:
                if self.verbose:
                    print(f"Preferred GBM '{self.preferred_gbm}' not available, falling back to XGBoost")
                    
                xgb_params = {'n_estimators': n_estimators, 'random_state': self.random_state, 'n_jobs': self.n_jobs}
                
                # Control verbosity - suppress training output unless explicitly verbose
                if not self.verbose:
                    xgb_params['verbosity'] = 0
                
                # Only use enable_categorical when categorical features are present
                if has_categorical:
                    xgb_params['enable_categorical'] = True
                    
                # Add GPU parameters if enabled and available
                if self.use_gpu:
                    if self._is_gpu_available_for_xgboost():
                        xgb_params.update({'tree_method': 'gpu_hist', 'gpu_id': 0})
                    elif self.verbose:
                        print("XGBoost GPU not available, using CPU")
                
                # For XGBoost, we might need to handle pure categorical data differently
                if has_categorical and not has_numerical:
                    # Convert the categorical values to numeric to ensure XGBoost can process them
                    from sklearn.preprocessing import OrdinalEncoder
                    X_encoded = X_train.copy()
                    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    
                    for col in categorical_columns:
                        X_encoded[col] = encoder.fit_transform(X_train[[col]])
                        
                    model_class = XGBClassifier if self.task_type == "classification" else XGBRegressor
                    xgb_model = model_class(**xgb_params)
                    xgb_model.fit(X_encoded, y)
                else:
                    model_class = XGBClassifier if self.task_type == "classification" else XGBRegressor
                    xgb_model = model_class(**xgb_params)
                    xgb_model.fit(X_train, y)
                    
                importances.append({feature: imp for feature, imp in zip(X.columns, xgb_model.feature_importances_)})
            else:
                # If no tree-based model is available, use a simple RandomForest
                if self.verbose:
                    print("No tree-based model available, using RandomForest for importance calculation")
                
                from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                rf_params = {'n_estimators': 50, 'random_state': self.random_state, 'n_jobs': self.n_jobs}
                
                # Convert categorical data if needed
                if has_categorical:
                    from sklearn.preprocessing import OrdinalEncoder
                    X_encoded = X_train.copy()
                    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    
                    for col in categorical_columns:
                        X_encoded[col] = encoder.fit_transform(X_train[[col]])
                        
                    model_class = RandomForestClassifier if self.task_type == "classification" else RandomForestRegressor
                    rf_model = model_class(**rf_params)
                    rf_model.fit(X_encoded, y)
                else:
                    model_class = RandomForestClassifier if self.task_type == "classification" else RandomForestRegressor
                    rf_model = model_class(**rf_params)
                    rf_model.fit(X_train, y)                    
                importances.append({feature: imp for feature, imp in zip(X.columns, rf_model.feature_importances_)})
                
        except Exception as e:
            if self.verbose:
                print(f"Error in tree importance calculation: {str(e)}")
                print(f"Data types: {X_train.dtypes}")
                print(f"Data shape: {X_train.shape}")
                print(f"Target shape: {y.shape}")
                import traceback
                traceback.print_exc()
            # In case of failure, assign equal importance to all features
            print("WARNING: Tree importance calculation failed, falling back to equal importance values")
            importances.append({col: 1.0/X.shape[1] for col in X.columns})
        
        # Aggregate tree importance - average across available models
        tree_results = {}
        for feature in X.columns:
            values = [imp.get(feature, 0) for imp in importances]
            tree_results[feature] = np.mean(values) if values else 0
        
        # Store results
        target = results_dict if results_dict is not None else self.results
        target["tree"] = tree_results
    
    def _calculate_permutation_importance(self, X, y, results_dict=None, X_val=None, y_val=None):
        """Calculate permutation importance using proper categorical feature handling."""
        # Use validation data if provided, otherwise use training data
        eval_X = X_val if X_val is not None else X
        eval_y = y_val if y_val is not None else y

        # Identify categorical columns
        categorical_columns = X.select_dtypes(exclude=['number']).columns
        numerical_columns = X.select_dtypes(include=['number']).columns
        
        # Create copies with proper categorical conversion
        X_proper = self._convert_to_categorical(X)
        eval_X_proper = self._convert_to_categorical(eval_X)
        
        # Use HistGradientBoosting models with proper categorical support
        model_class = HistGradientBoostingClassifier if self.task_type == 'classification' else HistGradientBoostingRegressor
        
        try:
            # Create categorical mask for HistGradientBoosting
            categorical_mask = [col in categorical_columns for col in X.columns]
            
            # Use native categorical support in newer scikit-learn versions
            model = model_class(
                max_iter=50, 
                random_state=self.random_state, 
                categorical_features=categorical_mask if any(categorical_mask) else None
            )
            
            model.fit(X_proper, y)
            
            # Scale number of repeats based on data size to manage computation time
            n_repeats = max(5, min(10, 30000 // (X.shape[0] * X.shape[1])))
            
            perm_importance = permutation_importance(
                model, eval_X_proper, eval_y, n_repeats=n_repeats, 
                random_state=self.random_state, n_jobs=self.n_jobs
            )
            
            perm_results = {col_name: perm_importance.importances_mean[col_idx] 
                            for col_idx, col_name in enumerate(X.columns)}
                
        except TypeError:
            # Fallback for older scikit-learn versions without categorical_features parameter
            if self.verbose:
                print("Using older scikit-learn version without native categorical support")
            
            # In this case, we do need to encode categorical features
            X_transformed = X.copy()
            eval_X_transformed = eval_X.copy()
            
            if len(categorical_columns) > 0:
                from sklearn.preprocessing import OrdinalEncoder
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                
                for col in categorical_columns:
                    X_transformed[col] = encoder.fit_transform(X[[col]])
                    eval_X_transformed[col] = encoder.transform(eval_X[[col]])
                
                if self.verbose:
                    print("Warning: Using OrdinalEncoder as fallback, which may introduce bias for categorical features")
                    
            model = model_class(max_iter=50, random_state=self.random_state)
            model.fit(X_transformed, y)
            
            n_repeats = max(5, min(10, 30000 // (X.shape[0] * X.shape[1])))
            perm_importance = permutation_importance(
                model, eval_X_transformed, eval_y, n_repeats=n_repeats, 
                random_state=self.random_state, n_jobs=self.n_jobs
            )
            
            perm_results = {col_name: perm_importance.importances_mean[col_idx] 
                           for col_idx, col_name in enumerate(X.columns)}
                
        except Exception as e:
            if self.verbose:
                print(f"Error in permutation importance calculation: {str(e)}")
            # Return empty results on error
            perm_results = {col: 0.0 for col in X.columns}
          # Store results
        target = results_dict if results_dict is not None else self.results
        target["permutation"] = perm_results
        
    def _calculate_shap_importance(self, X, y, results_dict=None, X_val=None, y_val=None):
        """Calculate SHAP importance with proper categorical feature handling and correct column order."""
        if not SHAP_AVAILABLE:
            if self.verbose:
                print("SHAP not available. Install with 'pip install shap'")
            return
            
        # Use validation data if provided, otherwise use training data
        eval_X = X_val if X_val is not None else X
        
        # Sample data if there are too many instances to speed up SHAP calculation
        sample_size = min(10000, eval_X.shape[0]) if eval_X.shape[0] > 1000 else eval_X.shape[0]
        eval_X_sample = eval_X.sample(sample_size, random_state=self.random_state) if sample_size < eval_X.shape[0] else eval_X
        
        # Handle NaNs and infs explicitly before passing to SHAP
        eval_X_sample = eval_X_sample.fillna(0)  # Replace NaNs with zeros
        
        # Identify categorical columns
        categorical_columns = X.select_dtypes(exclude=['number']).columns
        
        # Create copies with proper categorical dtype conversion
        X_proper = self._convert_to_categorical(X)
        eval_X_sample_proper = self._convert_to_categorical(eval_X_sample)
        
        # Handle fillna for categorical columns separately
        for col in X_proper.columns:
            if X_proper[col].dtype.name == 'category':
                # Add 0 to categories if not already present, then fillna
                if 0 not in X_proper[col].cat.categories:
                    X_proper[col] = X_proper[col].cat.add_categories([0])
                X_proper[col] = X_proper[col].fillna(0)
            else:
                # For non-categorical columns, fillna directly
                X_proper[col] = X_proper[col].fillna(0)
                
        for col in eval_X_sample_proper.columns:
            if eval_X_sample_proper[col].dtype.name == 'category':
                # Add 0 to categories if not already present, then fillna
                if 0 not in eval_X_sample_proper[col].cat.categories:
                    eval_X_sample_proper[col] = eval_X_sample_proper[col].cat.add_categories([0])
                eval_X_sample_proper[col] = eval_X_sample_proper[col].fillna(0)
            else:
                # For non-categorical columns, fillna directly
                eval_X_sample_proper[col] = eval_X_sample_proper[col].fillna(0)

        # --- Store original feature order to ensure consistency ---
        # This is the original feature order before any transformations
        original_feature_order = list(X.columns)
        
        # Track the current feature order used for model training
        # This will be updated in different code paths
        feature_order = list(X_proper.columns)
        
        # Verify feature orders match after conversion (they should, but let's be safe)
        if set(original_feature_order) != set(feature_order):
            if self.verbose:
                print("Warning: Feature order changed after categorical conversion")
                print(f"Original features: {original_feature_order}")
                print(f"Converted features: {feature_order}")
        
        try:
            # Choose appropriate model based on user preference and availability
            if self.preferred_gbm == 'xgboost' and XGBOOST_AVAILABLE:
                xgb_params = {
                    'n_estimators': 50, 
                    'random_state': self.random_state, 
                    'n_jobs': self.n_jobs
                }
                if not self.verbose:
                    xgb_params['verbosity'] = 0
                if len(categorical_columns) > 0:
                    xgb_params['enable_categorical'] = True
                if self.use_gpu:
                    if self._is_gpu_available_for_xgboost():
                        xgb_params.update({'tree_method': 'gpu_hist', 'gpu_id': 0})
                model_class = XGBClassifier if self.task_type == 'classification' else XGBRegressor
                model = model_class(**xgb_params)
                try:
                    model.fit(X_proper, y)
                    eval_X_for_shap = eval_X_sample_proper.copy()
                    feature_order = list(X_proper.columns)
                except Exception as e:
                    if self.verbose:
                        print(f"XGBoost categorical support failed: {str(e)}. Falling back to ordinal encoding.")
                    X_encoded = X.copy().fillna(0)  # Fill NaNs
                    eval_X_sample_encoded = eval_X_sample.copy().fillna(0)  # Fill NaNs
                    if len(categorical_columns) > 0:
                        from sklearn.preprocessing import OrdinalEncoder
                        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                        for col in categorical_columns:
                            X_encoded[col] = encoder.fit_transform(X[[col]].fillna('missing'))  # Handle NaNs in categorical
                            eval_X_sample_encoded[col] = encoder.transform(eval_X_sample[[col]].fillna('missing'))
                    model.fit(X_encoded, y)
                    # Ensure eval_X_for_shap uses the same column order as model
                    eval_X_for_shap = eval_X_sample_encoded.copy()
                    # Update feature_order to match the encoded data
                    feature_order = list(X_encoded.columns)
                    # Verify all original columns are still present
                    if set(original_feature_order) != set(feature_order):
                        if self.verbose:
                            print("Warning: Feature sets don't match after encoding")
                            print(f"Original features: {original_feature_order}")
                            print(f"Encoded features: {feature_order}")
            elif LIGHTGBM_AVAILABLE:
                lgb_params = {
                    'n_estimators': 50, 
                    'random_state': self.random_state, 
                    'n_jobs': self.n_jobs
                }
                if self.use_gpu:
                    if self._is_gpu_available_for_lightgbm():
                        lgb_params.update({'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0})
                model_class = LGBMClassifier if self.task_type == 'classification' else LGBMRegressor
                model = model_class(**lgb_params)
                model.fit(X_proper, y)
                eval_X_for_shap = eval_X_sample_proper[X_proper.columns]
                feature_order = list(X_proper.columns)
            else:
                try:
                    categorical_mask = [col in categorical_columns for col in X.columns]
                    model_class = HistGradientBoostingClassifier if self.task_type == 'classification' else HistGradientBoostingRegressor
                    model = model_class(
                        max_iter=100, 
                        random_state=self.random_state, 
                        categorical_features=categorical_mask if any(categorical_mask) else None
                    )
                    model.fit(X_proper, y)
                    eval_X_for_shap = eval_X_sample_proper[X_proper.columns]
                    feature_order = list(X_proper.columns)
                except TypeError:
                    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                    if self.verbose:
                        print("Using RandomForest as fallback for SHAP calculation")
                    X_encoded = X.copy().fillna(0)  # Fill NaNs
                    eval_X_sample_encoded = eval_X_sample.copy().fillna(0)  # Fill NaNs
                    if len(categorical_columns) > 0:
                        from sklearn.preprocessing import OrdinalEncoder
                        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                        for col in categorical_columns:
                            X_encoded[col] = encoder.fit_transform(X[[col]].fillna('missing'))  # Handle NaNs in categorical
                            eval_X_sample_encoded[col] = encoder.transform(eval_X_sample[[col]].fillna('missing'))
                    model_class = RandomForestClassifier if self.task_type == 'classification' else RandomForestRegressor
                    model = model_class(n_estimators=50, random_state=self.random_state, n_jobs=self.n_jobs)
                    model.fit(X_encoded, y)
                    eval_X_for_shap = eval_X_sample_encoded[X_encoded.columns]
                    feature_order = list(X_encoded.columns)
            
            # --- Ensure eval_X_for_shap columns match feature_order ---
            eval_X_for_shap = eval_X_for_shap[feature_order]
            
            # Try different SHAP approaches
            shap_values = None
            error_messages = []
            
            try:
                # Use TreeExplainer with DataFrame input
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(eval_X_for_shap)
            except Exception as e1:
                error_messages.append(str(e1))
                
                # Second attempt with numpy array
                try:
                    eval_X_for_shap_values = eval_X_for_shap.values
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(eval_X_for_shap_values)
                except Exception as e2:
                    error_messages.append(str(e2))
                    
                    # Third attempt with KernelExplainer
                    try:
                        background = shap.kmeans(eval_X_for_shap.values, 3)
                        explainer = shap.KernelExplainer(model.predict, background)
                        shap_values = explainer.shap_values(eval_X_for_shap.values[:50])
                    except Exception as e3:
                        error_messages.append(str(e3))
                        
                        # Fourth attempt with Explainer base class
                        try:
                            explainer = shap.Explainer(model)
                            shap_values = explainer(eval_X_for_shap)
                            # Extract values from Explanation object if needed
                            if hasattr(shap_values, 'values'):
                                shap_values = shap_values.values
                        except Exception as e4:
                            if self.verbose:
                                print(f"All SHAP approaches failed.")
                            # We'll handle this in the outer exception handler
            
            # Process SHAP values
            if shap_values is None:
                # Create dummy importance values
                shap_importance = np.ones(len(feature_order)) * 0.01
            else:
                try:
                    if isinstance(shap_values, list):
                        if self.task_type == 'classification' and len(shap_values) > 1:
                            # Multi-class case: average across all classes
                            clean_shap_values = []
                            for i in range(len(shap_values)):
                                clean_class = np.array(shap_values[i], dtype=np.float64)
                                clean_class = np.nan_to_num(clean_class, nan=0.0)
                                clean_shap_values.append(clean_class)
                            shap_importance = np.mean([np.abs(clean_class).mean(axis=0) for clean_class in clean_shap_values], axis=0)
                        else:
                            # Binary classification case or single array
                            if len(shap_values) > 1:
                                clean_shap = np.array(shap_values[-1], dtype=np.float64)  # Use positive class
                            else:
                                clean_shap = np.array(shap_values[0], dtype=np.float64)
                                
                            clean_shap = np.nan_to_num(clean_shap, nan=0.0)
                            shap_importance = np.abs(clean_shap).mean(axis=0)
                    else:
                        # Handle direct values (regression case or Explanation object)
                        if hasattr(shap_values, 'values'):  # It's an Explanation object
                            clean_shap = np.array(shap_values.values, dtype=np.float64)
                        else:
                            clean_shap = np.array(shap_values, dtype=np.float64)
                            
                        clean_shap = np.nan_to_num(clean_shap, nan=0.0)
                        shap_importance = np.abs(clean_shap).mean(axis=0)
                    
                    # Final NaN cleanup
                    shap_importance = np.nan_to_num(shap_importance, nan=0.01)
                        
                except Exception as e:
                    if self.verbose:
                        print(f"Error processing SHAP values: {str(e)}")
                    # Create dummy importance values
                    shap_importance = np.ones(len(feature_order)) * 0.01
            
            # Handle any remaining NaNs in the importance scores
            shap_importance = np.nan_to_num(shap_importance, nan=0.0)
            
            epsilon = 1e-10
                
            # Make sure dimensions match before creating the results dict
            if len(feature_order) == len(shap_importance):
                # Use feature_order to map SHAP values to correct feature names
                shap_results = {col_name: max(shap_importance[col_idx], epsilon) for col_idx, col_name in enumerate(feature_order)}
            else:
                # Fall back to using column names directly
                shap_results = {col_name: 0.01 for col_name in X.columns}
                
        except Exception as e:
            if self.verbose:
                print(f"Error in SHAP importance calculation: {str(e)}")
            shap_results = {col: 0.01 for col in X.columns}  # Use small non-zero value instead of 0.0
              # Store results
        target = results_dict if results_dict is not None else self.results
        target['shap'] = shap_results

    def _calculate_shap_interaction_values(self, X, y, max_pairs=200):
        """Calculate SHAP interaction values using TreeExplainer.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series): Target variable
            max_pairs (int): Maximum number of feature pairs to consider
            
        Returns:
            dict: Dictionary with feature pairs as keys and interaction strength as values
        """
        if not SHAP_AVAILABLE:
            if self.verbose:
                print("SHAP not available. Install with 'pip install shap'")
            return {}
            
        # Sample data if there are too many instances to speed up SHAP calculation
        sample_size = min(5000, X.shape[0]) if X.shape[0] > 1000 else X.shape[0]
        X_sample = X.sample(sample_size, random_state=self.random_state) if sample_size < X.shape[0] else X
        
        # Handle NaNs and infs explicitly before passing to SHAP
        X_sample = X_sample.fillna(0)  # Replace NaNs with zeros
        
        # Identify categorical columns
        categorical_columns = X.select_dtypes(exclude=['number']).columns
        
        # Create copies with proper categorical dtype conversion
        X_proper = self._convert_to_categorical(X)
        X_sample_proper = self._convert_to_categorical(X_sample)
        
        # Handle fillna for categorical columns separately
        for col in X_proper.columns:
            if X_proper[col].dtype.name == 'category':
                # Add 0 to categories if not already present, then fillna
                if 0 not in X_proper[col].cat.categories:
                    X_proper[col] = X_proper[col].cat.add_categories([0])
                X_proper[col] = X_proper[col].fillna(0)
                
                if 0 not in X_sample_proper[col].cat.categories:
                    X_sample_proper[col] = X_sample_proper[col].cat.add_categories([0])
                X_sample_proper[col] = X_sample_proper[col].fillna(0)
            else:
                # For non-categorical columns, fillna directly
                X_proper[col] = X_proper[col].fillna(0)
                X_sample_proper[col] = X_sample_proper[col].fillna(0)
        
        # Store original feature order to ensure consistency
        original_feature_order = list(X.columns)
        feature_order = list(X_proper.columns)
        
        try:
            # Choose appropriate model based on user preference and availability
            if self.preferred_gbm == 'xgboost' and XGBOOST_AVAILABLE:
                xgb_params = {
                    'n_estimators': 50, 
                    'random_state': self.random_state, 
                    'n_jobs': self.n_jobs
                }
                if not self.verbose:
                    xgb_params['verbosity'] = 0
                if len(categorical_columns) > 0:
                    xgb_params['enable_categorical'] = True
                if self.use_gpu:
                    if self._is_gpu_available_for_xgboost():
                        xgb_params.update({'tree_method': 'gpu_hist', 'gpu_id': 0})
                model_class = XGBClassifier if self.task_type == 'classification' else XGBRegressor
                model = model_class(**xgb_params)
                
                try:
                    model.fit(X_proper, y)
                    explainer = shap.TreeExplainer(model)
                    
                    # Calculate SHAP interaction values
                    shap_interaction_values = explainer.shap_interaction_values(X_sample_proper)
                    
                    # For classification tasks, we need to handle multiple classes
                    if self.task_type == 'classification':
                        if isinstance(shap_interaction_values, list):
                            # Multi-class case - average across classes
                            shap_interaction_values = np.abs(np.array(shap_interaction_values)).mean(axis=0)
                        else:
                            # Binary case - take absolute values
                            shap_interaction_values = np.abs(shap_interaction_values)
                    
                    # Create a dictionary to store feature pair interactions
                    feature_interactions = {}
                    for i in range(len(feature_order)):
                        for j in range(i+1, len(feature_order)):
                            # Sum the absolute interaction values in both directions (symmetrical matrix)
                            interaction_value = np.abs(shap_interaction_values[:, i, j]).mean() + np.abs(shap_interaction_values[:, j, i]).mean()
                            if interaction_value > 0:  # Only store non-zero interactions
                                pair_key = (feature_order[i], feature_order[j])
                                feature_interactions[pair_key] = float(interaction_value)
                    
                    # Sort by interaction strength and limit to max_pairs
                    sorted_interactions = sorted(feature_interactions.items(), key=lambda x: x[1], reverse=True)
                    top_interactions = dict(sorted_interactions[:max_pairs])
                    
                    if self.verbose:
                        print(f"Calculated {len(top_interactions)} feature interactions")
                    
                    return top_interactions
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Error calculating SHAP interaction values: {str(e)}")
                    return {}
            
            else:
                if self.verbose:
                    print("XGBoost is required for SHAP interaction values calculation")
                return {}
                
        except Exception as e:
            if self.verbose:
                print(f"Error in SHAP interaction values calculation: {str(e)}")
            return {}
            
    def get_feature_interactions(self, X, y, max_pairs=200):
        """Get feature interaction strengths using SHAP interaction values.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series): Target variable
            max_pairs (int): Maximum number of feature pairs to consider
            
        Returns:
            dict: Dictionary with feature pairs as keys and interaction strength as values
        """
        return self._calculate_shap_interaction_values(X, y, max_pairs=max_pairs)

    def get_importance(self, normalize=True, include_std=False):
        """
        Get aggregated feature importance based on configured weights.
        
        When used for visualization/reporting, setting normalize=True provides
        normalized values (0-1) for clearer comparisons and better plots.
          When used for feature selection or genetic feature generation, 
        setting normalize=False returns balanced importance values that maintain the relative
        ordering of features within each metric type but prevent any single metric (like SHAP)
        from overpowering others due to scale differences. This approach:
        1. Scales metrics relative to each other using their median values
        2. Preserves meaningful ratios within each importance method
        3. Ensures all metrics contribute to the final weighted importance according to their 
           configured weights, regardless of their original scale
        4. Avoids distortions from min-max normalization that would alter sampling distributions
        
        Parameters:
        -----------
        normalize : bool, default=True
            Whether to normalize importance values to [0,1] range.
            For visualization/reporting, this should be True.
            For feature selection where original scale matters, set to False.
        include_std : bool, default=False
            Whether to include standard deviation columns from cross-validation.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame of feature importance values with columns for each
            importance method and a weighted_importance column.
        """
        # Initialize DataFrame with all features
        all_features = set()
        for metric_type in self.results:
            all_features.update(self.results[metric_type].keys())
            
        importance_df = pd.DataFrame(index=list(all_features))
        
        # Add individual metric scores
        for metric_type in self.results:
            metric_scores = pd.Series(self.results[metric_type])
            
            # Store original unnormalized values in separate columns
            if not normalize:
                importance_df[f'{metric_type}_importance_original'] = metric_scores.copy()
            
            # Normalize each metric scores to [0, 1] range for combining methods
            # Using softmax-based normalization to better preserve relative importance
            if len(metric_scores) > 1 and metric_scores.max() > 0:
                # Softmax normalization better preserves relative differences compared to min-max scaling
                exp_scores = np.exp(metric_scores - np.max(metric_scores))  # Subtract max for numerical stability
                normalized_scores = exp_scores / np.sum(exp_scores)
            else:
                normalized_scores = metric_scores.copy()
                
            # For consistency in combining methods, always use normalized scores internally
            # but we may return unnormalized values based on the normalize parameter
            importance_df[f'{metric_type}_importance'] = normalized_scores if normalize else metric_scores
            
            # Add standard deviation from CV if available
            if include_std and self.n_cv_folds is not None and metric_type in self.cv_results:
                std_values = {feature: np.std(values) for feature, values in self.cv_results[metric_type].items() if values}
                std_series = pd.Series(std_values)
                if normalize and std_series.max() > 0:
                    std_series = std_series / normalized_scores.max() if normalized_scores.max() > 0 else std_series
                importance_df[f'{metric_type}_std'] = std_series
        
        # Calculate weighted importance using normalized values for consistent weighting
        weighted_importance = pd.Series(0.0, index=importance_df.index)
        
        # Track which methods were used for each feature
        feature_available_methods = {feature: [] for feature in importance_df.index}
        for metric_type in self.weights:
            col_name = f'{metric_type}_importance'
            if col_name in importance_df.columns:
                for feature in importance_df.index:
                    if pd.notna(importance_df.loc[feature, col_name]):
                        feature_available_methods[feature].append(metric_type)
                          # Calculate weighted importance with per-feature weight normalization
        weighted_importance_unnormalized = pd.Series(0.0, index=importance_df.index)
          # For unnormalized values, we need to normalize metrics relative to each other first
        # This prevents one metric (e.g., SHAP) from overpowering others due to scale differences
        # Example: SHAP values might be ~7.0 while tree importance is ~0.01, causing SHAP to dominate
        # even when its configured weight is the same as other metrics
        metric_scale_factors = {}
        if not normalize:
            for metric in self.weights:
                if metric in self.results and self.weights[metric] > 0:
                    # Get median of non-zero values for this metric to use as scale factor
                    values = pd.Series(self.results[metric])
                    positive_values = values[values > 0]
                    if len(positive_values) > 0:
                        # Use median as a robust measure of central tendency
                        metric_scale_factors[metric] = positive_values.median()
                    else:
                        metric_scale_factors[metric] = 1.0
            
            # Ensure no division by zero and normalize scale factors
            if metric_scale_factors:
                # If any metric has a very small scale factor, set a minimum
                min_scale = max(1e-6, min(metric_scale_factors.values()))
                for metric in metric_scale_factors:
                    metric_scale_factors[metric] = max(metric_scale_factors[metric], min_scale)
        
        for feature in importance_df.index:
            # Get available methods for this feature
            available_methods = feature_available_methods[feature]
            if not available_methods:
                continue
                
            # Calculate the sum of weights for available methods for this feature
            total_weight = sum(self.weights[metric] for metric in available_methods)
            
            # Apply weights, normalized by the sum of available weights
            if total_weight > 0:
                for metric in available_methods:
                    # Use normalized values for combining methods consistently
                    col_name = f'{metric}_importance'
                    normalized_weight = self.weights[metric] / total_weight
                    
                    # Apply the same weight to both normalized and unnormalized values
                    weighted_importance[feature] += importance_df.loc[feature, col_name] * normalized_weight
                    
                    # If we have original unnormalized values, scale them before weighting to prevent domination by one metric
                    if not normalize and f'{metric}_importance_original' in importance_df.columns:
                        orig_value = importance_df.loc[feature, f'{metric}_importance_original']
                        # Apply metric-specific scaling to balance the metrics
                        if metric in metric_scale_factors and metric_scale_factors[metric] > 0:
                            scaled_value = orig_value / metric_scale_factors[metric]
                        else:
                            scaled_value = orig_value
                        weighted_importance_unnormalized[feature] += scaled_value * normalized_weight
        
        # Store the weighted importance (either normalized or unnormalized)
        if not normalize and 'tree_importance_original' in importance_df.columns:
            # Use unnormalized weighted importance if we have original values and normalize=False
            importance_df['weighted_importance'] = weighted_importance_unnormalized
            # Clean up temporary columns
            for col in [c for c in importance_df.columns if c.endswith('_original')]:
                importance_df.drop(col, axis=1, inplace=True)
        else:
            # Store the normalized weighted importance
            importance_df['weighted_importance'] = weighted_importance
            
            # Normalize all features together if requested
            if normalize:
                if len(weighted_importance) > 1 and weighted_importance.max() > 0:
                    # Use softmax normalization to better preserve relative importance differences
                    exp_importance = np.exp(weighted_importance - np.max(weighted_importance))
                    importance_df['weighted_importance'] = exp_importance / np.sum(exp_importance)
                elif weighted_importance.max() > 0:
                    # Fallback to division by max if only one feature
                    importance_df['weighted_importance'] = weighted_importance / weighted_importance.max()
        
        # Sort by weighted importance
        return importance_df.sort_values('weighted_importance', ascending=False)

    def plot_importance(self, top_n=20, include_std=False, figsize=(14, 10), 
                       color_map='viridis', annotations=True, style='seaborn-v0_8-colorblind', 
                       grid=True, title=None, as_percent=True):
        """Plot feature importance chart for top N features with enhanced styling."""
        if not PLOTTING_AVAILABLE:
            print("Matplotlib is required for plotting. Install it with 'pip install matplotlib'")
            return None
            
        # Get importance dataframe and top features
        importance_df = self.get_importance(include_std=include_std)
        top_features = importance_df.head(top_n)
        
        # Set style and create figure
        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create colormap for gradient
            cmap = cm.get_cmap(color_map)
            norm = mpl.colors.Normalize(vmin=0, vmax=len(top_features))
            
            # Plot bars
            for i, (feature, row) in enumerate(top_features.iterrows()):
                color = cmap(norm(i))
                value = row['weighted_importance']
                display_value = value * 100 if as_percent else value
                
                # Plot bar with or without error bar
                if include_std and self.n_cv_folds is not None and 'weighted_std' in top_features.columns:
                    err = row.get('weighted_std', 0)
                    err_value = err * 100 if as_percent else err
                    ax.barh(feature, display_value, xerr=err_value, color=color, 
                           alpha=0.8, ecolor='black', capsize=5)
                else:
                    ax.barh(feature, display_value, color=color, alpha=0.8)
                
                # Add annotations
                if annotations:
                    ax.text(display_value + (2 if as_percent else 0.02), i, 
                           f"{display_value:.1f}{'%' if as_percent else ''}", 
                           va='center', fontweight='bold', fontsize=9)
    
            # Set title and labels
            default_title = f"Top {top_n} Features by Importance"
            if self.n_cv_folds and include_std:
                default_title += f" (with {self.n_cv_folds}-fold CV)"
            ax.set_title(title or default_title, fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Importance' + (' (%)' if as_percent else ''), fontsize=12, fontweight='bold')
            ax.set_ylabel('Features', fontsize=12, fontweight='bold')
            
            # Add grid and customize
            if grid:
                ax.grid(axis='x', linestyle='--', alpha=0.6)
            ax.tick_params(axis='both', which='major', labelsize=10)
            
            # Add legend and footer
            methods_used = [f"{method.capitalize()} ({weight:.2f})" for method, weight in self.weights.items() 
                            if f"{method}_importance" in importance_df.columns and weight > 0]
            if methods_used:
                fig.text(0.01, 0.01, f"Methods used: {', '.join(methods_used)}", fontsize=9, style='italic')
            
            if self.n_cv_folds:
                cv_type = "Stratified CV" if self.task_type == "classification" else "K-Fold CV"
                fig.text(0.99, 0.01, f"{cv_type} with {self.n_cv_folds} folds", fontsize=9, style='italic', ha='right')
                    
            plt.tight_layout()
            return fig  # Return the figure object instead of plt

    def plot_feature_importance_dashboard(self, top_n=15, figsize=(20, 16), save_path=None):
        """Create a comprehensive dashboard with multiple feature importance visualizations."""
        if not PLOTTING_AVAILABLE:
            print("Matplotlib is required for plotting. Install it with 'pip install matplotlib'")
            return None, None

        # Load seaborn if available
        try:
            import seaborn as sns
        except ImportError:
            pass
        
        # Set higher font size for readability
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10
        })
        
        # Get importance data 
        importance_df = self.get_importance(include_std=True)
        
        # Determine available plots
        has_cv = self.n_cv_folds is not None and self.cv_results
        has_multiple_methods = sum(1 for m, w in self.weights.items() 
                                   if w > 0 and f"{m}_importance" in importance_df.columns) > 1
        
        # Create 2x2 dashboard layout
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Plot 1: Main importance
        self._plot_importance_bars(axes[0], importance_df, top_n=top_n)
        
        # Plot 2: Method comparison if available, otherwise correlation heatmap
        if has_multiple_methods:
            self._plot_method_comparison(axes[1], importance_df, top_n=min(10, top_n))
        elif hasattr(self, '_feature_correlations'):
            self._plot_correlation_heatmap(axes[1], importance_df.index[:min(10, top_n)].tolist())
        else:
            axes[1].set_visible(False)
            
        # Plot 3: CV distributions if available
        if has_cv:
            self._plot_cv_boxplots(axes[2], importance_df.index[:min(8, top_n)].tolist())
        else:
            axes[2].set_visible(False)
            
        # Plot 4: Stability or correlation heatmap
        if has_cv:
            self._plot_stability_metrics(axes[3])
        elif hasattr(self, '_feature_correlations') and not has_multiple_methods:
            self._plot_correlation_heatmap(axes[3], importance_df.index[:min(10, top_n)].tolist())
        else:
            axes[3].set_visible(False)
            
        # Add title
        title = "Feature Importance Analysis Dashboard"
        if self.n_cv_folds:
            cv_type = "Stratified " if self.task_type == "classification" else ""
            title += f" ({cv_type}CV with {self.n_cv_folds} folds)"
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            
        return fig, axes

    # Helper methods for dashboard plots - simplified versions of the standalone plots
    def _plot_importance_bars(self, ax, importance_df, top_n=15, color_map='viridis', 
                             include_std=True, as_percent=True, annotations=True, grid=True):
        """Simplified helper for importance bars."""
        top_features = importance_df.head(top_n)
        cmap = cm.get_cmap(color_map)
        
        for i, (feature, row) in enumerate(top_features.iterrows()):
            color = cmap(0.2 + 0.6 * i / len(top_features))
            value = row['weighted_importance']
            display_value = value * 100 if as_percent else value
            
            if include_std and 'weighted_std' in top_features.columns:
                err = row.get('weighted_std', 0) * (100 if as_percent else 1)
                ax.barh(feature, display_value, xerr=err, color=color, alpha=0.8, ecolor='black', capsize=3)
            else:
                ax.barh(feature, display_value, color=color, alpha=0.8)
            
            if annotations:
                ax.text(display_value + (2 if as_percent else 0.02), i, 
                       f"{display_value:.1f}{'%' if as_percent else ''}", va='center', fontsize=9)
    
        ax.set_title("Top Feature Importance", fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance' + (' (%)' if as_percent else ''))
        if grid:
            ax.grid(axis='x', linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
            
    def _plot_method_comparison(self, ax, importance_df, top_n=10, as_percent=True, grid=True):
        """Simplified helper for method comparison."""
        top_features = importance_df.head(top_n).index.tolist()
        methods = [col.replace('_importance', '') for col in importance_df.columns 
                  if col.endswith('_importance') and col != 'weighted_importance']
        
        x = np.arange(len(top_features))
        width = 0.8 / len(methods)
        offsets = [(i - (len(methods) - 1) / 2) * width for i in range(len(methods))]
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        
        for i, method in enumerate(methods):
            values = [importance_df.loc[f, f"{method}_importance"] * (100 if as_percent else 1) 
                     for f in top_features]
            ax.bar(x + offsets[i], values, width * 0.9, label=method.capitalize(), color=colors[i], alpha=0.85)
        
        ax.set_title("Importance by Method", fontsize=14, fontweight='bold')
        ax.set_ylabel('Importance' + (' (%)' if as_percent else ''))
        ax.set_xticks(x)
        ax.set_xticklabels([f[:15] + ('...' if len(f) > 15 else '') for f in top_features], 
                           rotation=30, ha='right', fontsize=9)
        ax.legend(loc='upper right', framealpha=0.9)
        if grid:
            ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
            
    def _plot_cv_boxplots(self, ax, features, method='weighted', color_map='viridis', as_percent=True):
        """Simplified helper for CV boxplots."""
        if self.n_cv_folds is None or not self.cv_results:
            ax.text(0.5, 0.5, "Cross-validation data not available", ha='center', va='center')
            ax.set_axis_off()
            return
            
        boxplot_data = []
        labels = []
        
        # For each method, collect all feature values to determine max within each method
        method_max_values = {}
        for m_type in self.cv_results:
            method_values = []
            for feature in self.cv_results[m_type]:
                method_values.extend(self.cv_results[m_type][feature])
            if method_values:
                method_max_values[m_type] = max(method_values)
        
        for feature in features:
            feature_values = []
            
            # Check if this feature has any CV data
            has_data = any(feature in self.cv_results.get(m_type, {}) 
                          for m_type in self.weights if self.weights.get(m_type, 0) > 0)
            
            if has_data:
                # Collect normalized importance values for each fold
                for fold in range(self.n_cv_folds):
                    weighted_val = 0
                    for m_type, weight in self.weights.items():
                        if (m_type in self.cv_results and 
                            feature in self.cv_results[m_type] and 
                            fold < len(self.cv_results[m_type][feature])):
                            
                            # Normalize within each method by its maximum value
                            if method_max_values.get(m_type, 0) > 0:
                                fold_val = self.cv_results[m_type][feature][fold]
                                norm_val = fold_val / method_max_values[m_type]
                                weighted_val += norm_val * weight
                    
                    # Add normalized and weighted value for this fold
                    feature_values.append(weighted_val)
                
                if feature_values:  # Only add if we have data
                    # Apply percentage scaling if needed, without distorting relative variability
                    if as_percent:
                        feature_values = [val * 100 for val in feature_values]
                    boxplot_data.append(feature_values)
                    labels.append(feature)
        
        if boxplot_data:
            cmap = cm.get_cmap(color_map)
            bplot = ax.boxplot(boxplot_data, vert=False, patch_artist=True,
                              labels=[f[:15] + ('...' if len(f) > 15 else '') for f in labels], 
                              widths=0.6, showfliers=False)
            
            colors = [cmap(0.2 + 0.6 * i/len(boxplot_data)) for i in range(len(boxplot_data))]
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
        
            ax.set_title("CV Fold Distributions", fontsize=14, fontweight='bold')
            ax.set_xlabel('Normalized Importance' + (' (%)' if as_percent else ''))
            ax.grid(axis='x', linestyle='--', alpha=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        else:
            ax.text(0.5, 0.5, "No CV distribution data available", ha='center', va='center')
            ax.set_axis_off()
            
    def _plot_stability_metrics(self, ax, color_map='viridis'):
        """Simplified helper for stability metrics."""
        stability = self.get_cv_stability_score()
        if not stability:
            ax.text(0.5, 0.5, "Stability data not available", ha='center', va='center')
            ax.set_axis_off()
            return
            
        methods = [m for m in stability.keys() if m != 'overall']
        values = [stability[m] for m in methods]
        if 'overall' in stability:
            methods.append('overall')
            values.append(stability['overall'])
            
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(methods)))
        bars = ax.bar(methods, values, color=colors, alpha=0.85, width=0.7)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_title("Feature Ranking Stability", fontsize=14, fontweight='bold')
        ax.set_ylabel('Jaccard Similarity')

        ax.set_ylim(0, min(1.0, max(values) * 1.2))
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    def _plot_correlation_heatmap(self, ax, features, color_map='RdBu_r'):
        """Simplified helper for correlation heatmap."""
        if not hasattr(self, '_feature_correlations'):
            ax.text(0.5, 0.5, "Feature correlation data not available", ha='center', va='center')
            ax.set_axis_off()
            return
            
        features = [f for f in features if f in self._feature_correlations.index]
        if not features:
            ax.text(0.5, 0.5, "No numeric features available", ha='center', va='center')

            ax.set_axis_off()
            return
            
        correlation_matrix = self._feature_correlations.loc[features, features]
        
        try:
            import seaborn as sns
            sns.heatmap(correlation_matrix, cmap=color_map, annot=True, fmt=".2f", 
                       linewidths=0.5, cbar=True, ax=ax, annot_kws={"size": 9}, 
                       vmin=-1, vmax=1, square=True)
        except ImportError:
            im = ax.imshow(correlation_matrix, cmap=color_map, vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax)
            ax.set_xticks(np.arange(len(features)))
            ax.set_yticks(np.arange(len(features)))
            ax.set_xticklabels(features, rotation=45, ha='right')
            ax.set_yticklabels(features)
            
        ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight='bold')
        
    def save_plots(self, output_dir="cache/plots", formats=None):
        """Save multiple feature importance plots to an output directory."""
        if not PLOTTING_AVAILABLE:
            print("Matplotlib is required for plotting. Install it with 'pip install matplotlib'")
            return {}
            
        os.makedirs(output_dir, exist_ok=True)
        formats = formats or ['png']
        saved_paths = {}
        
        # Main dashboard
        fig, _ = self.plot_feature_importance_dashboard(save_path=None)
        for fmt in formats:
            path = os.path.join(output_dir, f"feature_importance_dashboard.{fmt}")
            fig.savefig(path, bbox_inches='tight', dpi=150)
            saved_paths.setdefault('dashboard', []).append(path)
        plt.close(fig)
        
        # Main importance plot
        fig = self.plot_importance()
        if fig:
            for fmt in formats:
                path = os.path.join(output_dir, f"feature_importance_importance.{fmt}")
                fig.savefig(path, bbox_inches='tight', dpi=150)
                saved_paths.setdefault('importance', []).append(path)
            plt.close(fig)  # Close the figure, not the plt
        
        # Get other components from dashboard
        importance_df = self.get_importance()
        
        # Method comparison
        if sum(1 for m, w in self.weights.items() if w > 0) > 1:
            fig, ax = plt.subplots(figsize=(12, 8))
            self._plot_method_comparison(ax, importance_df)
            for fmt in formats:
                path = os.path.join(output_dir, f"feature_importance_method_comparison.{fmt}")
                fig.savefig(path, bbox_inches='tight', dpi=150)
                saved_paths.setdefault('method_comparison', []).append(path)
            plt.close(fig)
            
        # CV distributions
        if self.n_cv_folds is not None and self.cv_results:
            fig, ax = plt.subplots(figsize=(12, 8))
            self._plot_cv_boxplots(ax, importance_df.index[:10].tolist())
            for fmt in formats:
                path = os.path.join(output_dir, f"feature_importance_cv_distribution.{fmt}")
                fig.savefig(path, bbox_inches='tight', dpi=150)
                saved_paths.setdefault('cv_distribution', []).append(path)
            plt.close(fig)
            
            # Stability plot
            fig, ax = plt.subplots(figsize=(10, 6))
            self._plot_stability_metrics(ax)
            for fmt in formats:
                path = os.path.join(output_dir, f"feature_importance_stability.{fmt}")
                fig.savefig(path, bbox_inches='tight', dpi=150)
                saved_paths.setdefault('stability', []).append(path)
            plt.close(fig)
            
        # Correlation heatmap
        if hasattr(self, '_feature_correlations'):
            fig, ax = plt.subplots(figsize=(12, 10))
            self._plot_correlation_heatmap(ax, importance_df.index[:10].tolist())
            for fmt in formats:
                path = os.path.join(output_dir, f"feature_importance_correlation.{fmt}")
                fig.savefig(path, bbox_inches='tight', dpi=150)
                saved_paths.setdefault('correlation', []).append(path)
            plt.close(fig)
            
        return saved_paths

    def get_cv_stability_score(self, top_k=10):
        """
        Calculate stability score for cross-validation feature rankings.
        
        Uses Jaccard similarity to measure stability of feature selection across folds.
        Higher scores indicate more stable feature selection.
        
        Args:
            top_k (int): Number of top features to consider for stability calculation
            
        Returns:
            dict: Stability scores for each method and overall weighted average
        """
        if self.n_cv_folds is None or not self.cv_results:
            return None
            
        stability_scores = {}
        
        # Calculate stability for each method
        for method, method_results in self.cv_results.items():
            if not method_results:
                continue
                
            # Get feature rankings for each fold
            fold_rankings = []
            for fold in range(self.n_cv_folds):
                # Get features and their importance for this fold
                fold_importance = {feature: values[fold] if fold < len(values) else 0 
                                  for feature, values in method_results.items()}
                
                # Sort features by importance and take top k
                top_features = sorted(fold_importance, key=fold_importance.get, reverse=True)[:top_k]
                fold_rankings.append(set(top_features))
            
            # Calculate Jaccard similarity between all pairs of fold rankings
            jaccard_scores = []
            for i in range(len(fold_rankings)):
                for j in range(i+1, len(fold_rankings)):
                    intersection = len(fold_rankings[i].intersection(fold_rankings[j]))
                    union = len(fold_rankings[i].union(fold_rankings[j]))
                    jaccard_scores.append(intersection / max(union, 1))  # Avoid division by zero
            
            # Average Jaccard score is our stability metric for this method
            stability_scores[method] = sum(jaccard_scores) / max(len(jaccard_scores), 1)
        
        # Calculate overall weighted stability score
        weighted_score = 0
        total_weight = 0
        for method, score in stability_scores.items():
            method_weight = self.weights.get(method, 0)
            weighted_score += score * method_weight
            total_weight += method_weight
            
        if total_weight > 0:
            stability_scores['overall'] = weighted_score / total_weight
        
        return stability_scores
    
    def plot_feature_interactions(self, X=None, y=None, top_n=20, figsize=(14, 12), color_map='viridis', title=None):
        """
        Plot top feature interactions from SHAP interaction values.
        
        Args:
            X (pd.DataFrame, optional): Input features. Only needed if interactions haven't been calculated yet.
            y (pd.Series, optional): Target variable. Only needed if interactions haven't been calculated yet.
            top_n (int): Number of top interactions to show
            figsize (tuple): Figure size
            color_map (str): Color map for visualization
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if not PLOTTING_AVAILABLE:
            print("Matplotlib is required for plotting. Install it with 'pip install matplotlib'")
            return None
            
        # Calculate interactions if not already done
        interactions = getattr(self, '_feature_interactions_values', None)
        if interactions is None and X is not None and y is not None:
            interactions = self._calculate_shap_interaction_values(X, y)
            self._feature_interactions_values = interactions
            
        if not interactions:
            print("No interaction values available. Run get_feature_interactions first.")
            return None
            
        # Sort interactions by strength and get top N
        sorted_interactions = sorted(interactions.items(), key=lambda x: x[1], reverse=True)
        top_interactions = sorted_interactions[:top_n]
        
        with plt.style.context('seaborn-v0_8-colorblind'):
            fig, ax = plt.subplots(figsize=figsize)
            
            # Prepare data for plotting
            feature_pairs = [f"{pair[0]}\n× {pair[1]}" for pair, _ in top_interactions]
            values = [value for _, value in top_interactions]
            
            # Create colormap
            cmap = cm.get_cmap(color_map)
            norm = mpl.colors.Normalize(vmin=min(values), vmax=max(values))
            colors = [cmap(norm(val)) for val in values]
            
            # Plot horizontal bars
            y_pos = range(len(feature_pairs))
            ax.barh(y_pos, values, color=colors, alpha=0.8)
            
            # Customize plot
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_pairs)
            ax.invert_yaxis()  # Highest values at the top
            ax.set_xlabel('Interaction Strength')
            ax.set_title(title or f'Top {top_n} Feature Interactions by SHAP', fontsize=16, fontweight='bold', pad=20)
            
            # Add grid
            ax.grid(axis='x', linestyle='--', alpha=0.6)
            
            # Add values as annotations
            for i, val in enumerate(values):
                ax.text(val + max(values)*0.02, i, f"{val:.4f}", va='center')
                
            plt.tight_layout()
            return fig

