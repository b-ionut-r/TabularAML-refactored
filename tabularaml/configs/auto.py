# -*- coding: utf-8 -*-

"""
Optimized Hyperparameter Search Space Generation Module
======================================================

Generates adaptive hyperparameter search spaces tailored to:
- Dataset characteristics (size, features, target, sparsity, correlations)
- Computational budget (categorical or time-based continuous scaling)
- Task type (classification/regression, multiclass, imbalance)

Includes automatic calibration of the time estimation heuristic (`BASE_TIME_UNIT`)
by running a quick benchmark training on the target hardware and data sample.

Key Principles:
1. Data-driven Scaling: Parameters like estimators, depth, leaves scale
   logarithmically or polynomially with dataset size.
2. Budget-aware Adaptation: Search ranges and model priorities are adjusted
   continuously based on the available budget. Low budgets focus search
   on faster, simpler regions; high budgets explore more complex settings.
3. Task-Specific Tuning: Different parameters/ranges emphasized for
   classification vs. regression, handling multiclass and imbalance (including eval_metric).
4. Feature-aware Adjustments: High dimensionality, categorical feature ratios,
   correlation, and sparsity influence regularization, feature subsampling, penalty types,
   and model choice/priority.
5. Interdependency Management: Coordinated adjustments for related parameters
   (e.g., depth/leaves, learning_rate/estimators with stricter capping).
6. Structured Configuration: Parameter definitions include metadata for
   easier and more consistent heuristic application.
7. Automatic Time Calibration: Estimates the hardware/data-specific time scaling
   factor (`BASE_TIME_UNIT`) via a benchmark run when using time budgets.

Main functions:
- auto_config_from_data: Generate search spaces from data and budget level.
- auto_config_from_data_with_time_budget: Time-aware generation with
  continuous scaling, automatic base time unit calibration (optional override),
  and optional GPU usage flag for calibration.

Requirements for Time Calibration:
- `lightgbm` library must be installed.
"""

import pandas as pd
import numpy as np
import math
import warnings
import time
import pprint
from typing import Dict, Tuple, List, Union, Any, Optional, Literal
from scipy import stats
from scipy.sparse import issparse, csr_matrix

# Attempt to import LightGBM for calibration, warn if unavailable
try:
    import lightgbm as lgb
    _lightgbm_available = True
except ImportError:
    lgb = None
    _lightgbm_available = False
    warnings.warn(
        "LightGBM library not found. Automatic time budget calibration will be skipped, "
        "falling back to default BASE_TIME_UNIT. Install lightgbm for calibration."
    )

# ==============================================================================
# --- Configuration Constants ---
# ==============================================================================

# --- General Scaling & Bounds ---
MIN_ESTIMATORS = 30
# Adjusted MAX_ESTIMATORS based on user requirement to avoid excessively high values except in limit cases
MAX_ESTIMATORS = 2500  # Lowered from 4000, limit cases handled by scaling/interdependency logic
MAX_DEPTH_HARD_CAP = 24
MAX_LEAVES_HARD_CAP = 2048 # 2**11
MIN_SAMPLES_PER_LEAF = 1
BASE_FEATURE_FRACTION_LOW = 0.5
BASE_FEATURE_FRACTION_HIGH = 1.0
BASE_SUBSAMPLE_LOW = 0.5
BASE_SUBSAMPLE_HIGH = 1.0

# Define strict bounds for parameters (used for clamping)
PARAM_BOUNDS = {
    # Fractions / Probabilities [0, 1] - Use small epsilon to avoid exact 0/1 for some models/samplers
    "subsample": (0.01, 1.0),
    "colsample_bytree": (0.01, 1.0),
    "colsample_bylevel": (0.01, 1.0),
    "feature_fraction": (0.01, 1.0), # LGB alias
    "bagging_fraction": (0.01, 1.0), # LGB alias
    "l1_ratio": (0.0, 1.0),
    "bagging_temperature": (0.0, 1.0),
    "max_features": (0.01, 1.0),
    # Non-negative Reals (use small value for log scale lower bound if needed)
    "gamma": (0.0, None),
    "reg_alpha": (1e-9, None), # L1: Use small positive for log
    "reg_lambda": (1e-9, None), # L2: Use small positive for log
    "lambda_l1": (1e-9, None), # LGB alias
    "lambda_l2": (1e-9, None), # LGB alias
    "alpha": (1e-9, None),     # SGD alias: Use small positive for log
    "l2_leaf_reg": (1e-9, None), # CatBoost needs > 0
    "random_strength": (1e-9, None), # Use small positive for log
    "min_child_weight": (0.0, None),
    "eta0": (1e-9, None), # Use small positive for log
    "power_t": (0.0, None),
    "tol": (1e-9, None), # Use small positive for log
    "scale_pos_weight": (1e-9, None), # Use small positive for log
    # Integers >= 1 or 2
    "min_samples_split": (2, None),
    "min_samples_leaf": (1, None),
    "min_child_samples": (1, None), # LGB alias for min_samples_leaf
    "bagging_freq": (0, None), # 0 means disable bagging
    "n_iter_no_change": (1, None),
    "border_count": (1, 255), # CatBoost constraint
    "iterations": (MIN_ESTIMATORS, MAX_ESTIMATORS),
    "n_estimators": (MIN_ESTIMATORS, MAX_ESTIMATORS),
    "max_depth": (1, MAX_DEPTH_HARD_CAP), # Actual min is 1 or 2, -1 handled specially for LGB
    "depth": (1, MAX_DEPTH_HARD_CAP), # CatBoost alias
    "num_leaves": (2, MAX_LEAVES_HARD_CAP),
    "max_iter": (50, None), # SGD iterations
}

# --- Data Characteristic Thresholds ---
HIGH_DIM_THRESHOLD = 150 # Features count considered high
VERY_HIGH_DIM_THRESHOLD = 1000
HIGH_DIM_RATIO = 1.5 # n_features > n_samples * ratio
LARGE_DATASET_THRESHOLD = 100_000 # Samples
SMALL_DATASET_THRESHOLD = 1_500
IMBALANCE_THRESHOLD_MODERATE = 3.0
IMBALANCE_THRESHOLD_HIGH = 10.0
HIGH_CORRELATION_THRESHOLD = 0.8
HIGH_CORRELATION_RATIO_THRESHOLD = 0.1 # % of feature pairs highly correlated
HIGH_CATEGORICAL_RATIO = 0.4
SPARSITY_THRESHOLD = 0.3 # If >30% zeros, consider sparse adjustments
CATEGORICAL_UNIQUE_COUNT_THRESHOLD = 50 # Max unique values for auto-detected int categorical
CATEGORICAL_UNIQUE_RATIO_THRESHOLD = 0.05 # Max unique ratio for auto-detected int categorical

# --- Budget Scaling Factors ---
BUDGET_SCALE_LOW = 0.6  # Shrink range width for low budget
BUDGET_SCALE_HIGH = 1.6 # Expand range width for high budget
BUDGET_CENTER_SHIFT_FACTOR = 0.15 # Shift center slightly for low/high budget

# --- Priority Adjustment Factors ---
PRIORITY_IMBALANCE_BOOST = 1.15
PRIORITY_MULTICLASS_BOOST_RF = 1.1
PRIORITY_HIGH_DIM_LINEAR_BOOST = 1.3
PRIORITY_HIGH_DIM_GBM_PENALTY = 0.85
PRIORITY_LARGE_DATA_LGBM_CAT_BOOST = 1.1
PRIORITY_LARGE_DATA_RF_PENALTY = 0.8
PRIORITY_SMALL_DATA_LINEAR_BOOST = 1.15
PRIORITY_SMALL_DATA_RF_BOOST = 1.1
PRIORITY_SPARSE_LINEAR_BOOST = 1.2 # Applied to SGD_LINEAR
PRIORITY_SPARSE_XGB_BOOST = 1.1 # Slight boost for XGB on sparse
PRIORITY_HIGH_CATEGORICAL_CATLGBM_BOOST = 1.15

# --- Time Budget & Calibration Constants ---
TIME_BUDGET_LOW_NORM = 0.0005 # Normalized time threshold for 'low' budget
TIME_BUDGET_MEDIUM_NORM = 0.005 # Normalized time threshold for 'medium' budget
FALLBACK_BASE_TIME_UNIT = 0.5  # Fallback if calibration fails or is skipped
CALIBRATION_MODEL = "LGB" # Model used for benchmarking (LGB recommended for speed)
CALIBRATION_SAMPLE_SIZE = 15000 # Max samples for calibration run
CALIBRATION_N_ESTIMATORS = 50
CALIBRATION_MAX_DEPTH = 6
CALIBRATION_LEARNING_RATE = 0.1
MIN_CALIBRATION_TIME = 0.01 # Minimum measured time to avoid division by zero/instability
BASE_TIME_UNIT_BOUNDS = (0.001, 10.0) # Reasonable bounds for the calibrated unit

# --- Parameter Meta-Definitions ---
PARAM_GROUP_COMPLEXITY = 'complexity' # Scales significantly with budget/data size (depth, estimators, leaves)
PARAM_GROUP_REGULARIZATION = 'regularization' # L1/L2, gamma, min_child_weight etc.
PARAM_GROUP_SAMPLING = 'sampling' # Subsample, colsample, feature_fraction etc.
PARAM_GROUP_LEARNING = 'learning' # Learning rate, eta etc.
PARAM_GROUP_STRUCTURAL = 'structural' # Fixed params, architecture choices (e.g., boosting_type)
PARAM_GROUP_TASK = 'task' # Objectives, loss functions, eval_metrics

# ==============================================================================
# --- Base Model Configuration Templates ---
# ==============================================================================
# Structure: param_name: (value_range_or_list_or_fixed, type, group, sensitivity=optional)
# type: 'int', 'float', 'float_log', 'cat', 'fixed'
# sensitivity: float (0 to ~2), hint for how much budget/scaling affects this param's range/center

BASE_MODEL_CONFIG_TEMPLATES = {
    "XGB": {
        "learning_rate": ((0.01, 0.3), "float_log", PARAM_GROUP_LEARNING, 0.8),
        "n_estimators": ((100, 1000), "int", PARAM_GROUP_COMPLEXITY, 1.0), # Range adjusted later
        "max_depth": ((3, 10), "int", PARAM_GROUP_COMPLEXITY, 1.0),
        "min_child_weight": ((0.1, 20), "float_log", PARAM_GROUP_REGULARIZATION, 0.5),
        "subsample": ((BASE_SUBSAMPLE_LOW, BASE_SUBSAMPLE_HIGH), "float", PARAM_GROUP_SAMPLING, 0.7),
        "colsample_bytree": ((BASE_FEATURE_FRACTION_LOW, BASE_FEATURE_FRACTION_HIGH), "float", PARAM_GROUP_SAMPLING, 0.7),
        "colsample_bylevel": ((BASE_FEATURE_FRACTION_LOW, BASE_FEATURE_FRACTION_HIGH), "float", PARAM_GROUP_SAMPLING, 0.7),
        "gamma": ((1e-8, 5.0), "float_log", PARAM_GROUP_REGULARIZATION, 0.8),
        "reg_alpha": ((1e-8, 10.0), "float_log", PARAM_GROUP_REGULARIZATION, 1.0), # L1
        "reg_lambda": ((1e-8, 10.0), "float_log", PARAM_GROUP_REGULARIZATION, 1.0), # L2
        "scale_pos_weight": (1.0, "fixed", PARAM_GROUP_TASK), # Adjusted later for imbalance
        "objective": (None, "fixed", PARAM_GROUP_TASK), # Set based on task
        "eval_metric": (None, "fixed", PARAM_GROUP_TASK), # Set based on task
        "booster": (["gbtree", "dart"], "cat", PARAM_GROUP_STRUCTURAL),
        "verbosity": (0, "fixed", PARAM_GROUP_STRUCTURAL),
        "n_jobs": (-1, "fixed", PARAM_GROUP_STRUCTURAL),
        "num_class": (None, "fixed", PARAM_GROUP_TASK), # Placeholder for multiclass
        "_priority": 120
    },
    "LGB": {
        "learning_rate": ((0.01, 0.3), "float_log", PARAM_GROUP_LEARNING, 0.8),
        "n_estimators": ((100, 1000), "int", PARAM_GROUP_COMPLEXITY, 1.0), # Range adjusted later
        "max_depth": ((-1, 12), "int", PARAM_GROUP_COMPLEXITY, 1.0), # -1 means no limit
        "num_leaves": ((20, 150), "int", PARAM_GROUP_COMPLEXITY, 1.2), # Sensitive
        "min_child_samples": ((5, 50), "int", PARAM_GROUP_REGULARIZATION, 0.5),
        "subsample": ((BASE_SUBSAMPLE_LOW, BASE_SUBSAMPLE_HIGH), "float", PARAM_GROUP_SAMPLING, 0.7), # Alias: bagging_fraction
        "colsample_bytree": ((BASE_FEATURE_FRACTION_LOW, BASE_FEATURE_FRACTION_HIGH), "float", PARAM_GROUP_SAMPLING, 0.7), # Alias: feature_fraction
        "reg_alpha": ((1e-8, 10.0), "float_log", PARAM_GROUP_REGULARIZATION, 1.0), # L1
        "reg_lambda": ((1e-8, 10.0), "float_log", PARAM_GROUP_REGULARIZATION, 1.0), # L2
        "scale_pos_weight": (1.0, "fixed", PARAM_GROUP_TASK), # Adjusted later for imbalance
        "is_unbalanced": (False, "fixed", PARAM_GROUP_TASK), # Alternative for imbalance
        "boosting_type": (["gbdt", "dart"], "cat", PARAM_GROUP_STRUCTURAL),
        "objective": (None, "fixed", PARAM_GROUP_TASK),
        "metric": (None, "fixed", PARAM_GROUP_TASK), # LGB uses 'metric'
        "max_cat_to_onehot": (4, "fixed", PARAM_GROUP_STRUCTURAL), # Consider adjusting based on cat_ratio?
        "verbosity": (-1, "fixed", PARAM_GROUP_STRUCTURAL),
        "n_jobs": (-1, "fixed", PARAM_GROUP_STRUCTURAL),
        "num_class": (None, "fixed", PARAM_GROUP_TASK), # Placeholder for multiclass
        "_priority": 125 # Slightly higher base prio for LGB often
    },
    "CAT": { # Requires catboost library
        "learning_rate": ((0.01, 0.2), "float_log", PARAM_GROUP_LEARNING, 0.8), # Often benefits from lower LR max
        "iterations": ((100, 1000), "int", PARAM_GROUP_COMPLEXITY, 1.0), # Alias n_estimators
        "depth": ((4, 10), "int", PARAM_GROUP_COMPLEXITY, 1.0), # Default max depth is lower
        "l2_leaf_reg": ((1.0, 10.0), "float_log", PARAM_GROUP_REGULARIZATION, 1.0), # Main L2 reg
        "random_strength": ((1e-8, 10.0), "float_log", PARAM_GROUP_REGULARIZATION, 0.5), # Noise
        "bagging_temperature": ((0.0, 1.0), "float", PARAM_GROUP_SAMPLING, 0.7), # Controls bagging exploration
        "border_count": ((32, 254), "int", PARAM_GROUP_STRUCTURAL, 0.5), # Affects discretization
        "scale_pos_weight": (1.0, "fixed", PARAM_GROUP_TASK), # Adjusted later
        "loss_function": (None, "fixed", PARAM_GROUP_TASK),
        "verbose": (False, "fixed", PARAM_GROUP_STRUCTURAL),
        "thread_count": (-1, "fixed", PARAM_GROUP_STRUCTURAL),
        "_priority": 115
    },
    "RF": { # Requires scikit-learn
        "n_estimators": ((100, 800), "int", PARAM_GROUP_COMPLEXITY, 0.8), # Often fewer estimators
        "max_depth": ((5, 30), "int", PARAM_GROUP_COMPLEXITY, 0.9),
        "min_samples_split": ((2, 20), "int", PARAM_GROUP_REGULARIZATION, 0.5),
        "min_samples_leaf": ((1, 20), "int", PARAM_GROUP_REGULARIZATION, 0.5),
        "max_features": ((0.1, 0.9), "float", PARAM_GROUP_SAMPLING, 0.6), # Often 'sqrt'/'log2' used
        "bootstrap": ([True, False], "cat", PARAM_GROUP_SAMPLING),
        "criterion": ([], "cat", PARAM_GROUP_TASK), # Placeholder
        "class_weight": ([], "cat", PARAM_GROUP_TASK), # Placeholder
        "n_jobs": (-1, "fixed", PARAM_GROUP_STRUCTURAL),
        "_priority": 90
    },
    "SGD_LINEAR": { # Requires scikit-learn
        "loss": ([], "cat", PARAM_GROUP_TASK), # Placeholder
        "penalty": (["l2", "l1", "elasticnet"], "cat", PARAM_GROUP_REGULARIZATION), # Adjusted for sparsity
        "alpha": ((1e-7, 1e-1), "float_log", PARAM_GROUP_REGULARIZATION, 1.0), # Reg strength
        "l1_ratio": ((0.0, 1.0), "float", PARAM_GROUP_REGULARIZATION, 0.8), # For elasticnet
        "fit_intercept": ([True, False], "cat", PARAM_GROUP_STRUCTURAL),
        "learning_rate": (["optimal", "invscaling", "adaptive"], "cat", PARAM_GROUP_LEARNING),
        "eta0": ((1e-5, 1e-1), "float_log", PARAM_GROUP_LEARNING, 0.7), # For invscaling/adaptive
        "power_t": ((0.1, 0.9), "float", PARAM_GROUP_LEARNING, 0.5), # For invscaling
        "max_iter": ((500, 3000), "int", PARAM_GROUP_COMPLEXITY, 1.1), # Scaled later, adjusted base cap
        "tol": ((1e-6, 1e-3), "float_log", PARAM_GROUP_STRUCTURAL),
        "early_stopping": (True, "fixed", PARAM_GROUP_STRUCTURAL),
        "validation_fraction": (0.1, "fixed", PARAM_GROUP_STRUCTURAL),
        "n_iter_no_change": (10, "fixed", PARAM_GROUP_STRUCTURAL),
        "_priority": 70 # Lower base priority, boosted if high-dim/sparse
    }
}

# ==============================================================================
# --- Helper Functions ---
# ==============================================================================

def _is_range_param(param_info: Dict[str, Any]) -> bool:
    """Check if parameter info dict defines a range."""
    return param_info.get('type') in ['int', 'float', 'float_log'] and 'low' in param_info and 'high' in param_info

def _get_param_info(param_config: Union[Tuple, List, Any]) -> Dict[str, Any]:
    """Extract structured info from parameter config tuple into a dictionary."""
    if not isinstance(param_config, tuple) or len(param_config) < 3:
        # Handle fixed values or simple lists for categorical (less structured format)
        p_type = 'cat' if isinstance(param_config, list) else 'fixed'
        return {'value': param_config, 'type': p_type, 'group': PARAM_GROUP_STRUCTURAL, 'sensitivity': 0}

    value, p_type, group = param_config[0], param_config[1], param_config[2]
    sensitivity = param_config[3] if len(param_config) > 3 else (1.0 if group == PARAM_GROUP_COMPLEXITY else 0.7)

    info = {'type': p_type, 'group': group, 'sensitivity': sensitivity}

    if p_type in ['int', 'float', 'float_log']:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            info['low'] = value[0]
            info['high'] = value[1]
        elif value is None and p_type == 'fixed': # Allow fixed None, e.g. num_class default
             info['value'] = None
             info['type'] = 'fixed' # Correct type if initially set as range but value is None
        else:
            raise ValueError(f"Range parameter requires a tuple/list of length 2 for value, got {value}")
    elif p_type == 'cat':
        if isinstance(value, list):
            info['values'] = value
        else:
            raise ValueError(f"Categorical parameter requires a list of values, got {value}")
    elif p_type == 'fixed':
        info['value'] = value
    else:
        raise ValueError(f"Unsupported parameter type '{p_type}' in template.")

    return info

def _apply_scaling(value: float, factor: float, scale_type: str = 'multiplicative', is_int: bool = False) -> float:
    """Apply scaling factor to a value, handling None and integer rounding."""
    if value is None:
        return None
    try:
        numeric_value = float(value)
        if scale_type == 'multiplicative':
            scaled = numeric_value * factor
        elif scale_type == 'additive':
            scaled = numeric_value + factor
        else:
            scaled = numeric_value

        return int(round(scaled)) if is_int else scaled
    except (ValueError, TypeError):
        warnings.warn(f"Could not apply scaling to non-numeric value {value}. Returning original.")
        return value

def _clamp(value: Union[int, float], min_val: Optional[Union[int, float]], max_val: Optional[Union[int, float]]) -> Union[int, float]:
    """Clamp value within optional min/max bounds."""
    if min_val is not None:
        value = max(min_val, value)
    if max_val is not None:
        value = min(max_val, value)
    return value

# ==============================================================================
# --- Data Characteristics Detection (Refactored) ---
# ==============================================================================

def _detect_basic_info(X) -> Dict[str, Any]:
    """Detect basic shape, sparsity."""
    info = {'n_samples': 0, 'n_features': 0, 'is_sparse': False, 'sparsity_ratio': 0.0}
    try:
        info['n_samples'] = X.shape[0]
        if info['n_samples'] == 0:
            warnings.warn("Input data X has 0 samples.")
            return info
        info['n_features'] = X.shape[1] if len(X.shape) > 1 else 1
        if info['n_features'] == 0:
            warnings.warn("Input data X has 0 features.")
            info['n_samples'] = X.shape[0] # Keep n_samples if only features=0
            return info

        info['is_sparse'] = issparse(X)
        num_elements = info['n_samples'] * info['n_features']
        if num_elements == 0: return info

        if info['is_sparse']:
            info['sparsity_ratio'] = 1.0 - (X.nnz / num_elements)
        elif isinstance(X, np.ndarray):
            info['sparsity_ratio'] = np.mean(X == 0)
        elif isinstance(X, pd.DataFrame):
            # Approx sparsity for mixed-type DataFrames (considers 0 as sparse)
            try:
                 info['sparsity_ratio'] = (X == 0).sum().sum() / num_elements
            except Exception: # Handle potential type errors
                 info['sparsity_ratio'] = 0.0 # Fallback
        else: # Other dense types
            try:
                 info['sparsity_ratio'] = np.mean(np.array(X) == 0) # Attempt conversion
            except Exception:
                 info['sparsity_ratio'] = 0.0
    except Exception as e:
        warnings.warn(f"Error detecting basic info: {e}")
    return info

def _detect_feature_types(X, basic_info: Dict, categorical_features_hint: Optional[List]) -> Dict[str, Any]:
    """Detect numeric and categorical features."""
    n_samples = basic_info['n_samples']
    n_features = basic_info['n_features']
    results = {'numeric_features': [], 'categorical_features': [], 'cat_feature_ratio': 0.0, 'is_pandas': False, 'X_analyzed': None}
    if n_features == 0: return results

    is_pandas = isinstance(X, pd.DataFrame)
    X_analyzed = X if is_pandas else None
    all_cols = list(X.columns) if is_pandas else list(range(n_features))

    numeric_cols = []
    categorical_cols = []

    if is_pandas:
        results['is_pandas'] = True
        results['X_analyzed'] = X
        numeric_cols_pd = X.select_dtypes(include=np.number).columns.tolist()

        if categorical_features_hint is None:
            # Auto-detect
            cat_cols_obj = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            cat_cols_int = []
            unique_thresh = min(max(25, int(np.sqrt(n_samples) * 0.5)), CATEGORICAL_UNIQUE_COUNT_THRESHOLD)
            ratio_thresh = CATEGORICAL_UNIQUE_RATIO_THRESHOLD
            for col in X.select_dtypes(include=['int', 'uint']).columns:
                try:
                    n_unique = X[col].nunique()
                    unique_ratio = n_unique / n_samples if n_samples > 0 else 0
                    # Heuristic: Low unique count OR low unique ratio and likely not sequential ID
                    if n_unique <= unique_thresh or (unique_ratio < ratio_thresh and n_unique < n_samples * 0.8): # Avoid treating IDs as categorical
                        cat_cols_int.append(col)
                except Exception:
                    warnings.warn(f"Could not analyze integer column '{col}' for categorical detection.")
            categorical_cols = list(set(cat_cols_obj + cat_cols_int))
            numeric_cols = [c for c in numeric_cols_pd if c not in categorical_cols]
        else:
            # Use hint
            categorical_cols = [c for c in categorical_features_hint if c in all_cols]
            if len(categorical_cols) != len(categorical_features_hint):
                warnings.warn("Some provided categorical features were not found in DataFrame columns.")
            numeric_cols = [c for c in all_cols if c not in categorical_cols]

    elif not basic_info['is_sparse']: # Numpy array
        # Try converting smaller arrays to pandas for better type detection
        can_convert_to_pd = n_samples * n_features < 10_000_000
        if can_convert_to_pd:
            try:
                col_names = [f'f_{i}' for i in range(n_features)]
                X_pd = pd.DataFrame(X, columns=col_names)
                # Recursively call with the converted DataFrame
                pd_results = _detect_feature_types(X_pd, basic_info, categorical_features_hint)
                # Convert back to indices if original hint was indices
                if categorical_features_hint and all(isinstance(i, int) for i in categorical_features_hint):
                    name_to_idx = {name: i for i, name in enumerate(col_names)}
                    pd_results['numeric_features'] = [name_to_idx[name] for name in pd_results['numeric_features'] if name in name_to_idx]
                    pd_results['categorical_features'] = [name_to_idx[name] for name in pd_results['categorical_features'] if name in name_to_idx]
                return pd_results
            except Exception as e:
                warnings.warn(f"Could not convert NumPy to DataFrame for analysis: {e}. Proceeding with index-based logic.")
        # Fallback: Handle NumPy directly (mainly using hint)
        results['is_pandas'] = False
        results['X_analyzed'] = X
        valid_cat_indices = []
        if categorical_features_hint:
            valid_cat_indices = [i for i in categorical_features_hint if isinstance(i, int) and 0 <= i < n_features]
            if len(valid_cat_indices) != len(categorical_features_hint):
                warnings.warn("Some provided categorical feature indices were invalid for NumPy array.")
        categorical_cols = valid_cat_indices
        numeric_cols = [i for i in all_cols if i not in categorical_cols]

    else: # Sparse matrix
        results['is_pandas'] = False
        results['X_analyzed'] = X
        valid_cat_indices = []
        if categorical_features_hint:
            valid_cat_indices = [i for i in categorical_features_hint if isinstance(i, int) and 0 <= i < n_features]
            if len(valid_cat_indices) != len(categorical_features_hint):
                 warnings.warn("Some provided categorical feature indices were invalid for sparse matrix.")
        categorical_cols = valid_cat_indices
        numeric_cols = [i for i in all_cols if i not in categorical_cols]

    results['numeric_features'] = numeric_cols
    results['categorical_features'] = categorical_cols
    results['cat_feature_ratio'] = len(categorical_cols) / n_features if n_features > 0 else 0.0
    return results

def _analyze_target(y: Optional[Union[pd.Series, np.ndarray]], n_samples: int) -> Dict[str, Any]:
    """Analyze target variable for task type, classes, balance, skewness."""
    results = {
        'task_type': 'classification', # Default guess
        'n_classes': 2,
        'class_balance': 1.0,
        'target_skewness': 0.0, # For regression
    }
    if y is None:
        warnings.warn("Target variable y is None. Defaulting to binary classification task.")
        return results

    try:
        y_series = y if isinstance(y, pd.Series) else pd.Series(y)
        y_nonan = y_series.dropna()

        if y_nonan.empty:
            warnings.warn("Target variable contains only NaN values after dropna(). Defaulting task.")
            return results

        n_unique = y_nonan.nunique()

        # --- Classification Check ---
        # Conditions: object/string/bool type, OR low-cardinality integer type
        is_object_or_string = pd.api.types.is_object_dtype(y_nonan) or pd.api.types.is_string_dtype(y_nonan)
        is_bool = pd.api.types.is_bool_dtype(y_nonan)
        is_integer_like = False
        if pd.api.types.is_numeric_dtype(y_nonan) and not pd.api.types.is_float_dtype(y_nonan):
            try:
                 # Check if all numbers are integers
                 if np.all(np.equal(np.mod(y_nonan, 1), 0)):
                      is_integer_like = True
            except TypeError: # Handle potential non-numeric data that passed initial checks
                 is_integer_like = False

        low_cardinality_threshold = min(max(30, int(n_samples * 0.05)), 100)
        is_low_cardinality_int = is_integer_like and n_unique <= low_cardinality_threshold

        if is_object_or_string or is_bool or is_low_cardinality_int:
            results['task_type'] = 'classification'
            results['n_classes'] = n_unique
            if n_unique > 1:
                counts = y_nonan.value_counts()
                if not counts.empty:
                    min_count, max_count = counts.min(), counts.max()
                    results['class_balance'] = max_count / min_count if min_count > 0 else np.inf
            else: # Only one class found
                results['class_balance'] = 1.0
        else: # --- Regression Check ---
            results['task_type'] = 'regression'
            results['n_classes'] = 0 # Using 0 for regression
            results['class_balance'] = 1.0 # Not applicable
            if pd.api.types.is_numeric_dtype(y_nonan):
                try:
                    results['target_skewness'] = abs(stats.skew(y_nonan.astype(float)))
                except Exception as skew_e:
                    warnings.warn(f"Could not calculate target skewness: {skew_e}")
            else:
                warnings.warn("Regression target seems non-numeric, skewness set to 0.")

    except Exception as e:
        warnings.warn(f"Target analysis failed: {e}. Defaulting to binary classification.")
        results = { 'task_type': 'classification', 'n_classes': 2, 'class_balance': 1.0, 'target_skewness': 0.0 }

    return results

def _calculate_correlations(X_sample_dense: Union[pd.DataFrame, np.ndarray],
                             numeric_features: List[Union[str, int]],
                             is_pandas: bool) -> Dict[str, float]:
    """Calculate mean correlation and high correlation ratio on numeric features."""
    results = {'feature_correlation_mean': 0.0, 'feature_correlation_high_ratio': 0.0}
    if X_sample_dense is None or len(numeric_features) < 2:
        return results

    try:
        if is_pandas:
            # Ensure numeric_features are valid column names present in the sample
            valid_numeric_features = [f for f in numeric_features if f in X_sample_dense.columns]
            if len(valid_numeric_features) < 2: return results
            # Select only numeric columns among the valid ones
            numeric_subset = X_sample_dense[valid_numeric_features].select_dtypes(include=np.number)
            if numeric_subset.shape[1] < 2: return results
            corr_matrix = numeric_subset.corr(method='pearson').abs()
        else: # Numpy array
            # Ensure numeric_features are valid indices
            valid_numeric_features = [i for i in numeric_features if isinstance(i, int) and 0 <= i < X_sample_dense.shape[1]]
            if len(valid_numeric_features) < 2: return results
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning) # Ignore warnings from std dev 0 columns etc.
                numeric_slice = X_sample_dense[:, valid_numeric_features].astype(float) # Convert to float
                if np.all(np.isnan(numeric_slice)): return results # Handle all-NaN slice
                # Basic imputation for correlation calc stability
                col_means = np.nanmean(numeric_slice, axis=0)
                inds = np.where(np.isnan(numeric_slice))
                if inds[0].size > 0:
                    numeric_slice[inds] = np.take(np.nan_to_num(col_means), inds[1]) # Use 0 if mean is NaN
                # Check for constant columns after imputation
                if np.any(np.nanstd(numeric_slice, axis=0) < 1e-9):
                    non_constant_mask = np.nanstd(numeric_slice, axis=0) >= 1e-9
                    if np.sum(non_constant_mask) < 2: return results # Not enough varying columns
                    numeric_slice = numeric_slice[:, non_constant_mask]

                corr_matrix = pd.DataFrame(numeric_slice).corr(method='pearson').abs() # Use pandas for easy handling

        if corr_matrix is not None and not corr_matrix.empty:
            np.fill_diagonal(corr_matrix.values, np.nan) # Exclude self-correlation
            valid_corrs = corr_matrix.unstack().dropna()
            if not valid_corrs.empty:
                results['feature_correlation_mean'] = valid_corrs.mean()
                results['feature_correlation_high_ratio'] = (valid_corrs > HIGH_CORRELATION_THRESHOLD).mean()

    except Exception as e:
        warnings.warn(f"Correlation analysis failed: {e}")

    return results

def _calculate_missing_ratio(X_analyzed: Union[pd.DataFrame, np.ndarray], basic_info: Dict) -> float:
    """Calculate the ratio of missing values (NaNs)."""
    if X_analyzed is None or basic_info['is_sparse']:
        return 0.0 # Assume no missing for sparse or unanalyzed

    try:
        total_cells = basic_info['n_samples'] * basic_info['n_features']
        if total_cells == 0: return 0.0

        if isinstance(X_analyzed, pd.DataFrame):
            missing_total = X_analyzed.isna().sum().sum()
        elif isinstance(X_analyzed, np.ndarray):
            missing_total = np.isnan(X_analyzed).sum()
        else:
            return 0.0 # Unknown type

        return missing_total / total_cells if total_cells > 0 else 0.0
    except Exception as e:
        warnings.warn(f"Missing value analysis failed: {e}")
        return 0.0

def _estimate_feature_importance_entropy(X_sample_dense: Union[pd.DataFrame, np.ndarray],
                                         y_sample: np.ndarray,
                                         numeric_features: List[Union[str, int]],
                                         task_type: str,
                                         is_pandas: bool) -> float:
    """Estimate feature importance concentration using Mutual Information entropy."""
    default_entropy = 0.5 # Default if calculation fails or not possible
    if X_sample_dense is None or y_sample is None or len(numeric_features) < 1:
        return default_entropy

    try:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    except ImportError:
        warnings.warn("Scikit-learn not installed. Skipping Mutual Information calculation.")
        return default_entropy

    try:
        X_numeric_sample = None
        if is_pandas:
            valid_numeric_features = [f for f in numeric_features if f in X_sample_dense.columns]
            if not valid_numeric_features: return default_entropy
            # Select only numeric columns among valid ones
            numeric_subset_df = X_sample_dense[valid_numeric_features].select_dtypes(include=np.number)
            if numeric_subset_df.empty: return default_entropy
            # Simple median imputation for MI stability
            X_numeric_sample = numeric_subset_df.fillna(numeric_subset_df.median()).to_numpy()
        else: # Numpy array
            valid_numeric_indices = [i for i in numeric_features if isinstance(i, int) and 0 <= i < X_sample_dense.shape[1]]
            if not valid_numeric_indices: return default_entropy
            X_numeric_slice = X_sample_dense[:, valid_numeric_indices].astype(float)
            col_medians = np.nanmedian(X_numeric_slice, axis=0)
            inds = np.where(np.isnan(X_numeric_slice))
            if inds[0].size > 0:
                X_numeric_slice[inds] = np.take(np.nan_to_num(col_medians), inds[1]) # Use 0 if median is NaN
            X_numeric_sample = X_numeric_slice

        if X_numeric_sample is None or X_numeric_sample.shape[1] == 0: return default_entropy

        # Prepare target for MI
        y_for_mi = y_sample
        discrete_target = False
        if task_type == 'classification':
            try:
                 y_for_mi, _ = pd.factorize(y_sample)
                 discrete_target = True
            except Exception as e:
                 warnings.warn(f"Could not factorize target for MI: {e}. Skipping MI.")
                 return default_entropy
        elif pd.api.types.is_object_dtype(y_sample) or pd.api.types.is_string_dtype(y_sample):
            warnings.warn("MI requires numeric or factorized target for regression. Skipping MI.")
            return default_entropy

        # Ensure target is finite
        if np.issubdtype(y_for_mi.dtype, np.number):
            finite_mask = np.isfinite(y_for_mi.astype(float))
        else: # Assume factorized or other non-numeric are ok
            finite_mask = np.ones(y_for_mi.shape[0], dtype=bool)

        if np.sum(finite_mask) < 10: # Need minimum samples
             warnings.warn(f"Too few finite target samples ({np.sum(finite_mask)}) for MI. Skipping.")
             return default_entropy

        y_finite = y_for_mi[finite_mask]
        X_finite = X_numeric_sample[finite_mask, :]

        if X_finite.shape[0] != y_finite.shape[0] or X_finite.shape[0] == 0:
             warnings.warn("Shape mismatch or zero samples after finite filtering for MI. Skipping.")
             return default_entropy

        # Check for zero variance features
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            variances = np.nanvar(X_finite, axis=0)
        if np.all(variances < 1e-9):
            warnings.warn("All numeric features in sample have zero variance. MI entropy is 0.")
            return 0.0 # If no variance, importance is concentrated (on nothing useful)

        # Calculate MI
        mi_func = mutual_info_classif if task_type == 'classification' else mutual_info_regression
        n_neighbors = 3 if X_finite.shape[0] < 20 else 5 # Adjust neighbors for small samples
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # Ignore potential sklearn/numpy warnings during MI calc
            mi_scores = mi_func(X_finite, y_finite, discrete_features='auto',
                                random_state=42, n_neighbors=n_neighbors,
                                n_jobs = -1) 

        mi_scores = np.maximum(0, mi_scores) # Ensure non-negative
        mi_sum = np.sum(mi_scores)

        if mi_sum < 1e-9:
             return 0.0 # If total MI is negligible, consider entropy 0

        mi_norm = mi_scores / mi_sum
        entropy = stats.entropy(mi_norm, base=2)
        max_entropy = np.log2(len(mi_norm)) if len(mi_norm) > 1 else 1 # Use 1 if only one feature
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        return normalized_entropy

    except Exception as e:
        warnings.warn(f"Mutual Information calculation failed: {e}")
        return default_entropy


def _calculate_data_complexity_score(chars: Dict[str, Any]) -> float:
    """Calculate a heuristic data complexity score (1-10)."""
    score = 5.0 # Base score

    # Size impact (log scale)
    score += np.clip(np.log10(max(100, chars['n_samples']) / 10000), -2, 2) if chars['n_samples'] > 0 else -2
    score += np.clip(np.log10(max(10, chars['n_features']) / 50), -2, 2) if chars['n_features'] > 0 else -2

    # Task impact
    if chars['task_type'] == 'classification':
        balance = chars['class_balance']
        score += min(2.5, np.log2(max(1, balance)) * 0.6) if balance != np.inf and balance > 1 else 0
        score += min(2.0, (chars['n_classes'] - 2) * 0.25) if chars['n_classes'] > 2 else 0
    else: # Regression
        skewness = chars['target_skewness']
        score += min(1.5, np.log1p(max(0, skewness - 1)) * 0.7) if skewness > 1 else 0

    # Feature characteristics impact
    score += chars['cat_feature_ratio'] * 1.5
    score += chars['missing_value_ratio'] * 2.0
    score += chars['feature_correlation_high_ratio'] * 2.0
    score += chars['sparsity_ratio'] * 0.5

    # Importance distribution impact (lower entropy -> higher complexity assumed)
    score += (1.0 - chars['feature_importance_entropy']) * 1.0

    # Clamp score to [1, 10]
    return max(1.0, min(10.0, score))


def detect_dataset_characteristics(
    X, y=None,
    categorical_features: Optional[List[Union[int, str]]] = None,
    max_samples_for_analysis: int = 20000
) -> Dict[str, Any]:
    """
    Detect dataset characteristics relevant for hyperparameter optimization.
    """
    results = {
        'n_samples': 0,
        'n_features': 0,
        'is_sparse': False,
        'sparsity_ratio': 0.0,
        'categorical_features': [],
        'numeric_features': [],
        'cat_feature_ratio': 0.0,
        'feature_correlation_mean': 0.0,
        'feature_correlation_high_ratio': 0.0,
        'missing_value_ratio': 0.0,
        'task_type': 'classification', # Default guess
        'n_classes': 2,
        'class_balance': 1.0,
        'target_skewness': 0.0, # For regression
        'feature_importance_entropy': 0.5, # Default normalized MI entropy
        'data_complexity_score': 5.0, # Simple overall score 1-10
    }

    # 1. Basic Info
    basic_info = _detect_basic_info(X)
    results.update(basic_info)
    if results['n_samples'] == 0 or results['n_features'] == 0:
        return results # Cannot proceed

    # 2. Feature Type Detection
    feature_info = _detect_feature_types(X, basic_info, categorical_features)
    results.update({
        'categorical_features': feature_info['categorical_features'],
        'numeric_features': feature_info['numeric_features'],
        'cat_feature_ratio': feature_info['cat_feature_ratio'],
    })
    is_pandas_analysis = feature_info['is_pandas']
    X_analyzed = feature_info['X_analyzed'] # May be original X or converted DF

    # 3. Target Analysis
    target_info = _analyze_target(y, results['n_samples'])
    results.update(target_info)

    # 4. Sample Data for Expensive Analysis
    analysis_rows = min(results['n_samples'], max_samples_for_analysis)
    idx = np.arange(results['n_samples']) # Default to all rows
    if analysis_rows < results['n_samples']:
        idx = np.random.choice(results['n_samples'], size=analysis_rows, replace=False)

    X_sample = None
    y_sample = None
    X_sample_dense = None # Only dense version for correlation/MI

    try:
        if is_pandas_analysis:
            X_sample = X_analyzed.iloc[idx]
            X_sample_dense = X_sample # Already dense
        elif not results['is_sparse']: # Numpy
            X_sample = X_analyzed[idx,:]
            X_sample_dense = X_sample
        else: # Sparse
            X_sample = X_analyzed[idx,:]
            # Cannot directly use sparse for correlation/MI analysis below
            warnings.warn("Correlation and MI analysis skipped for sparse data format.")

        if y is not None:
             y_np = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)
             y_sample = y_np[idx]

    except Exception as e:
         warnings.warn(f"Failed to sample data/target for analysis: {e}")
         X_sample = None # Ensure analysis using sample is skipped

    # 5. Correlation Analysis (on dense numeric sample)
    if X_sample_dense is not None:
         correlation_info = _calculate_correlations(X_sample_dense, results['numeric_features'], is_pandas_analysis)
         results.update(correlation_info)

    # 6. Missing Value Analysis (on potentially converted X_analyzed)
    results['missing_value_ratio'] = _calculate_missing_ratio(X_analyzed, basic_info)

    # 7. Feature Importance Entropy (on dense numeric sample)
    if X_sample_dense is not None and y_sample is not None:
        results['feature_importance_entropy'] = _estimate_feature_importance_entropy(
            X_sample_dense, y_sample, results['numeric_features'], results['task_type'], is_pandas_analysis
        )

    # 8. Complexity Score
    results['data_complexity_score'] = _calculate_data_complexity_score(results)

    # Cleanup potentially large intermediate objects if created
    del X_analyzed, X_sample, y_sample, X_sample_dense

    return results


# ==============================================================================
# --- Search Space Generation Logic ---
# ==============================================================================

def _apply_initial_data_scaling(config: Dict, data_char: Dict, model_name: str) -> Dict:
    """Apply scaling based on n_samples, n_features before budget adjustments."""
    scaled_config = config.copy()
    n_samples = data_char['n_samples']
    n_features = data_char['n_features']
    if n_samples <= 0 or n_features <= 0: return scaled_config # Avoid division by zero

    is_high_dim = n_features > HIGH_DIM_THRESHOLD
    is_very_high_dim = n_features > VERY_HIGH_DIM_THRESHOLD

    # Logarithmic scaling factors based on dataset size relative to baseline sizes
    log_sample_factor = math.log10(n_samples / 10000.0) if n_samples > 100 else math.log10(100 / 10000.0)
    log_feature_factor = math.log10(n_features / 50.0) if n_features > 10 else math.log10(10 / 50.0)

    for param, p_info in config.items():
        if not isinstance(p_info, dict) or p_info.get('type') not in ['int', 'float', 'float_log']:
            continue
        if 'low' not in p_info or 'high' not in p_info or p_info['low'] is None or p_info['high'] is None:
             # Skip fixed params or params without range, except LGB max_depth=-1 special case
             if not (param == 'max_depth' and model_name == "LGB" and p_info.get('low') == -1):
                  continue

        is_int = p_info['type'] == 'int'

        # Get current range
        low, high = p_info['low'], p_info['high']

        # --- Scale parameters based on group and data characteristics ---

        if p_info['group'] == PARAM_GROUP_COMPLEXITY:
            if param in ['n_estimators', 'iterations', 'max_iter']:
                # Scale range based on samples (more effect on high end)
                scale_high = 1.0 + np.clip(log_sample_factor, -0.5, 1.2)
                scale_low = 1.0 + np.clip(log_sample_factor * 0.5, -0.3, 0.6)
                high = _apply_scaling(high, scale_high, is_int=is_int)
                low = _apply_scaling(low, scale_low, is_int=is_int)
            elif param in ['max_depth', 'depth']:
                # Scale high end based on samples, less aggressively
                scale = 1.0 + np.clip(log_sample_factor * 0.5, -0.4, 0.6)
                if low != -1: # Don't scale low end if it's not -1
                     high = _apply_scaling(high, scale, is_int=is_int)
                # Ensure low is reasonable if not -1
                if low != -1: low = max(PARAM_BOUNDS.get(param, (1, None))[0] or 1, low)
            elif param == 'num_leaves':
                 # Scale high end based on samples, more aggressively
                 scale = 1.0 + np.clip(log_sample_factor * 0.7, -0.4, 0.8)
                 high = _apply_scaling(high, scale, is_int=is_int)
                 low = max(PARAM_BOUNDS.get(param, (2, None))[0] or 2, low) # Ensure sensible minimum

        elif p_info['group'] == PARAM_GROUP_REGULARIZATION:
            if param in ["min_samples_split", "min_samples_leaf", "min_child_samples"]:
                 # Scale range based on samples
                 scale = 1.0 + np.clip(log_sample_factor * 0.8, -0.3, 1.2)
                 high = _apply_scaling(high, scale, is_int=is_int)
                 low = _apply_scaling(low, scale * 0.8, is_int=is_int)
            elif param == 'min_child_weight':
                 # Scale range based on samples
                 scale = 1.0 + np.clip(log_sample_factor * 0.5, -0.2, 0.8)
                 high = _apply_scaling(high, scale, is_int=is_int)
                 low = _apply_scaling(low, scale * 0.7, is_int=is_int)
            # Scale up regularization strength for high dimensions
            if param in ['reg_alpha', 'reg_lambda', 'lambda_l1', 'lambda_l2', 'alpha', 'l2_leaf_reg']:
                if is_very_high_dim:
                    high_dim_factor = 1.0 + 1.0 * np.log10(n_features / VERY_HIGH_DIM_THRESHOLD)
                    high = _apply_scaling(high, high_dim_factor)
                elif is_high_dim:
                     high_dim_factor = 1.0 + 0.7 * np.log10(n_features / HIGH_DIM_THRESHOLD)
                     high = _apply_scaling(high, high_dim_factor)

        elif p_info['group'] == PARAM_GROUP_SAMPLING:
             if param == 'max_features': # RF specific
                 sqrt_features = math.sqrt(n_features)
                 default_rf_max_feat = max(0.01, min(1.0, sqrt_features / n_features)) if n_features > 1 else 0.5
                 # Center range around default_rf_max_feat, influenced by initial template range
                 low = max(0.01, min(default_rf_max_feat * 0.7, low * 0.9))
                 high = min(1.0, max(default_rf_max_feat * 1.3 + 0.1, high * 1.1))
                 # Reduce high end slightly if many categoricals (sqrt less meaningful)
                 if data_char['cat_feature_ratio'] > HIGH_CATEGORICAL_RATIO:
                     high = max(low + 0.01, high * 0.9)
             else: # Other sampling fractions (subsample, colsample_*)
                  # Slightly reduce sampling fraction lower bound for high features
                  scale_low = 1.0 - np.clip(log_feature_factor * 0.2, 0.0, 0.15)
                  low = _apply_scaling(low, scale_low)

        # --- Clamping and Validation ---
        min_b, max_b = PARAM_BOUNDS.get(param, (None, None))

        # Handle LGB max_depth = -1 case separately for clamping
        if not (param == 'max_depth' and low == -1):
            if min_b is not None: low = max(min_b, low)
            if max_b is not None: high = min(max_b, high)
            # Ensure low < high after scaling
            if low >= high:
                 if is_int: high = low + 1
                 else: high = low + 1e-6 # Add small epsilon for floats
                 # Re-clamp high if it exceeded max_b
                 if max_b is not None: high = min(max_b, high)

        # Special clamping for log scale lower bound
        if p_info['type'] == "float_log" and low <= 0:
            low = max(PARAM_BOUNDS.get(param, (1e-9, None))[0] or 1e-9, 1e-9)
            if low >= high: high = low * 10 # Ensure high > low

        # Update config dict
        scaled_config[param]['low'] = low
        scaled_config[param]['high'] = high

    return scaled_config


def _apply_budget_and_continuous_scaling(config: Dict, budget: Literal['low', 'medium', 'high'], scaling_factor: float) -> Dict:
    """Adjust ranges based on discrete budget level and continuous scaling factor."""
    adjusted_config = config.copy()

    for param, p_info in config.items():
        if param.startswith('_'):
            continue
        if not _is_range_param(p_info):
            continue

        low, high = p_info['low'], p_info['high']
        is_int = p_info['type'] == 'int'
        sensitivity = p_info.get('sensitivity', 1.0)

        # Handle LGB max_depth = -1 case (only scale high end)
        is_lgb_depth_neg1 = (param == 'max_depth' and low == -1)
        if is_lgb_depth_neg1:
             # Apply scaling only to the high end
             if scaling_factor != 1.0:
                 effective_scale = max(0.1, scaling_factor ** (0.7 * sensitivity))
                 center_shift_scale = 0.2 if p_info['group'] == PARAM_GROUP_COMPLEXITY else 0.0
                 center_shift = (scaling_factor - 1.0) * center_shift_scale * sensitivity * (high - (PARAM_BOUNDS.get(param,(None,5))[1] or 5)) # Shift relative to a baseline
                 high = high * effective_scale + center_shift

             # Apply discrete budget shift to high end
             shift = (high - (PARAM_BOUNDS.get(param,(None,5))[1] or 5)) * BUDGET_CENTER_SHIFT_FACTOR * sensitivity
             if budget == "low": high -= shift
             elif budget == "high": high += shift
             # Clamp high end
             min_b, max_b = PARAM_BOUNDS.get(param, (None, None))
             high = _clamp(high, min_b, max_b)
             high = max(2, high) # Ensure high is at least 2
             if is_int: high = math.floor(high)
             p_info['high'] = high
             continue # Skip rest of the logic for this specific case

        # --- Standard Range Scaling Logic ---
        range_width = high - low
        if range_width <= (1e-9 if not is_int else 0):
            continue # Skip if range is effectively zero

        # 1. Apply continuous scaling_factor (affects range width and center)
        if scaling_factor != 1.0:
            # Scale range width (more sensitive params scale more)
            effective_scale = max(0.1, scaling_factor ** (0.7 * sensitivity))
            new_range_width = range_width * effective_scale

            # Shift center based on scaling factor and parameter group/sensitivity
            center = (low + high) / 2
            center_shift_scale = 0.0
            if p_info['group'] == PARAM_GROUP_COMPLEXITY: center_shift_scale = 0.2
            elif p_info['group'] == PARAM_GROUP_LEARNING: center_shift_scale = 0.1
            elif p_info['group'] == PARAM_GROUP_REGULARIZATION: center_shift_scale = -0.1 # Less regularization for more budget
            center_shift = (scaling_factor - 1.0) * center_shift_scale * sensitivity * range_width
            new_center = center + center_shift

            # Calculate new low/high based on scaled width and shifted center
            low = new_center - new_range_width / 2
            high = new_center + new_range_width / 2

        # 2. Apply discrete budget level adjustments (shifts center further)
        range_width = high - low # Recalculate width after continuous scaling
        shift = range_width * BUDGET_CENTER_SHIFT_FACTOR * sensitivity
        if budget == "low":
            if p_info['group'] in [PARAM_GROUP_COMPLEXITY, PARAM_GROUP_LEARNING]:
                # Shift towards simpler/faster values
                high -= shift
                low -= shift
            elif p_info['group'] == PARAM_GROUP_REGULARIZATION:
                # Shift towards more regularization
                low += shift
                high += shift
        elif budget == "high":
            if p_info['group'] in [PARAM_GROUP_COMPLEXITY, PARAM_GROUP_LEARNING]:
                # Shift towards more complex/slower values
                high += shift
                low += shift
            elif p_info['group'] == PARAM_GROUP_REGULARIZATION:
                # Shift towards less regularization
                low -= shift
                high -= shift

        # --- Final Bounds Enforcement & Validation ---
        min_b, max_b = PARAM_BOUNDS.get(param, (None, None))

        # Clamp low and high to bounds
        low = _clamp(low, min_b, max_b)
        high = _clamp(high, min_b, max_b)

        # Ensure low < high after all adjustments
        if low >= high:
            if is_int:
                # Try to adjust high first, then low
                if max_b is None or high < max_b: high = low + 1
                elif min_b is None or low > min_b: low = high - 1
                else: high = low # Cannot adjust if bounds are tight
            else:
                epsilon = max(1e-9, abs(low * 1e-6))
                if max_b is None or high < max_b: high = low + epsilon
                elif min_b is None or low > min_b: low = high - epsilon
                else: high = low # Cannot adjust

        # Final check for log scale lower bound
        if p_info['type'] == "float_log" and low <= 0:
             log_min_bound = PARAM_BOUNDS.get(param, (1e-9, None))[0] or 1e-9
             low = max(log_min_bound, 1e-9)
             if low >= high: high = low * 10 # Ensure high > low after fixing low

        # Final rounding for integers
        if is_int:
             low = math.ceil(low)
             high = math.floor(high)
             high = max(low, high) # Ensure high >= low after rounding

        # Update config dict
        adjusted_config[param]['low'] = low
        adjusted_config[param]['high'] = high

    return adjusted_config


def _apply_interdependencies(config: Dict, model_name: str) -> Dict:
    """
    Adjust parameters based on their relationships within a model.
    Handles:
    - LGBM max_depth vs num_leaves.
    - GBM learning_rate vs n_estimators (boosts estimators for low LR, caps for high LR range).
    """
    adjusted_config = config.copy()
    HIGH_LR_RANGE_LOW_THRESHOLD = 0.18 # If lr_low is above this, cap estimators
    HIGH_LR_ESTIMATOR_CAP = 1200 # The maximum estimators if LR range is high
    VERY_LOW_LR_THRESHOLD = 0.02 # LR below this threshold may boost n_estimators

    # 1. LGBM: max_depth and num_leaves
    if model_name == "LGB":
        depth_param_name = 'max_depth'
        leaves_param_name = 'num_leaves'

        if depth_param_name in adjusted_config and leaves_param_name in adjusted_config:
            depth_info = adjusted_config[depth_param_name]
            leaves_info = adjusted_config[leaves_param_name]

            if _is_range_param(depth_info) and _is_range_param(leaves_info):
                md_low, md_high = depth_info['low'], depth_info['high']
                nl_low, nl_high = leaves_info['low'], leaves_info['high']

                # Estimate practical max leaves based on max_depth range
                # Use a reasonable upper bound if md_high is -1 or very large
                effective_md_high = md_high if 0 < md_high <= 16 else 16
                theoretical_max_leaves = 2**effective_md_high
                practical_max_leaves = min(MAX_LEAVES_HARD_CAP, int(theoretical_max_leaves * 0.8))
                nl_high = min(nl_high, practical_max_leaves)

                # Estimate min leaves based on min_depth range (if not -1)
                if md_low != -1:
                    effective_md_low = max(1, md_low)
                    min_leaves_from_depth = 2**(effective_md_low -1) if effective_md_low > 1 else 2
                    nl_low = max(nl_low, min_leaves_from_depth)

                # Ensure nl_high >= nl_low + 1 after adjustments
                nl_high = max(nl_low + 1, nl_high)

                # Clamp to final bounds
                min_nl, max_nl = PARAM_BOUNDS.get(leaves_param_name, (2, None))
                nl_low = _clamp(int(round(nl_low)), min_nl, max_nl)
                nl_high = _clamp(int(round(nl_high)), min_nl, max_nl)
                nl_high = max(nl_low, nl_high) # Ensure high >= low

                adjusted_config[leaves_param_name]['low'] = nl_low
                adjusted_config[leaves_param_name]['high'] = nl_high

    # 2. GBMs: learning_rate and n_estimators trade-off
    if model_name in ["XGB", "LGB", "CAT"]:
        lr_param_name = 'learning_rate'
        est_param_name = 'n_estimators' if model_name != "CAT" else 'iterations'

        if lr_param_name in adjusted_config and est_param_name in adjusted_config:
             lr_info = adjusted_config[lr_param_name]
             est_info = adjusted_config[est_param_name]

             if _is_range_param(lr_info) and _is_range_param(est_info):
                lr_low, lr_high = lr_info['low'], lr_info['high']
                ne_low, ne_high = est_info['low'], est_info['high']

                # Boost high end of n_estimators if learning rate *low* is very low
                if lr_low < VERY_LOW_LR_THRESHOLD:
                     boost_factor = min(7.0, (VERY_LOW_LR_THRESHOLD / max(1e-4, lr_low)) ** 0.8)
                     # Boost ne_high, but ensure it doesn't shrink the range too much
                     potential_ne_high = int(max(ne_high, ne_low * 1.5) * boost_factor)
                     ne_high = max(ne_high, potential_ne_high)

                # Cap n_estimators if the *entire* learning rate range is high
                if lr_low >= HIGH_LR_RANGE_LOW_THRESHOLD:
                    effective_cap = max(MIN_ESTIMATORS + 50, HIGH_LR_ESTIMATOR_CAP)
                    # Cap both ends, ensure ne_low < ne_high
                    ne_high = min(ne_high, effective_cap)
                    ne_low = min(ne_low, ne_high - 1) # Ensure ne_low is below capped ne_high
                    print(f"INFO [{model_name}]: Learning rate range ({lr_low:.3f}-{lr_high:.3f}) is high. Capping n_estimators range to ({ne_low}, {ne_high}).")

                # Final clamping to global MIN/MAX bounds and ensure range validity
                min_ne, max_ne = PARAM_BOUNDS.get(est_param_name, (MIN_ESTIMATORS, MAX_ESTIMATORS))
                ne_high = _clamp(int(round(ne_high)), min_ne, max_ne)
                ne_low = _clamp(int(round(ne_low)), min_ne, max_ne)

                # Ensure ne_high >= ne_low + reasonable_gap
                min_gap = 50 # Prefer a minimum range gap
                if ne_high < ne_low + min_gap:
                     # Try expanding high first, then shrinking low
                     if ne_high + min_gap <= max_ne:
                         ne_high = ne_low + min_gap
                     elif ne_low - min_gap >= min_ne:
                         ne_low = ne_high - min_gap
                     else: # Cannot maintain gap, just ensure high >= low
                         ne_high = max(ne_low, ne_high)

                # Re-clamp after potential adjustments
                ne_high = _clamp(ne_high, min_ne, max_ne)
                ne_low = _clamp(ne_low, min_ne, max_ne)
                ne_high = max(ne_low, ne_high) # Final check

                adjusted_config[est_param_name]['low'] = ne_low
                adjusted_config[est_param_name]['high'] = ne_high

    return adjusted_config


def _adjust_priorities(config: Dict, data_char: Dict, budget: str, scaling_factor: float) -> Dict:
    """Adjust model priorities based on data, budget, continuous scaling."""
    adjusted_config = config.copy()
    base_priority = config.get("_priority", 100)
    model_name = config.get("_model_name", "") # Get model name stored internally
    if not model_name: return adjusted_config # Should not happen

    model_type = 'unknown'
    if model_name in ["XGB", "LGB", "CAT"]: model_type = 'gbm'
    elif model_name == "RF": model_type = 'rf'
    elif model_name == "SGD_LINEAR": model_type = 'linear'

    priority = float(base_priority)

    # --- Budget Level Impact ---
    if budget == 'low':
        if model_type == 'linear': priority *= 1.25
        elif model_type == 'rf': priority *= 1.1
        elif model_type == 'gbm': priority *= 0.85
    elif budget == 'high':
        if model_type == 'linear': priority *= 0.8
        elif model_type == 'gbm': priority *= 1.15

    # --- Continuous Scaling Factor Impact (Time budget proxy) ---
    if scaling_factor < 0.7: # More constrained time -> favor faster models
        if model_type == 'linear': priority *= max(1.0, (1.0 + (1.0 - scaling_factor) * 0.6))
        elif model_type == 'rf': priority *= max(1.0, (1.0 + (1.0 - scaling_factor) * 0.3))
        elif model_type == 'gbm': priority *= max(0.5, scaling_factor * 0.9) # Reduce priority significantly
    elif scaling_factor > 1.5: # Ample time -> slightly favor more complex models
         if model_type == 'gbm': priority *= max(1.0, (1.0 + (scaling_factor - 1.0) * 0.3))
         if model_type == 'linear': priority *= 0.9

    # --- Data Characteristics Impact ---
    is_high_dim = (data_char['n_features'] > HIGH_DIM_THRESHOLD or
                  (data_char['n_samples'] > 0 and data_char['n_features'] / data_char['n_samples'] > HIGH_DIM_RATIO))
    is_very_high_dim = data_char['n_features'] > VERY_HIGH_DIM_THRESHOLD
    is_sparse = data_char['is_sparse'] and data_char['sparsity_ratio'] > SPARSITY_THRESHOLD

    # High Dimensionality
    if is_high_dim or is_very_high_dim:
        if model_type == 'linear': priority *= PRIORITY_HIGH_DIM_LINEAR_BOOST
        elif model_type == 'gbm': priority *= PRIORITY_HIGH_DIM_GBM_PENALTY
        if is_very_high_dim and model_type == 'rf': priority *= 0.7 # RF less suitable for very high dim

    # Data Size
    if data_char['n_samples'] > LARGE_DATASET_THRESHOLD:
        if model_type == 'rf': priority *= PRIORITY_LARGE_DATA_RF_PENALTY
        if model_type == 'gbm' and model_name in ["LGB", "CAT"]: priority *= PRIORITY_LARGE_DATA_LGBM_CAT_BOOST
        if model_type == 'linear': priority *= 0.9 # Linear less likely to be best on very large data
    elif data_char['n_samples'] < SMALL_DATASET_THRESHOLD:
        if model_type == 'linear': priority *= PRIORITY_SMALL_DATA_LINEAR_BOOST
        if model_type == 'rf': priority *= PRIORITY_SMALL_DATA_RF_BOOST
        if model_type == 'gbm' and model_name == "XGB" : priority *= 0.9 # XGB can overfit small data

    # Categorical Features
    if data_char['cat_feature_ratio'] > HIGH_CATEGORICAL_RATIO:
        if model_type == 'gbm' and model_name in ["LGB", "CAT"]: priority *= PRIORITY_HIGH_CATEGORICAL_CATLGBM_BOOST
        elif model_type == 'rf': priority *= 0.9 # RF handles cats, but less sophisticatedly than Cat/LGB
        elif model_type == 'linear': priority *= 0.85 # Linear requires encoding

    # Sparsity
    if is_sparse:
         if model_type == 'linear': priority *= PRIORITY_SPARSE_LINEAR_BOOST
         if model_type == 'gbm' and model_name == "XGB": priority *= PRIORITY_SPARSE_XGB_BOOST
         if model_type == 'rf': priority *= 0.8 # RF less ideal for sparse

    # Task Specifics (Classification)
    if data_char['task_type'] == 'classification':
        is_imbalanced = data_char['class_balance'] >= IMBALANCE_THRESHOLD_MODERATE
        is_multiclass = data_char['n_classes'] > 2

        if is_imbalanced:
            if model_type != 'linear': priority *= PRIORITY_IMBALANCE_BOOST # Boost tree models
            else: priority *= 0.9 # Linear often needs careful weighting/sampling

        if is_multiclass:
             if model_type == 'rf': priority *= PRIORITY_MULTICLASS_BOOST_RF
             if model_type == 'linear' and data_char['n_classes'] > 10: priority *= 0.8 # Linear OVA can be slow

    # Ensure priority is at least 1
    adjusted_config['_priority'] = int(round(max(1, priority)))
    return adjusted_config

def _finalize_config_structure(config: Dict) -> Dict:
    """Convert internal parameter representation to final Optuna-like output format."""
    final_params = {}
    priority = config.get('_priority', 100) # Retrieve adjusted priority

    for param_name, p_info in config.items():
        if param_name.startswith('_'): # Skip internal keys like _priority, _model_name
             continue

        if not isinstance(p_info, dict):
            warnings.warn(f"Parameter '{param_name}' has unexpected format: {p_info}. Skipping.")
            continue

        p_type = p_info.get('type')
        if p_type in ['int', 'float', 'float_log']:
            if 'low' in p_info and 'high' in p_info and p_info['low'] is not None and p_info['high'] is not None:
                # Handle LGB max_depth=-1 special case
                if param_name == 'max_depth' and p_info['low'] == -1:
                    # Optuna doesn't directly support (-1, X) range.
                    # Represent as categorical choice between -1 and a range [min_depth, high_depth]
                    # Or, simplify: If -1 is possible, just offer a range from reasonable min to high.
                    # Let's choose the latter for simplicity: offer [min_bound, high] range.
                    min_bound = PARAM_BOUNDS.get(param_name, (1, None))[0] or 1
                    low = min_bound
                    high = p_info['high']
                    # Ensure range is valid
                    if high < low: high = low
                    final_params[param_name] = (int(low), int(high))
                    continue # Move to next param

                # Standard range processing
                low = int(p_info['low']) if p_type == 'int' else float(p_info['low'])
                high = int(p_info['high']) if p_type == 'int' else float(p_info['high'])

                # Final sanity check: low <= high
                if low > high:
                    warnings.warn(f"Final range for {param_name} is invalid ({low}, {high}). Setting low = high.")
                    low = high

                # Handle collapsed range -> treat as fixed value list
                if low == high:
                    final_params[param_name] = [low]
                elif p_type == 'float_log':
                    if low <= 0: # Should be handled earlier, but final check
                        log_min_bound = PARAM_BOUNDS.get(param_name, (1e-9, None))[0] or 1e-9
                        low = max(log_min_bound, 1e-9)
                        if high <= low: high = low * 10
                        warnings.warn(f"Corrected non-positive log lower bound for {param_name} to {low}.")
                    final_params[param_name] = (low, high, True) # Optuna log flag
                else: # int or float
                    final_params[param_name] = (low, high)

            elif 'value' in p_info and p_info['value'] is not None: # Handle fixed value from template
                 final_params[param_name] = [p_info['value']]
            # else: Param had range type but no valid range (shouldn't happen if logic is correct)

        elif p_type == 'cat':
            if 'values' in p_info and isinstance(p_info['values'], list) and p_info['values']:
                 # If only one value, treat as fixed
                 if len(p_info['values']) == 1:
                      final_params[param_name] = p_info['values']
                 else:
                     final_params[param_name] = p_info['values']
            # else: Empty list [] placeholder was not filled, skip parameter

        elif p_type == 'fixed':
            if 'value' in p_info and p_info['value'] is not None:
                 final_params[param_name] = [p_info['value']] # Represent as categorical list of one item
            # else: Fixed value was None (e.g. num_class for binary), skip parameter

        else:
             warnings.warn(f"Unsupported parameter type '{p_type}' for {param_name} during finalization.")

    # Only add non-empty parameter sets
    if final_params:
        final_params['_priority'] = int(round(priority))
        return final_params
    else:
        return {} # Return empty dict if no valid parameters were generated


# ==============================================================================
# --- Core Generation Function ---
# ==============================================================================

def generate_auto_config(
    data_char: Dict,
    budget: Literal['low', 'medium', 'high'] = "medium",
    scaling_factor: float = 1.0
) -> Dict[str, Dict[str, Any]]:
    """
    Core logic to generate search space configurations for all models.
    """
    final_config = {}
    is_classification = data_char['task_type'] == 'classification'
    is_multiclass = is_classification and data_char['n_classes'] > 2
    is_binary = is_classification and data_char['n_classes'] == 2
    is_imbalanced = data_char['class_balance'] >= IMBALANCE_THRESHOLD_MODERATE
    imbalance_ratio = data_char['class_balance'] if data_char['class_balance'] != np.inf else 100.0 # Cap effective ratio
    is_sparse = data_char['is_sparse'] and data_char['sparsity_ratio'] > SPARSITY_THRESHOLD

    for model_name, template_config in BASE_MODEL_CONFIG_TEMPLATES.items():
        # --- Initialize config from template ---
        current_config = {}
        base_priority = template_config.get("_priority", 100)
        for param, conf in template_config.items():
            if param == "_priority": continue
            try:
                 current_config[param] = _get_param_info(conf)
            except ValueError as e:
                 warnings.warn(f"Error processing template for {model_name}/{param}: {e}. Skipping param.")
                 continue
        current_config['_priority'] = base_priority # Store base priority
        current_config['_model_name'] = model_name # Store model name for internal use

        params_to_delete = [] # Track params to remove based on task/data

        # --- Task-Specific Adjustments (Objectives, Metrics, Task-Params) ---
        task_params = {} # Collect task-specific values
        if is_classification:
            obj_map = {"XGB": "binary:logistic", "LGB": "binary", "CAT": "Logloss", "RF": None, "SGD_LINEAR": "log_loss"}
            eval_map = {"XGB": "logloss", "LGB": "binary_logloss", "CAT": "Logloss", "RF": "neg_log_loss", "SGD_LINEAR": None} # RF/SGD handled by 'loss'/'criterion'
            crit_map = {"RF": ["gini", "entropy"]} # RF criterion
            loss_map = {"SGD_LINEAR": ["log_loss", "modified_huber"]} # SGD loss

            if is_multiclass:
                 obj_map = {"XGB": "multi:softprob", "LGB": "multiclass", "CAT": "MultiClass", "RF": None, "SGD_LINEAR": "log_loss"} # SGD uses log_loss for multi
                 eval_map = {"XGB": "mlogloss", "LGB": "multi_logloss", "CAT": "MultiClass", "RF": "neg_log_loss", "SGD_LINEAR": None}
                 task_params['num_class'] = data_char['n_classes'] # For XGB/LGB
            else: # Binary
                 params_to_delete.append('num_class')

            task_params['objective'] = obj_map.get(model_name)
            task_params['loss_function'] = obj_map.get(model_name) # CatBoost alias
            task_params['eval_metric'] = eval_map.get(model_name) # XGB alias
            task_params['metric'] = eval_map.get(model_name) # LGB alias
            task_params['criterion'] = crit_map.get(model_name, [])
            task_params['loss'] = loss_map.get(model_name, [])

            # Imbalance Handling
            if is_imbalanced:
                if is_binary: # scale_pos_weight usually only for binary
                    if model_name in ["XGB", "LGB", "CAT"]:
                         spw_low = max(1.0, imbalance_ratio * 0.4)
                         spw_high = min(100.0, imbalance_ratio * 2.5)
                         spw_high = max(spw_low + 0.1, spw_high)
                         task_params['scale_pos_weight'] = {'low': spw_low, 'high': spw_high, 'type': 'float_log', 'group': PARAM_GROUP_TASK, 'sensitivity': 0.3}
                    else: params_to_delete.append("scale_pos_weight")
                else: params_to_delete.append("scale_pos_weight") # Not typically used for multiclass

                if model_name == "LGB": task_params['is_unbalanced'] = {'values': [True, False], 'type': 'cat', 'group': PARAM_GROUP_TASK}
                else: params_to_delete.append("is_unbalanced")

                if model_name in ["RF", "SGD_LINEAR"]:
                    weights = ["balanced", None]
                    if model_name == "RF": weights.append("balanced_subsample")
                    task_params['class_weight'] = {'values': weights, 'type': 'cat', 'group': PARAM_GROUP_TASK}
                else: params_to_delete.append("class_weight")
            else: # Balanced or Regression
                 params_to_delete.extend(["scale_pos_weight", "is_unbalanced", "class_weight"])

        else: # Regression
            obj_map = {"XGB": "reg:squarederror", "LGB": "regression_l2", "CAT": "RMSE", "RF": None, "SGD_LINEAR": "squared_error"}
            eval_map = {"XGB": "rmse", "LGB": "l2", "CAT": "RMSE", "RF": "neg_mean_squared_error", "SGD_LINEAR": None}
            crit_map = {"RF": ["squared_error", "absolute_error", "friedman_mse"]} # Removed poisson for general regression
            loss_map = {"SGD_LINEAR": ["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"]}

            task_params['objective'] = obj_map.get(model_name)
            task_params['loss_function'] = obj_map.get(model_name)
            task_params['eval_metric'] = eval_map.get(model_name)
            task_params['metric'] = eval_map.get(model_name)
            task_params['criterion'] = crit_map.get(model_name, [])
            task_params['loss'] = loss_map.get(model_name, [])
            params_to_delete.extend(["scale_pos_weight", "class_weight", "is_unbalanced", "num_class"])

        # --- Apply Task Parameters ---
        for param, value_info in task_params.items():
             if param in current_config:
                 if isinstance(value_info, dict): # It's a complex definition (e.g., range for spw)
                      current_config[param].update(value_info)
                 else: # It's a simple value (objective, metric, etc.)
                      if current_config[param]['type'] == 'cat':
                          if isinstance(value_info, list): current_config[param]['values'] = value_info
                          else: params_to_delete.append(param) # Mismatch
                      elif current_config[param]['type'] == 'fixed':
                          current_config[param]['value'] = value_info
                      else: # Should not happen if templates are correct
                           params_to_delete.append(param)
             # else: Param not in template, ignore

        # --- Data Specific Adjustments ---
        # Sparse Data Handling for SGD Penalty
        if is_sparse and model_name == "SGD_LINEAR":
            if "penalty" in current_config and current_config["penalty"]['type'] == 'cat':
                 current_config["penalty"]['values'] = ["l1", "elasticnet"] # Prioritize L1/ElasticNet
                 print(f"INFO [{model_name}]: Emphasizing L1/ElasticNet penalty for SGD due to sparsity ({data_char['sparsity_ratio']:.2f})")
                 # Keep l1_ratio active
            else:
                 # If penalty somehow not 'cat', remove it and l1_ratio
                 params_to_delete.extend(["penalty", "l1_ratio"])
        elif model_name == "SGD_LINEAR" and "penalty" in current_config and current_config["penalty"]['type'] == 'cat':
             # If not sparse, ensure l2 is an option
             if "l2" not in current_config["penalty"]['values']:
                  current_config["penalty"]['values'].append("l2")


        # --- Delete inapplicable parameters ---
        for param_name in set(params_to_delete):
            current_config.pop(param_name, None)
            # Also remove related params if primary is deleted (e.g., l1_ratio if penalty gone)
            if param_name == 'penalty' and model_name == 'SGD_LINEAR':
                 current_config.pop('l1_ratio', None)

        # --- Apply Scaling & Adjustments ---
        if current_config: # Only proceed if config is not empty
            current_config = _apply_initial_data_scaling(current_config, data_char, model_name)
            current_config = _apply_budget_and_continuous_scaling(current_config, budget, scaling_factor)
            current_config = _apply_interdependencies(current_config, model_name)
            current_config = _adjust_priorities(current_config, data_char, budget, scaling_factor)

            # --- Finalize Structure ---
            final_model_config = _finalize_config_structure(current_config)
            if final_model_config: # Check if finalizing resulted in non-empty config
                final_config[model_name] = final_model_config

    # Sort final config by priority (descending)
    sorted_config = dict(sorted(final_config.items(), key=lambda item: item[1].get('_priority', 0), reverse=True))

    return sorted_config


# ==============================================================================
# --- Time Budget Calculation and Calibration ---
# ==============================================================================

def _calculate_complexity_factor(n_samples: int, n_features: int, data_complexity_score: float) -> float:
    """Calculates a heuristic cost factor based on data size and complexity."""
    # Use log scale for size, with minimums to avoid log(<=0) and ensure some base cost
    log_n = max(1.0, np.log10(max(100, n_samples)))
    log_p = max(1.0, np.log10(max(10, n_features)))
    # Complexity score modifier (higher score increases cost factor)
    complexity_modifier = max(1.0, data_complexity_score**1.2)
    # Combine factors
    cost_factor = (log_n**1.5) * (log_p**1.1) * complexity_modifier # Adjusted exponents for more sample impact
    return max(1e-6, cost_factor) # Avoid zero/negative


def _calibrate_base_time_unit(
    X, y,
    data_char: Dict,
    use_gpu: bool = False
) -> float:
    """
    Performs a benchmark training run using LightGBM to estimate BASE_TIME_UNIT.
    """
    if not _lightgbm_available:
        warnings.warn("LightGBM not available, skipping calibration. Using fallback base time unit.", RuntimeWarning)
        return FALLBACK_BASE_TIME_UNIT

    start_calibration_time = time.perf_counter()
    n_samples_orig = data_char['n_samples']
    n_features = data_char['n_features']
    task_type = data_char['task_type']
    data_complexity = data_char['data_complexity_score'] # Use pre-calculated complexity

    if n_samples_orig < 50 or n_features == 0:
        warnings.warn(f"Too few samples ({n_samples_orig}) or features ({n_features}) for reliable calibration. Using fallback.", RuntimeWarning)
        return FALLBACK_BASE_TIME_UNIT

    # --- Sampling & Data Prep ---
    n_samples_calib = min(n_samples_orig, CALIBRATION_SAMPLE_SIZE)
    idx = np.random.choice(n_samples_orig, size=n_samples_calib, replace=False)

    try:
        X_calib = X[idx, :] if not isinstance(X, pd.DataFrame) else X.iloc[idx]
        y_calib_orig = y[idx] if not isinstance(y, pd.Series) else y.iloc[idx]

        # Prepare y for LGBM (handle NaNs, factorize classification targets)
        y_series = pd.Series(y_calib_orig)
        nan_mask = y_series.notna()
        if not nan_mask.all():
            y_series = y_series[nan_mask]
            X_calib = X_calib[nan_mask] if not isinstance(X_calib, (pd.DataFrame, pd.Series, csr_matrix)) else X_calib[nan_mask,:]
            warnings.warn(f"NaNs found in calibration target; {X_calib.shape[0]} samples used after filtering.")

        if X_calib.shape[0] < 50:
             warnings.warn(f"Too few samples ({X_calib.shape[0]}) remaining after NaN removal for calibration. Using fallback.", RuntimeWarning)
             return FALLBACK_BASE_TIME_UNIT

        y_calib_np = y_series.to_numpy()

        if task_type == 'classification':
             y_calib_np, _ = pd.factorize(y_calib_np) # Factorize handles different types robustly
        elif task_type == 'regression':
             y_calib_np = y_calib_np.astype(float) # Ensure float for regression
        else:
             warnings.warn(f"Unsupported task type '{task_type}' for calibration. Using fallback.", RuntimeWarning)
             return FALLBACK_BASE_TIME_UNIT

    except Exception as e:
        warnings.warn(f"Error preparing data for calibration: {e}. Using fallback.", RuntimeWarning)
        return FALLBACK_BASE_TIME_UNIT


    # --- Model Setup ---
    model_params = {
        'objective': 'binary' if task_type == 'classification' and len(np.unique(y_calib_np)) <= 2 else \
                     'multiclass' if task_type == 'classification' else \
                     'regression_l1', # L1 less sensitive to outliers
        'metric': 'binary_logloss' if task_type == 'classification' and len(np.unique(y_calib_np)) <= 2 else \
                  'multi_logloss' if task_type == 'classification' else 'l1',
        'n_estimators': CALIBRATION_N_ESTIMATORS,
        'learning_rate': CALIBRATION_LEARNING_RATE,
        'max_depth': CALIBRATION_MAX_DEPTH,
        'num_leaves': 2**CALIBRATION_MAX_DEPTH - 1, # Typical relation
        'n_jobs': -1,
        'verbosity': -1,
        'seed': 42,
    }
    if model_params['objective'] == 'multiclass':
         model_params['num_class'] = len(np.unique(y_calib_np))

    gpu_params = {}
    if use_gpu:
        # Basic GPU params, more might be needed depending on setup (platform_id, device_id)
        gpu_params = {'device': 'gpu'}
        print("INFO: Attempting calibration using GPU.")
    else:
         print("INFO: Running calibration using CPU.")
    model_params.update(gpu_params)

    try:
        model = lgb.LGBMModel(**model_params) # Use base model to avoid classifier/regressor type mismatch

        # --- Identify Categorical Features for LGBM ---
        categorical_feature_arg = 'auto' # Let LGBM try auto detection
        if data_char.get('categorical_features'):
             cat_features_in_data = data_char['categorical_features']
             if isinstance(X_calib, pd.DataFrame):
                 # Filter features present in the sampled dataframe columns
                 cat_cols_present = [f for f in cat_features_in_data if f in X_calib.columns]
                 if cat_cols_present: categorical_feature_arg = cat_cols_present
             elif isinstance(X_calib, (np.ndarray, csr_matrix)):
                  # Filter indices valid for the (potentially modified) X_calib shape
                  cat_indices_present = [f for f in cat_features_in_data if isinstance(f, int) and 0 <= f < X_calib.shape[1]]
                  if cat_indices_present: categorical_feature_arg = cat_indices_present

             if isinstance(categorical_feature_arg, list) and categorical_feature_arg:
                  print(f"INFO: Passing {len(categorical_feature_arg)} identified categorical features to LGBM for calibration.")
             else:
                  print("INFO: Using 'auto' categorical feature detection in LGBM for calibration.")


        # --- Timing the Fit ---
        fit_start_time = time.perf_counter()
        model.fit(X_calib, y_calib_np, categorical_feature=categorical_feature_arg)
        fit_end_time = time.perf_counter()
        benchmark_time = fit_end_time - fit_start_time
        benchmark_time = max(MIN_CALIBRATION_TIME, benchmark_time) # Ensure minimum time

        # --- Calculate Base Time Unit ---
        # Use complexity factor based on calibration data size, but original full data complexity score
        cost_factor_calib = _calculate_complexity_factor(X_calib.shape[0], X_calib.shape[1], data_complexity)

        if cost_factor_calib <= 1e-6:
             warnings.warn("Calibration cost factor is near zero. Cannot reliably estimate time unit. Using fallback.", RuntimeWarning)
             return FALLBACK_BASE_TIME_UNIT

        calculated_base_time_unit = benchmark_time / cost_factor_calib

        # Apply bounds
        calculated_base_time_unit = _clamp(calculated_base_time_unit, BASE_TIME_UNIT_BOUNDS[0], BASE_TIME_UNIT_BOUNDS[1])

        total_calibration_time = time.perf_counter() - start_calibration_time
        print(f"INFO: Calibration successful. Benchmark time: {benchmark_time:.4f}s on {X_calib.shape[0]} samples. Estimated base_time_unit: {calculated_base_time_unit:.4f} (Total calibration overhead: {total_calibration_time:.2f}s)")
        return calculated_base_time_unit

    except Exception as e:
        warnings.warn(f"Calibration run failed: {e}. Using fallback base time unit.", RuntimeWarning)
        # Hint for GPU failure
        if use_gpu and ("GPU" in str(e) or "device" in str(e).lower()):
             warnings.warn("GPU calibration failed. Ensure LightGBM is compiled with GPU support and necessary drivers/runtime are installed.", RuntimeWarning)
        return FALLBACK_BASE_TIME_UNIT


def time_budget_to_level_and_scaling(
    time_budget_seconds: Optional[float],
    n_samples: int,
    n_features: int,
    data_complexity: float,
    base_time_unit: float # Pass calibrated or fallback unit directly
) -> Tuple[Literal['low', 'medium', 'high'], float]:
    """Convert time budget to discrete level and continuous scaling factor."""
    if time_budget_seconds is None or time_budget_seconds <= 0:
        warnings.warn("Invalid time_budget_seconds provided. Using default 'medium' budget and scaling=1.0.", RuntimeWarning)
        return "medium", 1.0

    # Calculate cost factor for the *full* dataset
    cost_factor = _calculate_complexity_factor(n_samples, n_features, data_complexity)

    # Normalize budget using the cost factor and base time unit
    denominator = cost_factor * base_time_unit
    denominator = max(1e-6, denominator) # Avoid division by zero
    normalized_budget = time_budget_seconds / denominator

    # --- Determine Level and Scaling Factor ---
    # Define thresholds for normalized budget
    low_thresh = TIME_BUDGET_LOW_NORM
    med_thresh = TIME_BUDGET_MEDIUM_NORM

    # Define scaling factor range endpoints
    min_scale = 0.5 # Corresponds to very low budget
    low_med_scale = 0.8 # Boundary between low and medium
    med_high_scale = 1.2 # Boundary between medium and high
    max_scale = 2.5 # Corresponds to very high budget

    if normalized_budget < low_thresh:
        level = "low"
        # Linear interpolation between min_scale and low_med_scale within the low range
        progress = normalized_budget / max(1e-9, low_thresh)
        scaling = min_scale + (low_med_scale - min_scale) * progress
    elif normalized_budget < med_thresh:
        level = "medium"
        # Linear interpolation between low_med_scale and med_high_scale within the medium range
        progress = (normalized_budget - low_thresh) / max(1e-6, (med_thresh - low_thresh))
        scaling = low_med_scale + (med_high_scale - low_med_scale) * progress
    else: # normalized_budget >= med_thresh
        level = "high"
        # Logarithmic interpolation from med_high_scale towards max_scale for high budget
        # Scaled relative progress beyond the medium threshold
        relative_budget = (normalized_budget - med_thresh) / max(1e-6, med_thresh)
        # Log scale reaches max_scale asymptotically, adjust factor 'k' for desired speed
        k = 0.5 # Controls how quickly scaling approaches max_scale
        scaling = med_high_scale + (max_scale - med_high_scale) * (1 - np.exp(-k * relative_budget))

    # Final clamp on scaling factor
    scaling = max(min_scale, min(max_scale, scaling))

    return level, scaling


# ==============================================================================
# --- Main Public API Functions ---
# ==============================================================================

def auto_config_from_data(
    X, y=None,
    categorical_features: Optional[List[Union[int, str]]] = None,
    budget: Literal['low', 'medium', 'high'] = "medium"
) -> Dict[str, Dict[str, Any]]:
    """
    Generate hyperparameter search space config from data and discrete budget.

    Args:
        X: Input features (Pandas DataFrame, NumPy array, or sparse matrix).
        y: Target variable (Pandas Series, NumPy array). Optional.
        categorical_features: List of column names or indices considered categorical.
                              If None, attempts auto-detection.
        budget: Discrete budget level ('low', 'medium', 'high').

    Returns:
        Dictionary where keys are model names (e.g., "XGB", "LGB") and values
        are dictionaries representing the search space for that model, including
        a '_priority' key. Parameter ranges are tuples (low, high) or
        (low, high, is_log_scale=True), categorical choices are lists. Fixed values
        are represented as a single-item list.
    """
    print("--- Generating Config from Data (Discrete Budget) ---")
    start_time = time.perf_counter()
    data_char = detect_dataset_characteristics(X, y, categorical_features)
    detection_time = time.perf_counter() - start_time
    print(f"Data Chars (Detection took {detection_time:.2f}s):")
    pprint.pprint({k: v for k, v in data_char.items() if k != 'numeric_features' and k != 'categorical_features'}, indent=2, width=100)
    print(f"  Num Numeric Feats: {len(data_char.get('numeric_features',[]))}, Num Categorical Feats: {len(data_char.get('categorical_features',[]))}")


    # For discrete budget, use scaling factor = 1.0; 'budget' label drives discrete shifts
    scaling_factor = 1.0
    print(f"Discrete Budget: '{budget}' -> Applying direct budget level adjustments.")

    config = generate_auto_config(data_char, budget, scaling_factor)
    gen_time = time.perf_counter() - start_time - detection_time
    print(f"--- Config Generation Complete (took {gen_time:.2f}s) ---")
    return config

def auto_config_from_data_with_time_budget(
    X, y=None,
    categorical_features: Optional[List[Union[int, str]]] = None,
    time_budget_seconds: Optional[float] = None,
    base_time_unit_override: Optional[float] = None,
    use_gpu: bool = False # Flag to attempt GPU calibration
) -> Dict[str, Dict[str, Any]]:
    """
    Generate hyperparameter search space config using data and time budget,
    with optional automatic base time unit calibration.

    Args:
        X: Input features (Pandas DataFrame, NumPy array, or sparse matrix).
        y: Target variable (Pandas Series, NumPy array). Optional.
        categorical_features: List of column names or indices considered categorical.
                              If None, attempts auto-detection.
        time_budget_seconds: Allowed time budget in seconds. If None or <= 0,
                             defaults to 'medium' budget behaviour (no calibration, scaling=1.0).
        base_time_unit_override: Optional float to manually set the base time unit.
                                 If provided, automatic calibration is skipped.
        use_gpu: If True, attempts GPU calibration benchmark (requires compatible
                 LightGBM install and hardware). Affects only calibration step.

    Returns:
        Dictionary of model search spaces similar to `auto_config_from_data`.

    Requires:
        - `lightgbm` library for automatic calibration (if not overridden).
    """
    print("--- Generating Config from Data (Time Budget) ---")
    start_time = time.perf_counter()
    data_char = detect_dataset_characteristics(X, y, categorical_features)
    detection_time = time.perf_counter() - start_time
    print(f"Data Chars (Detection took {detection_time:.2f}s):")
    pprint.pprint({k: v for k, v in data_char.items() if k != 'numeric_features' and k != 'categorical_features'}, indent=2, width=100)
    print(f"  Num Numeric Feats: {len(data_char.get('numeric_features',[]))}, Num Categorical Feats: {len(data_char.get('categorical_features',[]))}")

    effective_base_time_unit = FALLBACK_BASE_TIME_UNIT
    calibration_performed = False

    if base_time_unit_override is not None:
        if isinstance(base_time_unit_override, (int, float)) and base_time_unit_override > 0:
             effective_base_time_unit = float(base_time_unit_override)
             print(f"INFO: Using provided base_time_unit_override: {effective_base_time_unit:.4f}. Skipping calibration.")
        else:
            warnings.warn(f"Invalid base_time_unit_override ({base_time_unit_override}). Must be positive number. Using fallback/attempting calibration.", RuntimeWarning)
            # Fall through to potential calibration if override was invalid

    # Attempt calibration only if no valid override was given AND time budget is positive
    if effective_base_time_unit == FALLBACK_BASE_TIME_UNIT and time_budget_seconds is not None and time_budget_seconds > 0:
        print("INFO: Attempting automatic base_time_unit calibration...")
        calib_start_time = time.perf_counter()
        # Pass use_gpu flag to calibration function
        effective_base_time_unit = _calibrate_base_time_unit(X, y, data_char, use_gpu=use_gpu)
        calib_duration = time.perf_counter() - calib_start_time
        calibration_performed = True
        # Note: _calibrate_base_time_unit returns FALLBACK_BASE_TIME_UNIT on failure/skip
        if effective_base_time_unit == FALLBACK_BASE_TIME_UNIT:
            if _lightgbm_available:
                 print(f"INFO: Calibration failed or was skipped (duration: {calib_duration:.2f}s), using fallback base_time_unit: {FALLBACK_BASE_TIME_UNIT:.4f}")
            else:
                 print(f"INFO: LightGBM not found (calibration skipped), using fallback base_time_unit: {FALLBACK_BASE_TIME_UNIT:.4f}")
        # Success message is printed inside _calibrate_base_time_unit
    elif effective_base_time_unit == FALLBACK_BASE_TIME_UNIT: # No override, no (valid) time budget
         print(f"INFO: No time budget or override provided, or time budget <= 0. Using fallback base_time_unit: {FALLBACK_BASE_TIME_UNIT:.4f} for defaults.")

    # Calculate budget level and scaling factor using the determined base time unit
    budget_level, scaling_factor = time_budget_to_level_and_scaling(
        time_budget_seconds,
        data_char['n_samples'],
        data_char['n_features'],
        data_char['data_complexity_score'],
        base_time_unit=effective_base_time_unit # Use calibrated or fallback/override
    )

    print(f"Time Budget: {time_budget_seconds}s -> Level='{budget_level}', Scaling Factor={scaling_factor:.3f} (using base_time_unit={effective_base_time_unit:.4f})")

    config = generate_auto_config(data_char, budget_level, scaling_factor)
    gen_time = time.perf_counter() - start_time - detection_time - (calib_duration if calibration_performed else 0)
    print(f"--- Config Generation Complete (took {gen_time:.2f}s + {(calib_duration if calibration_performed else 0):.2f}s calibration) ---")
    return config


# ==============================================================================
# --- Example Usage ---
# ==============================================================================
if __name__ == "__main__":
    # Configure warnings for example run
    warnings.simplefilter("default", category=RuntimeWarning) # Show runtime warnings (like fallback)
    warnings.simplefilter("ignore", category=UserWarning) # Ignore general user warnings for cleaner output
    pp = pprint.PrettyPrinter(indent=2, width=120) # Setup pretty printer

    # --- Example 1: Small Classification Dataset - Medium Budget ---
    print("\n" + "="*60)
    print(" Example 1: Small Classification (Discrete Medium Budget)")
    print("="*60)
    from sklearn.datasets import make_classification
    X_small_cls, y_small_cls = make_classification(n_samples=800, n_features=15, n_informative=10, n_redundant=2,
                                                   n_classes=2, random_state=42, flip_y=0.05, class_sep=0.8)
    X_small_cls_pd = pd.DataFrame(X_small_cls, columns=[f'f_{i}' for i in range(15)])
    # Make one feature explicitly categorical for testing
    X_small_cls_pd['f_cat_explicit'] = pd.cut(X_small_cls_pd['f_14'], bins=5, labels=False).astype('category')
    # Test auto-detection of categorical integer
    X_small_cls_pd['f_cat_int'] = np.random.randint(0, 4, size=800)


    print("\n* Generating Config: Budget='medium' *")
    # Let it auto-detect categorical features including the integer one
    config_med = auto_config_from_data(X_small_cls_pd, y_small_cls, budget="medium")

    print("\n--- Generated Configuration (Medium Budget) ---")
    pp.pprint(config_med)

    # --- Example 2: Larger Regression High-Dim - Time Budget (Calibration CPU) ---
    print("\n" + "="*60)
    print(" Example 2: Larger Regression High-Dim (Time Budget - CPU Calibration)")
    print("="*60)
    from sklearn.datasets import make_regression
    X_large_reg, y_large_reg = make_regression(n_samples=25000, n_features=500, n_informative=80,
                                                 random_state=42, noise=10.0)
    X_large_reg_pd = pd.DataFrame(X_large_reg, columns=[f'f_{i}' for i in range(500)])

    time_med = 1800 # 30 mins
    print(f"\n* Generating Config: Time Budget={time_med}s (CPU Calibration) *")
    config_med_time_cpu = auto_config_from_data_with_time_budget(
        X_large_reg_pd, y_large_reg,
        time_budget_seconds=time_med,
        use_gpu=False # Explicitly use CPU
    )

    print(f"\n--- Generated Configuration (Time Budget: {time_med}s, CPU Calibrated Base Time) ---")
    pp.pprint(config_med_time_cpu)
    # Check n_estimators range in the output for XGB/LGB/CAT - should be scaled but capped by MAX_ESTIMATORS (2500)

    # --- Example 3: Multiclass Imbalanced - Low Time Budget & GPU Calibration Attempt ---
    print("\n" + "="*60)
    print(" Example 3: Multiclass Imbalanced (Low Time Budget - GPU Calibration Attempt)")
    print("="*60)
    X_multi, y_multi = make_classification(n_samples=5000, n_features=50, n_informative=25, n_redundant=5,
                                           n_classes=4, weights=[0.6, 0.2, 0.1, 0.1], # Imbalanced
                                           random_state=42, flip_y=0.02, class_sep=0.9)
    X_multi_pd = pd.DataFrame(X_multi, columns=[f'f_{i}' for i in range(50)])
    # Explicitly pass categorical feature indices (example)
    cat_feat_indices = [40, 41]
    X_multi_pd[f'f_{cat_feat_indices[0]}'] = np.random.choice([0, 1, 2], size=5000, p=[0.8, 0.15, 0.05]) # Low cardinality int
    X_multi_pd[f'f_{cat_feat_indices[1]}'] = pd.Categorical(
    np.random.choice(['A', 'B', 'C'], size=5000, p=[0.9, 0.07, 0.03])
)

    time_low = 300 # 5 minutes
    print(f"\n* Generating Config: Time Budget={time_low}s (GPU Calibration Attempt) *")
    # Note: This will only work if LightGBM is compiled with GPU support and a GPU is available/configured
    # Otherwise, it should gracefully fail calibration or LightGBM itself might error/warn, then fallback.
    config_low_time_gpu = auto_config_from_data_with_time_budget(
        X_multi_pd, y_multi,
        categorical_features=cat_feat_indices, # Pass categorical feature indices/names
        time_budget_seconds=time_low,
        use_gpu=True # Attempt GPU calibration
    )

    print(f"\n--- Generated Configuration (Time Budget: {time_low}s, GPU Calibrated Base Time Attempt) ---")
    pp.pprint(config_low_time_gpu)

    # --- Example 4: Sparse Data - Time Budget with Override ---
    print("\n" + "="*60)
    print(" Example 4: Sparse Data (Time Budget with Override)")
    print("="*60)
    from scipy.sparse import random as sparse_random
    X_sparse = sparse_random(10000, 500, density=0.05, format='csr', random_state=42, dtype=np.float32)
    # Ensure y has same number of samples as X_sparse
    y_sparse = np.random.randint(0, 2, X_sparse.shape[0])

    time_high = 3600 # 1 hour
    manual_time_unit = 0.05 # Assume we calibrated this manually before or know it from experience
    print(f"\n* Generating Config: Time Budget={time_high}s (Manual Override: {manual_time_unit}) *")
    config_sparse_override = auto_config_from_data_with_time_budget(
        X_sparse, y_sparse,
        time_budget_seconds=time_high,
        base_time_unit_override=manual_time_unit
    )

    print(f"\n--- Generated Configuration (Time Budget: {time_high}s, Override Base Time) ---")
    pp.pprint(config_sparse_override)
    # Check SGD_LINEAR penalty param - should favor l1/elasticnet

    # --- Example 5: Test high LR capping ---
    print("\n" + "="*60)
    print(" Example 5: Test High LR Range Capping")
    print("="*60)
    # Create data chars that might push LR high, but not extremely large dataset
    # Use high budget and high scaling factor to push ranges up, potentially triggering the cap
    X_dummy, y_dummy = make_classification(n_samples=2000, n_features=20, n_informative=15, random_state=43)

    time_very_high = 10000 # Very high budget -> high scaling factor
    # Run with a known (or overridden) base_time_unit to ensure high scaling factor
    print(f"\n* Generating Config: Time Budget={time_very_high}s (Force High Scaling) *")
    config_high_lr_test = auto_config_from_data_with_time_budget(
         X_dummy, y_dummy,
         time_budget_seconds=time_very_high,
         base_time_unit_override=0.01 # Force a high normalized budget -> high scaling factor
    )

    print("\n--- Generated Configuration (High Budget/Scaling) ---")
    pp.pprint(config_high_lr_test)
    # Check the output for LGB/XGB:
    # - Expect learning_rate range to be pushed higher by the large budget/scaling.
    # - If lr_low >= HIGH_LR_RANGE_LOW_THRESHOLD (0.18), expect an INFO message and
    #   n_estimators range capped around HIGH_LR_ESTIMATOR_CAP (1200).
    # - Otherwise, n_estimators should still be capped by MAX_ESTIMATORS (2500).

    # --- Example 6: Test low LR boosting n_estimators ---
    print("\n" + "="*60)
    print(" Example 6: Test Low LR Range Boosting n_estimators")
    print("="*60)
    # Use low budget and low scaling factor to push LR low
    time_very_low = 50 # Very low budget
    print(f"\n* Generating Config: Time Budget={time_very_low}s (Force Low Scaling) *")
    config_low_lr_test = auto_config_from_data_with_time_budget(
         X_dummy, y_dummy, # Use same dummy data
         time_budget_seconds=time_very_low,
         base_time_unit_override=0.5 # Force a low normalized budget -> low scaling factor
    )
    print("\n--- Generated Configuration (Low Budget/Scaling) ---")
    pp.pprint(config_low_lr_test)
    # Check the output for LGB/XGB:
    # - Expect learning_rate range to be pushed lower.
    # - If lr_low < VERY_LOW_LR_THRESHOLD (0.02), expect n_estimators high end
    #   to be potentially boosted beyond the default scaled range, but still capped by MAX_ESTIMATORS (2500).


    print("\n" + "="*60)
    print(" Example Testing Complete")
    print("="*60)
