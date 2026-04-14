import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import KFold
from sklearn.model_selection import BaseCrossValidator
from typing import Tuple, Any, Union, Dict, Callable

# Suppress specific numpy warnings to avoid cluttering output
warnings.filterwarnings('ignore', category=RuntimeWarning)

def _safe_power(base_series, exp_series):
    """
    Safely compute base ** exp, handling edge cases that cause pandas errors
    """
    # Convert to float to avoid integer power issues
    base = base_series.astype(float)
    exp = exp_series.astype(float)
    
    # Create result array filled with NaN
    result = pd.Series(np.nan, index=base.index)
    
    # Define safe conditions
    safe_mask = (
        np.isfinite(base) & 
        np.isfinite(exp) & 
        (np.abs(exp) < 10) & 
        (np.abs(base) < 1e10) &
        (
            (base >= 0) |  # Positive base is always safe
            (np.abs(exp - np.round(exp)) < 1e-10)  # Negative base with integer exponent
        ) &
        ~((base == 0) & (exp < 0))  # Avoid 0^negative
    )
    
    if safe_mask.any():
        # Compute power only for safe values
        try:
            safe_results = np.power(base[safe_mask], exp[safe_mask])
            # Check if results are reasonable
            finite_results = np.isfinite(safe_results) & (np.abs(safe_results) < 1e15)
            result.loc[safe_mask] = np.where(finite_results, safe_results, np.nan)
        except:
            # If there's still an error, leave as NaN
            pass
    
    return result

# --- OPS dict ---
OPS = {
    "num": {
        "unary": ["neg", "abs", "square", "sqrt",
                  "log", "log1p", "exp", "inv",
                  "cube", "sin", "cos", "tan",
                  "sigmoid", "tanh", "reciprocal_sqrt",
                  "cbrt", "floor", "ceil",
                  "round", "sign", "arcsin", "arccos", "arctan"
                  ],
        "binary": ["add", "absdiff", "mul",
                   "div", "logmul", "diff_ratio",
                   "sub", "pow", "mod", "max", "min",
                   "geometric_mean", "harmonic_mean",
                   "relative_diff", "log_ratio",
                   "angle_between", "weighted_sum", "weighted_diff"
                   ]
    },
    "cat": {
        "unary": ["target", "freq", "count"],  # encodings
        "binary": ["concat"]
    },
}

# OPS = {
#     "num": {
#         "unary": [
#             "neg", "square", "sqrt", "exp", "inv", "cube",
#             "cos", "tan", "sigmoid", "tanh",
#             "log10", "log2", "floor", "round",
#             "arcsin", "arccos"
#         ],
#         "binary": [
#             "add", "absdiff", "mul", "div", "logmul",
#             "sub", "pow", "mod", "max", "min",
#             "mean", "ratio"
#         ]
#     },
#     "cat": {
#         "unary": ["target", "freq", "count"],  # encodings
#         "binary": ["concat"]
#     },
# }



# --- Numeric Operations Lambdas ---
# Enhanced with better overflow protection and edge case handling

NUM_OPS_LAMBDAS: Dict[str, Callable[..., Tuple[str, pd.Series]]] = {
    # Unary ops
    "neg": lambda df, a: (f"{a}_neg", -df[a]),
    
    "abs": lambda df, a: (f"{a}_abs", df[a].abs()),
    
    "square": lambda df, a: (
        f"{a}_square", 
        np.where(
            np.abs(df[a]) < 1e7,  # Prevent overflow with large values
            df[a] ** 2, 
            np.nan  # Use nan instead of inf
        )
    ),

    "cube": lambda df, a: (
        f"{a}_cube", 
        np.where(
            np.abs(df[a]) < 1e7,  # Prevent overflow with large values
            df[a] ** 3, 
            np.nan  # Use nan instead of inf
        )
    ),
    
    "sqrt": lambda df, a: (
        f"{a}_sqrt", 
        np.sqrt(np.where(df[a] >= 0, df[a], np.nan))
    ),
    
    "log": lambda df, a: (
        f"{a}_log",
        np.log(np.where(df[a] > 0, df[a], np.nan))
    ),

    "log1p": lambda df, a: (
        f"{a}_log1p",
        np.where(df[a] >= 0, np.log1p(df[a]), np.nan)
    ),

    "exp": lambda df, a: (
        f"{a}_exp", 
        # First clip values to safe range, *then* calculate exp
        # This prevents overflow warnings during calculation
        np.where(
            np.abs(df[a]) <= 50,  # Safe range for exp
            np.exp(df[a]),
            np.nan  # Replace potential overflow with nan
        )
    ),
    
    "inv": lambda df, a: (
        f"{a}_inv", 
        np.where(
            np.abs(df[a]) > 1e-15,  # More conservative epsilon
            np.where(
                np.abs(1 / df[a]) < 1e15,  # Check if result would be too large
                1 / df[a],
                np.nan  # Replace potential extreme values with nan
            ),
            np.nan
        )
    ),
  
    "sin": lambda df, a: (f"{a}_sin", np.sin(df[a])),   # For cyclical patterns
    "cos": lambda df, a: (f"{a}_cos", np.cos(df[a])),
    
    # New unary operations
    "tan": lambda df, a: (
        f"{a}_tan", 
        np.where(
            np.abs(np.tan(df[a])) < 1e10,  # Prevent extreme tangent values
            np.tan(df[a]),
            np.nan
        )
    ),
    
    "sigmoid": lambda df, a: (
        f"{a}_sigmoid", 
        np.where(
            np.abs(df[a]) <= 50,
            1 / (1 + np.exp(-df[a])),
            np.where(df[a] > 50, 1.0, 0.0)  # Handle extreme values
        )
    ),
    
    "tanh": lambda df, a: (f"{a}_tanh", np.tanh(df[a])),
    
    "reciprocal_sqrt": lambda df, a: (
        f"{a}_reciprocal_sqrt", 
        np.where(
            df[a] > 1e-15,
            np.where(
                df[a] < 1e15,  # Prevent 1/sqrt of very small numbers
                1 / np.sqrt(df[a]),
                np.nan
            ),
            np.nan
        )
    ),
    
    "cbrt": lambda df, a: (f"{a}_cbrt", np.cbrt(df[a])),  # Cube root
    
    "floor": lambda df, a: (f"{a}_floor", np.floor(df[a])),
    
    "ceil": lambda df, a: (f"{a}_ceil", np.ceil(df[a])),
    
    "round": lambda df, a: (f"{a}_round", np.round(df[a])),
    
    "sign": lambda df, a: (f"{a}_sign", np.sign(df[a])),
    
    "arcsin": lambda df, a: (
        f"{a}_arcsin",
        np.where((df[a] >= -1) & (df[a] <= 1), np.arcsin(df[a]), np.nan)
    ),

    "arccos": lambda df, a: (
        f"{a}_arccos",
        np.where((df[a] >= -1) & (df[a] <= 1), np.arccos(df[a]), np.nan)
    ),
    
    "arctan": lambda df, a: (f"{a}_arctan", np.arctan(df[a])),

    # Binary ops
    "add": lambda df, a, b: (
        f"{a}_add_{b}", 
        df[a] + df[b]
    ),
    
    "sub": lambda df, a, b: (
        f"{a}_sub_{b}", 
        df[a] - df[b]
    ),
    
    "absdiff": lambda df, a, b: (
        f"{a}_absdiff_{b}", 
        np.abs(df[a] - df[b])
    ),

    "diff_ratio": lambda df, a, b: (
        f"{a}_diff_ratio_{b}", 
        np.where(
            (df[a] + df[b]) != 0,
            (df[a] - df[b]) / (np.abs(df[a] + df[b]) + 1e-15),
            np.nan
        )
    ),
    
    "mul": lambda df, a, b: (
        f"{a}_mul_{b}", 
        np.where(
            (np.abs(df[a] * df[b])) < 1e15,  # Check if result would be too large
            df[a] * df[b],
            np.nan  # Replace potential overflow with nan
        )
    ),
    
    "div": lambda df, a, b: (
        f"{a}_div_{b}", 
        np.where(
            np.abs(df[b]) > 1e-15,  # More conservative epsilon for division
            np.where(
                np.abs(df[a] / df[b]) < 1e15,  # Check if result would be too large
                df[a] / df[b],
                np.nan  # Replace extreme values with nan
            ),
            np.nan
        )
    ),
    
    "logmul": lambda df, a, b: (
        f"{a}_logmul_{b}", 
        np.where(
            (df[a] > 0) & (df[b] > 0) & (df[a] * df[b] < 1e15),
            np.log1p(np.abs(df[a] * df[b])),
            np.nan
        )
    ),
    
    # Fixed power operation
    "pow": lambda df, a, b: (
        f"{a}_pow_{b}", 
        _safe_power(df[a], df[b])
    ),
    
    "mod": lambda df, a, b: (
        f"{a}_mod_{b}", 
        np.where(
            np.abs(df[b]) > 1e-15,
            np.mod(df[a], df[b]),
            np.nan
        )
    ),
    
    "max": lambda df, a, b: (f"{a}_max_{b}", np.maximum(df[a], df[b])),
    
    "min": lambda df, a, b: (f"{a}_min_{b}", np.minimum(df[a], df[b])),
    
    "geometric_mean": lambda df, a, b: (
        f"{a}_geometric_mean_{b}", 
        np.where(
            (df[a] > 0) & (df[b] > 0),
            np.sqrt(df[a] * df[b]),
            np.nan
        )
    ),
    
    "harmonic_mean": lambda df, a, b: (
        f"{a}_harmonic_mean_{b}", 
        np.where(
            (df[a] > 0) & (df[b] > 0),
            2 / (1/df[a] + 1/df[b]),
            np.nan
        )
    ),
    
    "relative_diff": lambda df, a, b: (
        f"{a}_relative_diff_{b}",
        np.where(
            np.abs(df[b]) > 1e-15,
            (df[a] - df[b]) / np.abs(df[b]),
            np.nan
        )
    ),

    "log_ratio": lambda df, a, b: (
        f"{a}_log_ratio_{b}",
        np.where(
            (df[a] > 0) & (df[b] > 0),
            np.log(df[a] / df[b]),
            np.nan
        )
    ),

    "angle_between": lambda df, a, b: (
        f"{a}_angle_between_{b}", 
        np.arctan2(df[b], df[a])
    ),
    
    "weighted_sum": lambda df, a, b: (
        f"{a}_weighted_sum_{b}", 
        0.7 * df[a] + 0.3 * df[b]
    ),
    
    "weighted_diff": lambda df, a, b: (
        f"{a}_weighted_diff_{b}", 
        0.7 * df[a] - 0.3 * df[b]
    ),

}

# --- Categorical Operations Lambdas ---
CAT_OPS_LAMBDAS: Dict[str, Callable[..., Tuple[str, pd.Series]]] = {
    # --- Unary operations on categorical columns ---
    # Handled with custom encoders during pipeline (cv) to avoid data leakage
    "target": None,
    "freq": None,
    "count": None,

    # --- Binary operations involving categorical columns ---
    # Concatenate one categorical column with another (categorical or not)
    "concat": lambda df, cat_col1, cat_col2: (
        f"{cat_col1}_concat_{cat_col2}", 
        df[cat_col1].astype(str) + "_" + df[cat_col2].astype(str)
    ),
}

ALL_OPS_LAMBDAS = NUM_OPS_LAMBDAS.copy()
ALL_OPS_LAMBDAS.update(CAT_OPS_LAMBDAS)

# --- Group-By Aggregation Operations ---
# These take (df, cat_col, num_col) — a different signature from row-wise ops.
# They must be computed inside CV folds (pipeline_required=True) to prevent leakage.
AGG_OPS = {
    "groupby_mean": lambda df, cat_col, num_col:
        df.groupby(cat_col)[num_col].transform("mean"),
    "groupby_std": lambda df, cat_col, num_col:
        df.groupby(cat_col)[num_col].transform("std"),
    "groupby_median": lambda df, cat_col, num_col:
        df.groupby(cat_col)[num_col].transform("median"),
    "groupby_min": lambda df, cat_col, num_col:
        df.groupby(cat_col)[num_col].transform("min"),
    "groupby_max": lambda df, cat_col, num_col:
        df.groupby(cat_col)[num_col].transform("max"),
    "groupby_count": lambda df, cat_col, num_col:
        df.groupby(cat_col)[num_col].transform("count"),
    "groupby_rank": lambda df, cat_col, num_col:
        df.groupby(cat_col)[num_col].transform("rank", pct=True),
    "groupby_zscore": lambda df, cat_col, num_col:
        (df[num_col] - df.groupby(cat_col)[num_col].transform("mean")) /
        (df.groupby(cat_col)[num_col].transform("std") + 1e-8),
}

# Register aggregation ops in the OPS dict
OPS["agg"] = {"binary": list(AGG_OPS.keys())}
# --- Temporal / Lag Operations ---
# These take (df, col, id_col, time_col) — a 4-arg signature.
# They must be computed inside CV folds (pipeline_required=True) after sorting by time.
# Only enabled when time_col is provided to FeatureGenerator.
#
# Window sizes are parameterized. The default [1, 4] produces the 6 ops recommended
# by the upgrade report. Users can override via FeatureGenerator(temporal_windows=...)
# to explore domain-specific windows (e.g., [1, 5, 20, 60] for daily financial data).

DEFAULT_TEMPORAL_WINDOWS = [1, 4]

def _make_lag(w):
    return lambda df, col, id_col, time_col: (
        df.sort_values(time_col).groupby(id_col)[col].shift(w)
    )

def _make_rolling_mean(w):
    return lambda df, col, id_col, time_col: (
        df.sort_values(time_col).groupby(id_col)[col].transform(
            lambda x: x.rolling(w, min_periods=1).mean())
    )

def _make_rolling_std(w):
    return lambda df, col, id_col, time_col: (
        df.sort_values(time_col).groupby(id_col)[col].transform(
            lambda x: x.rolling(w, min_periods=1).std())
    )

def _make_momentum(w):
    return lambda df, col, id_col, time_col: (
        df[col] - df.sort_values(time_col).groupby(id_col)[col].shift(w)
    )

def _make_pct_change(w):
    return lambda df, col, id_col, time_col: (
        df.sort_values(time_col).groupby(id_col)[col].pct_change(w)
    )


def build_temporal_ops(windows=None):
    """Build TEMPORAL_OPS dict for given window sizes.
    
    Default [1, 4] produces exactly the 6 ops from the upgrade report:
      lag_1, lag_4, rolling_mean_4, rolling_std_4, momentum_4, pct_change_1
    
    Custom windows (e.g., [1, 5, 20]) let the genetic search explore
    domain-specific lookback periods.
    """
    if windows is None:
        windows = DEFAULT_TEMPORAL_WINDOWS
    ops = {}
    for w in windows:
        ops[f"lag_{w}"] = _make_lag(w)
        if w >= 2:  # rolling/momentum/pct_change need window >= 2
            ops[f"rolling_mean_{w}"] = _make_rolling_mean(w)
            ops[f"rolling_std_{w}"] = _make_rolling_std(w)
            ops[f"momentum_{w}"] = _make_momentum(w)
            ops[f"pct_change_{w}"] = _make_pct_change(w)
        else:  # w == 1: only lag and pct_change make sense
            ops[f"pct_change_{w}"] = _make_pct_change(w)
    return ops


# Default ops: exactly the 6 from the report
TEMPORAL_OPS = build_temporal_ops(DEFAULT_TEMPORAL_WINDOWS)

# Register temporal ops in the OPS dict
OPS["temporal"] = {"unary": list(TEMPORAL_OPS.keys())}

# Utility function to clean dataframe after feature engineering
def clean_dataframe_for_xgboost(df):
    """
    Clean dataframe to ensure compatibility with XGBoost by replacing inf values
    """
    # Replace inf and -inf with nan, then fill nan with 0
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.fillna(0)
    return df_clean