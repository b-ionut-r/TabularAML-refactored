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
                  "log", "exp", "inv",
                  "cube", "sin", "cos", "tan",
                  "sigmoid", "tanh", "reciprocal_sqrt",
                  "log10", "log2", "cbrt", "floor", "ceil",
                  "round", "sign", "arcsin", "arccos", "arctan"
                  ],
        "binary": ["add", "absdiff", "mul", 
                   "div", "logmul", "diff_ratio",
                   "sub", "pow", "mod", "max", "min",
                   "mean", "geometric_mean", "harmonic_mean",
                   "relative_diff", "percent_change", "ratio",
                   "distance_euclidean", "distance_manhattan",
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
    
    "log10": lambda df, a: (
        f"{a}_log10", 
        np.log10(np.where(df[a] > 0, df[a], np.nan))
    ),
    
    "log2": lambda df, a: (
        f"{a}_log2", 
        np.log2(np.where(df[a] > 0, df[a], np.nan))
    ),
    
    "cbrt": lambda df, a: (f"{a}_cbrt", np.cbrt(df[a])),  # Cube root
    
    "floor": lambda df, a: (f"{a}_floor", np.floor(df[a])),
    
    "ceil": lambda df, a: (f"{a}_ceil", np.ceil(df[a])),
    
    "round": lambda df, a: (f"{a}_round", np.round(df[a])),
    
    "sign": lambda df, a: (f"{a}_sign", np.sign(df[a])),
    
    "arcsin": lambda df, a: (
        f"{a}_arcsin", 
        np.arcsin(np.clip(df[a], -1, 1))  # Clip to valid range for arcsin
    ),
    
    "arccos": lambda df, a: (
        f"{a}_arccos", 
        np.arccos(np.clip(df[a], -1, 1))  # Clip to valid range for arccos
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
    
    "mean": lambda df, a, b: (f"{a}_mean_{b}", (df[a] + df[b]) / 2),
    
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
    
    "percent_change": lambda df, a, b: (
        f"{a}_percent_change_{b}", 
        np.where(
            np.abs(df[b]) > 1e-15,
            100 * (df[a] - df[b]) / np.abs(df[b]),
            np.nan
        )
    ),
    
    "ratio": lambda df, a, b: (
        f"{a}_ratio_{b}", 
        np.where(
            np.abs(df[b]) > 1e-15,
            np.where(
                np.abs(df[a] / df[b]) < 1e15,
                df[a] / df[b],
                np.nan
            ),
            np.nan
        )
    ),
    
    # Spatial operations (useful for lat/lon data)
    "distance_euclidean": lambda df, a, b: (
        f"{a}_distance_euclidean_{b}", 
        np.sqrt(np.where(
            np.isfinite(df[a]) & np.isfinite(df[b]),
            (df[a] - df[b]) ** 2,
            np.nan
        ))
    ),
    
    "distance_manhattan": lambda df, a, b: (
        f"{a}_distance_manhattan_{b}", 
        np.abs(df[a] - df[b])
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

# Utility function to clean dataframe after feature engineering
def clean_dataframe_for_xgboost(df):
    """
    Clean dataframe to ensure compatibility with XGBoost by replacing inf values
    """
    # Replace inf and -inf with nan, then fill nan with 0
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.fillna(0)
    return df_clean