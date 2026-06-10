import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Dict, Callable

warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Configuration ---

DEFAULT_TEMPORAL_WINDOWS = [1, 4, 7, 14, 30]

# --- Registry: selected op names per category ---

OPS = {
    "num": {
        "unary": [
            "neg", "abs", "square", "sqrt", "log", "log1p", "exp", "inv",
            "cube", "sin", "cos", "tan", "sigmoid", "tanh", "reciprocal_sqrt",
            "cbrt", "floor", "ceil", "round", "sign", "arcsin", "arccos", "arctan",
        ],
        "binary": [
            "add", "absdiff", "mul", "div", "logmul", "diff_ratio",
            "sub", "pow", "mod", "max", "min", "geometric_mean", 
            "harmonic_mean", "relative_diff", "log_ratio", "angle_between", 
            "weighted_sum", "weighted_diff",
        ],
    },
    "cat": {
        "unary": ["target", "freq", "count"],
        "binary": ["concat"],
    },
    "agg": {
        "binary": [
            "groupby_mean", "groupby_std", "groupby_median",
            "groupby_min", "groupby_max", "groupby_count",
            "groupby_rank", "groupby_zscore",
        ],
    },
    "temporal": {
        "unary": [
            f"{op}_{w}"
            for w in DEFAULT_TEMPORAL_WINDOWS
            for op in (
                ["lag", "pct_change"] if w < 2 else
                ["lag", "rolling_mean", "rolling_std", "momentum", "pct_change"]
            )
        ],
    },
    # Global (whole-column) transforms: fitted on the train fold inside the
    # pipeline (rank maps / bin edges / winsor bounds), applied leakage-free
    # to validation and test rows.
    "global": {
        "unary": ["rank_pct", "qbin", "zscore_winsor", "log_rank"],
    },
}

GLOBAL_OPS = set(OPS["global"]["unary"])

# --- Helpers ---

def _safe_power(base_series, exp_series):
    base, exp = base_series.astype(float), exp_series.astype(float)
    result = pd.Series(np.nan, index=base.index)
    
    safe_mask = (
        np.isfinite(base) & np.isfinite(exp) &
        (np.abs(exp) < 10) & (np.abs(base) < 1e10) &
        ((base >= 0) | (np.abs(exp - np.round(exp)) < 1e-10)) &
        ~((base == 0) & (exp < 0))
    )
    
    if safe_mask.any():
        try:
            res = np.power(base[safe_mask], exp[safe_mask])
            valid = np.isfinite(res) & (np.abs(res) < 1e15)
            result.loc[safe_mask] = np.where(valid, res, np.nan)
        except Exception:
            pass
    return result

# --- Numeric Operations ---

NUM_OPS: Dict[str, Callable[..., Tuple[str, pd.Series]]] = {
    "neg": lambda df, a: (f"{a}_neg", -df[a]),
    "abs": lambda df, a: (f"{a}_abs", df[a].abs()),
    "square": lambda df, a: (f"{a}_square", np.where(np.abs(df[a]) < 1e7, df[a]**2, np.nan)),
    "cube":   lambda df, a: (f"{a}_cube",   np.where(np.abs(df[a]) < 1e7, df[a]**3, np.nan)),
    "sqrt":   lambda df, a: (f"{a}_sqrt",   np.sqrt(np.where(df[a] >= 0, df[a], np.nan))),
    "log":    lambda df, a: (f"{a}_log",    np.log(np.where(df[a] > 0, df[a], np.nan))),
    "log1p":  lambda df, a: (f"{a}_log1p",  np.where(df[a] >= 0, np.log1p(df[a]), np.nan)),
    "exp":    lambda df, a: (f"{a}_exp",    np.where(np.abs(df[a]) <= 50, np.exp(df[a]), np.nan)),
    "inv":    lambda df, a: (
        f"{a}_inv", 
        np.where(np.abs(df[a]) > 1e-15, np.where(np.abs(1/df[a]) < 1e15, 1/df[a], np.nan), np.nan)
    ),
    "sin":    lambda df, a: (f"{a}_sin", np.sin(df[a])),
    "cos":    lambda df, a: (f"{a}_cos", np.cos(df[a])),
    "tan":    lambda df, a: (f"{a}_tan", np.where(np.abs(np.tan(df[a])) < 1e10, np.tan(df[a]), np.nan)),
    "sigmoid": lambda df, a: (
        f"{a}_sigmoid",
        np.where(np.abs(df[a]) <= 50, 1/(1 + np.exp(-df[a])), np.where(df[a] > 50, 1.0, 0.0))
    ),
    "tanh":   lambda df, a: (f"{a}_tanh", np.tanh(df[a])),
    "reciprocal_sqrt": lambda df, a: (
        f"{a}_reciprocal_sqrt",
        np.where((df[a] > 1e-15) & (df[a] < 1e15), 1/np.sqrt(df[a]), np.nan)
    ),
    "cbrt":   lambda df, a: (f"{a}_cbrt",  np.cbrt(df[a])),
    "floor":  lambda df, a: (f"{a}_floor", np.floor(df[a])),
    "ceil":   lambda df, a: (f"{a}_ceil",  np.ceil(df[a])),
    "round":  lambda df, a: (f"{a}_round", np.round(df[a])),
    "sign":   lambda df, a: (f"{a}_sign",  np.sign(df[a])),
    "arcsin": lambda df, a: (f"{a}_arcsin", np.where(df[a].between(-1, 1), np.arcsin(df[a]), np.nan)),
    "arccos": lambda df, a: (f"{a}_arccos", np.where(df[a].between(-1, 1), np.arccos(df[a]), np.nan)),
    "arctan": lambda df, a: (f"{a}_arctan", np.arctan(df[a])),

    # Binary
    "add":     lambda df, a, b: (f"{a}_add_{b}", df[a] + df[b]),
    "sub":     lambda df, a, b: (f"{a}_sub_{b}", df[a] - df[b]),
    "absdiff": lambda df, a, b: (f"{a}_absdiff_{b}", np.abs(df[a] - df[b])),
    "mul":     lambda df, a, b: (f"{a}_mul_{b}", np.where(np.abs(df[a]*df[b]) < 1e15, df[a]*df[b], np.nan)),
    "pow":     lambda df, a, b: (f"{a}_pow_{b}", _safe_power(df[a], df[b])),
    "mod":     lambda df, a, b: (f"{a}_mod_{b}", np.where(np.abs(df[b]) > 1e-15, np.mod(df[a], df[b]), np.nan)),
    "max":     lambda df, a, b: (f"{a}_max_{b}", np.maximum(df[a], df[b])),
    "min":     lambda df, a, b: (f"{a}_min_{b}", np.minimum(df[a], df[b])),
    "diff_ratio": lambda df, a, b: (
        f"{a}_diff_ratio_{b}",
        np.where((df[a] + df[b]) != 0, (df[a] - df[b]) / (np.abs(df[a] + df[b]) + 1e-15), np.nan)
    ),
    "div": lambda df, a, b: (
        f"{a}_div_{b}",
        np.where(np.abs(df[b]) > 1e-15, np.where(np.abs(df[a]/df[b]) < 1e15, df[a]/df[b], np.nan), np.nan)
    ),
    "logmul": lambda df, a, b: (
        f"{a}_logmul_{b}",
        np.where((df[a] > 0) & (df[b] > 0) & (df[a]*df[b] < 1e15), np.log1p(np.abs(df[a]*df[b])), np.nan)
    ),
    "geometric_mean": lambda df, a, b: (
        f"{a}_geometric_mean_{b}",
        np.where((df[a] > 0) & (df[b] > 0), np.sqrt(df[a]*df[b]), np.nan)
    ),
    "harmonic_mean": lambda df, a, b: (
        f"{a}_harmonic_mean_{b}",
        np.where((df[a] > 0) & (df[b] > 0), 2 / (1/df[a] + 1/df[b]), np.nan)
    ),
    "relative_diff": lambda df, a, b: (
        f"{a}_relative_diff_{b}",
        np.where(np.abs(df[b]) > 1e-15, (df[a] - df[b]) / np.abs(df[b]), np.nan)
    ),
    "log_ratio": lambda df, a, b: (
        f"{a}_log_ratio_{b}",
        np.where((df[a] > 0) & (df[b] > 0), np.log(df[a]/df[b]), np.nan)
    ),
    "angle_between": lambda df, a, b: (f"{a}_angle_between_{b}", np.arctan2(df[b], df[a])),
    "weighted_sum":  lambda df, a, b: (f"{a}_weighted_sum_{b}",  0.7 * df[a] + 0.3 * df[b]),
    "weighted_diff": lambda df, a, b: (f"{a}_weighted_diff_{b}", 0.7 * df[a] - 0.3 * df[b]),
}

# --- Categorical & Aggregation ---

CAT_OPS: Dict[str, Callable] = {
    "target": None, "freq": None, "count": None,
    "concat": lambda df, a, b: (f"{a}_concat_{b}", df[a].astype(str) + "_" + df[b].astype(str)),
}

AGG_OPS: Dict[str, Callable] = {
    "groupby_mean":   lambda df, c, n: df.groupby(c)[n].transform("mean"),
    "groupby_std":    lambda df, c, n: df.groupby(c)[n].transform("std"),
    "groupby_median": lambda df, c, n: df.groupby(c)[n].transform("median"),
    "groupby_min":    lambda df, c, n: df.groupby(c)[n].transform("min"),
    "groupby_max":    lambda df, c, n: df.groupby(c)[n].transform("max"),
    "groupby_count":  lambda df, c, n: df.groupby(c)[n].transform("count"),
    "groupby_rank":   lambda df, c, n: df.groupby(c)[n].transform("rank", pct=True),
    "groupby_zscore": lambda df, c, n: (
        (df[n] - df.groupby(c)[n].transform("mean")) / (df.groupby(c)[n].transform("std") + 1e-8)
    ),
}

# --- Temporal operations ---

def _make_lag(w): 
    return lambda df, c, i, t: df.sort_values(t).groupby(i)[c].shift(w)

def _make_rolling_mean(w): 
    return lambda df, c, i, t: df.sort_values(t).groupby(i)[c].transform(lambda x: x.rolling(w, min_periods=1).mean())

def _make_rolling_std(w): 
    return lambda df, c, i, t: df.sort_values(t).groupby(i)[c].transform(lambda x: x.rolling(w, min_periods=1).std())

def _make_momentum(w): 
    return lambda df, c, i, t: df[c] - df.sort_values(t).groupby(i)[c].shift(w)

def _make_pct_change(w): 
    return lambda df, c, i, t: df.sort_values(t).groupby(i)[c].pct_change(w)

def build_temporal_ops(windows=None) -> Dict[str, Callable]:
    windows = windows or DEFAULT_TEMPORAL_WINDOWS
    ops = {}
    for w in windows:
        ops[f"lag_{w}"] = _make_lag(w)
        ops[f"pct_change_{w}"] = _make_pct_change(w)
        if w >= 2:
            ops[f"rolling_mean_{w}"] = _make_rolling_mean(w)
            ops[f"rolling_std_{w}"]  = _make_rolling_std(w)
            ops[f"momentum_{w}"]     = _make_momentum(w)
    return ops

TEMPORAL_OPS = build_temporal_ops(DEFAULT_TEMPORAL_WINDOWS)

# --- Exports & Utilities ---

ALL_OPS = {**NUM_OPS, **CAT_OPS, **AGG_OPS, **TEMPORAL_OPS}
NUM_OPS_LAMBDAS, CAT_OPS_LAMBDAS, ALL_OPS_LAMBDAS = NUM_OPS, CAT_OPS, ALL_OPS

def clean_dataframe_for_xgboost(df):
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)