"""
Real-dataset integration test using the NYC Taxis dataset (seaborn built-in).
6,433 real taxi trips from March 2019.

Why this dataset exercises all 6 enhancements:
  - Categorical columns (pickup_borough, dropoff_borough, color, payment)
    -> GroupBy agg ops: "avg fare in this pickup borough" etc.
  - Real timestamps -> sequential entity/time structure for temporal/lag ops
  - 6k rows triggers meta-val split (>2000 threshold)
  - Multi-generation search exercises caching + proxy eval
  - Multiple generated features trigger regularized post-selection

Target: tip (regression)
"""
import sys
import time
import numpy as np
import pandas as pd
np.NaN = np.nan

import seaborn as sns
from tabularaml.generate.features import FeatureGenerator

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"
checks = {}

def check(label, ok, detail="", warn_only=False):
    if ok:
        tag = PASS
    elif warn_only:
        tag = WARN
        ok  = True
    else:
        tag = FAIL
    sfx = f" — {detail}" if detail else ""
    print(f"  [{tag}] {label}{sfx}")
    checks[label] = ok
    return ok

# ---------------------------------------------------------------------------
# Load and prepare the dataset
# ---------------------------------------------------------------------------
print("Loading NYC Taxis dataset (seaborn)...")
df = sns.load_dataset("taxis")

# Drop rows with missing categoricals or target
df = df.dropna(subset=["tip", "payment", "pickup_zone", "dropoff_zone",
                        "pickup_borough", "dropoff_borough"])
df = df.reset_index(drop=True)

# Feature engineering from timestamps (before search)
df["hour"]        = df["pickup"].dt.hour
df["day_of_week"] = df["pickup"].dt.dayofweek
df["trip_duration_min"] = (df["dropoff"] - df["pickup"]).dt.total_seconds() / 60

# Target
y = df["tip"].copy()

# Feature matrix — drop leaky/non-feature columns
X = df.drop(columns=["pickup", "dropoff", "tip",
                      "total"])   # total = fare+tip+tolls, leaky

# For temporal structure: sort by pickup time within each borough,
# use borough-as-entity + sequential rank-within-borough as time step
df_sorted = df.sort_values("pickup")
X["entity_id"] = df_sorted["pickup_borough"].map(
    {b: i for i, b in enumerate(df_sorted["pickup_borough"].unique())}
).values
X["time_step"] = df_sorted.groupby("pickup_borough").cumcount().values
y = y.loc[X.index].reset_index(drop=True)
X = X.reset_index(drop=True)

n = len(X)
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if X[c].dtype in [np.float64, np.int64]]

print(f"  Shape        : {X.shape}")
print(f"  Numeric cols : {num_cols}")
print(f"  Categorical  : {cat_cols}")
print(f"  Entity/time  : entity_id ({X['entity_id'].nunique()} boroughs), "
      f"time_step (0–{X['time_step'].max()})")
print(f"  Target       : tip  mean={y.mean():.2f}  std={y.std():.2f}\n")

# ---------------------------------------------------------------------------
# Monkey-patch to capture scorer info before it is reset to None post-search
# ---------------------------------------------------------------------------
_info = {}
_orig_search = FeatureGenerator.search
def _patched_search(self, Xa, ya, **kw):
    result = _orig_search(self, Xa, ya, **kw)
    _info['pct_gain']    = getattr(self, 'pct_gain', 0.0)
    _info['gain']        = getattr(self, 'gain', 0.0)
    _info['initial_val'] = getattr(self, 'initial_val_metric', None)
    _info['final_val']   = getattr(self, 'final_metric', None)
    _info['scorer_name'] = getattr(self, 'scorer', None)
    return result
FeatureGenerator.search = _patched_search

# ---------------------------------------------------------------------------
# Run FeatureGenerator
# ---------------------------------------------------------------------------
print("=" * 60)
print("Running FeatureGenerator (mode=lite, 10-min budget)...")
print("=" * 60)

fg = FeatureGenerator(
    mode="lite",
    time_budget=600,          # 10 minutes
    n_children=30,            # 30 children/gen — faster than default 90
    use_proxy_evaluation=True,
    proxy_top_pct=0.20,
    meta_validation_frac=0.15,
    rotate_cv_folds=True,
    fold_rotation_period=4,
    final_selection=True,
    cache_size_mb=1000,
    time_col="time_step",
    id_col="entity_id",
)

t0 = time.time()
X_new, pipeline, generation, interactions = fg.search(X.copy(), y.copy())
elapsed = time.time() - t0

# ---------------------------------------------------------------------------
# Collect stats
# ---------------------------------------------------------------------------
init_cols  = set(fg.initial_features)
new_cols   = [c for c in X_new.columns if c not in init_cols]
agg_cols   = [c for c in new_cols if c.startswith("groupby_")]
temp_cols  = [c for c in new_cols
              if any(c.startswith(p)
                     for p in ["lag_", "rolling_", "momentum_", "pct_change_"])]
other_cols = [c for c in new_cols if c not in agg_cols and c not in temp_cols]

pct_gain  = _info.get('pct_gain', 0.0) * 100
init_val  = _info.get('initial_val')
final_val = _info.get('final_val')
cache     = fg._feature_cache

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
print(f"\nTiming         : {elapsed:.0f}s")
print(f"Initial feats  : {len(init_cols)}")
print(f"Generated feats: {len(new_cols)}  "
      f"(groupby={len(agg_cols)}, temporal={len(temp_cols)}, other={len(other_cols)})")
if init_val is not None:
    print(f"Baseline metric: {init_val:.5f}")
if final_val is not None:
    print(f"Final metric   : {final_val:.5f}")
print(f"Gain           : {pct_gain:+.3f}%")
print(f"Cache          : {cache.hits} hits / {cache.misses} misses "
      f"/ hit_rate={cache.hit_rate:.3f}\n")

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

# E1: Proxy evaluation
check("1. Proxy eval: OOF preds computed",
      hasattr(fg, '_current_oof_preds') and fg._current_oof_preds is not None,
      f"oof_shape={getattr(fg,'_current_oof_preds', np.array([])).shape}")

check("1. Proxy eval: candidates were screened (cache populated)",
      cache.misses > 0,
      f"cache_misses={cache.misses}")

# E2: GroupBy ops
from tabularaml.generate.ops import AGG_OPS, OPS
check("2. AGG_OPS: 8 ops registered in OPS dict",
      len(AGG_OPS) == 8 and "agg" in OPS)

check("2. GroupBy features accepted by search",
      len(agg_cols) > 0,
      f"accepted: {agg_cols}",
      warn_only=True)

# E3: Meta-val split
check("3. Meta-val split applied (search rows < total rows)",
      fg.n_samples < n,
      f"search={fg.n_samples} / total={n}")

check("3. CV fold rotation ran (cv is no longer plain int)",
      not isinstance(fg.cv, int),
      f"cv_type={type(fg.cv).__name__}",
      warn_only=True)

# E4: Cache
check("4. Feature cache had hits across generations",
      cache.hits > 0,
      f"hits={cache.hits}, rate={cache.hit_rate:.3f}")

# E5: Regularized post-selection ran without crash
check("5. Search completed with final_selection=True",
      len(X_new.columns) > 0,
      f"output_cols={len(X_new.columns)}")

# E6: Temporal ops
from tabularaml.generate.ops import TEMPORAL_OPS
check("6. TEMPORAL_OPS registered in OPS dict",
      "temporal" in OPS and len(TEMPORAL_OPS) >= 6)

check("6. Temporal features accepted by search",
      len(temp_cols) > 0,
      f"accepted: {temp_cols}",
      warn_only=True)

# Overall
check("Overall: positive gain over baseline",
      pct_gain > 0,
      f"{pct_gain:+.3f}%")

# ---------------------------------------------------------------------------
# Print generated features
# ---------------------------------------------------------------------------
if new_cols:
    print(f"\nGenerated features ({len(new_cols)}):")
    for c in sorted(new_cols):
        tag = " [groupby]"  if c in agg_cols  else \
              " [temporal]" if c in temp_cols  else ""
        print(f"    {c}{tag}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
n_pass  = sum(checks.values())
n_total = len(checks)
print("\n" + "=" * 60)
print(f"Result: {n_pass}/{n_total} checks passed")
if n_pass < n_total:
    print("Failed:")
    for name, ok in checks.items():
        if not ok:
            print(f"  - {name}")
print("=" * 60)
sys.exit(0 if all(checks.values()) else 1)
