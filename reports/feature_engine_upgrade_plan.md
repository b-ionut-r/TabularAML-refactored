# SOTA Feature Engineering Upgrade Plan

## Context

6 highest-impact enhancements to the genetic feature engine in `tabularaml/generate/`. All target the genetic search core with minimal architectural changes. Each change documented in a unified `CHANGES.md` at project root.

## Critical Files

| File | Role |
|---|---|
| `tabularaml/generate/features.py` (2049 lines) | `FeatureGenerator`, `Feature`, `Interaction`, `ImprovedAdaptiveController` |
| `tabularaml/generate/ops.py` (375 lines) | `OPS`, `NUM_OPS_LAMBDAS`, `CAT_OPS_LAMBDAS`, `ALL_OPS_LAMBDAS` |
| `tabularaml/preprocessing/encoders.py` (158 lines) | `CategoricalEncoder` |
| `tabularaml/configs/feature_gen.py` (58 lines) | `PRESET_PARAMS` |

---

## 1. FeatureBoost Proxy Evaluation (HIGH impact — 50x throughput)

**Why**: This single change would increase candidate throughput by ~50x, enabling exploration of the search space that currently requires days in hours. OpenFE's core innovation demonstrates that residual-based scoring correlates highly with full-CV evaluation while being orders of magnitude faster.

**Files**: `features.py`, `requirements.hf.txt`

**Where to modify**: `features.py` → `_select_elites()` (line 1030) and the evaluation calls within `search()`.

- Add `_train_base_model_and_get_residuals(X, y, cv)` — trains LightGBM on current features, returns OOF predictions. Recomputed each generation.
- Add `_featureboost_score(candidate_values, y, oof_preds, cv)` — trains a tiny single-feature LightGBM with `init_score=oof_preds` (OpenFE's core trick). `num_leaves=16, n_estimators=50`. Falls back to XGBoost if LGB unavailable.
- Modify `_select_elites()` to two-phase:
  - **Phase 1**: Score all ranked candidates via `_featureboost_score`. Keep top ~15%.
  - **Phase 2**: Full-CV `_eval_baseline` only on survivors.
- Pipeline-required features (target/count/freq encoding) skip proxy and go directly to full CV.
- New params: `use_proxy_evaluation: bool = True`, `proxy_top_pct: float = 0.15`
- Add `lightgbm>=4.0.0` to `requirements.hf.txt`

```python
# In FeatureGenerator, add a method:
def _train_base_model_and_get_residuals(self, X, y, cv):
    """Train base model on current features, return OOF predictions."""
    import lightgbm as lgb
    oof_preds = np.zeros(len(y))
    for train_idx, val_idx in cv.split(X, y, groups=self._groups):
        dtrain = lgb.Dataset(X.iloc[train_idx], y.iloc[train_idx])
        model = lgb.train({"objective": self._objective, "verbosity": -1,
                           "n_estimators": 200, "learning_rate": 0.1},
                          dtrain, num_boost_round=200)
        oof_preds[val_idx] = model.predict(X.iloc[val_idx])
    return oof_preds

def _featureboost_score(self, candidate_values, y, oof_preds, cv):
    """Score a single candidate feature via residual-based incremental training."""
    import lightgbm as lgb
    scores = []
    for train_idx, val_idx in cv.split(candidate_values, y, groups=self._groups):
        dtrain = lgb.Dataset(
            candidate_values.iloc[train_idx].values.reshape(-1, 1),
            y.iloc[train_idx],
            init_score=oof_preds[train_idx]  # KEY: base model predictions as offset
        )
        dval = lgb.Dataset(
            candidate_values.iloc[val_idx].values.reshape(-1, 1),
            y.iloc[val_idx],
            init_score=oof_preds[val_idx],
            reference=dtrain
        )
        model = lgb.train(
            {"objective": self._objective, "num_leaves": 16,
             "n_estimators": 50, "verbosity": -1},
            dtrain, valid_sets=[dval],
            callbacks=[lgb.early_stopping(10, verbose=False)]
        )
        # Score improvement = base_loss - (base_loss + residual_model)
        base_score = self._metric(y.iloc[val_idx], oof_preds[val_idx])
        new_score = self._metric(
            y.iloc[val_idx],
            oof_preds[val_idx] + model.predict(candidate_values.iloc[val_idx].values.reshape(-1,1))
        )
        scores.append(new_score - base_score)
    return np.mean(scores)
```

**Two-phase search integration** in `search()`:

```python
# Phase 1: FeatureBoost screening (fast)
oof_preds = self._train_base_model_and_get_residuals(X_current, y, cv)
fb_scores = {c: self._featureboost_score(c.values, y, oof_preds, cv) 
             for c in all_candidates}
top_candidates = sorted(fb_scores, key=fb_scores.get, reverse=True)[:n_children // 10]

# Phase 2: Full CV validation (expensive, only top candidates)  
accepted = self._select_elites(top_candidates, X_current, y, ...)
```

---

## 2. Group-By Aggregation Operators (HIGH impact — most powerful missing feature class)

**Why**: GroupBy features are the single most powerful feature class in Kaggle winning solutions. For panel data, features like "stock's feature value relative to industry mean" capture cross-sectional structure that row-wise ops cannot. OpenFE includes 6 GroupBy operators; TabularAML has zero.

**Files**: `ops.py`, `features.py`, `encoders.py`

**Where to modify**: `ops.py` → add to `OPS` dict and create `AGG_OPS`; `features.py` → `_sample_children_with_creativity` to generate groupby candidates; `encoders.py` → new `GroupByEncoder` for pipeline-required aggregations.

**In `ops.py`**:
- Add `AGG_OPS` dict with 8 ops
- Add `OPS["agg"] = {"binary": list(AGG_OPS.keys())}` to the `OPS` dict

```python
# In ops.py, add:
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
```

**In `encoders.py`**:
- Add `GroupByEncoder(BaseEstimator, TransformerMixin)` — leakage-safe, fitted only on train fold inside CV.

```python
# In encoders.py, add GroupByEncoder:
class GroupByEncoder:
    """Fit-transform group-by statistics within CV folds to prevent leakage."""
    def __init__(self, cat_col, num_col, agg_func):
        self.cat_col = cat_col
        self.num_col = num_col
        self.agg_func = agg_func
        self.mapping_ = None
        
    def fit(self, X, y=None):
        self.mapping_ = X.groupby(self.cat_col)[self.num_col].agg(self.agg_func)
        return self
        
    def transform(self, X):
        result = X[self.cat_col].map(self.mapping_)
        # Handle unseen categories with global statistic
        global_val = self.mapping_.mean() if self.agg_func != "count" else 0
        return result.fillna(global_val)
```

**In `features.py`**:
- Extend `Interaction` (line 104): handle `agg` type where `feature_1` is categorical, `feature_2` is numeric. Set `require_pipeline=True`. Name format: `groupby_{op}({cat}, {num})`
- Extend `_sample_children_with_creativity` (line 845): generate agg interactions pairing categorical parents with numeric parents
- Extend `_prepare_pipeline()` (line 985): collect agg features, configure `GroupByEncoder` instances in the pipeline's `ColumnTransformer`
- Adaptive controller tracks agg ops via `("agg", "binary", op_name)` key — no changes needed, it already supports arbitrary keys
- The `Feature` dataclass needs a new field for `aggregation_type` and the child generation logic needs to handle the `(cat_col, num_col, agg_func)` triple

---

## 3. CV Selection Bias Fix (MEDIUM-HIGH impact — correctness)

**Why**: Ambroise & McLachlan (2002) demonstrated that using the same CV folds for selection and evaluation produces near-zero error rates even on scrambled labels when many candidates are tested. Cawley & Talbot (2010) showed the degradation can be "comparable in magnitude to differences between learning algorithms." Over 80 generations with hundreds of candidates each, TabularAML is particularly susceptible.

**Files**: `features.py`

**Where to modify**: `features.py` → `search()` method, `_select_elites()`.

- **Meta-validation split** in `search()` before generation loop (~line 1410):
  - Reserve 15% of data as held-out meta-validation (stratified/group-aware)
  - All search uses remaining 85%
  - After search: evaluate full feature set on meta split. Log gap as overfitting diagnostic.
  - If gap exceeds threshold, aggressively prune weakest features
  - Only apply when `len(X) > 2000` (small datasets can't afford the split)
- **CV fold rotation** inside generation loop:
  - Every 5 generations, recreate CV with `random_state=generation`
  - Prevents features from overfitting to specific fold boundary patterns
- New params: `meta_validation_frac: float = 0.15`, `rotate_cv_folds: bool = True`, `fold_rotation_period: int = 5`

```python
# Fix 1: Held-out meta-validation split
# In search(), before the generation loop:
if self.meta_validation:
    # Reserve 15-20% of data as meta-validation, never seen during search
    meta_idx = stratified_sample(len(X), frac=0.2, y=y, groups=groups)
    search_idx = ~meta_idx
    X_search, y_search = X.iloc[search_idx], y.iloc[search_idx]
    X_meta, y_meta = X.iloc[meta_idx], y.iloc[meta_idx]
    # All search happens on X_search; final feature set validated on X_meta
    # After search completes, re-evaluate all selected features on meta split

# Fix 2: Rotate CV folds across generations
# In search(), at each generation:
if generation % fold_rotation_period == 0:
    cv = self._create_cv(n_splits=self.n_splits, random_state=generation)
    # Different fold assignment prevents features from overfitting 
    # to a specific fold pattern
```

---

## 4. Feature Value Caching (MEDIUM impact — easy speed win)

**Why**: Currently, each candidate feature's column values are recomputed from parent columns every time they're needed. When the same parent pair with the same operation appears in multiple generations (or in elite re-evaluation), the computation is wasted. With 360 children per generation over 80 generations, cache hit rates above 30% are expected due to the genetic algorithm's tendency to revisit productive regions.

**Files**: `features.py`

**Where to modify**: `features.py` → add a `FeatureCache` class.

- Add `FeatureCache` class with hash-based LRU eviction, configurable max memory (default 2GB)
- Integrate into `Interaction.generate()` (line 117): check cache before computing `ALL_OPS_LAMBDAS[op](...)`
- Initialize `self._feature_cache = FeatureCache(max_size_mb)` in `FeatureGenerator.__init__`
- Clear on `_partial_restart` (preserve high-value entries)
- New param: `cache_size_mb: int = 2000`

```python
import hashlib

class FeatureCache:
    def __init__(self, max_size_mb=2000):
        self._cache = {}
        self._max_bytes = max_size_mb * 1024 * 1024
        self._current_bytes = 0
    
    def _key(self, parent_names, op_name):
        return hashlib.md5(f"{sorted(parent_names)}_{op_name}".encode()).hexdigest()
    
    def get_or_compute(self, parent_names, op_name, compute_fn):
        key = self._key(parent_names, op_name)
        if key in self._cache:
            return self._cache[key]
        result = compute_fn()
        nbytes = result.nbytes if hasattr(result, 'nbytes') else 0
        if self._current_bytes + nbytes < self._max_bytes:
            self._cache[key] = result
            self._current_bytes += nbytes
        return result
```

---

## 5. Regularized Wrapper Post-Selection (MEDIUM impact — better final feature set)

**Why**: After generating hundreds of features across generations, fitting a single L1-regularized model to select the joint-optimal subset is faster and often better than the greedy forward-selection path that produced them. This catches features that the greedy path missed (jointly valuable but individually weak) and removes features that were accepted early but became redundant as later features were added.

**Files**: `features.py`

**Where to modify**: `features.py` → add `_final_regularized_selection()` called after `search()` completes (~line 1660).

- Fit `LassoCV` (regression) or `LogisticRegressionCV` with L1 penalty (classification) on all selected features
- Also fit XGBoost and extract tree-based importance
- Final set = union of non-zero L1 coefficients AND top-K tree-important features
- Log which features added/removed vs. greedy path
- New param: `final_selection: bool = True`

```python
def _final_regularized_selection(self, X_with_generated, y):
    """After search, use L1 regularization to jointly select the best feature subset."""
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_with_generated)
    
    lasso = LassoCV(cv=5, alphas=np.logspace(-4, 1, 50), max_iter=10000)
    lasso.fit(X_scaled, y)
    
    # Keep features with non-zero coefficients
    selected_mask = np.abs(lasso.coef_) > 1e-6
    selected_features = X_with_generated.columns[selected_mask].tolist()
    
    # Also run tree-based importance as cross-check
    import xgboost as xgb
    xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=6)
    xgb_model.fit(X_with_generated, y)
    tree_imp = pd.Series(xgb_model.feature_importances_, 
                         index=X_with_generated.columns)
    
    # Final set: union of L1-selected and top-K tree-important
    final = set(selected_features) | set(tree_imp.nlargest(len(selected_features)).index)
    return list(final)
```

---

## 6. Temporal / Lag Operators (HIGH impact for time-series/panel data)

**Why**: For panel data with time periods (e.g. DataCrunch 2's weekly "moons"), lag features (`feature_value_at_moon_t-1`), rolling statistics (`mean_of_last_4_moons`), and momentum features (`value_at_t - value_at_t-4`) are standard in quant finance and capture temporal dynamics that cross-sectional row-wise ops miss entirely.

**Files**: `ops.py`, `features.py`

**Where to modify**: `ops.py` → new `TEMPORAL_OPS`; `features.py` → child generation logic with temporal awareness.

**In `ops.py`**:
- Add `TEMPORAL_OPS` dict with 6 operations
- Add `OPS["temporal"] = {"unary": list(TEMPORAL_OPS.keys())}` to the `OPS` dict

```python
# In ops.py:
TEMPORAL_OPS = {
    "lag_1": lambda df, col, id_col, time_col: 
        df.groupby(id_col)[col].shift(1),
    "lag_4": lambda df, col, id_col, time_col: 
        df.groupby(id_col)[col].shift(4),
    "rolling_mean_4": lambda df, col, id_col, time_col:
        df.sort_values(time_col).groupby(id_col)[col].transform(
            lambda x: x.rolling(4, min_periods=1).mean()),
    "rolling_std_4": lambda df, col, id_col, time_col:
        df.sort_values(time_col).groupby(id_col)[col].transform(
            lambda x: x.rolling(4, min_periods=1).std()),
    "momentum_4": lambda df, col, id_col, time_col:
        df[col] - df.groupby(id_col)[col].shift(4),
    "pct_change_1": lambda df, col, id_col, time_col:
        df.groupby(id_col)[col].pct_change(1),
}
```

**In `features.py`**:
- Add `time_col: str = None` and `id_col: str = None` params to `FeatureGenerator.__init__`
- When `time_col` is set, enable temporal operators in candidate generation
- Temporal ops are `pipeline_required=True` (they use `.shift()` and `.rolling()` which need proper time ordering and group-aware computation inside CV folds)
- Add temporal-aware encoder similar to `GroupByEncoder` that sorts by time, groups by ID, and applies the temporal transform on train fold only
- Handle NaN from insufficient history (first rows of each group)

**Leakage safety**: These ops use `.shift()` and backward-looking `.rolling()`, so they only reference past values. They must be marked `pipeline_required=True` and computed after sorting by time. Only activate when `time_col` is provided. Skip entirely for cross-sectional datasets.

---

## Implementation Order

4 (caching, simplest) → 1 (FeatureBoost, highest impact) → 2 (GroupBy ops) → 3 (CV fix) → 5 (regularized selection) → 6 (temporal ops)

## Verification

After each change: run `FeatureGenerator.search()` with `mode="lite"` on a small dataset to verify no crashes. Final integration test: `mode="medium"` on a dataset with categoricals, verify proxy eval + groupby ops + CV rotation + caching + regularized selection all work together.

## Changes Documentation

Create `CHANGES.md` at project root after each priority. Each entry: title, files modified, new params with defaults, technical summary.
