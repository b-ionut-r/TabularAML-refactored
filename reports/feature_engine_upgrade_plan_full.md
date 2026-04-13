# SOTA Feature Engineering — Full Upgrade Plan (All 10 Priorities)

## Context

The genetic feature engine in `tabularaml/generate/` is well-architected (adaptive stagnation, SHAP-guided parents, pattern memory) but has key gaps vs. published SOTA. This plan covers all 10 enhancements from the technical audit, targeting the genetic search core with minimal architectural rewrites. Each change documented in a unified `CHANGES.md` at project root.

## Critical Files

| File | Lines | Role |
|---|---|---|
| `tabularaml/generate/features.py` | 2049 | `FeatureGenerator`, `Feature`, `Interaction`, `ImprovedAdaptiveController`, `StagnationLevel` |
| `tabularaml/generate/ops.py` | 375 | `OPS` dict, `NUM_OPS_LAMBDAS`, `CAT_OPS_LAMBDAS`, `ALL_OPS_LAMBDAS` |
| `tabularaml/preprocessing/encoders.py` | 158 | `CategoricalEncoder` (target/count/freq encoding inside CV) |
| `tabularaml/preprocessing/pipeline.py` | 198 | `PipelineWrapper` |
| `tabularaml/eval/cv.py` | 206 | `cross_val_score` with pipeline support |
| `tabularaml/eval/scorers.py` | 433 | `Scorer` class, predefined scorers |
| `tabularaml/configs/feature_gen.py` | 58 | `PRESET_PARAMS` (lite/medium/best/extreme) |
| `tabularaml/utils/gpu.py` | — | `is_gpu_available()` |
| `requirements.hf.txt` | — | Dependencies |

---

## Priority 1: FeatureBoost Proxy Evaluation (HIGH impact — 50x throughput)

**Why**: This single change would increase candidate throughput by ~50x, enabling exploration of the search space that currently requires days in hours. OpenFE's core innovation demonstrates that residual-based scoring correlates highly with full-CV evaluation while being orders of magnitude faster.

**Files**: `features.py`, `requirements.hf.txt`

**Where to modify**: `features.py` → `_select_elites()` (line 1030) and the evaluation calls within `search()`.

- Add `_train_base_model_and_get_residuals(X, y, cv)` — trains LightGBM on current features, returns OOF predictions. Recomputed each generation.
- Add `_featureboost_score(candidate_values, y, oof_preds, cv)` — trains a tiny single-feature LightGBM with `init_score=oof_preds` (OpenFE's core trick). Falls back to XGBoost if LGB unavailable.
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

## Priority 2: Group-By Aggregation Operators (HIGH impact — most powerful missing feature class)

**Why**: GroupBy features are the single most powerful feature class in Kaggle winning solutions (Chris Deotte's 1st-place solutions routinely generate 10,000+ groupby candidates). OpenFE includes 6 GroupBy operators; TabularAML has zero. For panel data, features like "stock's feature value relative to industry mean" capture cross-sectional structure that row-wise ops cannot.

**Files**: `ops.py`, `features.py`, `encoders.py`

**Where to modify**: `ops.py` → add to `OPS` dict and create `AGG_OPS`; `features.py` → `_sample_children_with_creativity` to generate groupby candidates; `encoders.py` → new `GroupByEncoder` for pipeline-required aggregations.

**In `ops.py`**:

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

## Priority 3: Fix the CV Selection Bias (MEDIUM-HIGH impact — correctness)

**Why**: Ambroise & McLachlan (2002) demonstrated that using the same CV folds for selection and evaluation produces near-zero error rates even on scrambled labels when many candidates are tested. Cawley & Talbot (2010) showed the degradation can be "comparable in magnitude to differences between learning algorithms." Over 80 generations with hundreds of candidates each, TabularAML is particularly susceptible.

**Files**: `features.py`

**Where to modify**: `features.py` → `search()` method, `_select_elites()`.

- **Meta-validation split** before generation loop (~line 1410). Only apply when `len(X) > 2000`.
- **CV fold rotation** every 5 generations inside the loop.
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

The meta-validation split is the more robust fix. After the full search completes, retrain on the search portion and evaluate the entire feature set on the meta portion. If the meta-validation score is substantially lower than the search CV score, the feature set is overfit. Use the gap as a regularization signal to prune aggressively.

---

## Priority 4: Batch Multi-Feature Evaluation (MEDIUM impact)

**Why**: Pure greedy forward selection fails on suppressor variables (features useless alone but valuable in combination) and correlated feature groups (selects one and misses the complementary set). Evaluating features in batches partially addresses this.

**Files**: `features.py`

**Where to modify**: `features.py` → `_select_elites()`.

- Add `_select_elites_batch()` method alongside existing `_select_elites()`
- Integrates into the two-phase flow from Priority 1
- New params: `batch_evaluation: bool = True`, `batch_size: int = 5`
- During severe stagnation, increase batch_size to 8-10

```python
def _select_elites_batch(self, candidates, X, y, batch_size=5):
    """Evaluate candidates in batches, using permutation importance 
    to identify the best subset within each batch."""
    from sklearn.inspection import permutation_importance
    
    accepted = []
    for batch in chunk(candidates, batch_size):
        # Add all batch features to current X
        X_augmented = pd.concat([X] + [c.values for c in batch], axis=1)
        
        # Train model once on augmented features
        model = self._fit_model(X_augmented, y)
        
        # Permutation importance identifies which batch features help
        perm_imp = permutation_importance(model, X_augmented, y, 
                                          n_repeats=5, scoring=self.scorer)
        
        # Keep batch features with positive importance above threshold
        for i, candidate in enumerate(batch):
            col_idx = len(X.columns) + i
            if perm_imp.importances_mean[col_idx] > self.adaptive_min_gain:
                accepted.append(candidate)
                X = pd.concat([X, candidate.values], axis=1)
    
    return accepted
```

This trains **1 model per batch** instead of 1 per candidate, reducing total evaluations by `batch_size`x while allowing complementary features to be discovered together.

---

## Priority 5: Feature Value Caching (MEDIUM impact — easy speed win)

**Why**: Currently, each candidate feature's column values are recomputed from parent columns every time they're needed. When the same parent pair with the same operation appears in multiple generations (or in elite re-evaluation), the computation is wasted. With 360 children per generation over 80 generations, cache hit rates above 30% are expected.

**Files**: `features.py`

**Where to modify**: `features.py` → add a `FeatureCache` class. Integrate into `Interaction.generate()` (line 117).

- New param: `cache_size_mb: int = 2000`
- Clear on `_partial_restart` (preserve high-value entries)

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

## Priority 6: GPU-Accelerated Evaluation (MEDIUM impact — 5-20x training speedup)

**Why**: XGBoost's `tree_method='gpu_hist'` provides 5-20x training speedup. Combined with cuDF for feature computation, the total search time compresses proportionally. NVIDIA benchmarks show cuDF delivers up to 150x speedup for groupby operations on A100 GPUs.

**Files**: `features.py`, `scorers.py`, `ops.py`

**Where to modify**: `features.py` → `_set_defaults()` (line 1751) to wire `use_gpu` param to model creation via existing `tabularaml/utils/gpu.py` (`is_gpu_available()`). `scorers.py` → default XGBoost params. `ops.py` → optional cuDF acceleration.

```python
# In scorers.py, modify default XGBoost params:
class XGBScorer(Scorer):
    def __init__(self, **kwargs):
        default_params = {
            "tree_method": "gpu_hist",  # GPU training
            "device": "cuda",
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 6,
        }
        default_params.update(kwargs)
        super().__init__(xgb.XGBRegressor(**default_params))

# In ops.py, optional cuDF acceleration:
try:
    import cudf
    def gpu_groupby_mean(df, cat_col, num_col):
        gdf = cudf.from_pandas(df[[cat_col, num_col]])
        return gdf.groupby(cat_col)[num_col].transform("mean").to_pandas()
except ImportError:
    pass
```

---

## Priority 7: Temporal / Lag Operators (HIGH impact for time-series/panel data)

**Why**: For panel data with time periods (e.g. DataCrunch 2's weekly "moons"), lag features, rolling statistics, and momentum features are standard in quant finance and capture temporal dynamics that cross-sectional row-wise ops miss entirely.

**Files**: `ops.py`, `features.py`

**Where to modify**: `ops.py` → new `TEMPORAL_OPS`; `features.py` → child generation logic with temporal awareness.

- New params: `time_col: str = None`, `id_col: str = None`
- Only activate when `time_col` is provided. Skip entirely for cross-sectional datasets.
- All temporal ops are `pipeline_required=True`

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

**Leakage safety**: These ops use `.shift()` and backward-looking `.rolling()`, so they only reference past values. They must be marked `pipeline_required=True` and computed after sorting by time, with proper handling of the first rows (NaN from insufficient history).

---

## Priority 8: LLM-Assisted Feature Proposal Injection (MEDIUM impact)

**Why**: CAAFE (NeurIPS 2023) and OCTree (NeurIPS 2024) demonstrate that LLMs can propose domain-appropriate features that systematic search misses. For anonymized features (DataCrunch 2) this is less useful, but for competitions with named columns, LLM proposals complement the genetic search.

**Files**: `features.py`

**Where to modify**: `features.py` → add `_inject_llm_proposals()` called at generation 0.

- New params: `llm_seed: bool = False`, `llm_provider: str = None`
- Guard with `try/except` — purely optional, never blocks the search
- Skip automatically if columns are anonymized (detect via naming patterns)

```python
def _inject_llm_proposals(self, column_names, dtypes, task_description):
    """Use an LLM to propose initial feature formulas, seeding gen 0."""
    import openai  # or local LLM
    prompt = f"""Dataset columns: {list(zip(column_names, dtypes))}
    Task: {task_description}
    Propose 20 useful engineered features as Python expressions using column names.
    Format: feature_name = expression
    Only use: +, -, *, /, log, sqrt, abs, and groupby aggregations."""
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
    )
    # Parse response into Feature candidates and inject into generation[0]
    proposals = self._parse_llm_features(response.choices[0].message.content)
    return proposals
```

---

## Priority 9: Regularized Wrapper Post-Selection (MEDIUM impact — better final feature set)

**Why**: After generating hundreds of features across generations, fitting a single L1-regularized model to select the joint-optimal subset is faster and often better than the greedy forward-selection path that produced them. This catches features that the greedy path missed (jointly valuable but individually weak) and removes features that were accepted early but became redundant.

**Files**: `features.py`

**Where to modify**: `features.py` → add `_final_regularized_selection()` called after `search()` completes (~line 1660).

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

## Priority 10: Meta-Learning Warm Start (MEDIUM-LOW impact)

**Why**: For repeated competition submissions (DataCrunch 2 runs weekly), storing which operators succeeded on past moons and using that to bias initial exploration saves generations of cold-start search. The dataset structure is stable across moons, so operator effectiveness transfers strongly.

**Files**: `features.py` (specifically `ImprovedAdaptiveController`)

**Where to modify**: `features.py` → `ImprovedAdaptiveController`, add serialization methods.

- New param: `meta_knowledge_path: str = None` on `FeatureGenerator.__init__`
- If set, load on init and save after search completes

```python
# In ImprovedAdaptiveController:
def save_meta_knowledge(self, filepath):
    """Serialize learned op_stats, successful_patterns for future warm-start."""
    meta = {
        "op_stats": dict(self.op_stats),
        "successful_patterns": list(self.successful_patterns),
        "feature_as_parent_success": dict(self.feature_as_parent_success),
    }
    import json
    with open(filepath, "w") as f:
        json.dump(meta, f, default=str)

def load_meta_knowledge(self, filepath):
    """Initialize from previous run's learned knowledge."""
    import json
    with open(filepath) as f:
        meta = json.load(f)
    # Decay old knowledge (50% weight) to allow adaptation
    for key, stats in meta["op_stats"].items():
        self.op_stats[tuple(key)] = {
            k: v * 0.5 for k, v in stats.items()
        }
    self.successful_patterns = meta["successful_patterns"][-10:]  # Keep recent
```

---

## Priority Ranking Summary

| Priority | Recommendation | Impact | Complexity | Reason |
|---|---|---|---|---|
| **1** | FeatureBoost proxy evaluation | HIGH | 2–3 days | 50x throughput increase; enables exploring vastly larger search space |
| **2** | Group-by aggregation operators | HIGH | 3–4 days | Missing the single most powerful feature class; critical for panel data |
| **3** | CV selection bias fix | MEDIUM-HIGH | 1–2 days | Prevents overfitting feature set to validation noise; improves OOS transfer |
| **4** | Batch multi-feature evaluation | MEDIUM | 1–2 days | Discovers complementary features greedy selection misses |
| **5** | Feature value caching | MEDIUM | 0.5 days | Free speedup with minimal code changes |
| **6** | GPU acceleration | MEDIUM | 1–2 days | Leverages existing CUDA hardware; large speedup |
| **7** | Temporal operators | HIGH (panel data) | 2–3 days | Captures time dynamics missing from current op set |
| **8** | LLM feature injection | MEDIUM | 1 day | Useful on named-column datasets; skip for anonymized competitions |
| **9** | Regularized wrapper post-selection | MEDIUM | 1 day | Better joint optimization than greedy path alone |
| **10** | Meta-learning warm start | MEDIUM-LOW | 0.5 days | Compounds advantage on weekly competitions |

## Implementation Order

5 (caching) → 1 (FeatureBoost) → 6 (GPU) → 2 (GroupBy ops) → 3 (CV fix) → 4 (batch eval) → 9 (regularized selection) → 7 (temporal ops) → 8 (LLM injection) → 10 (meta-learning)

## Verification

After each change: run `FeatureGenerator.search()` with `mode="lite"` on a small dataset to verify no crashes. Final integration test: `mode="medium"` on a dataset with categoricals, verify all new features work together end-to-end.

## Changes Documentation

Create `CHANGES.md` at project root after each priority. Each entry: title, files modified, new params with defaults, technical summary.
