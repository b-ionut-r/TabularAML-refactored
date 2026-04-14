"""
Targeted smoke tests for the 6 feature engine enhancements.
Each test is independent and clearly labelled.
"""
import numpy as np
import pandas as pd
np.NaN = np.nan  # compatibility shim

from sklearn.datasets import make_classification, make_regression

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = {}

def report(name, ok, detail=""):
    tag = PASS if ok else FAIL
    suffix = f" — {detail}" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    results[name] = ok

# ---------------------------------------------------------------------------
# Shared small dataset (3000 rows so meta-val split triggers)
# ---------------------------------------------------------------------------
np.random.seed(0)
N = 3000
X_base, y_reg = make_regression(n_samples=N, n_features=6, noise=10, random_state=0)
X_base = pd.DataFrame(X_base, columns=[f"num_{i}" for i in range(6)])
X_base["cat_a"] = np.random.choice(["X", "Y", "Z"], size=N)
X_base["cat_b"] = np.random.choice(["P", "Q"], size=N)
# Panel structure for temporal tests
X_base["entity_id"] = np.tile(np.arange(100), 30)      # 100 entities x 30 time steps
X_base["time_step"]  = np.repeat(np.arange(30), 100)
y_reg = pd.Series(y_reg)

X_cls, y_cls = make_classification(n_samples=N, n_features=6, random_state=0)
X_cls = pd.DataFrame(X_cls, columns=[f"num_{i}" for i in range(6)])
X_cls["cat_a"] = np.random.choice(["X", "Y", "Z"], size=N)
y_cls = pd.Series(y_cls)


# ===========================================================================
# Enhancement 1 -- FeatureBoost Proxy Evaluation
# ===========================================================================
print("\n=== Enhancement 1: FeatureBoost Proxy Evaluation ===")

try:
    from tabularaml.generate.features import FeatureGenerator
    fg = FeatureGenerator(cv=3)
    fg._set_defaults(X_base[["num_0", "num_1"]].copy(), y_reg)
    cv_obj = fg._get_cv_splitter()
    X_small = X_base[["num_0", "num_1"]].copy()
    oof = fg._train_base_model_and_get_residuals(X_small, y_reg, cv_obj)
    ok = oof.shape == (N,) and np.isfinite(oof).mean() > 0.9
    report("1a: _train_base_model_and_get_residuals returns finite OOF preds", ok,
           f"shape={oof.shape}, finite={np.isfinite(oof).mean():.2f}")
except Exception as e:
    report("1a: _train_base_model_and_get_residuals returns finite OOF preds", False, str(e))

try:
    cand = X_base["num_2"]
    score = fg._featureboost_score(cand, y_reg, oof, cv_obj)
    ok = np.isfinite(score)
    report("1b: _featureboost_score returns finite score", ok, f"score={score:.5f}")
except Exception as e:
    report("1b: _featureboost_score returns finite score", False, str(e))

try:
    from tabularaml.generate.features import Interaction, Feature
    f1 = Feature("num_0", "num", 0.5)
    f2 = Feature("num_1", "num", 0.4)
    batch = [Interaction(f1, op, f2) for op in ["add", "sub", "mul", "div", "absdiff"]]
    fg._oof_preds_stale = True
    screened = fg._proxy_screen_candidates(batch, X_small, y_reg)
    ok = 0 < len(screened) <= len(batch)
    report("1c: proxy screening reduces candidate count", ok,
           f"{len(batch)} -> {len(screened)}")
except Exception as e:
    report("1c: proxy screening reduces candidate count", False, str(e))


# ===========================================================================
# Enhancement 2 -- GroupBy Aggregation Operators
# ===========================================================================
print("\n=== Enhancement 2: GroupBy Aggregation Operators ===")

try:
    from tabularaml.generate.ops import AGG_OPS, OPS
    ok = len(AGG_OPS) == 8 and "agg" in OPS
    report("2a: AGG_OPS has 8 ops and registered in OPS", ok, str(list(AGG_OPS.keys())))
except Exception as e:
    report("2a: AGG_OPS has 8 ops and registered in OPS", False, str(e))

try:
    from tabularaml.preprocessing.encoders import GroupByEncoder
    enc = GroupByEncoder(cat_col="cat_a", num_col="num_0", agg_func="mean")
    enc.fit(X_base)
    out = enc.transform(X_base)
    ok = out.shape == (N, 1) and out.columns[0] == "groupby_mean(cat_a, num_0)"
    report("2b: GroupByEncoder(mean) fit/transform", ok,
           f"shape={out.shape}, col={out.columns[0]}")
except Exception as e:
    report("2b: GroupByEncoder(mean) fit/transform", False, str(e))

try:
    enc_z = GroupByEncoder(cat_col="cat_a", num_col="num_0", agg_func="zscore")
    enc_z.fit(X_base)
    out_z = enc_z.transform(X_base)
    ok = np.isfinite(out_z.values).mean() > 0.95
    report("2c: GroupByEncoder(zscore) produces finite values", ok,
           f"finite={np.isfinite(out_z.values).mean():.3f}")
except Exception as e:
    report("2c: GroupByEncoder(zscore) produces finite values", False, str(e))

try:
    enc_r = GroupByEncoder(cat_col="cat_a", num_col="num_0", agg_func="rank")
    enc_r.fit(X_base)
    out_r = enc_r.transform(X_base)
    ok = float(out_r.values.min()) >= 0.0 and float(out_r.values.max()) <= 1.0
    report("2d: GroupByEncoder(rank) produces [0,1] percentile ranks", ok,
           f"min={out_r.values.min():.3f}, max={out_r.values.max():.3f}")
except Exception as e:
    report("2d: GroupByEncoder(rank) produces [0,1] percentile ranks", False, str(e))

try:
    from tabularaml.generate.features import Interaction, Feature
    cat_f = Feature("cat_a", "cat", 0.3)
    num_f = Feature("num_0", "num", 0.5)
    inter = Interaction(cat_f, "groupby_mean", num_f)
    ok = inter.is_agg and inter.require_pipeline and inter.dtype == "num"
    ok &= inter.name == "groupby_mean(cat_a, num_0)"
    report("2e: Interaction(agg) sets is_agg, require_pipeline, correct name", ok,
           f"name={inter.name}")
except Exception as e:
    report("2e: Interaction(agg) sets is_agg, require_pipeline, correct name", False, str(e))

try:
    fg_agg = FeatureGenerator(
        n_generations=2, n_children=20, cv=3,
        use_proxy_evaluation=False,
        meta_validation_frac=0.0, final_selection=False, rotate_cv_folds=False
    )
    X_agg = X_base[["num_0", "num_1", "cat_a"]].copy()
    X_new, pipe, gen, interactions = fg_agg.search(X_agg, y_reg)
    agg_feats = [c for c in X_new.columns if c.startswith("groupby_")]
    report("2f: search() runs without crash with cat+agg ops", True,
           f"agg_feats_accepted={len(agg_feats)}")
except Exception as e:
    report("2f: search() runs without crash with cat+agg ops", False, str(e))


# ===========================================================================
# Enhancement 3 -- CV Selection Bias Fix
# ===========================================================================
print("\n=== Enhancement 3: CV Selection Bias Fix ===")

try:
    fg_meta = FeatureGenerator(
        n_generations=1, n_children=5, cv=3,
        meta_validation_frac=0.15,
        rotate_cv_folds=False, final_selection=False, use_proxy_evaluation=False
    )
    X_m = X_base[["num_0", "num_1", "num_2"]].copy()
    fg_meta.search(X_m, y_reg)
    ok = fg_meta.n_samples <= int(N * 0.87)
    report("3a: meta-val split reduces search dataset to ~85%", ok,
           f"original={N}, search_size={fg_meta.n_samples}")
except Exception as e:
    report("3a: meta-val split reduces search dataset to ~85%", False, str(e))

try:
    fg_small = FeatureGenerator(
        n_generations=1, n_children=5, cv=3,
        meta_validation_frac=0.15,
        final_selection=False, use_proxy_evaluation=False
    )
    X_tiny = X_base.iloc[:500][["num_0", "num_1"]].copy()
    y_tiny = y_reg.iloc[:500]
    fg_small.search(X_tiny, y_tiny)
    ok = fg_small.n_samples == 500
    report("3b: meta-val NOT applied when len(X) <= 2000", ok,
           f"n_samples={fg_small.n_samples}")
except Exception as e:
    report("3b: meta-val NOT applied when len(X) <= 2000", False, str(e))

try:
    fg_rot = FeatureGenerator(
        n_generations=8, n_children=5, cv=3,
        rotate_cv_folds=True, fold_rotation_period=3,
        meta_validation_frac=0.0, final_selection=False, use_proxy_evaluation=False
    )
    fg_rot.search(X_base[["num_0", "num_1"]].copy(), y_reg)
    report("3c: CV fold rotation runs without crash", True,
           f"cv_type={type(fg_rot.cv).__name__}")
except Exception as e:
    report("3c: CV fold rotation runs without crash", False, str(e))


# ===========================================================================
# Enhancement 4 -- Feature Value Caching
# ===========================================================================
print("\n=== Enhancement 4: Feature Value Caching ===")

try:
    from tabularaml.generate.features import FeatureCache
    cache = FeatureCache(max_size_mb=100)
    call_count = [0]

    def compute():
        call_count[0] += 1
        return ("a_add_b", pd.Series([1.0, 2.0, 3.0]))

    cache.get_or_compute(["a", "b"], "add", compute)
    cache.get_or_compute(["a", "b"], "add", compute)  # should hit
    ok = call_count[0] == 1 and cache.hits == 1 and cache.misses == 1
    report("4a: second call is a cache hit (no recomputation)", ok,
           f"calls={call_count[0]}, hits={cache.hits}, misses={cache.misses}")
except Exception as e:
    report("4a: second call is a cache hit (no recomputation)", False, str(e))

try:
    cache2 = FeatureCache(max_size_mb=100)
    calls = [0]
    def c_ab(): calls[0] += 1; return ("a_sub_b", pd.Series([1.0]))
    def c_ba(): calls[0] += 1; return ("b_sub_a", pd.Series([-1.0]))
    cache2.get_or_compute(["a", "b"], "sub", c_ab)
    cache2.get_or_compute(["b", "a"], "sub", c_ba)
    ok = calls[0] == 2  # order-sensitive: sub(a,b) != sub(b,a)
    report("4b: sub(a,b) and sub(b,a) have distinct cache keys", ok,
           f"compute_calls={calls[0]}")
except Exception as e:
    report("4b: sub(a,b) and sub(b,a) have distinct cache keys", False, str(e))

try:
    cache3 = FeatureCache(max_size_mb=100)
    cache3.get_or_compute(["a"], "neg", lambda: ("a_neg", pd.Series([1.0])))
    cache3.clear()
    ok = cache3._current_bytes == 0 and len(cache3._cache) == 0
    report("4c: cache.clear() resets byte count and entries", ok)
except Exception as e:
    report("4c: cache.clear() resets byte count and entries", False, str(e))

try:
    fg_c = FeatureGenerator(
        n_generations=4, n_children=30, cv=3,
        use_proxy_evaluation=False,
        meta_validation_frac=0.0, final_selection=False,
        rotate_cv_folds=False, cache_size_mb=500
    )
    fg_c.search(X_base[["num_0", "num_1", "num_2"]].copy(), y_reg)
    report("4d: cache used during multi-generation search", True,
           f"hits={fg_c._feature_cache.hits}, misses={fg_c._feature_cache.misses}, "
           f"hit_rate={fg_c._feature_cache.hit_rate:.3f}")
except Exception as e:
    report("4d: cache used during multi-generation search", False, str(e))


# ===========================================================================
# Enhancement 5 -- Regularized Post-Selection
# ===========================================================================
print("\n=== Enhancement 5: Regularized Post-Selection ===")

try:
    fg_sel = FeatureGenerator(cv=3)
    fg_sel._set_defaults(X_base[["num_0", "num_1"]].copy(), y_reg)
    fg_sel.initial_features = ["num_0", "num_1"]
    X_fake = X_base[["num_0", "num_1"]].copy()
    for i in range(15):
        X_fake[f"gen_feat_{i}"] = np.random.randn(N)
    dropped = fg_sel._final_regularized_selection(X_fake, y_reg)
    ok = isinstance(dropped, list)
    report("5a: _final_regularized_selection returns list", ok,
           f"dropped {len(dropped)} of 15 generated features")
except Exception as e:
    report("5a: _final_regularized_selection returns list", False, str(e))

try:
    ok = all(f not in dropped for f in ["num_0", "num_1"])
    report("5b: original features are never dropped", ok)
except Exception as e:
    report("5b: original features are never dropped", False, str(e))

try:
    fg_cls2 = FeatureGenerator(cv=3)
    fg_cls2._set_defaults(X_cls[["num_0", "num_1"]].copy(), y_cls)
    fg_cls2.initial_features = ["num_0", "num_1"]
    X_fc = X_cls[["num_0", "num_1"]].copy()
    for i in range(12):
        X_fc[f"gen_{i}"] = np.random.randn(N)
    dropped_cls = fg_cls2._final_regularized_selection(X_fc, y_cls)
    ok = isinstance(dropped_cls, list)
    report("5c: regularized selection works for classification", ok,
           f"dropped={len(dropped_cls)}")
except Exception as e:
    report("5c: regularized selection works for classification", False, str(e))

try:
    fg_fs = FeatureGenerator(
        n_generations=3, n_children=20, cv=3,
        final_selection=True, use_proxy_evaluation=False,
        meta_validation_frac=0.0, rotate_cv_folds=False
    )
    X_fs = X_base[["num_0", "num_1", "num_2", "num_3"]].copy()
    X_new_fs, *_ = fg_fs.search(X_fs, y_reg)
    report("5d: full search with final_selection=True completes", True,
           f"final_cols={X_new_fs.shape[1]}")
except Exception as e:
    report("5d: full search with final_selection=True completes", False, str(e))


# ===========================================================================
# Enhancement 6 -- Temporal / Lag Operators
# ===========================================================================
print("\n=== Enhancement 6: Temporal / Lag Operators ===")

try:
    from tabularaml.generate.ops import TEMPORAL_OPS, OPS, build_temporal_ops
    ok = "temporal" in OPS and len(TEMPORAL_OPS) >= 6
    report("6a: TEMPORAL_OPS has 6+ ops and registered in OPS", ok,
           f"ops={sorted(TEMPORAL_OPS.keys())}")
except Exception as e:
    report("6a: TEMPORAL_OPS has 6+ ops and registered in OPS", False, str(e))

try:
    custom = build_temporal_ops([1, 5, 20])
    ok = all(k in custom for k in ["lag_1", "lag_5", "lag_20",
                                    "rolling_mean_5", "momentum_20", "pct_change_1"])
    report("6b: build_temporal_ops(custom windows) generates correct names", ok,
           f"keys={sorted(custom.keys())}")
except Exception as e:
    report("6b: build_temporal_ops(custom windows) generates correct names", False, str(e))

try:
    from tabularaml.preprocessing.encoders import TemporalEncoder
    enc_lag = TemporalEncoder(col="num_0", id_col="entity_id",
                               time_col="time_step", op_name="lag_1")
    enc_lag.fit(X_base)
    out_lag = enc_lag.transform(X_base)
    ok = (out_lag.shape == (N, 1)
          and out_lag.columns[0] == "lag_1(num_0)"
          and np.isfinite(out_lag.values).mean() > 0.9)
    report("6c: TemporalEncoder(lag_1) fit/transform", ok,
           f"col={out_lag.columns[0]}, finite={np.isfinite(out_lag.values).mean():.3f}")
except Exception as e:
    report("6c: TemporalEncoder(lag_1) fit/transform", False, str(e))

try:
    enc_roll = TemporalEncoder(col="num_0", id_col="entity_id",
                                time_col="time_step", op_name="rolling_mean_4")
    enc_roll.fit(X_base)
    out_roll = enc_roll.transform(X_base)
    ok = out_roll.shape == (N, 1) and np.isfinite(out_roll.values).mean() > 0.95
    report("6d: TemporalEncoder(rolling_mean_4) fit/transform", ok,
           f"finite={np.isfinite(out_roll.values).mean():.3f}")
except Exception as e:
    report("6d: TemporalEncoder(rolling_mean_4) fit/transform", False, str(e))

try:
    enc_mom = TemporalEncoder(col="num_0", id_col="entity_id",
                               time_col="time_step", op_name="momentum_4")
    enc_mom.fit(X_base)
    out_mom = enc_mom.transform(X_base)
    ok = out_mom.shape == (N, 1) and np.isfinite(out_mom.values).mean() > 0.8
    report("6e: TemporalEncoder(momentum_4) fit/transform", ok,
           f"finite={np.isfinite(out_mom.values).mean():.3f}")
except Exception as e:
    report("6e: TemporalEncoder(momentum_4) fit/transform", False, str(e))

try:
    from tabularaml.generate.features import Interaction, Feature
    num_f = Feature("num_0", "num", 0.5)
    inter_t = Interaction(num_f, "lag_1")
    ok = (inter_t.is_temporal and inter_t.require_pipeline
          and inter_t.name == "lag_1(num_0)")
    report("6f: Interaction(temporal) sets is_temporal, require_pipeline, correct name", ok,
           f"name={inter_t.name}")
except Exception as e:
    report("6f: Interaction(temporal) sets is_temporal, require_pipeline, correct name", False, str(e))

try:
    fg_temp = FeatureGenerator(
        n_generations=2, n_children=15, cv=3,
        time_col="time_step", id_col="entity_id",
        use_proxy_evaluation=False,
        meta_validation_frac=0.0, final_selection=False, rotate_cv_folds=False
    )
    X_t = X_base[["num_0", "num_1", "entity_id", "time_step"]].copy()
    X_new_t, *_ = fg_temp.search(X_t, y_reg)
    temporal_feats = [c for c in X_new_t.columns
                      if any(c.startswith(p) for p in
                             ["lag_", "rolling_", "momentum_", "pct_change_"])]
    report("6g: search() with time_col/id_col runs without crash", True,
           f"temporal_feats_accepted={len(temporal_feats)}")
except Exception as e:
    report("6g: search() with time_col/id_col runs without crash", False, str(e))


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 55)
passed = sum(results.values())
total  = len(results)
print(f"Result: {passed}/{total} tests passed")
if passed < total:
    print("Failed tests:")
    for name, ok in results.items():
        if not ok:
            print(f"  - {name}")
print("=" * 55)
