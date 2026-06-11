"""Tests for the budget-aware proxy degradation ladder and row cap."""
import time
import numpy as np
import pandas as pd
import pytest

from tabularaml.generate.features import FeatureGenerator, Feature, Interaction
from tabularaml.eval.splitters import RotatedGroupKFold


def _setup(seed=0, n=400, proxy_top_pct=0.25, **gen_kwargs):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(size=n), "b": rng.normal(size=n),
        "d1": rng.normal(size=n), "d2": rng.normal(size=n),
    })
    y = pd.Series(X["a"] * X["b"] + rng.normal(size=n) * 0.05)
    gen = FeatureGenerator(log_file=None, cv=3, proxy_top_pct=proxy_top_pct,
                           use_proxy_evaluation=True, proxy_mode="batched", **gen_kwargs)
    gen._set_defaults(X, y)
    gen._current_y = y
    gen._groups_active = None
    gen._oof_preds_stale = True
    feats = {c: Feature(c, "num", 1.0) for c in X.columns}
    batch = [
        Interaction(feats["a"], "mul", feats["b"]),
        Interaction(feats["d1"], "add", feats["d2"]),
        Interaction(feats["d1"], "sin"),
        Interaction(feats["d2"], "abs"),
        Interaction(feats["d1"], "div", feats["d2"]),
        Interaction(feats["d2"], "square"),
        Interaction(feats["d1"], "tanh"),
        Interaction(feats["d2"], "cos"),
    ]
    return gen, X, y, batch


def _spy(gen, monkeypatch, name):
    calls = {"n": 0}
    orig = getattr(gen, name)

    def wrapper(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(gen, name, wrapper)
    return calls


def test_no_deadline_is_identity_behavior(monkeypatch):
    gen, X, y, batch = _setup()
    assert gen._search_deadline is None
    rank_calls = _spy(gen, monkeypatch, "_batched_proxy_rank")
    fb_calls = _spy(gen, monkeypatch, "_featureboost_screen")
    kept = gen._proxy_screen_candidates(batch, X, y)
    assert rank_calls["n"] == 1 and fb_calls["n"] == 1
    assert any(i.name == "a_mul_b" for i in kept)


def test_ladder_skips_featureboost_in_coarse_only_band(monkeypatch):
    gen, X, y, batch = _setup()
    cv = gen._get_cv_splitter()
    gen._current_oof_preds = gen._train_base_model_and_get_residuals(X, y, cv)
    gen._oof_preds_stale = False
    gen._proxy_time_ema = 10.0
    gen._cv_eval_time_ema = 0.01
    gen._gen_start_remaining = 15.0
    # avail = 15 - reserve(3) = 12 -> in [10, 20): coarse-only
    gen._search_deadline = time.time() + 15.0
    fb_calls = _spy(gen, monkeypatch, "_featureboost_screen")
    rank_calls = _spy(gen, monkeypatch, "_batched_proxy_rank")
    kept = gen._proxy_screen_candidates(batch, X, y)
    assert rank_calls["n"] == 1
    assert fb_calls["n"] == 0
    n_final = max(3, int(len(batch) * gen.proxy_top_pct))
    assert len([i for i in kept if not i.require_pipeline]) <= n_final
    assert any(i.name == "a_mul_b" for i in kept)


def test_ladder_skips_proxy_entirely_below_reserve(monkeypatch):
    gen, X, y, batch = _setup()
    cv = gen._get_cv_splitter()
    gen._current_oof_preds = gen._train_base_model_and_get_residuals(X, y, cv)
    gen._oof_preds_stale = False
    gen._proxy_time_ema = 10.0
    gen._cv_eval_time_ema = 0.01
    gen._gen_start_remaining = 5.0
    gen._search_deadline = time.time() + 5.0  # avail = 5 - 1 = 4 < 10
    rank_calls = _spy(gen, monkeypatch, "_batched_proxy_rank")
    fb_calls = _spy(gen, monkeypatch, "_featureboost_screen")
    kept = gen._proxy_screen_candidates(batch, X, y)
    assert rank_calls["n"] == 0 and fb_calls["n"] == 0
    assert kept is batch  # untouched pass-through


def test_featureboost_partial_break_honors_deadline_and_quota(monkeypatch):
    gen, X, y, _ = _setup(proxy_top_pct=0.5)
    cv = gen._get_cv_splitter()
    gen._current_oof_preds = gen._train_base_model_and_get_residuals(X, y, cv)
    gen._oof_preds_stale = False
    gen._cv_eval_time_ema = 0.001
    gen._gen_start_remaining = 0.4
    gen._search_deadline = time.time() + 0.4

    # 20 candidates x 0.06s each = 1.2s if run fully; deadline allows ~0.32s
    feats = {c: Feature(c, "num", 1.0) for c in X.columns}
    ops = ["square", "abs", "sin", "cos", "tanh", "sqrt", "log1p", "exp", "neg", "cube"]
    batch = [Interaction(feats[c], op) for c in ("a", "b") for op in ops]

    def slow_score(*a, **k):
        time.sleep(0.06)
        return 0.5

    monkeypatch.setattr(gen, "_featureboost_score", slow_score)
    t0 = time.time()
    kept = gen._featureboost_screen(batch, X, y, cv, n_keep=5)
    assert len(kept) == 5  # quota preserved via unscored top-up
    assert time.time() - t0 < 0.06 * len(batch) * 0.75  # broke well before the end


def test_row_cap_groups_and_multiclass_oof_shape():
    rng = np.random.default_rng(7)
    n_eras, per_era = 40, 100
    n = n_eras * per_era
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n),
                      "f3": rng.normal(size=n)})
    y = pd.Series(rng.integers(0, 3, size=n))
    groups = np.repeat(np.arange(n_eras), per_era)
    gen = FeatureGenerator(log_file=None, task="classification",
                           proxy_row_cap=1000, use_proxy_evaluation=True)
    gen._set_defaults(X, y)
    gen.cv = RotatedGroupKFold(4, rotation=0)
    gen._current_y = y
    gen._groups_active = groups
    gen._oof_preds_stale = True
    gen.state = {"counters": {"current_gen": 0}}

    feats = {c: Feature(c, "num", 1.0) for c in X.columns}
    batch = [Interaction(feats["f1"], op) for op in
             ("square", "abs", "sin", "tanh", "cos", "sqrt")]
    kept = gen._proxy_screen_candidates(batch, X, y)
    assert kept  # screen ran without alignment errors

    sub = gen._proxy_row_idx
    assert sub is not None
    # whole eras only, and enough of them for grouped CV
    counts = pd.Series(groups[sub]).value_counts()
    assert (counts == per_era).all()
    assert len(counts) >= 8
    # multiclass OOF has full class width on the subset
    assert gen._current_oof_preds.shape == (len(sub), 3)


def test_search_guarantees_evals_under_tight_budget():
    """The starvation regression test: with budget ~ tens of CV evals, at
    least one generation must actually evaluate candidates."""
    rng = np.random.default_rng(3)
    n = 400
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                      "c": rng.normal(size=n), "d": rng.normal(size=n)})
    y = pd.Series(X["a"] * X["b"] + 0.5 * X["c"] + rng.normal(size=n) * 0.1)
    gen = FeatureGenerator(n_generations=4, n_parents=6, n_children=40, cv=3,
                           log_file=None, use_proxy_evaluation=True,
                           proxy_mode="batched", final_selection=False,
                           meta_validation_frac=0.0, early_stopping_iter=10,
                           expand_row_stats=False, time_budget=25)
    t0 = time.time()
    gen.search(X, y)
    assert gen.state["counters"].get("last_gen_evals", 0) > 0
    assert time.time() - t0 < 25 * 2.5  # post-search steps add bounded overhead
