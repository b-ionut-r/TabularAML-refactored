"""Tests for batched proxy screening, fallback behavior, pipeline-candidate cap,
and budget-aware generation sizing."""
import time
import numpy as np
import pandas as pd
import pytest

from tabularaml.generate.features import FeatureGenerator, Feature, Interaction


def _signal_setup(seed=0, n=400):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(size=n), "b": rng.normal(size=n),
        "d1": rng.normal(size=n), "d2": rng.normal(size=n),
    })
    y = pd.Series(X["a"] * X["b"] + rng.normal(size=n) * 0.05)
    gen = FeatureGenerator(log_file=None, cv=3, proxy_top_pct=0.25,
                           use_proxy_evaluation=True, proxy_mode="batched")
    gen._set_defaults(X, y)
    gen._current_y = y
    gen._groups_active = None
    gen._oof_preds_stale = True
    feats = {c: Feature(c, "num", 1.0) for c in X.columns}
    batch = [
        Interaction(feats["a"], "mul", feats["b"]),     # true signal
        Interaction(feats["d1"], "add", feats["d2"]),
        Interaction(feats["d1"], "sin"),
        Interaction(feats["d2"], "abs"),
        Interaction(feats["d1"], "div", feats["d2"]),
        Interaction(feats["d2"], "square"),
        Interaction(feats["d1"], "tanh"),
        Interaction(feats["d2"], "cos"),
    ]
    return gen, X, y, batch


def test_batched_proxy_keeps_planted_signal():
    gen, X, y, batch = _signal_setup()
    kept = gen._proxy_screen_candidates(batch, X, y)
    kept_names = {i.name for i in kept}
    assert "a_mul_b" in kept_names
    assert len(kept) < len(batch)  # actually filtered something


def test_batched_proxy_coarse_stage_keeps_signal_drops_zero_gain():
    gen, X, y, batch = _signal_setup()
    cv = gen._get_cv_splitter()
    gen._current_oof_preds = gen._train_base_model_and_get_residuals(X, y, cv)
    gen._oof_preds_stale = False
    coarse = gen._batched_proxy_rank(batch, X, y, cv)
    assert coarse is not None
    # signal survives the coarse stage; coarse quota is ~3x the final quota
    assert any(i.name == "a_mul_b" for i in coarse)
    n_final = max(3, int(len(batch) * gen.proxy_top_pct))
    assert len(coarse) <= min(len(batch), max(15, 3 * n_final))


def test_dispatcher_final_quota_after_refinement():
    gen, X, y, batch = _signal_setup()
    kept = gen._proxy_screen_candidates(batch, X, y)
    kept_scored = [i for i in kept if not i.require_pipeline]
    assert len(kept_scored) <= max(3, int(len(batch) * gen.proxy_top_pct))
    assert any(i.name == "a_mul_b" for i in kept_scored)


def test_batched_proxy_falls_back_to_featureboost(monkeypatch):
    gen, X, y, batch = _signal_setup()
    calls = {"fb": 0}

    def boom(*a, **k):
        raise RuntimeError("forced")

    orig_fb = gen._featureboost_screen

    def counting_fb(*a, **k):
        calls["fb"] += 1
        return orig_fb(*a, **k)

    monkeypatch.setattr(gen, "_batched_proxy_rank", boom)
    monkeypatch.setattr(gen, "_featureboost_screen", counting_fb)
    kept = gen._proxy_screen_candidates(batch, X, y)
    assert calls["fb"] == 1
    assert any(i.name == "a_mul_b" for i in kept)


def test_batched_proxy_respects_tiny_ram_budget():
    gen, X, y, batch = _signal_setup(n=2000)
    gen.proxy_ram_budget_mb = 0.001  # force the row-subsample guard
    kept = gen._proxy_screen_candidates(batch, X, y)
    assert any(i.name == "a_mul_b" for i in kept)


def test_pipeline_candidate_cap():
    gen, X, y, batch = _signal_setup()
    # Flood with pipeline-required candidates (target encodings of fake cats)
    cat_feats = [Feature(f"c{i}", "cat", 1.0) for i in range(40)]
    pipe_cands = [Interaction(cf, "target") for cf in cat_feats]
    assert all(c.require_pipeline for c in pipe_cands)
    kept = gen._proxy_screen_candidates(batch + pipe_cands, X, y)
    kept_pipe = [i for i in kept if i.require_pipeline]
    kept_scored = [i for i in kept if not i.require_pipeline]
    assert len(kept_pipe) <= max(10, len(kept_scored) // 2)


def test_budget_scaled_sizes_no_budget_is_identity():
    gen = FeatureGenerator(log_file=None, n_children=100, time_budget=None)
    assert gen._budget_scaled_sizes(time.time(), 0) == (100, None)


def test_budget_scaled_sizes_shrinks_when_behind():
    gen = FeatureGenerator(log_file=None, n_children=100, n_generations=10,
                           early_stopping_child_eval=30, time_budget=100)
    # 80% of budget burned by generation 2 (i.e. 70% of generations remain)
    start = time.time() - 80
    n_children, early_thr = gen._budget_scaled_sizes(start, 2)
    assert n_children < 100
    assert n_children >= 20
    assert early_thr is not None and early_thr >= 8


def test_budget_scaled_sizes_on_track_is_identity():
    gen = FeatureGenerator(log_file=None, n_children=100, n_generations=10,
                           time_budget=1000)
    start = time.time() - 10  # barely any time burned
    assert gen._budget_scaled_sizes(start, 2) == (100, None)
