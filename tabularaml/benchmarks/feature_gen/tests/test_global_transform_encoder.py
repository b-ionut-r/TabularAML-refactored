"""Tests for GlobalTransformEncoder and its pipeline integration."""
import numpy as np
import pandas as pd
import pytest

from tabularaml.preprocessing.encoders import GlobalTransformEncoder
from tabularaml.generate.features import FeatureGenerator, Feature, Interaction


def _train_test(seed=0, n=300):
    rng = np.random.default_rng(seed)
    X_train = pd.DataFrame({"v": rng.normal(10, 3, size=n)})
    X_test = pd.DataFrame({"v": rng.normal(10, 3, size=80)})
    return X_train, X_test


def test_rank_pct_batch_independent():
    X_train, X_test = _train_test()
    enc = GlobalTransformEncoder("v", "rank_pct").fit(X_train)
    full = enc.transform(X_test)
    parts = pd.concat([enc.transform(X_test.iloc[:40]), enc.transform(X_test.iloc[40:])])
    pd.testing.assert_frame_equal(full, parts)
    # percentiles are computed against the TRAIN distribution
    med = X_train["v"].median()
    probe = enc.transform(pd.DataFrame({"v": [med]}))
    assert 0.4 < probe["rank_pct_v"].iloc[0] < 0.6


def test_qbin_edges_from_train_only():
    X_train, X_test = _train_test()
    enc = GlobalTransformEncoder("v", "qbin", n_bins=10).fit(X_train)
    out = enc.transform(X_test)["qbin_v"]
    assert out.between(0, 10).all()
    # NaN falls into the median bin
    nan_out = enc.transform(pd.DataFrame({"v": [np.nan]}))["qbin_v"].iloc[0]
    assert nan_out == enc.median_bin_


def test_zscore_winsor_clips_to_train_bounds():
    X_train, _ = _train_test()
    enc = GlobalTransformEncoder("v", "zscore_winsor").fit(X_train)
    extreme = enc.transform(pd.DataFrame({"v": [1e9, -1e9]}))["zscore_winsor_v"]
    bounded = (np.array([enc.p99_, enc.p1_]) - enc.mean_) / enc.std_
    np.testing.assert_allclose(extreme.values, bounded, rtol=1e-6)


def test_log_rank_monotone():
    X_train, X_test = _train_test()
    enc = GlobalTransformEncoder("v", "log_rank").fit(X_train)
    out = enc.transform(X_test.sort_values("v"))["log_rank_v"].values
    assert (np.diff(out) >= 0).all()


def test_missing_column_fallback():
    X_train, _ = _train_test()
    enc = GlobalTransformEncoder("v", "rank_pct").fit(X_train)
    out = enc.transform(pd.DataFrame({"other": [1, 2]}))
    assert (out["rank_pct_v"] == 0.5).all()


def test_global_interaction_through_pipeline(toy_reg):
    """A rank_pct interaction must flow through _prepare/_extend_pipeline into
    the fitted sklearn pipeline and produce its column leakage-free."""
    X, y = toy_reg
    gen = FeatureGenerator(log_file=None)
    gen._set_defaults(X, y)
    feat = Feature("num_a", "num", 1.0)
    inter = Interaction(feat, "rank_pct")
    assert inter.require_pipeline and getattr(inter, "is_global", False)
    assert inter.name == "rank_pct_num_a"

    pipe_w = gen._extend_pipeline(gen.pipeline, gen._prepare_pipeline([inter]))
    assert len(pipe_w.global_encoders) == 1
    skl = pipe_w.get_pipeline(X, y)
    X_tr = X.iloc[:150]
    X_va = X.iloc[150:]
    out_tr = skl.fit_transform(X_tr, y.iloc[:150])
    out_va = skl.transform(X_va)
    assert "rank_pct_num_a" in out_tr.columns and "rank_pct_num_a" in out_va.columns
    # validation percentile of the train median must be ~0.5 (train-fitted map)
    med = X_tr["num_a"].median()
    probe = skl.transform(pd.DataFrame({**{c: X_va[c].iloc[:1] for c in X_va.columns}}).assign(num_a=med))
    assert 0.35 < probe["rank_pct_num_a"].iloc[0] < 0.65


def test_old_pickled_interactions_lack_is_global(toy_reg):
    """Consumers must tolerate Interaction objects without the new attribute."""
    X, y = toy_reg
    gen = FeatureGenerator(log_file=None)
    gen._set_defaults(X, y)
    inter = Interaction(Feature("num_a", "num", 1.0), "square")
    del inter.__dict__["is_global"]
    # _prepare_pipeline and update_operation_stats must not raise
    pipe = gen._prepare_pipeline([inter])
    assert pipe.global_encoders == []
    gen.adaptive_controller.update_operation_stats(inter, success=True, gain=0.01)
