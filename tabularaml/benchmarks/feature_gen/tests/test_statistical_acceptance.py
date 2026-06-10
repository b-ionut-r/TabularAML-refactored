"""Tests for the statistical paired-fold acceptance gate."""
import numpy as np
import pandas as pd
import pytest

from tabularaml.generate.features import FeatureGenerator, StagnationLevel


def _make_gen(acceptance="statistical", folds_frac=0.7, min_pct_gain=0.001):
    gen = FeatureGenerator(acceptance=acceptance, acceptance_folds_frac=folds_frac,
                           min_pct_gain=min_pct_gain, log_file=None)
    return gen


def _set_stagnation(gen, level):
    gen.adaptive_controller.state.stagnation_level = level


def test_gate_rejects_below_mean_threshold():
    gen = _make_gen()
    # gain below adaptive min gain -> rejected regardless of folds
    assert gen._acceptance_gate(0.0, np.array([1.0, 1.0, 1.0, 1.0])) is False
    assert gen._acceptance_gate(-0.5, None) is False


def test_gate_sign_test_k4():
    gen = _make_gen()  # frac=0.7 -> k_req = ceil(2.8) = 3 of 4
    gain = 0.01
    assert gen._acceptance_gate(gain, np.array([1, 1, 1, -1.0])) is True   # 3/4
    assert gen._acceptance_gate(gain, np.array([1, 1, -1, -1.0])) is False  # 2/4
    assert gen._acceptance_gate(gain, np.array([1, 1, 1, 1.0])) is True    # 4/4


def test_gate_sign_test_k5():
    gen = _make_gen()  # k_req = ceil(3.5) = 4 of 5
    gain = 0.01
    assert gen._acceptance_gate(gain, np.array([1, 1, 1, 1, -1.0])) is True   # 4/5
    assert gen._acceptance_gate(gain, np.array([1, 1, 1, -1, -1.0])) is False  # 3/5


def test_gate_stagnation_relaxation():
    gen = _make_gen()
    gain = 0.01
    deltas_3of5 = np.array([1, 1, 1, -1, -1.0])
    assert gen._acceptance_gate(gain, deltas_3of5) is False
    _set_stagnation(gen, StagnationLevel.SEVERE)
    # k_req relaxes 4 -> 3 (still a majority)
    assert gen._acceptance_gate(gain, deltas_3of5) is True
    # but never below majority: 2/4 stays rejected even at CRITICAL
    _set_stagnation(gen, StagnationLevel.CRITICAL)
    assert gen._acceptance_gate(gain, np.array([1, 1, -1, -1.0])) is False


def test_gate_fallback_small_k_and_missing_folds():
    gen = _make_gen()
    gain = 0.01
    # K < 3: no statistical power -> mean rule decides
    assert gen._acceptance_gate(gain, np.array([1.0, -1.0])) is True
    # missing fold vector (splitter mismatch) -> mean rule decides
    assert gen._acceptance_gate(gain, None) is True


def test_gate_mean_mode_ignores_folds():
    gen = _make_gen(acceptance="mean")
    gain = 0.01
    assert gen._acceptance_gate(gain, np.array([1, -1, -1, -1.0])) is True


def test_zero_gain_respects_adaptive_threshold():
    gen = _make_gen(min_pct_gain=0.001)
    # exactly at threshold passes the mean check
    thr = gen.adaptive_controller.get_adaptive_min_gain()
    assert gen._acceptance_gate(thr, np.array([1, 1, 1, 1.0])) is True
    assert gen._acceptance_gate(thr * 0.99, np.array([1, 1, 1, 1.0])) is False


def _noise_search(acceptance, seed=7):
    rng = np.random.default_rng(seed)
    n = 240
    X = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(6)})
    y = pd.Series(rng.normal(size=n))  # pure noise target
    gen = FeatureGenerator(n_generations=3, n_parents=6, n_children=24, cv=4,
                           acceptance=acceptance, log_file=None,
                           use_proxy_evaluation=False, final_selection=False,
                           meta_validation_frac=0.0, adaptive=True,
                           random_state=seed, time_budget=240)
    X_out, _, _, interactions = gen.search(X, y)
    return len(interactions)


def test_statistical_accepts_fewer_noise_features():
    """On a pure-noise target every accepted feature is a false positive; the
    paired sign-test gate must not accept more of them than the mean rule."""
    n_stat = _noise_search("statistical")
    n_mean = _noise_search("mean")
    assert n_stat <= n_mean
