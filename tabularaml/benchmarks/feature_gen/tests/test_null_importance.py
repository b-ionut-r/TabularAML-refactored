"""Tests for null-importance (target permutation) post-selection."""
import numpy as np
import pandas as pd
import pytest

from tabularaml.generate.features import FeatureGenerator


def _setup(seed=0, n=500):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"orig_a": rng.normal(size=n), "orig_b": rng.normal(size=n)})
    y = pd.Series(2.0 * X["orig_a"] + rng.normal(size=n) * 0.3)
    # generated features: one informative, eleven pure noise
    X["gen_signal"] = X["orig_a"] * 2 + rng.normal(size=n) * 0.05
    for i in range(11):
        X[f"gen_noise_{i}"] = rng.normal(size=n)
    gen = FeatureGenerator(log_file=None, null_importance_selection=True,
                           null_importance_n_perm=4)
    gen._set_defaults(X, y)
    gen.initial_features = ["orig_a", "orig_b"]
    return gen, X, y


def test_null_importance_drops_noise_keeps_signal():
    gen, X, y = _setup()
    drop = gen._null_importance_selection(X, y)
    assert "gen_signal" not in drop
    assert len([f for f in drop if f.startswith("gen_noise_")]) >= 6  # most noise dies
    # originals are never dropped
    assert not any(f.startswith("orig_") for f in drop)


def test_null_importance_deterministic():
    gen1, X, y = _setup()
    gen2, _, _ = _setup()
    assert gen1._null_importance_selection(X, y) == gen2._null_importance_selection(X, y)


def test_null_importance_disabled_or_too_few():
    gen, X, y = _setup()
    gen.null_importance_selection = False
    assert gen._null_importance_selection(X, y) == []
    gen.null_importance_selection = True
    # fewer than 10 generated features -> no-op
    X_small = X[["orig_a", "orig_b", "gen_signal", "gen_noise_0"]]
    assert gen._null_importance_selection(X_small, y) == []
