"""Tests for FoldEvalState caching and invalidation (cv_epoch / rows / columns)."""
import numpy as np
import pandas as pd
import pytest

from tabularaml.generate.features import FeatureGenerator, FoldEvalState
from tabularaml.eval.cv import FoldScores


@pytest.fixture
def fitted_gen(toy_reg):
    X, y = toy_reg
    gen = FeatureGenerator(log_file=None, cv=4)
    gen._set_defaults(X, y)
    return gen, X, y


def _count_cv_calls(gen, monkeypatch):
    calls = {"n": 0}
    orig = gen._eval_cv_light

    def counting(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(gen, "_eval_cv_light", counting)
    return calls


def test_cache_hit_on_unchanged_state(fitted_gen, monkeypatch):
    gen, X, y = fitted_gen
    calls = _count_cv_calls(gen, monkeypatch)
    first = gen._get_baseline_fold_scores(X, y)
    assert calls["n"] == 1
    second = gen._get_baseline_fold_scores(X, y)
    assert calls["n"] == 1  # cache hit, no extra CV
    assert np.allclose(first.fold_scores, second.fold_scores)


def test_epoch_bump_forces_recompute(fitted_gen, monkeypatch):
    gen, X, y = fitted_gen
    calls = _count_cv_calls(gen, monkeypatch)
    gen._get_baseline_fold_scores(X, y)
    gen._bump_cv_epoch("rotation")
    gen._get_baseline_fold_scores(X, y)
    assert calls["n"] == 2


def test_column_change_forces_recompute(fitted_gen, monkeypatch):
    gen, X, y = fitted_gen
    calls = _count_cv_calls(gen, monkeypatch)
    gen._get_baseline_fold_scores(X, y)
    X2 = X.copy()
    X2["extra"] = X2["num_a"] * 2
    gen._get_baseline_fold_scores(X2, y)
    assert calls["n"] == 2


def test_row_change_forces_recompute(fitted_gen, monkeypatch):
    gen, X, y = fitted_gen
    calls = _count_cv_calls(gen, monkeypatch)
    gen._get_baseline_fold_scores(X, y)
    gen._get_baseline_fold_scores(X.iloc[:150], y.iloc[:150])
    assert calls["n"] == 2


def test_best_state_roundtrips_fold_vector(fitted_gen):
    gen, X, y = fitted_gen
    res = gen._get_baseline_fold_scores(X, y)
    epoch = gen._cv_epoch

    # Simulate the search-loop bookkeeping around _save_current_as_best
    gen.X, gen.generation, gen.interactions = X, [], []
    gen.pruned_features = set()
    gen.state = {"best": dict(gen_num=0, val_score=res.mean_val, train_score=0,
                              X=None, pipeline=None, generation=[],
                              pruned_features=set(), interactions=[])}
    gen.save_path = None
    gen._save_current_as_best()
    assert gen.state["best"]["val_fold_scores"] is not None
    assert gen.state["best"]["fold_cv_epoch"] == epoch

    # Poison the live cache, then revert: vector must be restored and valid
    gen._best_fold_state = FoldEvalState(fold_scores=np.array([99.0]), cv_epoch=123,
                                         n_rows=1, cols_hash=0)
    assert gen._revert_to_best() is True
    assert np.allclose(gen._best_fold_state.fold_scores, res.fold_scores)
    assert gen._best_fold_state.matches(epoch, X)

    # After a rotation, the restored vector is stale and must not match
    gen._bump_cv_epoch("rotation")
    assert not gen._best_fold_state.matches(gen._cv_epoch, X)
