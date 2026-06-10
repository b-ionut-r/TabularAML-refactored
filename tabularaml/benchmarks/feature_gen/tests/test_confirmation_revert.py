"""Tests for multi-seed generation confirmation and its revert behavior."""
import numpy as np
import pandas as pd
import pytest

import tabularaml.generate.features as fg
from tabularaml.generate.features import FeatureGenerator
from tabularaml.eval.cv import FoldScores


def _gen_with_best(toy_reg, seeds=1):
    X, y = toy_reg
    gen = FeatureGenerator(log_file=None, cv=4, confirmation_seeds=seeds)
    gen._set_defaults(X, y)
    gen._cv_int_hint = 4
    gen._groups_active = None
    gen.state = {"best": dict(X=X.copy(), pipeline=None, val_score=1.0)}
    return gen, X, y


def test_confirmation_seeds_zero_skips(toy_reg, monkeypatch):
    gen, X, y = _gen_with_best(toy_reg, seeds=0)

    def boom(*a, **k):
        raise AssertionError("CV must not run when confirmation is disabled")

    monkeypatch.setattr(fg, "cross_val_fold_scores", boom)
    assert gen._confirm_generation(X, y, None) is True


def test_confirmation_paired_sign(toy_reg, monkeypatch):
    gen, X, y = _gen_with_best(toy_reg, seeds=2)
    results = {"new": np.array([0.5, 0.5, 0.5, 0.5]),   # rmse: lower is better
               "best": np.array([0.6, 0.6, 0.6, 0.6])}
    seen_cvs = []

    def fake_scores(model, X_eval, y_eval, scorer, cv=None, **kwargs):
        seen_cvs.append(cv)
        key = "new" if len(X_eval.columns) != len(gen.state["best"]["X"].columns) else "best"
        v = results[key]
        return FoldScores(mean_val=float(np.mean(v)), fold_scores=v)

    monkeypatch.setattr(fg, "cross_val_fold_scores", fake_scores)
    X_new = X.copy()
    X_new["extra"] = 1.0
    # new has LOWER rmse on every alternate fold -> confirmed
    assert gen._confirm_generation(X_new, y, None) is True
    # both eval calls of a seed must use a distinct (fresh) splitter object
    assert len(seen_cvs) == 4 and seen_cvs[0] is not seen_cvs[1]

    # flip: new is worse -> rejected
    results["new"], results["best"] = results["best"], results["new"]
    assert gen._confirm_generation(X_new, y, None) is False


def test_confirmation_skips_unreseedable_splitter(toy_reg, monkeypatch):
    gen, X, y = _gen_with_best(toy_reg, seeds=1)
    gen._cv_int_hint = None

    class OpaqueSplitter:
        def split(self, X, y=None, groups=None):
            yield np.arange(10), np.arange(10, 20)

    gen.cv = OpaqueSplitter()

    def boom(*a, **k):
        raise AssertionError("CV must not run when splitter cannot be reseeded")

    monkeypatch.setattr(fg, "cross_val_fold_scores", boom)
    X_new = X.copy()
    X_new["extra"] = 1.0
    assert gen._confirm_generation(X_new, y, None) is True


def test_failed_confirmation_reverts_everything():
    """A search whose every improvement fails confirmation must end with the
    original feature set (all generations reverted)."""
    rng = np.random.default_rng(3)
    n = 300
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                      "c": rng.normal(size=n)})
    y = pd.Series(X["a"] * X["b"] + rng.normal(size=n) * 0.1)
    gen = FeatureGenerator(n_generations=3, n_parents=6, n_children=24, cv=3,
                           log_file=None, use_proxy_evaluation=False,
                           final_selection=False, meta_validation_frac=0.0,
                           early_stopping_iter=10, time_budget=150)
    gen._confirm_generation = lambda *a, **k: False
    X_out, _, _, interactions = gen.search(X, y)
    assert len(interactions) == 0
    assert list(X_out.columns) == ["a", "b", "c"]
