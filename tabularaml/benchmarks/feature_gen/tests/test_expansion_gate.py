"""Tests for the paired-fold gate on base-expansion blocks."""
import numpy as np
import pandas as pd
import pytest

from tabularaml.generate.features import FeatureGenerator
from tabularaml.generate.expanders import BaselineFeatureExpander
from tabularaml.eval.cv import FoldScores


def _gen_with_expansion(seed=0, n=200):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                      "c": rng.normal(size=n)})
    y = pd.Series(X["a"] + rng.normal(size=n) * 0.2)
    gen = FeatureGenerator(log_file=None, cv=4)
    gen._set_defaults(X, y)
    expander = BaselineFeatureExpander(datetime_features=False, row_stats=True).fit(X)
    X_exp = expander.transform(X)
    gen.base_expander = expander
    assert "row_mean" in X_exp.columns
    return gen, X_exp, y


def _patch_scores(gen, monkeypatch, rowstat_delta):
    """Scores depend on whether row-stat columns are present (rmse: lower=better)."""
    base = np.array([1.0, 1.0, 1.0, 1.0])

    def fake(X, y, pipeline=None, groups=None):
        v = base + (rowstat_delta if "row_mean" in X.columns else 0.0)
        return FoldScores(mean_val=float(np.mean(v)), fold_scores=v)

    monkeypatch.setattr(gen, "_eval_cv_light", fake)


def test_gate_drops_harmful_row_stats(monkeypatch):
    gen, X_exp, y = _gen_with_expansion()
    _patch_scores(gen, monkeypatch, rowstat_delta=+0.1)  # worse on every fold
    X_out = gen._gate_base_expansion(X_exp, y)
    assert "row_mean" not in X_out.columns
    # nothing left to expand (no datetime cols either) -> expander removed
    # entirely, so transform() is a clean no-op on fresh data
    assert gen.base_expander is None


def test_gate_keeps_helpful_row_stats(monkeypatch):
    gen, X_exp, y = _gen_with_expansion()
    _patch_scores(gen, monkeypatch, rowstat_delta=-0.1)  # better on every fold
    X_out = gen._gate_base_expansion(X_exp, y)
    assert "row_mean" in X_out.columns
    assert "row_mean" in gen.base_expander.row_stat_outputs_


def test_gate_drops_neutral_row_stats(monkeypatch):
    """Burden of proof is on the block: CV-neutral expansions are dropped
    (they previously passed and then hurt holdout on some datasets)."""
    gen, X_exp, y = _gen_with_expansion()
    _patch_scores(gen, monkeypatch, rowstat_delta=0.0)  # identical -> no win -> drop
    X_out = gen._gate_base_expansion(X_exp, y)
    assert "row_mean" not in X_out.columns


def test_search_schema_consistent_after_gate_drop(monkeypatch):
    """End-to-end: when the gate drops row stats, search output and transform
    output agree on the schema."""
    rng = np.random.default_rng(4)
    n = 300
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                      "c": rng.normal(size=n)})
    y = pd.Series(X["a"] * X["b"] + rng.normal(size=n) * 0.1)
    gen = FeatureGenerator(n_generations=2, n_parents=5, n_children=16, cv=3,
                           log_file=None, use_proxy_evaluation=False,
                           final_selection=False, meta_validation_frac=0.0,
                           early_stopping_iter=10, time_budget=90)

    orig_gate = gen._gate_base_expansion

    def forcing_gate(X_in, y_in):
        # force-fail the row-stat block by faking degraded fold scores
        real_eval = gen._eval_cv_light

        def biased(Xe, ye, pipeline=None, groups=None):
            res = real_eval(Xe, ye, pipeline)
            if "row_mean" in Xe.columns:
                res = FoldScores(mean_val=res.mean_val + 1.0,
                                 fold_scores=res.fold_scores + 1.0)
            return res

        gen._eval_cv_light = biased
        try:
            return orig_gate(X_in, y_in)
        finally:
            gen._eval_cv_light = real_eval

    gen._gate_base_expansion = forcing_gate
    X_out, _, _, _ = gen.search(X, y)
    assert "row_mean" not in X_out.columns
    assert "row_mean" not in gen.initial_features
    fresh = gen.transform(X.iloc[:30].copy())
    assert "row_mean" not in fresh.columns
