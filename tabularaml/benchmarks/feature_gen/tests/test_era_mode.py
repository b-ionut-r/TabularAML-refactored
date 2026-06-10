"""Tests for era mode: scorers, whole-era sampling, grouped CV, stability gate."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from tabularaml.eval.scorers import (PREDEFINED_REG_SCORERS, GroupAwareScorer,
                                     _spearman_correlation_score)
from tabularaml.eval.splitters import RotatedGroupKFold
from tabularaml.generate.features import FeatureGenerator


def _era_data(n_eras=40, per_era=30, seed=0):
    rng = np.random.default_rng(seed)
    n = n_eras * per_era
    eras = np.repeat([f"era{i:03d}" for i in range(n_eras)], per_era)
    X = pd.DataFrame({
        "era": eras,
        "f1": rng.normal(size=n), "f2": rng.normal(size=n),
        "f3": rng.normal(size=n), "f4": rng.normal(size=n),
    })
    y = pd.Series(X["f1"] * 0.5 + X["f2"] * X["f3"] * 0.5 + rng.normal(size=n) * 0.5)
    return X, y


def test_era_spearman_matches_manual():
    rng = np.random.default_rng(1)
    n = 200
    groups = np.repeat(np.arange(10), 20)
    y_true = rng.normal(size=n)
    y_pred = y_true * 0.5 + rng.normal(size=n)
    scorer = PREDEFINED_REG_SCORERS["era_spearman"]
    got = scorer.score(y_true, y_pred, groups=groups)
    manual = np.mean([spearmanr(y_true[groups == g], y_pred[groups == g])[0]
                      for g in range(10)])
    assert got == pytest.approx(manual, abs=1e-9)


def test_era_spearman_sharpe_penalizes_instability():
    rng = np.random.default_rng(2)
    groups = np.repeat(np.arange(10), 30)
    y_true = rng.normal(size=300)
    # stable predictor: same modest skill in every era
    stable = y_true + rng.normal(size=300) * 1.0
    sharpe = PREDEFINED_REG_SCORERS["era_spearman_sharpe"]
    mean_sc = PREDEFINED_REG_SCORERS["era_spearman"]
    # unstable predictor: perfect in half the eras, anti-correlated in the rest
    unstable = np.where(np.isin(groups, np.arange(5)), y_true, -y_true)
    assert mean_sc.score(y_true, unstable, groups=groups) < 0.2  # near zero mean
    assert sharpe.score(y_true, stable, groups=groups) > sharpe.score(y_true, unstable, groups=groups)


def test_group_aware_scorer_global_fallback_warns_once():
    scorer = GroupAwareScorer("t", _spearman_correlation_score, True, {}, aggregation="mean")
    y = np.arange(50, dtype=float)
    with pytest.warns(UserWarning):
        a = scorer.score(y, y)
    assert a == pytest.approx(1.0)


def test_whole_era_subsampling_never_splits_eras():
    X, y = _era_data(n_eras=40, per_era=30)
    gen = FeatureGenerator(log_file=None, era_col="era", cv=4)
    groups = X["era"].values
    Xs, ys, gs = gen._create_search_subsample(X.drop(columns=["era"]), y, 400, groups)
    counts = pd.Series(gs).value_counts()
    assert (counts == 30).all()           # only complete eras
    assert len(Xs) >= 400                  # met the row budget


def test_era_gate_rejects_narrow_winners():
    gen = FeatureGenerator(log_file=None, era_col="era", era_acceptance_frac=0.55)
    gain = 0.05
    folds_ok = np.array([1.0, 1.0, 1.0, 1.0])
    # helps in 4 of 10 eras -> below 55% -> rejected despite good folds
    era_deltas = np.array([1] * 4 + [-1] * 6, dtype=float)
    assert gen._acceptance_gate(gain, folds_ok, era_deltas) is False
    # helps in 7 of 10 -> accepted
    era_deltas = np.array([1] * 7 + [-1] * 3, dtype=float)
    assert gen._acceptance_gate(gain, folds_ok, era_deltas) is True
    # too few shared eras -> era gate silent, fold gate decides
    assert gen._acceptance_gate(gain, folds_ok, np.array([-1.0, -1.0])) is True


def test_era_search_end_to_end():
    X, y = _era_data(n_eras=30, per_era=20, seed=5)
    gen = FeatureGenerator(n_generations=2, n_parents=6, n_children=24, cv=3,
                           log_file=None, era_col="era",
                           scorer=PREDEFINED_REG_SCORERS["era_spearman"],
                           task="regression",
                           use_proxy_evaluation=False, final_selection=False,
                           meta_validation_frac=0.0, early_stopping_iter=10,
                           expand_row_stats=False, time_budget=150)
    X_out, _, _, _ = gen.search(X, y)
    # era column is consumed as grouping, never a feature
    assert "era" not in X_out.columns
    assert isinstance(gen.cv, RotatedGroupKFold)
    # per-era baseline vector was captured for the stability gate
    assert gen._best_fold_state.per_era is None or len(gen._best_fold_state.per_era) > 0
