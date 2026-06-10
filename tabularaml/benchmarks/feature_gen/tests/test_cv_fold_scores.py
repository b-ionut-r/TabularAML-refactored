"""Tests for the refactored cross_val_score: per-fold vectors, light path,
early-stopping leak fix, parallel folds, and group-aware scorer threading."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from tabularaml.eval.cv import (cross_val_score, cross_val_fold_scores,
                                FoldScores, make_cv_splitter)
from tabularaml.eval.scorers import PREDEFINED_REG_SCORERS, Scorer

rmse = PREDEFINED_REG_SCORERS["rmse"]


def _num(X):
    """Numeric columns only (Ridge cannot consume categoricals)."""
    return X.select_dtypes("number")


class RecordingModel:
    """Mean-predicting model that records fit/predict calls across fold deepcopies
    via class-level lists (deepcopy shares the class object)."""
    fit_records = []
    predict_records = []

    def __init__(self, early_stopping_rounds=None):
        self.early_stopping_rounds = early_stopping_rounds
        self._mean = 0.0

    def get_params(self, deep=True):
        return {"early_stopping_rounds": self.early_stopping_rounds}

    def fit(self, X, y, eval_set=None, verbose=None, **kwargs):
        RecordingModel.fit_records.append({
            "train_idx": list(X.index) if hasattr(X, "index") else None,
            "eval_idx": (list(eval_set[0][0].index)
                         if eval_set is not None and hasattr(eval_set[0][0], "index")
                         else None),
        })
        self._mean = float(np.mean(y))
        return self

    def predict(self, X):
        RecordingModel.predict_records.append(
            {"idx": list(X.index) if hasattr(X, "index") else None, "n": len(X)})
        return np.full(len(X), self._mean)

    @classmethod
    def reset(cls):
        cls.fit_records = []
        cls.predict_records = []


@pytest.fixture(autouse=True)
def _reset_recording():
    RecordingModel.reset()
    yield
    RecordingModel.reset()


def test_fold_scores_match_legacy_mean(toy_reg):
    X, y = toy_reg
    X = _num(X)
    model = Ridge()
    mean_score = cross_val_score(model, X, y, rmse, cv=5)
    res = cross_val_fold_scores(model, X, y, rmse, cv=5)
    assert isinstance(res, FoldScores)
    assert len(res.fold_scores) == 5
    assert res.mean_val == pytest.approx(mean_score)
    assert np.mean(res.fold_scores) == pytest.approx(mean_score)


def test_return_dict_exposes_fold_val_scores(toy_reg):
    X, y = toy_reg
    out = cross_val_score(Ridge(), _num(X), y, rmse, cv=4, return_dict=True)
    assert "fold_val_scores" in out and len(out["fold_val_scores"]) == 4
    assert out["mean_val_score"] == pytest.approx(np.mean(out["fold_val_scores"]))
    # legacy keys intact
    assert "mean_train_score" in out and out["mean_train_score"] is not None
    assert "fold_0" in out and "train_score" in out["fold_0"] and "model" in out["fold_0"]


def test_fold_pairing_determinism(toy_reg):
    """Two runs under the same int cv produce identical splits → pairable vectors."""
    X, y = toy_reg
    a = cross_val_fold_scores(Ridge(), _num(X), y, rmse, cv=4)
    b = cross_val_fold_scores(Ridge(), _num(X), y, rmse, cv=4)
    assert np.allclose(a.fold_scores, b.fold_scores)


def test_light_path_skips_train_predictions(toy_reg):
    X, y = toy_reg
    cross_val_fold_scores(RecordingModel(), X, y, rmse, cv=5)
    light_predicts = len(RecordingModel.predict_records)
    sizes = [r["n"] for r in RecordingModel.predict_records]
    assert light_predicts == 5
    assert all(s < len(X) for s in sizes)  # val folds only, never full train

    RecordingModel.reset()
    cross_val_score(RecordingModel(), X, y, rmse, cv=5, return_dict=True)
    assert len(RecordingModel.predict_records) == 10  # train + val per fold


def test_parallel_folds_match_serial(toy_reg):
    X, y = toy_reg
    serial = cross_val_fold_scores(Ridge(), _num(X), y, rmse, cv=5, n_jobs_folds=1)
    parallel = cross_val_fold_scores(Ridge(), _num(X), y, rmse, cv=5, n_jobs_folds=2)
    assert np.allclose(serial.fold_scores, parallel.fold_scores)


def test_auto_policy_no_es_no_eval_set(toy_reg):
    X, y = toy_reg
    cross_val_score(RecordingModel(early_stopping_rounds=None), X, y, rmse, cv=4)
    assert all(r["eval_idx"] is None for r in RecordingModel.fit_records)


def test_auto_policy_es_eval_set_disjoint_from_val_fold(toy_reg):
    X, y = toy_reg
    cross_val_score(RecordingModel(early_stopping_rounds=10), X, y, rmse, cv=4)
    assert len(RecordingModel.fit_records) == 4
    for fit_rec, pred_rec in zip(RecordingModel.fit_records, RecordingModel.predict_records):
        es_idx = fit_rec["eval_idx"]
        val_idx = pred_rec["idx"]
        assert es_idx is not None and len(es_idx) > 0
        # ES rows never overlap the validation fold (no optimistic bias)
        assert set(es_idx).isdisjoint(set(val_idx))
        # ES rows are excluded from the rows actually trained on
        assert set(es_idx).isdisjoint(set(fit_rec["train_idx"]))


def test_legacy_policy_passes_val_fold(toy_cls):
    # Discrete target: the legacy unseen-label guard passes and the val fold
    # itself is handed to fit as eval_set (the pre-existing leaky behavior).
    X, y = toy_cls
    cross_val_score(RecordingModel(), X, y.astype(float), rmse, cv=4,
                    eval_set_policy="legacy")
    for fit_rec, pred_rec in zip(RecordingModel.fit_records, RecordingModel.predict_records):
        assert fit_rec["eval_idx"] == pred_rec["idx"]


def test_none_policy_never_passes_eval_set(toy_reg):
    X, y = toy_reg
    cross_val_score(RecordingModel(early_stopping_rounds=10), X, y, rmse, cv=4,
                    eval_set_policy="none")
    assert all(r["eval_idx"] is None for r in RecordingModel.fit_records)


class _GroupRecordingScorer:
    """Minimal group-aware scorer (mirrors the GroupAwareScorer contract)."""
    name = "group_recording"
    from_probs = False
    greater_is_better = False
    needs_groups = True
    received = []

    def score(self, y_true, y_pred, groups=None):
        _GroupRecordingScorer.received.append(np.asarray(groups))
        return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

    def score_per_group(self, y_true, y_pred, groups):
        out = {}
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        for g in np.unique(groups):
            m = groups == g
            out[g] = float(np.mean((y_true[m] - y_pred[m]) ** 2))
        return out


def test_group_aware_scorer_receives_val_group_slices(toy_reg):
    X, y = toy_reg
    X = _num(X)
    groups = np.repeat(np.arange(10), len(X) // 10)
    _GroupRecordingScorer.received = []
    out = cross_val_score(Ridge(), X, y, _GroupRecordingScorer(), cv=5,
                          groups=groups, return_dict=True,
                          compute_train_scores=False, keep_fold_models=False)
    # Each fold's scorer call got exactly the val-slice of groups
    assert len(_GroupRecordingScorer.received) == 5
    seen = np.concatenate(_GroupRecordingScorer.received)
    assert sorted(np.unique(seen)) == sorted(np.unique(groups))
    # per-group scores merged across folds: every era present exactly once (GroupKFold)
    assert "val_group_scores" in out
    assert sorted(out["val_group_scores"].keys()) == sorted(np.unique(groups))
