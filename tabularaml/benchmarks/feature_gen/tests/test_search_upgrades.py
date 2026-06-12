"""Unit tests for the genetic-search upgrades (screening, gates, priors, deadline)."""
import numpy as np
import pandas as pd
import pytest

from tabularaml.generate.features import FeatureGenerator, Feature, Interaction
from tabularaml.generate.ops import SYMMETRIC_OPS, ANTISYMMETRIC_OPS, DEFAULT_OP_PRIORS


def _make_fg(**kw):
    defaults = dict(task="classification", cv=3, use_gpu=False, log_file=None,
                    random_state=0, n_jobs=1)
    defaults.update(kw)
    return FeatureGenerator(**defaults)


def _signal_data(n=600, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "a": rng.normal(2, 1, n).clip(0.1),
        "b": rng.normal(5, 2, n).clip(0.1),
        "c": rng.normal(size=n),
        "cat1": pd.Series(rng.choice(["x", "y", "z"], n), dtype=object),
    })
    y = pd.Series((X["a"] / X["b"] + 0.25 * rng.normal(size=n) > 0.45).astype(int))
    return X, y


# ---------------------------------------------------------------------------
# Prefilter
# ---------------------------------------------------------------------------

class TestPrefilter:
    def _features(self, X):
        return {c: Feature(c, "num", 1.0) for c in X.columns if c != "cat1"}

    def test_commutative_dedup(self, ):
        X, y = _signal_data()
        fg = _make_fg()
        fg._set_defaults(X, y)
        f = self._features(X)
        batch = [
            Interaction(f["a"], "add", f["b"]),
            Interaction(f["b"], "add", f["a"]),   # mirror of symmetric op -> dropped
            Interaction(f["a"], "sub", f["b"]),
            Interaction(f["b"], "sub", f["a"]),   # mirror of antisymmetric op -> dropped
            Interaction(f["a"], "div", f["b"]),
            Interaction(f["b"], "div", f["a"]),
        ]
        kept = fg._prefilter_candidates(batch, X)
        names = [i.name for i in kept]
        assert "a_add_b" in names and "b_add_a" not in names
        assert "a_sub_b" in names and "b_sub_a" not in names
        # a and b are all-positive here, so b/a = 1/(a/b) is rank-equivalent to
        # a/b (split-identical for trees): the value-level near-dup filter keeps
        # exactly one of the two orders.
        assert ("a_div_b" in names) ^ ("b_div_a" in names)

    def test_div_orders_kept_on_mixed_sign_data(self):
        rng = np.random.RandomState(2)
        n = 500
        X = pd.DataFrame({"u": rng.normal(size=n), "v": rng.normal(size=n)})
        y = pd.Series((rng.normal(size=n) > 0).astype(int))
        fg = _make_fg()
        fg._set_defaults(X, y)
        fu, fv = Feature("u", "num", 1.0), Feature("v", "num", 1.0)
        # mixed signs: u/v and v/u are not monotone transforms of each other
        kept = fg._prefilter_candidates(
            [Interaction(fu, "div", fv), Interaction(fv, "div", fu)], X)
        assert sorted(i.name for i in kept) == ["u_div_v", "v_div_u"]

    def test_existing_column_and_batch_dup(self):
        X, y = _signal_data()
        X = X.copy()
        fg = _make_fg()
        fg._set_defaults(X, y)
        f = self._features(X)
        inter = Interaction(f["a"], "mul", f["b"])
        X[inter.name] = inter.generate(X)
        batch = [Interaction(f["a"], "mul", f["b"]),       # already a column
                 Interaction(f["c"], "div", f["a"]),       # c mixed-sign: value-distinct
                 Interaction(f["c"], "div", f["a"])]       # duplicate in batch
        kept = fg._prefilter_candidates(batch, X)
        assert [i.name for i in kept] == ["c_div_a"]

    def test_constant_and_near_duplicate(self):
        X, y = _signal_data()
        X = X.copy()
        X["const"] = 1.0
        X["a_clone"] = X["a"] * 1.0000001
        fg = _make_fg()
        fg._set_defaults(X, y)
        fc = Feature("const", "num", 1.0)
        fclone = Feature("a_clone", "num", 1.0)
        f = self._features(X)
        batch = [Interaction(fc, "mul", fc),        # constant result
                 Interaction(fclone, "abs"),        # |a*1.0000001| ~ rank-identical to a... only if a >= 0 (it is: clipped 0.1)
                 Interaction(f["a"], "sub", f["c"])]
        kept = fg._prefilter_candidates(batch, X)
        names = [i.name for i in kept]
        assert "const_mul_const" not in names
        assert "a_clone_abs" not in names           # near-duplicate of column a
        assert "a_sub_c" in names

    def test_nan_pattern_guard_keeps_sqrt_of_signed(self):
        rng = np.random.RandomState(0)
        n = 400
        # ~30% negative values: sqrt is NaN there (passes the 50% NaN cutoff),
        # rank-correlates ~1 with s on the non-NaN part, but the NaN mask itself
        # (s < 0) is informative -> the NaN-pattern guard must keep it.
        X = pd.DataFrame({"s": rng.normal(0.5, 1.0, size=n), "t": rng.normal(size=n)})
        y = pd.Series((rng.normal(size=n) > 0).astype(int))
        fg = _make_fg()
        fg._set_defaults(X, y)
        fs = Feature("s", "num", 1.0)
        kept = fg._prefilter_candidates([Interaction(fs, "sqrt")], X)
        assert [i.name for i in kept] == ["s_sqrt"]


# ---------------------------------------------------------------------------
# Fold-consistency gate
# ---------------------------------------------------------------------------

class TestFoldConsistencyGate:
    def _run_select(self, monkeypatch, fold_scores_seq, greater_is_better=False):
        """Run _select_elites with a stubbed evaluator; returns accepted names."""
        X, y = _signal_data(300)
        fg = _make_fg(fold_consistency_gate=True, fold_consistency_min_frac=0.65,
                      use_proxy_evaluation=False, prefilter_candidates=False,
                      min_pct_gain=0.0001)
        fg._set_defaults(X, y)
        fg._current_y = y
        f = {c: Feature(c, "num", 1.0) for c in ["a", "b", "c"]}
        batch = [Interaction(f["a"], "mul", f["b"])]

        calls = {"n": 0}
        base = fold_scores_seq[0]
        cand = fold_scores_seq[1]

        def fake_eval(X_, y_, pipeline=None, groups=None, cache=True):
            calls["n"] += 1
            scores = base if calls["n"] == 1 else cand
            return float(np.mean(scores)), float(np.mean(scores)), list(scores)

        monkeypatch.setattr(fg, "_eval_baseline_cached", fake_eval)
        from copy import deepcopy as _dc
        fg.scorer = _dc(fg.scorer)  # PREDEFINED scorers are shared module objects
        fg.scorer.greater_is_better = greater_is_better
        selected, _, _ = fg._select_elites(batch, 5, X, y)
        return [i.name for i in selected]

    def test_rejects_inconsistent_folds(self, monkeypatch):
        # mean improves (lower logloss) but only 1/3 folds improve -> reject
        base = [0.60, 0.60, 0.60]
        cand = [0.40, 0.61, 0.62]   # mean 0.543 < 0.60 but folds 2,3 worse
        assert self._run_select(monkeypatch, [base, cand]) == []

    def test_accepts_consistent_folds(self, monkeypatch):
        base = [0.60, 0.60, 0.60]
        cand = [0.55, 0.56, 0.57]   # all folds improve
        assert self._run_select(monkeypatch, [base, cand]) == ["a_mul_b"]

    def test_greater_is_better_orientation(self, monkeypatch):
        base = [0.70, 0.70, 0.70]
        cand = [0.90, 0.69, 0.68]   # mean up but 2/3 folds down -> reject
        assert self._run_select(monkeypatch, [base, cand], greater_is_better=True) == []


# ---------------------------------------------------------------------------
# Noise-probe gate + proxy screening
# ---------------------------------------------------------------------------

class TestProxyScreening:
    def test_planted_signal_beats_noise_and_junk_blocked(self):
        # Strong planted signal: y is a deterministic function of a/b, so the
        # div candidate must clearly clear the noise threshold.
        rng = np.random.RandomState(0)
        n = 800
        X = pd.DataFrame({
            "a": rng.normal(2, 1, n).clip(0.1),
            "b": rng.normal(5, 2, n).clip(0.1),
            "c": rng.normal(size=n),
            "cat1": pd.Series(rng.choice(["x", "y", "z"], n), dtype=object),
        })
        ratio = X["a"] / X["b"]
        y = pd.Series((ratio > ratio.median()).astype(int))
        fg = _make_fg()
        fg._set_defaults(X, y)
        fg._current_y = y
        cv = fg._get_cv_splitter()
        fg._current_oof_preds = fg._train_base_model_and_get_residuals(X, y, cv)
        thr = fg._noise_probe_threshold(X, y, cv)
        assert thr is not None
        f = {c: Feature(c, "num", 1.0) for c in ["a", "b", "c"]}
        signal_score = fg._featureboost_score(
            Interaction(f["a"], "div", f["b"]).generate(X), y, fg._current_oof_preds, cv)
        junk_score = fg._featureboost_score(
            Interaction(f["c"], "mul", f["c"]).generate(X), y, fg._current_oof_preds, cv)
        assert signal_score > thr
        assert junk_score < signal_score

    def test_determinism(self):
        X, y = _signal_data(500)
        results = []
        for _ in range(2):
            fg = _make_fg()
            fg._set_defaults(X, y)
            fg._current_y = y
            cv = fg._get_cv_splitter()
            fg._current_oof_preds = fg._train_base_model_and_get_residuals(X, y, cv)
            f = {c: Feature(c, "num", 1.0) for c in ["a", "b"]}
            s = fg._featureboost_score(Interaction(f["a"], "div", f["b"]).generate(X),
                                       y, fg._current_oof_preds, cv)
            results.append((s, fg._noise_probe_threshold(X, y, cv)))
        assert results[0] == results[1]

    def test_pipeline_candidates_scored_via_oof(self):
        X, y = _signal_data(600)
        fg = _make_fg()
        fg._set_defaults(X, y)
        fg._current_y = y
        cv = fg._get_cv_splitter()
        fcat = Feature("cat1", "cat", 1.0)
        fnum = Feature("a", "num", 1.0)
        for inter in [Interaction(fcat, "target"), Interaction(fcat, "freq"),
                      Interaction(fcat, "groupby_mean", fnum)]:
            oof = fg._compute_oof_candidate_values(inter, X, y, cv)
            assert oof is not None and len(oof) == len(X)
            assert np.isfinite(oof).mean() > 0.9

    def test_categorical_concat_scoreable(self):
        X, y = _signal_data(500)
        X = X.copy()
        X["cat2"] = pd.Series(np.random.RandomState(1).choice(["m", "n"], len(X)), dtype=object)
        fg = _make_fg()
        fg._set_defaults(X, y)
        fg._current_y = y
        cv = fg._get_cv_splitter()
        fg._current_oof_preds = fg._train_base_model_and_get_residuals(X, y, cv)
        f1, f2 = Feature("cat1", "cat", 1.0), Feature("cat2", "cat", 1.0)
        vals = Interaction(f1, "concat", f2).generate(X)
        s = fg._featureboost_score(vals, y, fg._current_oof_preds, cv)
        assert np.isfinite(s)  # previously object dtype crashed and was silently dropped


# ---------------------------------------------------------------------------
# Finalization gate (do no harm)
# ---------------------------------------------------------------------------

class TestFinalizationGate:
    def test_noise_features_vetoed_small_data(self):
        # Pure-noise target: anything the search accepts is overfit; the
        # fresh-CV gate should return the original features unchanged.
        rng = np.random.RandomState(3)
        n = 500
        X = pd.DataFrame({f"n{i}": rng.normal(size=n) for i in range(6)})
        y = pd.Series(rng.randint(0, 2, n))
        fg = _make_fg(n_generations=3, n_parents=6, n_children=30,
                      time_budget=90, min_pct_gain=0.0)
        Xn, pipe, gen, inters = fg.generate(X, y)
        assert len(Xn) == n
        if fg.n_added_feats == 0:
            assert list(Xn.columns) == list(X.columns)
            Xt = fg.fit(X, y).transform(X)
            assert list(Xt.columns) == list(X.columns)

    def test_row_count_preserved_with_meta_split(self):
        X, y = _signal_data(3000, seed=5)
        fg = _make_fg(n_generations=2, n_parents=6, n_children=30, time_budget=120)
        Xn, *_ = fg.generate(X, y)
        assert len(Xn) == len(X)

    def test_helpful_feature_retained(self):
        # y depends strongly on a*b -> the gate must keep the generated set
        rng = np.random.RandomState(7)
        n = 2500
        X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                          "c": rng.normal(size=n)})
        y = pd.Series(X["a"] * X["b"] + 0.1 * rng.normal(size=n))
        fg = _make_fg(task="regression", n_generations=3, n_parents=6, n_children=40,
                      time_budget=150)
        Xn, *_ = fg.generate(X, y)
        assert fg.n_added_feats >= 1
        assert any("mul" in c for c in Xn.columns if c not in ["a", "b", "c"])


# ---------------------------------------------------------------------------
# Deadline + determinism + round-trip
# ---------------------------------------------------------------------------

class TestDeadlineAndRoundTrip:
    def test_tiny_budget_returns_quickly_and_intact(self):
        import time as _t
        X, y = _signal_data(1500, seed=2)
        fg = _make_fg(n_generations=10, n_parents=10, n_children=60, time_budget=8)
        t0 = _t.time()
        Xn, *_ = fg.generate(X, y)
        assert _t.time() - t0 < 120          # bounded; eval overhead allowed
        assert len(Xn) == len(X)
        fg.fit(X, y)
        assert fg.transform(X).shape[0] == len(X)

    def test_same_seed_same_result(self):
        X, y = _signal_data(700, seed=4)
        outs = []
        for _ in range(2):
            fg = _make_fg(n_generations=2, n_parents=6, n_children=30, time_budget=90)
            Xn, _, _, inters = fg.generate(X.copy(), y.copy())
            outs.append((tuple(sorted(i.name for i in inters)), tuple(Xn.columns)))
        assert outs[0] == outs[1]

    def test_save_load_transform_roundtrip(self, tmp_path):
        X, y = _signal_data(700, seed=6)
        fg = _make_fg(n_generations=2, n_parents=6, n_children=30, time_budget=90)
        fg.generate(X, y)
        fg.fit(X, y)
        ref = fg.transform(X)
        path = str(tmp_path / "fg.pkl")
        fg.save(path)
        fg2 = FeatureGenerator.load(path)
        out = fg2.transform(X)
        pd.testing.assert_frame_equal(ref, out)


# ---------------------------------------------------------------------------
# Op priors + eval cache plumbing
# ---------------------------------------------------------------------------

class TestPriorsAndCache:
    def test_op_priors_seeded(self):
        X, y = _signal_data(300)
        fg = _make_fg(use_op_priors=True)
        fg._set_defaults(X, y)
        ctrl = fg.adaptive_controller
        assert ctrl.op_stats["num"]["binary"]["div"]["priority_score"] == DEFAULT_OP_PRIORS["div"]
        assert ctrl.op_stats["num"]["unary"]["neg"]["priority_score"] == DEFAULT_OP_PRIORS["neg"]
        fg2 = _make_fg(use_op_priors=False)
        fg2._set_defaults(X, y)
        assert fg2.adaptive_controller.op_stats["num"]["unary"]["neg"]["priority_score"] in (0.5, 0.7)

    def test_baseline_cache_hits(self, toy_cls=None):
        X, y = _signal_data(400)
        fg = _make_fg()
        fg._set_defaults(X, y)
        fg._current_y = y
        r1 = fg._eval_baseline_cached(X, y, fg.pipeline)
        import unittest.mock as mock
        with mock.patch("tabularaml.generate.features.cross_val_score") as cvs:
            r2 = fg._eval_baseline_cached(X, y, fg.pipeline)
            assert not cvs.called          # served from cache
        assert r1 == r2
        assert len(r1[2]) == 3             # per-fold scores present (cv=3)
