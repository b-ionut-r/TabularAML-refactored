"""Leakage probe for the OOF proxy scoring of pipeline-required candidates.

A target encoding of a unique-id-like column is the canonical leakage trap: a
transductive (fit-on-all) encoding memorizes y perfectly, while a proper
fold-fitted OOF encoding of the same column carries no signal. The OOF values
used for proxy screening must behave like the latter.
"""
import numpy as np
import pandas as pd

from tabularaml.generate.features import FeatureGenerator, Feature, Interaction


def _id_leak_data(n=900, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({
        "id_like": pd.Series([f"row_{i}" for i in range(n)], dtype=object),  # unique per row
        "num_a": rng.normal(size=n),
        "num_b": rng.normal(size=n),
    })
    y = pd.Series(rng.randint(0, 2, n))  # pure noise target
    return X, y


def test_oof_target_encoding_of_id_column_scores_like_noise():
    X, y = _id_leak_data()
    fg = FeatureGenerator(task="classification", cv=3, use_gpu=False, log_file=None,
                          random_state=0, n_jobs=1)
    fg._set_defaults(X, y)
    fg._current_y = y
    cv = fg._get_cv_splitter()
    fg._current_oof_preds = fg._train_base_model_and_get_residuals(X, y, cv)

    fid = Feature("id_like", "cat", 1.0)
    inter = Interaction(fid, "target")
    oof_vals = fg._compute_oof_candidate_values(inter, X, y, cv)
    assert oof_vals is not None

    # The transductive (leaky) encoding correlates strongly with y...
    import category_encoders as ce
    leaky = ce.TargetEncoder(cols=["id_like"]).fit(X[["id_like"]], y).transform(X[["id_like"]])["id_like"]
    leaky_corr = abs(np.corrcoef(leaky, y)[0, 1])
    assert leaky_corr > 0.9

    # ...while the OOF values must not (unseen categories fall back to the prior).
    finite = np.isfinite(oof_vals)
    if finite.sum() > 10 and np.nanstd(oof_vals) > 1e-12:
        oof_corr = abs(np.corrcoef(oof_vals[finite], y[finite])[0, 1])
        assert oof_corr < 0.15
    # And its proxy score must not beat the noise threshold.
    score = fg._featureboost_score(oof_vals, y, fg._current_oof_preds, cv)
    thr = fg._noise_probe_threshold(X, y, cv)
    assert score <= max(thr, 0.0) + 1e-12


def test_oof_groupby_of_y_derived_column_no_leak():
    rng = np.random.RandomState(1)
    n = 900
    y = pd.Series(rng.randint(0, 2, n))
    X = pd.DataFrame({
        "grp": pd.Series([f"g{i}" for i in range(n)], dtype=object),  # singleton groups
        "num_a": rng.normal(size=n),
    })
    fg = FeatureGenerator(task="classification", cv=3, use_gpu=False, log_file=None,
                          random_state=0, n_jobs=1)
    fg._set_defaults(X, y)
    fg._current_y = y
    cv = fg._get_cv_splitter()
    fg._current_oof_preds = fg._train_base_model_and_get_residuals(X, y, cv)

    fgrp, fnum = Feature("grp", "cat", 1.0), Feature("num_a", "num", 1.0)
    oof_vals = fg._compute_oof_candidate_values(Interaction(fgrp, "groupby_mean", fnum), X, y, cv)
    assert oof_vals is not None
    score = fg._featureboost_score(oof_vals, y, fg._current_oof_preds, cv)
    thr = fg._noise_probe_threshold(X, y, cv)
    assert score <= max(thr, 0.0) + 1e-12
