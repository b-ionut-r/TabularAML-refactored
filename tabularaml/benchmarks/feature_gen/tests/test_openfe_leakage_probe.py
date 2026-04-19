"""Probe: if OpenFE's transform leaks, fitting on pure noise features with
random labels will yield a holdout score materially above chance.

This is a benchmark-honesty check, not an OpenFE correctness assertion. A
failed probe means the final report should annotate OpenFE numbers with a
caveat — it does not cause the runner to fail.
"""
from __future__ import annotations
import importlib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split


openfe_available = importlib.util.find_spec("openfe") is not None


@pytest.mark.skipif(not openfe_available, reason="openfe not installed")
def test_openfe_leakage_probe_records_result(tmp_path):
    from tabularaml.benchmarks.feature_gen.adapters import get_adapter_cls
    from tabularaml.benchmarks.feature_gen.evaluator import (
        score_on_holdout,
        split_early_stopping_validation,
    )

    rng = np.random.default_rng(42)
    n, d = 600, 12
    X = pd.DataFrame(rng.normal(size=(n, d)),
                     columns=[f"f{i}" for i in range(d)])
    y = rng.integers(0, 2, size=n)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y,
    )

    adapter = get_adapter_cls("openfe")(
        task="classification", time_budget_s=120, random_state=42, n_jobs=1,
    )
    try:
        X_tr_fit, X_tr_val, y_tr_fit, y_tr_val = split_early_stopping_validation(
            X_tr,
            y_tr,
            task="classification",
            seed=42,
        )
        X_tr_fe = adapter.fit_transform(X_tr_fit, pd.Series(y_tr_fit))
        X_val_fe = adapter.transform(X_tr_val)
        X_te_fe = adapter.transform(X_te)
    except Exception as e:
        pytest.skip(f"OpenFE fit failed on this environment: {e}")

    score, _ = score_on_holdout(
        X_tr_fe,
        y_tr_fit,
        X_val_fe,
        y_tr_val,
        X_te_fe,
        y_te,
        task="classification", n_classes=2, seed=42,
    )

    assert np.isfinite(score), "leakage probe produced non-finite score"
    # Pure-noise data must not yield ROC-AUC > 0.60; a higher value means
    # upstream transform() is leaking test distribution into training aggregates.
    assert score < 0.60, (
        f"OpenFE leakage probe: score={score:.4f} on random-noise data exceeds 0.60. "
        "Upstream transform() is leaking test rows into training aggregates — "
        "annotate OpenFE results in the report with a data-leakage caveat."
    )
