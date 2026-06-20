import numpy as np
import pandas as pd
import pytest

from tabularaml.inspect.adversarial import AdversarialValidator


def _shared_frame(n, seed):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "a": rng.randn(n),
        "b": rng.randn(n),
        "c": rng.choice(["x", "y", "z"], n),
    }), rng


def test_detects_injected_shift_and_flags_feature():
    n = 800
    train, rng = _shared_frame(n, 0)
    test, _ = _shared_frame(n, 1)
    # Inject a strong distribution shift into a single column.
    train["shifted"] = rng.normal(0.0, 1.0, n)
    test["shifted"] = np.random.RandomState(2).normal(5.0, 1.0, n)

    av = AdversarialValidator(cv=3, random_state=0).fit(train, test)
    drift = av.feature_drift_scores()

    assert av.auc_ > 0.7
    assert drift["shifted"] == max(drift.values())
    assert drift["shifted"] > 0.5
    assert av.oof_test_likeness().shape == (n,)


def test_matched_distributions_have_low_drift():
    n = 800
    train, _ = _shared_frame(n, 10)
    test, _ = _shared_frame(n, 11)
    train["extra"] = np.random.RandomState(12).randn(n)
    test["extra"] = np.random.RandomState(13).randn(n)

    av = AdversarialValidator(cv=3, random_state=0).fit(train, test)
    # No real shift -> classifier near chance, drift penalties near zero.
    assert av.auc_ < 0.6
    assert max(av.feature_drift_scores().values()) < 0.2


def test_handles_disjoint_columns_gracefully():
    train = pd.DataFrame({"a": np.arange(50.0)})
    test = pd.DataFrame({"b": np.arange(50.0)})
    av = AdversarialValidator(cv=3, random_state=0).fit(train, test)
    assert av.auc_ == 0.5
    assert av.feature_drift_scores() == {}
