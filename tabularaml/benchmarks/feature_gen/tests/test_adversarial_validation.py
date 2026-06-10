"""Tests for the adversarial (train-vs-test) validation hook."""
import numpy as np
import pandas as pd
import pytest

from tabularaml.generate.features import FeatureGenerator


def _shifted_setup(seed=0, n=600):
    rng = np.random.default_rng(seed)
    X_train = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    X_test = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(X_train["a"] + rng.normal(size=n) * 0.1)
    gen = FeatureGenerator(log_file=None)
    gen._set_defaults(X_train, y)
    gen.initial_features = ["a", "b"]
    gen.interactions = []
    return gen, X_train, X_test, y, rng


def test_planted_shift_feature_tops_report():
    gen, X_train, X_test, y, rng = _shifted_setup()
    # generated feature with a large distribution shift between train and test
    X_train = X_train.copy()
    X_train["gen_shift"] = rng.normal(0, 1, size=len(X_train))
    X_test = X_test.copy()
    X_test["gen_shift"] = rng.normal(5, 1, size=len(X_test))
    drop = gen._adversarial_validation_report(X_train, X_test)
    rep = gen.adversarial_report
    assert rep["auc"] > 0.9
    top_name, top_share, is_gen = rep["top_shift_features"][0]
    assert top_name == "gen_shift" and is_gen
    # default adversarial_drop=False -> nothing dropped despite the warning
    assert drop == []


def test_drop_only_when_enabled():
    gen, X_train, X_test, y, rng = _shifted_setup(seed=1)
    X_train = X_train.copy()
    X_train["gen_shift"] = rng.normal(0, 1, size=len(X_train))
    X_test = X_test.copy()
    X_test["gen_shift"] = rng.normal(5, 1, size=len(X_test))
    gen.adversarial_drop = True
    drop = gen._adversarial_validation_report(X_train, X_test)
    assert drop == ["gen_shift"]


def test_no_shift_no_warning_auc_near_half():
    gen, X_train, X_test, y, _ = _shifted_setup(seed=2)
    drop = gen._adversarial_validation_report(X_train, X_test)
    assert gen.adversarial_report["auc"] < 0.65
    assert drop == []
