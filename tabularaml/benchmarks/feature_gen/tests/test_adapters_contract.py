"""Every adapter must produce matching train/test columns and no NaN in train."""
from __future__ import annotations
import importlib
import pytest
from sklearn.model_selection import train_test_split

from tabularaml.benchmarks.feature_gen.adapters import get_adapter_cls
from tabularaml.benchmarks.feature_gen.adapters.base import _check_contract


def _pkg(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


_CASES = [
    ("nofe", True),
    ("tabularaml", True),
    ("openfe", _pkg("openfe")),
    ("autofeat", _pkg("autofeat")),
    ("featuretools", _pkg("featuretools")),
]


@pytest.mark.parametrize("framework,available", _CASES)
def test_contract_classification(framework, available, toy_cls):
    if not available:
        pytest.skip(f"{framework} not installed")
    X, y = toy_cls
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

    cls = get_adapter_cls(framework)
    kwargs = {}
    if framework == "tabularaml":
        kwargs = {"mode": "lite", "use_gpu": False}
    adapter = cls(task="classification", time_budget_s=90, random_state=0, n_jobs=1, **kwargs)

    X_tr_fe = adapter.fit_transform(X_tr, y_tr)
    X_te_fe = adapter.transform(X_te)
    _check_contract(X_tr_fe, X_te_fe, len(X_tr), len(X_te))


@pytest.mark.parametrize("framework,available", _CASES)
def test_contract_regression(framework, available, toy_reg):
    if not available:
        pytest.skip(f"{framework} not installed")
    X, y = toy_reg
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.25, random_state=0)

    cls = get_adapter_cls(framework)
    kwargs = {}
    if framework == "tabularaml":
        kwargs = {"mode": "lite", "use_gpu": False}
    adapter = cls(task="regression", time_budget_s=90, random_state=0, n_jobs=1, **kwargs)

    X_tr_fe = adapter.fit_transform(X_tr, y_tr)
    X_te_fe = adapter.transform(X_te)
    _check_contract(X_tr_fe, X_te_fe, len(X_tr), len(X_te))
