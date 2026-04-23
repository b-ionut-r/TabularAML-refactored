from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from tabularaml.benchmarks.feature_gen.adapters.featuretools_adapter import (
    FeaturetoolsAdapter,
    _FeaturetoolsUpstreamBugError,
)
from tabularaml.benchmarks.feature_gen.adapters.openfe_adapter import (
    OpenFEAdapter,
    _OpenFEUpstreamBugError,
)
from tabularaml.benchmarks.feature_gen.targeted._worker import _preprocess as targeted_preprocess
from tabularaml.benchmarks.feature_gen.targeted.loader import load_dataset
from tabularaml.benchmarks.feature_gen.targeted.registry import DatasetSpec
from tabularaml.generate.features import _restore_missing_pipeline_columns


def test_targeted_loader_falls_back_from_unknown_task_to_dataset(monkeypatch):
    class _FakeTaskAPI:
        @staticmethod
        def get_task(task_id):
            raise RuntimeError(
                f"https://www.openml.org/api/v1/xml/task/{task_id} returned code 151: Unknown task - None"
            )

    class _FakeDataset:
        default_target_attribute = "target"

        def get_data(self, *, dataset_format="dataframe", target=None):
            assert dataset_format == "dataframe"
            assert target == "target"
            X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
            y = pd.Series(["no", "yes"], name="target")
            return X, y, [False, False], ["a", "b"]

    class _FakeDatasetAPI:
        @staticmethod
        def get_dataset(dataset_id, **kwargs):
            assert dataset_id == 1169
            assert kwargs["download_data"] is True
            return _FakeDataset()

    fake_openml = types.SimpleNamespace(
        tasks=_FakeTaskAPI,
        datasets=_FakeDatasetAPI,
    )
    monkeypatch.setitem(sys.modules, "openml", fake_openml)

    spec = DatasetSpec("1169", "Airlines", "openml_task", "classification", "stress_test")
    loaded = load_dataset(spec)

    assert loaded.name == "Airlines"
    assert loaded.task == "classification"
    assert loaded.n_classes == 2
    assert list(loaded.X.columns) == ["a", "b"]


def test_targeted_preprocess_replaces_nonfinite_numeric_values():
    X = pd.DataFrame(
        {
            "num": [1.0, np.inf, -np.inf],
            "cat": ["a", "b", "c"],
        }
    )
    Xt, y, n_classes = targeted_preprocess(X, pd.Series([0, 1, 0]), "classification", 2)

    assert Xt["num"].isna().sum() == 2
    assert str(Xt["cat"].dtype) == "category"
    assert n_classes == 2
    np.testing.assert_array_equal(y, np.array([0, 1, 0]))


def test_restore_missing_pipeline_columns_adds_numeric_and_categorical_inputs():
    class _FakeScaling:
        feature_names_in_ = np.array(["num", "cat", "generated"])

    class _FakeImputer:
        numerical_columns_ = ["num", "generated"]
        categorical_columns_ = ["cat"]

    class _FakePipeline:
        named_steps = {
            "scaling_encoding": _FakeScaling(),
            "imputing": _FakeImputer(),
        }

    restored = _restore_missing_pipeline_columns(
        pd.DataFrame({"num": [1.0]}),
        _FakePipeline(),
    )

    assert set(restored.columns) == {"num", "cat", "generated"}
    assert restored["generated"].isna().all()
    assert str(restored["cat"].dtype) == "category"


def test_restore_missing_pipeline_columns_logs_restoration():
    class _FakeScaling:
        feature_names_in_ = np.array(["num", "cat", "generated"])

    class _FakeImputer:
        numerical_columns_ = ["num", "generated"]
        categorical_columns_ = ["cat"]

    class _FakePipeline:
        named_steps = {
            "scaling_encoding": _FakeScaling(),
            "imputing": _FakeImputer(),
        }

    logs = []
    _restore_missing_pipeline_columns(
        pd.DataFrame({"num": [1.0]}),
        _FakePipeline(),
        log_fn=logs.append,
    )

    assert len(logs) == 1
    assert "restoring 2 missing pipeline column(s)" in logs[0]
    assert "cat, generated" in logs[0]


def test_featuretools_adapter_maps_boolean_na_bug_to_upstream_status():
    adapter = FeaturetoolsAdapter(
        task="classification",
        time_budget_s=1,
        random_state=0,
        n_jobs=1,
    )

    def _raise_boolean_na(**kwargs):
        raise TypeError("boolean value of NA is ambiguous")

    adapter._dfs = lambda frame: (types.SimpleNamespace(dfs=_raise_boolean_na), object())

    with pytest.raises(_FeaturetoolsUpstreamBugError):
        adapter.fit_transform(
            pd.DataFrame({"flag": pd.Series([True, None, False], dtype="object")}),
            pd.Series([0, 1, 0]),
        )


def test_openfe_adapter_converts_system_exit_to_upstream_bug(monkeypatch):
    adapter = OpenFEAdapter(
        task="classification",
        time_budget_s=1,
        random_state=0,
        n_jobs=1,
    )

    monkeypatch.setattr(OpenFEAdapter, "_patch_init_score_flatten", staticmethod(lambda: None))
    monkeypatch.setattr(OpenFEAdapter, "_patch_sklearn_mse", staticmethod(lambda: None))

    def _boom(X_train, y_train):
        raise SystemExit("multiclass bug")

    monkeypatch.setattr(adapter, "_fit_transform_inner", _boom)

    with pytest.raises(_OpenFEUpstreamBugError):
        adapter.fit_transform(
            pd.DataFrame({"a": [1.0, 2.0]}),
            pd.Series([0, 1]),
        )
