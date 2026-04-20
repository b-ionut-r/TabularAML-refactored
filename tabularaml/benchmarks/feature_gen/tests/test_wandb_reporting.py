from __future__ import annotations

import importlib.util
import sys
import types
import warnings
from pathlib import Path

import pandas as pd
import pytest

_MOD_PATH = Path(__file__).resolve().parents[1] / "wandb_logger.py"
_SPEC = importlib.util.spec_from_file_location("wandb_logger_under_test", _MOD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

_build_per_dataset_frame = _MOD._build_per_dataset_frame
_build_scorer_summary_frame = _MOD._build_scorer_summary_frame
_build_task_summary_frame = _MOD._build_task_summary_frame
_build_failure_rate_figure = _MOD._build_failure_rate_figure
_build_pareto_figure = _MOD._build_pareto_figure
_normalize_results_frame = _MOD._normalize_results_frame
_to_wandb_table = _MOD._to_wandb_table
OrchestratorRun = _MOD.OrchestratorRun
log_media_placeholder = _MOD.log_media_placeholder

_TARGETED_MOD_PATH = Path(__file__).resolve().parents[1] / "targeted" / "wandb_suite.py"


class _FakeImage:
    def __init__(self, payload, caption=None):
        self.payload = payload
        self.caption = caption


class _FakeTable:
    def __init__(self, columns=None, data=None, log_mode=None):
        self.columns = list(columns or [])
        self.data = [list(row) for row in (data or [])]
        self.log_mode = log_mode

    def add_data(self, *row):
        self.data.append(list(row))


class _FakeArtifact:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.files = []
        self.dirs = []

    def add_file(self, path):
        self.files.append(path)

    def add_dir(self, path):
        self.dirs.append(path)


class _FakeSummary:
    def __init__(self):
        self.updates = []

    def update(self, payload):
        self.updates.append(dict(payload))


class _FakeRun:
    def __init__(self, init_kwargs):
        self.init_kwargs = init_kwargs
        self.logged = []
        self.summary = _FakeSummary()

    def log(self, payload, step=None):
        self.logged.append({"payload": dict(payload), "step": step})


class _FakeSettings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeWandb:
    Image = _FakeImage
    Table = _FakeTable
    Artifact = _FakeArtifact
    Settings = _FakeSettings

    def __init__(self):
        self.run = None
        self.runs = []
        self.logged_artifacts = []

    def init(self, **kwargs):
        run = _FakeRun(kwargs)
        self.run = run
        self.runs.append(run)
        return run

    def finish(self):
        self.run = None

    def log_artifact(self, artifact):
        self.logged_artifacts.append(artifact)


def _install_fake_wandb(monkeypatch):
    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def _load_targeted_suite_module():
    tabularaml_pkg = sys.modules.setdefault("tabularaml", types.ModuleType("tabularaml"))
    tabularaml_pkg.__path__ = [str(Path(__file__).resolve().parents[3])]

    benchmarks_pkg = sys.modules.setdefault("tabularaml.benchmarks", types.ModuleType("tabularaml.benchmarks"))
    benchmarks_pkg.__path__ = [str(Path(__file__).resolve().parents[2])]

    feature_gen_pkg = sys.modules.setdefault(
        "tabularaml.benchmarks.feature_gen",
        types.ModuleType("tabularaml.benchmarks.feature_gen"),
    )
    feature_gen_pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules["tabularaml.benchmarks.feature_gen.wandb_logger"] = _MOD

    targeted_pkg = sys.modules.setdefault(
        "tabularaml.benchmarks.feature_gen.targeted",
        types.ModuleType("tabularaml.benchmarks.feature_gen.targeted"),
    )
    targeted_pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "targeted")]

    spec = importlib.util.spec_from_file_location("targeted_wandb_suite_under_test", _TARGETED_MOD_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _base_rows():
    return [
        {
            "dataset_id": 1,
            "task": "classification",
            "framework": "nofe",
            "seed": 0,
            "scorer_name": "binary_roc_auc",
            "score_holdout": 0.70,
            "score_nofe_same_seed": 0.70,
            "pct_improvement": 0.0,
            "n_train": 80,
            "n_test": 20,
            "n_features_before": 10,
            "n_features_after": 10,
            "n_added": 0,
            "wall_time_fit": 1.0,
            "wall_time_transform": 0.1,
            "wall_time_total": 1.1,
            "peak_rss_mb": 100.0,
            "n_boost_rounds": 12,
            "status": "ok",
            "error_msg": "",
        },
        {
            "dataset_id": 1,
            "task": "classification",
            "framework": "openfe",
            "seed": 0,
            "scorer_name": "binary_roc_auc",
            "score_holdout": 0.77,
            "score_nofe_same_seed": 0.70,
            "pct_improvement": 0.1,
            "n_train": 80,
            "n_test": 20,
            "n_features_before": 10,
            "n_features_after": 18,
            "n_added": 8,
            "wall_time_fit": 3.2,
            "wall_time_transform": 0.5,
            "wall_time_total": 3.7,
            "peak_rss_mb": 140.0,
            "n_boost_rounds": 16,
            "status": "ok",
            "error_msg": "",
        },
    ]


def _targeted_rows():
    return [
        {
            "dataset_id": "adult",
            "dataset_name": "adult",
            "dataset_source": "pmlb",
            "suite": "amlb",
            "task": "classification",
            "framework": "nofe",
            "seed": 0,
            "scorer_name": "binary_roc_auc",
            "score_holdout": 0.71,
            "score_nofe_same_seed": 0.71,
            "pct_improvement": 0.0,
            "n_train": 100,
            "n_test": 25,
            "n_features_before": 8,
            "n_features_after": 8,
            "n_added": 0,
            "wall_time_fit": 1.0,
            "wall_time_transform": 0.1,
            "wall_time_total": 1.1,
            "peak_rss_mb": 90.0,
            "n_boost_rounds": 8,
            "status": "ok",
            "error_msg": "",
        },
        {
            "dataset_id": "adult",
            "dataset_name": "adult",
            "dataset_source": "pmlb",
            "suite": "amlb",
            "task": "classification",
            "framework": "openfe",
            "seed": 0,
            "scorer_name": "binary_roc_auc",
            "score_holdout": 0.77,
            "score_nofe_same_seed": 0.71,
            "pct_improvement": 0.0845,
            "n_train": 100,
            "n_test": 25,
            "n_features_before": 8,
            "n_features_after": 18,
            "n_added": 10,
            "wall_time_fit": 3.0,
            "wall_time_transform": 0.4,
            "wall_time_total": 3.4,
            "peak_rss_mb": 120.0,
            "n_boost_rounds": 10,
            "status": "ok",
            "error_msg": "",
        },
        {
            "dataset_id": "credit",
            "dataset_name": "credit",
            "dataset_source": "pmlb",
            "suite": "pmlb",
            "task": "classification",
            "framework": "nofe",
            "seed": 0,
            "scorer_name": "binary_roc_auc",
            "score_holdout": 0.73,
            "score_nofe_same_seed": 0.73,
            "pct_improvement": 0.0,
            "n_train": 120,
            "n_test": 30,
            "n_features_before": 9,
            "n_features_after": 9,
            "n_added": 0,
            "wall_time_fit": 1.1,
            "wall_time_transform": 0.1,
            "wall_time_total": 1.2,
            "peak_rss_mb": 92.0,
            "n_boost_rounds": 9,
            "status": "ok",
            "error_msg": "",
        },
        {
            "dataset_id": "credit",
            "dataset_name": "credit",
            "dataset_source": "pmlb",
            "suite": "pmlb",
            "task": "classification",
            "framework": "openfe",
            "seed": 0,
            "scorer_name": "binary_roc_auc",
            "score_holdout": 0.79,
            "score_nofe_same_seed": 0.73,
            "pct_improvement": 0.0822,
            "n_train": 120,
            "n_test": 30,
            "n_features_before": 9,
            "n_features_after": 19,
            "n_added": 10,
            "wall_time_fit": 3.4,
            "wall_time_transform": 0.5,
            "wall_time_total": 3.9,
            "peak_rss_mb": 128.0,
            "n_boost_rounds": 11,
            "status": "ok",
            "error_msg": "",
        },
    ]


def test_task_summary_uses_per_dataset_means_not_per_run_means():
    raw = pd.DataFrame([
        {
            "dataset_id": 1,
            "task": "classification",
            "framework": "openfe",
            "seed": 0,
            "scorer_name": "binary_roc_auc",
            "score_holdout": 0.72,
            "pct_improvement": 0.5,
            "wall_time_total": 10.0,
            "wall_time_fit": 8.0,
            "peak_rss_mb": 100.0,
            "n_added": 20,
            "status": "ok",
        },
        {
            "dataset_id": 1,
            "task": "classification",
            "framework": "openfe",
            "seed": 1,
            "scorer_name": "binary_roc_auc",
            "score_holdout": 0.48,
            "pct_improvement": -0.5,
            "wall_time_total": 12.0,
            "wall_time_fit": 9.0,
            "peak_rss_mb": 120.0,
            "n_added": 22,
            "status": "ok",
        },
        {
            "dataset_id": 2,
            "task": "classification",
            "framework": "openfe",
            "seed": 0,
            "scorer_name": "binary_roc_auc",
            "score_holdout": 0.83,
            "pct_improvement": 1.0,
            "wall_time_total": 6.0,
            "wall_time_fit": 5.0,
            "peak_rss_mb": 90.0,
            "n_added": 18,
            "status": "ok",
        },
    ])

    frame = _normalize_results_frame(raw)
    per_dataset = _build_per_dataset_frame(frame)
    summary = _build_task_summary_frame(frame, per_dataset)

    row = summary[(summary["framework"] == "openfe") & (summary["task"] == "classification")].iloc[0]
    assert int(row["n_datasets"]) == 2
    assert int(row["n_ok_runs"]) == 3
    assert row["pct_improvement_mean"] == pytest.approx(0.5)


def test_reporting_keeps_failure_rates_and_scorers_separate():
    raw = pd.DataFrame([
        {
            "dataset_id": 10,
            "task": "classification",
            "framework": "featuretools",
            "seed": 0,
            "scorer_name": "binary_roc_auc",
            "score_holdout": 0.81,
            "pct_improvement": 0.1,
            "wall_time_total": 4.0,
            "wall_time_fit": 3.5,
            "peak_rss_mb": 75.0,
            "n_added": 11,
            "status": "ok",
        },
        {
            "dataset_id": 10,
            "task": "classification",
            "framework": "featuretools",
            "seed": 1,
            "scorer_name": "binary_roc_auc",
            "status": "timeout",
            "error_msg": "wall clock exceeded",
        },
        {
            "dataset_id": 11,
            "task": "classification",
            "framework": "featuretools",
            "seed": 0,
            "scorer_name": "categorical_crossentropy",
            "score_holdout": 0.62,
            "pct_improvement": 0.2,
            "wall_time_total": 7.0,
            "wall_time_fit": 6.2,
            "peak_rss_mb": 82.0,
            "n_added": 14,
            "status": "ok",
        },
    ])

    frame = _normalize_results_frame(raw)
    per_dataset = _build_per_dataset_frame(frame)
    summary = _build_task_summary_frame(frame, per_dataset)
    scorer_summary = _build_scorer_summary_frame(per_dataset)

    task_row = summary[(summary["framework"] == "featuretools") & (summary["task"] == "classification")].iloc[0]
    assert int(task_row["n_attempts"]) == 3
    assert int(task_row["n_non_ok_runs"]) == 1
    assert task_row["non_ok_rate"] == pytest.approx(1 / 3)

    scorer_rows = scorer_summary[scorer_summary["framework"] == "featuretools"]
    assert set(scorer_rows["scorer_name"]) == {"binary_roc_auc", "categorical_crossentropy"}


def test_crowded_media_figures_avoid_tight_layout_warnings(monkeypatch):
    _install_fake_wandb(monkeypatch)

    per_dataset = pd.DataFrame([
        {
            "dataset_id": i,
            "task": task,
            "framework": framework,
            "wall_time_total_mean": 10.0 + i,
            "pct_improvement_mean": (-0.2 + 0.05 * i),
        }
        for i, (task, framework) in enumerate([
            ("classification", "featuretools-extra-long"),
            ("classification", "openfe"),
            ("classification", "autofeat"),
            ("regression", "featurewiz"),
            ("regression", "tabularaml"),
            ("regression", "baseline-nofe"),
        ], start=1)
    ])
    task_summary = pd.DataFrame([
        {
            "framework": framework,
            "task": task,
            "non_ok_rate": rate,
            "n_non_ok_runs": int(rate * 10),
            "n_attempts": 10,
        }
        for task, framework, rate in [
            ("classification", "featuretools-extra-long", 0.4),
            ("classification", "openfe", 0.2),
            ("classification", "autofeat", 0.1),
            ("regression", "featurewiz", 0.3),
            ("regression", "tabularaml", 0.2),
            ("regression", "baseline-nofe", 0.05),
        ]
    ])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _build_pareto_figure(per_dataset) is not None
        assert _build_failure_rate_figure(task_summary) is not None

    assert not any("Tight layout not applied" in str(w.message) for w in caught)


def test_log_media_placeholder_still_works_when_called(monkeypatch):
    # log_media_placeholder is kept for API compatibility but workers no longer call it.
    # Verify the function itself still behaves correctly if invoked externally.
    _install_fake_wandb(monkeypatch)

    class FakeRun:
        def __init__(self):
            self.logged = []

        def log(self, payload):
            self.logged.append(payload)

    run = FakeRun()
    log_media_placeholder(run, key="some_key", caption="test caption")

    assert len(run.logged) == 1
    assert "some_key" in run.logged[0]
    assert isinstance(run.logged[0]["some_key"], _FakeImage)


def test_to_wandb_table_creates_correct_table(monkeypatch):
    _install_fake_wandb(monkeypatch)

    table = _to_wandb_table(pd.DataFrame([{"a": 1, "b": 2}]))

    assert isinstance(table, _FakeTable)
    assert table.columns == ["a", "b"]
    assert table.data == [[1, 2]]


def test_orchestrator_run_logs_scalars_tables_figures_with_step(tmp_path, monkeypatch):
    fake = _install_fake_wandb(monkeypatch)
    rows = _base_rows()
    master_csv = tmp_path / "master.csv"

    pd.DataFrame([rows[0]]).to_csv(master_csv, index=False)

    with OrchestratorRun(project="proj", entity=None, artifact_name="bench-results", enabled=True) as orch:
        orch.append_result(rows[0])
        assert orch.push([master_csv], min_interval_s=0)

        pd.DataFrame([rows[1]]).to_csv(master_csv, mode="a", header=False, index=False)
        orch.append_result(rows[1])
        assert orch.push([master_csv], min_interval_s=0)

        assert orch.push([master_csv], force=True, min_interval_s=0)

    run = fake.runs[0]

    # Three pushes → three step-based log entries
    assert [entry["step"] for entry in run.logged] == [1, 2, 3]

    # Each step-log has scalars + lightweight tables (no heavy per-run table)
    for entry in run.logged:
        payload = entry["payload"]
        assert "n_rows_total" in payload
        assert "n_ok_rows" in payload
        assert isinstance(payload.get("results_aggregated"), _FakeTable)
        assert "results_per_run" not in payload  # only in summary, not history

    # Summary persists latest state including the heavy per-run table (final push only)
    assert len(run.summary.updates) == 3
    final_summary = run.summary.updates[2]
    assert isinstance(final_summary.get("results_aggregated"), _FakeTable)
    assert isinstance(final_summary.get("results_per_run"), _FakeTable)
    assert final_summary["n_rows_total"] == 2
    assert final_summary["n_ok_rows"] == 2

    # Intermediate push summaries do not include the per-run table
    first_summary = run.summary.updates[0]
    assert isinstance(first_summary.get("results_aggregated"), _FakeTable)
    assert "results_per_run" not in first_summary


def test_targeted_orchestrator_logs_suite_summary_and_rank_figure(tmp_path, monkeypatch):
    fake = _install_fake_wandb(monkeypatch)
    targeted_mod = _load_targeted_suite_module()
    rows = _targeted_rows()
    master_csv = tmp_path / "master.csv"
    pd.DataFrame(rows).to_csv(master_csv, index=False)

    with targeted_mod.TargetedOrchestratorRun(
        project="proj",
        entity=None,
        artifact_name="targeted-results",
        suite="all",
        enabled=True,
    ) as orch:
        assert orch.push([master_csv], force=True, min_interval_s=0)

    run = fake.runs[0]

    # Step-logged payload has suite summary table (lightweight → goes to workspace panel)
    step_payload = run.logged[0]["payload"]
    assert isinstance(step_payload.get("results_suite_summary"), _FakeTable)
    assert step_payload["n_suites_started"] == 2

    # Summary also has the heavy per-run table
    summary = run.summary.updates[0]
    assert isinstance(summary.get("results_per_run"), _FakeTable)
    assert isinstance(summary.get("results_suite_summary"), _FakeTable)
    assert summary["n_suites_started"] == 2
