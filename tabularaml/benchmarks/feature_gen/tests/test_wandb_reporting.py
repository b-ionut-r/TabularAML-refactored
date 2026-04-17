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
log_media_placeholder = _MOD.log_media_placeholder


class _FakeImage:
    def __init__(self, payload, caption=None):
        self.payload = payload
        self.caption = caption


def _install_fake_wandb(monkeypatch):
    fake = types.SimpleNamespace(Image=_FakeImage)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


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


def test_log_media_placeholder_logs_runtime_key(monkeypatch):
    _install_fake_wandb(monkeypatch)

    class FakeRun:
        def __init__(self):
            self.logged = []

        def log(self, payload):
            self.logged.append(payload)

    run = FakeRun()
    log_media_placeholder(
        run,
        key="figure_runtime_vs_improvement",
        caption="Aggregated runtime-vs-improvement media is logged on the orchestrator run.",
    )

    assert len(run.logged) == 1
    payload = run.logged[0]
    assert "figure_runtime_vs_improvement" in payload
    image = payload["figure_runtime_vs_improvement"]
    assert isinstance(image, _FakeImage)
    assert image.caption == "Aggregated runtime-vs-improvement media is logged on the orchestrator run."
