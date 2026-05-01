from __future__ import annotations

import json

import pandas as pd
import pytest

from tabularaml.benchmarks.feature_gen.targeted.runner import (
    TargetedBenchmarkRunner,
    _done_key_set,
)


def _runner(tmp_path) -> TargetedBenchmarkRunner:
    return TargetedBenchmarkRunner(
        suite="pmlb",
        frameworks=["nofe"],
        seeds=[0],
        results_dir=tmp_path,
        wandb_enabled=False,
    )


def test_targeted_pct_uses_matching_baseline_metric(tmp_path):
    runner = _runner(tmp_path)
    row = {
        "dataset_source": "pmlb",
        "dataset_id": "adult",
        "suite": "pmlb",
        "task": "classification",
        "framework": "openfe",
        "seed": 0,
        "score_holdout": 0.4,
        "scorer_name": "binary_crossentropy",
        "all_metrics_json": json.dumps({
            "binary_crossentropy": 0.4,
            "binary_roc_auc": 0.8,
        }),
    }
    nofe_lookup = {
        ("pmlb", "adult", "pmlb", 0): {
            "score_holdout": 0.9,
            "scorer_name": "binary_roc_auc",
            "all_metrics": {
                "binary_crossentropy": 0.5,
                "binary_roc_auc": 0.9,
            },
        },
    }

    runner._attach_pct_improvement(row, nofe_lookup)

    assert row["score_nofe_same_seed"] == 0.5
    assert row["pct_improvement"] == pytest.approx(0.2)
    gains = json.loads(row["metric_gains_json"])
    assert gains["binary_crossentropy"] == pytest.approx(0.2)
    assert gains["binary_roc_auc"] == pytest.approx((0.8 - 0.9) / 0.9)


def test_targeted_pct_is_missing_when_headline_metric_is_missing(tmp_path):
    runner = _runner(tmp_path)
    row = {
        "dataset_source": "openml_task",
        "dataset_id": "219",
        "suite": "amlb",
        "task": "classification",
        "framework": "openfe",
        "seed": 0,
        "score_holdout": float("nan"),
        "scorer_name": "binary_crossentropy",
        "all_metrics_json": json.dumps({
            "binary_crossentropy": None,
            "binary_roc_auc": 0.8,
        }),
    }
    nofe_lookup = {
        ("openml_task", "219", "amlb", 0): {
            "score_holdout": None,
            "scorer_name": "binary_crossentropy",
            "all_metrics": {
                "binary_crossentropy": None,
                "binary_roc_auc": 0.9,
            },
        },
    }

    runner._attach_pct_improvement(row, nofe_lookup)

    assert row["score_nofe_same_seed"] is None
    assert row["pct_improvement"] is None
    gains = json.loads(row["metric_gains_json"])
    assert gains["binary_crossentropy"] is None
    assert gains["binary_roc_auc"] == pytest.approx((0.8 - 0.9) / 0.9)


def test_targeted_resume_key_distinguishes_repeated_dataset_across_suites():
    master = pd.DataFrame([
        {
            "dataset_source": "pmlb",
            "dataset_id": "adult",
            "suite": "pmlb",
            "task": "classification",
            "framework": "nofe",
            "seed": 0,
            "status": "ok",
        },
    ])

    done = _done_key_set(master)

    assert ("pmlb", "adult", "pmlb", 0, "nofe") in done
    assert ("pmlb", "adult", "stress_test", 0, "nofe") not in done
