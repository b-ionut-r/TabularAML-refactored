"""Resume semantics: when master.csv contains terminal rows, the second run
must schedule zero new subprocesses for the same (dataset, framework, seed).
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

from tabularaml.benchmarks.feature_gen.runner import (
    BenchmarkRunner, RESULT_COLUMNS,
)


def _fake_manifest(path: Path) -> Path:
    pd.DataFrame([
        {"tid": 101, "task": "classification", "n_rows": 500, "n_cols": 5,
         "n_numeric": 5, "n_categorical": 0, "n_classes": 2, "pct_missing": 0.0, "did": 1, "name": "t"},
        {"tid": 102, "task": "regression", "n_rows": 500, "n_cols": 5,
         "n_numeric": 5, "n_categorical": 0, "n_classes": 0, "pct_missing": 0.0, "did": 2, "name": "t"},
    ]).to_csv(path, index=False)
    return path


def _fake_master(path: Path, specs) -> None:
    rows = []
    for dsid, fw, seed in specs:
        row = {c: None for c in RESULT_COLUMNS}
        row.update(dict(
            dataset_id=dsid, task="classification", framework=fw, seed=seed,
            status="ok", score_holdout=0.5, score_nofe_same_seed=0.5,
            pct_improvement=0.0, wall_time_total=1.0,
        ))
        rows.append(row)
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(path, index=False)


def test_resume_skips_done_rows(tmp_path):
    manifest = _fake_manifest(tmp_path / "m.csv")
    results = tmp_path / "results"
    results.mkdir()
    _fake_master(results / "master.csv", [(101, "nofe", 0), (102, "nofe", 0)])

    runner = BenchmarkRunner(
        manifest_path=manifest,
        frameworks=["nofe"],
        seeds=[0],
        results_dir=results,
        subset="full",
        wandb_enabled=False,
    )
    plan = runner.build_run_plan()
    assert plan == [], f"Expected zero runs when all rows already present, got {plan}"
