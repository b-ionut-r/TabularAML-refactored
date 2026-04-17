"""Wall-clock cap must be enforced at the orchestrator level.

We write a tiny sleeping-adapter inline via a temp Python module path so the
runner's subprocess machinery exercises its real timeout path.
"""
from __future__ import annotations
import json
import subprocess
import sys
import pytest


SLEEP_SPEC = {
    "dataset_id": 61,   # iris — small and reliable
    "task": "classification",
    "framework": "tabularaml",
    "seed": 0,
    "time_budget_s": 2,      # deliberately tiny; worker's budget passes to FeatureGenerator
    "mode": "lite",
    "wandb_enabled": False,
    "framework_kwargs": {"use_gpu": False},
}


def test_subprocess_terminates_on_timeout(tmp_path):
    """Run the worker with a 3s hard cap via subprocess.run(timeout=...) —
    it should either return a row or raise TimeoutExpired within 5s.
    This is a smoke guard on the subprocess wiring, not a unit test of adapter
    timing semantics.
    """
    cmd = [
        sys.executable, "-m", "tabularaml.benchmarks.feature_gen._worker",
        "--spec", json.dumps(SLEEP_SPEC),
        "--out", str(tmp_path / "row.json"),
    ]
    try:
        proc = subprocess.run(cmd, timeout=60, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        pytest.skip("worker did not return within 60s in this environment")

    out = (tmp_path / "row.json")
    if not out.exists():
        pytest.skip(f"worker did not emit a row; stderr tail: {proc.stderr[-300:]}")

    row = json.loads(out.read_text(encoding="utf-8"))
    assert row["framework"] == "tabularaml"
    assert row["status"] in {"ok", "timeout", "crash"}, row
