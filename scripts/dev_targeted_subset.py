"""Dev-loop driver: run the frozen targeted benchmark on a subset of PMLB datasets.

Exists because the frozen CLI (`targeted/__main__.py`) has no per-dataset filter and
restricts --tabularaml-mode choices. This subclass only filters the run plan and
forwards an arbitrary mode; the protocol (worker, evaluator, scoring, master.csv)
is inherited untouched, so dev numbers are directly comparable to the full benchmark.

Usage (note: cwd + PYTHONPATH must point at the SAME worktree; --worktree enforces both):

    python scripts/dev_targeted_subset.py --worktree /home/user/before-main \
        --datasets monk1 titanic sonar spambase car_evaluation 529_pollen 1028_SWD \
        --mode lite --seeds 0 1 --time-budget 420 --n-workers 2 \
        --results-dir /home/user/results/dev/before
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_DEV_DATASETS = [
    "monk1",           # pure feature-interaction signal (parity-style)
    "titanic",         # categoricals + missing values
    "sonar",           # wide numeric, tiny n=208
    "spambase",        # 4.6k rows -> exercises meta-validation split path
    "car_evaluation",  # all-categorical multiclass
    "529_pollen",      # regression
    "1028_SWD",        # regression (ordinal-ish target)
]


def _make_runner_class():
    # Imported lazily so --worktree can be placed on sys.path first; defined at
    # module level (via global) so ProcessPoolExecutor can pickle runner instances.
    from tabularaml.benchmarks.feature_gen.targeted.runner import TargetedBenchmarkRunner

    global DevSubsetRunner

    class DevSubsetRunner(TargetedBenchmarkRunner):
        wanted: set = set()

        def build_run_plan(self):
            specs = super().build_run_plan()
            return [s for s in specs if s.dataset_id in self.wanted]

    # Pickle resolves classes via __module__ + __qualname__; strip the local scope.
    DevSubsetRunner.__qualname__ = "DevSubsetRunner"
    return DevSubsetRunner


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worktree", required=True, help="Repo root whose tabularaml package to benchmark")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DEV_DATASETS)
    p.add_argument("--frameworks", nargs="+", default=["nofe", "tabularaml"])
    p.add_argument("--mode", default="lite")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--time-budget", type=int, default=420)
    p.add_argument("--n-workers", type=int, default=2)
    p.add_argument("--results-dir", required=True)
    args = p.parse_args()

    worktree = str(Path(args.worktree).resolve())
    # Workers are spawned with `python -m ...`, which puts cwd first on sys.path:
    # chdir + PYTHONPATH guarantee both this process and workers import the same tree.
    os.chdir(worktree)
    os.environ["PYTHONPATH"] = worktree
    sys.path.insert(0, worktree)

    runner_cls = _make_runner_class()
    runner_cls.wanted = set(args.datasets)

    runner = runner_cls(
        suite="pmlb",
        frameworks=args.frameworks,
        seeds=args.seeds,
        results_dir=Path(args.results_dir),
        time_budget_s=args.time_budget,
        n_workers=args.n_workers,
        tabularaml_mode=args.mode,
        wandb_enabled=False,
        artifact_sync=False,
    )
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
