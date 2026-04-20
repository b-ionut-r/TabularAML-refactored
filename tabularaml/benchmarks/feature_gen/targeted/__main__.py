"""CLI entry point for the targeted benchmark suite.

Usage::

    python -m tabularaml.benchmarks.feature_gen.targeted \\
        --suite amlb \\
        --frameworks nofe tabularaml openfe autofeat featuretools \\
        --seeds 0 1 2 \\
        --n-workers 4 \\
        --time-budget 600 \\
        --results-dir ./results/targeted/amlb \\
        --wandb-project tabularaml-targeted-benchmark

    # Dry run (no W&B, single seed, stress-test suite)
    python -m tabularaml.benchmarks.feature_gen.targeted \\
        --suite stress_test --frameworks nofe tabularaml \\
        --seeds 0 --n-workers 1 --no-wandb

Available suites: amlb, pmlb, stress_test, all
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .runner import TargetedBenchmarkRunner


def _parse(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run targeted FE benchmark on curated dataset suites.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--suite", required=True,
        choices=["amlb", "pmlb", "stress_test", "all"],
        help="Dataset suite to run.",
    )
    p.add_argument(
        "--frameworks", nargs="+",
        default=["nofe", "tabularaml", "openfe", "autofeat", "featuretools"],
        metavar="FW",
        help="Feature engineering frameworks to evaluate.",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=[0, 1, 2],
        metavar="SEED",
        help="Random seeds for train/test splits.",
    )
    p.add_argument(
        "--n-workers", type=int, default=4,
        help="Parallel worker processes (default: 4).",
    )
    p.add_argument(
        "--time-budget", type=int, default=600,
        help="Per-adapter wall-clock budget in seconds (default: 600).",
    )
    p.add_argument(
        "--results-dir", type=Path, default=None,
        help="Directory for master.csv and raw/ CSVs. "
             "Defaults to ./results/targeted/<suite>.",
    )
    p.add_argument(
        "--tabularaml-mode", default="medium",
        choices=["fast", "medium", "best"],
        help="TabularAML preset (default: medium).",
    )
    p.add_argument(
        "--wandb-project", default="tabularaml-targeted-benchmark",
        help="W&B project name.",
    )
    p.add_argument(
        "--wandb-entity", default=None,
        help="W&B entity (team or username). Omit to use default.",
    )
    p.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B logging entirely.",
    )
    p.add_argument(
        "--no-resume", action="store_true",
        help="Ignore existing results and re-run everything.",
    )
    p.add_argument(
        "--retry-crashes", action="store_true",
        help="Retry runs that previously crashed.",
    )
    p.add_argument(
        "--artifact-name", default=None,
        help="W&B artifact name for results (default: targeted-<suite>-results).",
    )
    p.add_argument(
        "--sync-every", type=int, default=5,
        help="Push W&B artifact after every N completed rows (default: 5).",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse(argv)

    results_dir = args.results_dir or Path(f"./results/targeted/{args.suite}")
    results_dir.mkdir(parents=True, exist_ok=True)

    runner = TargetedBenchmarkRunner(
        suite=args.suite,
        frameworks=args.frameworks,
        seeds=args.seeds,
        results_dir=results_dir,
        time_budget_s=args.time_budget,
        n_workers=args.n_workers,
        tabularaml_mode=args.tabularaml_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_enabled=not args.no_wandb,
        skip_existing=not args.no_resume,
        retry_crashes=args.retry_crashes,
        artifact_name=args.artifact_name,
        sync_every_rows=args.sync_every,
    )
    runner.run()


if __name__ == "__main__":
    main()
