"""CLI for the cross-framework FE benchmark runner.

Example:
    python scripts/run_benchmark.py \
        --manifest tabularaml/benchmarks/feature_gen/manifest.csv \
        --frameworks nofe openfe autofeat featuretools tabularaml \
        --seeds 0 1 2 \
        --time-budget 3600 \
        --subset full \
        --n-workers 1 \
        --wandb-project tabularaml-fe-benchmark \
        --results-dir tabularaml/benchmarks/feature_gen/results
"""
from __future__ import annotations
import argparse
from pathlib import Path

from tabularaml.benchmarks.feature_gen.runner import BenchmarkRunner
from tabularaml.benchmarks.feature_gen.adapters import ADAPTER_NAMES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="tabularaml/benchmarks/feature_gen/manifest.csv")
    p.add_argument("--frameworks", nargs="+", default=sorted(ADAPTER_NAMES),
                   help=f"Any subset of {sorted(ADAPTER_NAMES)}")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--time-budget", type=int, default=3600)
    p.add_argument("--subset", choices=["smoke", "small", "medium", "full"], default="full")
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--results-dir", default="tabularaml/benchmarks/feature_gen/results")
    p.add_argument("--tabularaml-mode", default="medium",
                   choices=["lite", "medium", "best", "extreme"])
    p.add_argument("--wandb-project", default="tabularaml-fe-benchmark")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--artifact-name", default="benchmark_results",
                   help="W&B artifact that stores master.csv + raw/*.csv across sessions")
    p.add_argument("--no-artifact-sync", action="store_true",
                   help="Skip pulling prior artifact at startup / pushing during run")
    p.add_argument("--sync-every", type=int, default=5,
                   help="Push a new artifact version every N completed rows")
    p.add_argument("--sync-min-interval", type=float, default=30.0,
                   help="Minimum seconds between artifact pushes (rate limit guard)")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Re-run even if a row already exists in master.csv")
    p.add_argument("--retry-crashes", action="store_true")
    args = p.parse_args()

    unknown = [f for f in args.frameworks if f not in ADAPTER_NAMES]
    if unknown:
        raise SystemExit(f"Unknown frameworks: {unknown}")

    runner = BenchmarkRunner(
        manifest_path=Path(args.manifest),
        frameworks=args.frameworks,
        seeds=args.seeds,
        results_dir=Path(args.results_dir),
        time_budget_s=args.time_budget,
        n_workers=args.n_workers,
        subset=args.subset,
        tabularaml_mode=args.tabularaml_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_enabled=not args.no_wandb,
        skip_existing=not args.no_skip_existing,
        retry_crashes=args.retry_crashes,
        artifact_name=args.artifact_name,
        artifact_sync=not args.no_artifact_sync,
        sync_every_rows=args.sync_every,
        sync_min_interval_s=args.sync_min_interval,
    )
    runner.run()


if __name__ == "__main__":
    main()
