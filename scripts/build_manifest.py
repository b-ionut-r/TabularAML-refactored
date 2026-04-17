"""One-shot manifest builder.

Usage:
    python scripts/build_manifest.py \
        --out tabularaml/benchmarks/feature_gen/manifest.csv \
        --min-rows 500 --max-rows 50000 --max-features 200
"""
from __future__ import annotations
import argparse
from pathlib import Path

from tabularaml.benchmarks.feature_gen.manifest import build_manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tabularaml/benchmarks/feature_gen/manifest.csv")
    p.add_argument("--min-rows", type=int, default=500)
    p.add_argument("--max-rows", type=int, default=50_000)
    p.add_argument("--max-features", type=int, default=200)
    p.add_argument("--max-missing-pct", type=float, default=0.5)
    p.add_argument("--min-numeric-features", type=int, default=1)
    p.add_argument("--max-classes-multiclass", type=int, default=20)
    p.add_argument("--no-incremental", action="store_true",
                   help="Re-probe all tids even if an existing manifest is present")
    args = p.parse_args()

    df = build_manifest(
        min_rows=args.min_rows,
        max_rows=args.max_rows,
        max_features=args.max_features,
        max_missing_pct=args.max_missing_pct,
        min_numeric_features=args.min_numeric_features,
        max_classes_multiclass=args.max_classes_multiclass,
        out_path=Path(args.out),
        incremental=not args.no_incremental,
    )
    print(f"Manifest written to {args.out} with {len(df)} datasets "
          f"({(df['task'] == 'classification').sum()} cls, "
          f"{(df['task'] == 'regression').sum()} reg)")


if __name__ == "__main__":
    main()
