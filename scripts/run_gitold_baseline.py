#!/usr/bin/env python
"""Append a TRUE pre-session baseline arm (git worktree of 85ba69b) to the
comprehensive benchmark CSV. Each run executes scripts/_gitold_worker.py in a
subprocess so the old tabularaml shadows the new one cleanly.

Usage:
  python scripts/run_gitold_baseline.py --out reports/comprehensive_ab.csv \
      --time-budget 600 --parallel 2
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_REPO = "/tmp/old_tree"
CACHE_DIR = str(REPO_ROOT / "cache" / "pmlb")
WORKER = str(REPO_ROOT / "scripts" / "_gitold_worker.py")

# Full grid: identical to the new arm's datasets + the era showcase
DATASETS = {
    "churn": "classification",
    "spambase": "classification",
    "hypothyroid": "classification",
    "coil2000": "classification",
    "satimage": "classification",
    "splice": "classification",
    "optdigits": "classification",
    "ann_thyroid": "classification",
    "503_wind": "regression",
    "537_houses": "regression",
    "573_cpu_act": "regression",
    "4544_GeographicalOriginalofMusic": "regression",
    "synthetic_era": "era",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/final_old_vs_new.csv")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--time-budget", type=int, default=600)
    ap.add_argument("--parallel", type=int, default=2)
    args = ap.parse_args()

    out = Path(args.out)
    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = {(r.dataset, int(r.seed), r.config) for r in prev.itertuples()
                if r.config == "gitold"
                and not (hasattr(r, "error") and isinstance(r.error, str) and r.error)}

    seeds = [int(s) for s in args.seeds.split(",")]
    n_jobs = max(1, 4 // max(1, args.parallel))
    specs = [(n, s) for n in DATASETS for s in seeds if (n, s, "gitold") not in done]
    print(f"{len(specs)} gitold runs to do ({len(done)} already complete)")

    def write_row(row):
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"  done: {row.get('dataset')}/{row.get('seed')}/gitold "
              f"test_gain={row.get('test_gain')} err={row.get('error', '')}")

    def launch(spec):
        name, seed = spec
        return subprocess.Popen(
            [sys.executable, WORKER, name, str(seed), str(args.time_budget),
             str(n_jobs), OLD_REPO, CACHE_DIR, DATASETS[name]],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

    def collect(proc, spec):
        out_text, _ = proc.communicate()
        for line in out_text.splitlines():
            if line.startswith("GITOLD_RESULT "):
                write_row(json.loads(line[len("GITOLD_RESULT "):]))
                return
        write_row({"dataset": spec[0], "seed": spec[1], "config": "gitold",
                   "error": "no result line (crash before reporting)"})

    running = []
    for spec in specs:
        while len(running) >= args.parallel:
            for i, (p, s) in enumerate(running):
                if p.poll() is not None:
                    collect(p, s)
                    running.pop(i)
                    break
            else:
                running[0][0].wait()
        running.append((launch(spec), spec))
    for p, s in running:
        collect(p, s)
    print("gitold arm complete")


if __name__ == "__main__":
    main()
