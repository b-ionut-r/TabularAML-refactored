"""Summarise ``bench_fe.py`` results: holdout lift of each arm over raw features.

    python scripts/summarize_fe.py reports/fe_bench.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

GREATER_IS_BETTER = {"auc", "accuracy", "r2"}


def lift_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.status == "ok"]
    raw = df[df.arm == "raw"].set_index(["dataset", "seed"])["test_score"]
    rows = []
    for arm, g in df[df.arm != "raw"].groupby("arm"):
        for _, r in g.iterrows():
            key = (r.dataset, r.seed)
            if key not in raw.index:
                continue
            base = raw.loc[key]
            sign = 1 if r.metric in GREATER_IS_BETTER else -1
            rows.append(dict(arm=arm, dataset=r.dataset, seed=r.seed, metric=r.metric,
                             raw=base, fe=r.test_score,
                             lift_pct=100 * sign * (r.test_score - base) / abs(base),
                             fe_seconds=r.fe_seconds, n_added=r.n_added, gate=r.gate))
    return pd.DataFrame(rows)


def main(path):
    t = lift_table(pd.read_csv(path))
    if t.empty:
        print("no paired results yet")
        return
    pd.set_option("display.width", 200)
    for arm, g in t.groupby("arm"):
        per_ds = g.groupby("dataset").agg(lift_pct=("lift_pct", "mean"), n=("seed", "size"),
                                          fe_seconds=("fe_seconds", "mean"), n_added=("n_added", "mean"))
        print(f"\n=== {arm}: {len(g)} runs on {g.dataset.nunique()} datasets ===")
        print(per_ds.sort_values("lift_pct", ascending=False).round(2).to_string())
        print(f"mean lift {g.lift_pct.mean():+.2f}%  median {g.lift_pct.median():+.2f}%  "
              f"wins {(g.lift_pct > 0).mean():.0%}  losses<-0.5% {(g.lift_pct < -0.5).mean():.0%}  "
              f"mean FE time {g.fe_seconds.mean():.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "reports/fe_bench.csv")
