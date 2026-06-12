"""Paired before/after comparison of two targeted-benchmark result sets.

Joins the two master.csv files on (dataset_source, dataset_id, suite, seed) and
compares the `tabularaml` arm of each run, using each run's own NoFE baseline.

Includes an identity check: NoFE holdout scores must match across the two runs
(same seeds, same frozen protocol) — any drift means the comparison is
contaminated (different package versions, harness edits, etc.).

Usage:
    python scripts/compare_targeted_runs.py \
        --before /home/user/results/before-pmlb/master.csv \
        --after  /home/user/results/after-pmlb/master.csv \
        --out    /home/user/results/before_after_report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["dataset_source", "dataset_id", "suite", "seed"]
FRAMEWORK = "tabularaml"


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["dataset_id"] = df["dataset_id"].astype(str)
    df["seed"] = df["seed"].astype(int)
    return df


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    # keep the last row per (key, framework) — resume reruns append
    return df.drop_duplicates(subset=KEY + ["framework"], keep="last")


def nofe_identity_check(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    b = before[(before.framework == "nofe") & (before.status == "ok")]
    a = after[(after.framework == "nofe") & (after.status == "ok")]
    m = b.merge(a, on=KEY, suffixes=("_before", "_after"))
    m["nofe_abs_diff"] = (m["score_holdout_before"] - m["score_holdout_after"]).abs()
    return m[KEY + ["dataset_name_before", "score_holdout_before", "score_holdout_after", "nofe_abs_diff"]]


def paired_table(before: pd.DataFrame, after: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    b = before[before.framework == FRAMEWORK]
    a = after[after.framework == FRAMEWORK]
    m = b.merge(a, on=KEY, suffixes=("_before", "_after"), how="outer", indicator=True)
    ok = m[(m.status_before == "ok") & (m.status_after == "ok")
           & m.pct_improvement_before.notna() & m.pct_improvement_after.notna()].copy()
    ok["delta"] = ok["pct_improvement_after"] - ok["pct_improvement_before"]
    unpaired = m[(m.status_before != "ok") | (m.status_after != "ok")
                 | m.pct_improvement_before.isna() | m.pct_improvement_after.isna()]
    return ok, unpaired


def wilcoxon_safe(x: np.ndarray) -> dict:
    from scipy.stats import wilcoxon
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    out = {"n": int(len(x)), "median": float(np.median(x)) if len(x) else None,
           "mean": float(np.mean(x)) if len(x) else None}
    nonzero = x[x != 0]
    if len(nonzero) < 5:
        out.update(stat=None, p_two_sided=None, p_after_greater=None,
                   note="too few non-zero pairs for Wilcoxon")
        return out
    try:
        st, p2 = wilcoxon(x, zero_method="pratt")
        _, pg = wilcoxon(x, zero_method="pratt", alternative="greater")
        out.update(stat=float(st), p_two_sided=float(p2), p_after_greater=float(pg), note="")
    except Exception as e:  # noqa: BLE001
        out.update(stat=None, p_two_sided=None, p_after_greater=None, note=str(e))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tie-eps", type=float, default=1e-6)
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    before = _dedupe(_load(args.before))
    after = _dedupe(_load(args.after))

    # --- identity check -----------------------------------------------------
    ident = nofe_identity_check(before, after)
    n_drift = int((ident["nofe_abs_diff"] > 1e-9).sum())
    ident.to_csv(out_dir / "nofe_identity_check.csv", index=False)

    # --- paired comparison ---------------------------------------------------
    pairs, unpaired = paired_table(before, after)
    pairs_out = pairs[KEY + ["dataset_name_before", "task_before",
                             "pct_improvement_before", "pct_improvement_after", "delta",
                             "n_added_before", "n_added_after",
                             "wall_time_total_before", "wall_time_total_after"]]
    pairs_out = pairs_out.sort_values("delta")
    pairs_out.to_csv(out_dir / "paired_runs.csv", index=False)
    unpaired_cols = [c for c in (KEY + ["dataset_name_before", "dataset_name_after",
                                        "status_before", "status_after",
                                        "error_msg_before", "error_msg_after"]) if c in unpaired.columns]
    unpaired[unpaired_cols].to_csv(out_dir / "unpaired_runs.csv", index=False)

    wins = int((pairs.delta > args.tie_eps).sum())
    losses = int((pairs.delta < -args.tie_eps).sum())
    ties = int(len(pairs) - wins - losses)

    pair_level = wilcoxon_safe(pairs.delta.values)

    ds = (pairs.groupby(["dataset_source", "dataset_id", "dataset_name_before"], as_index=False)
          .agg(pct_before=("pct_improvement_before", "mean"),
               pct_after=("pct_improvement_after", "mean"),
               delta=("delta", "mean"), n_seeds=("delta", "size")))
    ds = ds.sort_values("delta")
    ds.to_csv(out_dir / "per_dataset.csv", index=False)
    ds_level = wilcoxon_safe(ds.delta.values)
    ds_wins = int((ds.delta > args.tie_eps).sum())
    ds_losses = int((ds.delta < -args.tie_eps).sum())
    ds_ties = int(len(ds) - ds_wins - ds_losses)

    def status_counts(df: pd.DataFrame) -> dict:
        return df[df.framework == FRAMEWORK].status.value_counts().to_dict()

    summary = {
        "n_paired_runs": int(len(pairs)),
        "n_unpaired_runs": int(len(unpaired)),
        "nofe_identity_drift_rows": n_drift,
        "pair_level": {**pair_level, "wins": wins, "losses": losses, "ties": ties,
                       "median_pct_before": float(pairs.pct_improvement_before.median()) if len(pairs) else None,
                       "median_pct_after": float(pairs.pct_improvement_after.median()) if len(pairs) else None,
                       "mean_pct_before": float(pairs.pct_improvement_before.mean()) if len(pairs) else None,
                       "mean_pct_after": float(pairs.pct_improvement_after.mean()) if len(pairs) else None},
        "dataset_level": {**ds_level, "wins": ds_wins, "losses": ds_losses, "ties": ds_ties},
        "status_counts_before": status_counts(before),
        "status_counts_after": status_counts(after),
        "mean_wall_time_before": float(pairs.wall_time_total_before.mean()) if len(pairs) else None,
        "mean_wall_time_after": float(pairs.wall_time_total_after.mean()) if len(pairs) else None,
        "mean_n_added_before": float(pairs.n_added_before.mean()) if len(pairs) else None,
        "mean_n_added_after": float(pairs.n_added_after.mean()) if len(pairs) else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # --- markdown report ------------------------------------------------------
    md = ["# Before vs After — paired targeted-benchmark comparison\n"]
    md.append(f"- Paired (status=ok both arms): **{len(pairs)}** runs over **{len(ds)}** datasets; "
              f"unpaired/failed rows: {len(unpaired)} (see unpaired_runs.csv)")
    md.append(f"- NoFE identity check: {n_drift} rows drifted (> 1e-9) out of {len(ident)} "
              + ("✅" if n_drift == 0 else "⚠️ INVESTIGATE — protocol contamination"))
    pl, dl = summary["pair_level"], summary["dataset_level"]
    md.append("\n## Run level (dataset × seed)\n")
    md.append(f"| | before | after |\n|---|---|---|")
    md.append(f"| median pct_improvement vs NoFE | {pl['median_pct_before']:.5f} | {pl['median_pct_after']:.5f} |")
    md.append(f"| mean pct_improvement vs NoFE | {pl['mean_pct_before']:.5f} | {pl['mean_pct_after']:.5f} |")
    md.append(f"\n- Δ(after−before): median {pl['median']:.5f}, mean {pl['mean']:.5f}")
    md.append(f"- Wins / losses / ties: **{pl['wins']} / {pl['losses']} / {pl['ties']}**")
    if pl["p_after_greater"] is not None:
        md.append(f"- Wilcoxon signed-rank (after > before): p = **{pl['p_after_greater']:.4g}** "
                  f"(two-sided p = {pl['p_two_sided']:.4g}, n = {pl['n']})")
    md.append("\n## Dataset level (mean over seeds)\n")
    md.append(f"- Δ median {dl['median']:.5f}, mean {dl['mean']:.5f}; "
              f"wins/losses/ties: **{dl['wins']} / {dl['losses']} / {dl['ties']}** (n = {dl['n']})")
    if dl["p_after_greater"] is not None:
        md.append(f"- Wilcoxon (after > before): p = **{dl['p_after_greater']:.4g}**")
    md.append("\n## Reliability and cost\n")
    md.append(f"- Status counts before: `{summary['status_counts_before']}`")
    md.append(f"- Status counts after: `{summary['status_counts_after']}`")
    md.append(f"- Mean wall time (paired runs): {summary['mean_wall_time_before']:.0f}s → "
              f"{summary['mean_wall_time_after']:.0f}s")
    md.append(f"- Mean features added: {summary['mean_n_added_before']:.1f} → {summary['mean_n_added_after']:.1f}")
    md.append("\n## Per-dataset deltas (mean over seeds, sorted)\n")
    md.append("| dataset | task seeds | pct before | pct after | Δ |\n|---|---|---|---|---|")
    for r in ds.itertuples(index=False):
        md.append(f"| {r.dataset_name_before} | {r.n_seeds} | {r.pct_before:.5f} | {r.pct_after:.5f} | {r.delta:+.5f} |")
    (out_dir / "report.md").write_text("\n".join(md) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"\nReport written to {out_dir}/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
