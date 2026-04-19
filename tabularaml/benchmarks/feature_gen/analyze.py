"""Post-hoc analysis of master.csv: stats, plots, report.md.

Reads a completed (or partial) benchmark result table and emits:
    summary.csv
    wilcoxon.csv
    cd_plot.png                 (critical-difference diagram)
    pareto_scatter.png
    win_matrix.png
    per_dataset_heatmap.png
    report.md                   (stitches everything together)

Designed to be resilient: works on partial data, silently drops frameworks
without enough datasets for statistical tests.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import pandas as pd


def load_master(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Guard against empty/partial runs.
    for col in ["dataset_id", "seed", "score_holdout", "pct_improvement",
                "wall_time_total"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def per_dataset_means(df: pd.DataFrame) -> pd.DataFrame:
    """Average score / improvement over seeds, one row per (dataset_id, framework)."""
    ok = df[df["status"] == "ok"].copy()
    if len(ok) == 0:
        return pd.DataFrame()
    g = ok.groupby(["dataset_id", "task", "framework"], as_index=False).agg(
        score_holdout=("score_holdout", "mean"),
        pct_improvement=("pct_improvement", "mean"),
        wall_time_total=("wall_time_total", "mean"),
        n_added=("n_added", "mean"),
        n_seeds=("seed", "nunique"),
    )
    return g


def summary_by_framework(pdm: pd.DataFrame) -> pd.DataFrame:
    if len(pdm) == 0:
        return pd.DataFrame()
    out = pdm.groupby("framework").agg(
        n_datasets=("dataset_id", "nunique"),
        median_pct_improvement=("pct_improvement", "median"),
        mean_pct_improvement=("pct_improvement", "mean"),
        median_wall_time=("wall_time_total", "median"),
        mean_wall_time=("wall_time_total", "mean"),
        mean_added=("n_added", "mean"),
    ).reset_index().sort_values("median_pct_improvement", ascending=False)
    return out


def win_matrix(pdm: pd.DataFrame) -> pd.DataFrame:
    """M[i,j] = fraction of datasets where framework i beat framework j (by pct_improvement)."""
    if len(pdm) == 0:
        return pd.DataFrame()
    pivot = pdm.pivot_table(index="dataset_id", columns="framework",
                            values="pct_improvement", aggfunc="mean")
    frameworks = list(pivot.columns)
    out = pd.DataFrame(index=frameworks, columns=frameworks, dtype=float)
    for i in frameworks:
        for j in frameworks:
            common = pivot[[i, j]].dropna()
            if i == j or len(common) == 0:
                out.loc[i, j] = np.nan
            else:
                out.loc[i, j] = float((common[i] > common[j]).mean())
    return out


def wilcoxon_vs_baseline(pdm: pd.DataFrame, baseline: str) -> pd.DataFrame:
    from scipy.stats import wilcoxon
    if len(pdm) == 0 or baseline not in set(pdm["framework"]):
        return pd.DataFrame()

    pivot = pdm.pivot_table(index="dataset_id", columns="framework",
                            values="pct_improvement", aggfunc="mean")
    rows = []
    for fw in [c for c in pivot.columns if c != baseline]:
        common = pivot[[fw, baseline]].dropna()
        if len(common) < 10:
            continue
        diff = common[fw] - common[baseline]
        if (diff == 0).all():
            rows.append({"framework": fw, "baseline": baseline,
                         "n": len(common), "median_diff": 0.0,
                         "wilcoxon_stat": 0.0, "p_value": 1.0})
            continue
        try:
            stat, pval = wilcoxon(diff.values, alternative="greater", zero_method="zsplit")
        except ValueError:
            continue
        rows.append({
            "framework": fw, "baseline": baseline,
            "n": len(common),
            "median_diff": float(diff.median()),
            "wilcoxon_stat": float(stat),
            "p_value": float(pval),
        })
    out = pd.DataFrame(rows)
    if len(out):
        # Holm correction.
        out = out.sort_values("p_value").reset_index(drop=True)
        m = len(out)
        out["p_holm"] = [min(1.0, out["p_value"].iloc[k] * (m - k)) for k in range(m)]
    return out


def critical_difference(pdm: pd.DataFrame, out_png: Path) -> Optional[dict]:
    """Friedman + Nemenyi CD diagram via autorank. Returns summary dict or None."""
    if len(pdm) == 0:
        return None
    pivot = pdm.pivot_table(index="dataset_id", columns="framework",
                            values="pct_improvement", aggfunc="mean")
    pivot = pivot.dropna(axis=0, how="any")
    if len(pivot) < 10 or pivot.shape[1] < 3:
        return None
    try:
        import autorank
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = autorank.autorank(pivot, alpha=0.05, verbose=False,
                                       order="descending")
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(10, 4))
            autorank.plot_stats(result)
            plt.tight_layout()
            plt.savefig(out_png, dpi=120, bbox_inches="tight")
            plt.close(fig)
        return {
            "pvalue": float(result.pvalue),
            "omnibus": str(result.omnibus),
            "posthoc": str(result.posthoc),
            "cd": float(getattr(result, "cd", np.nan) or np.nan),
            "n_datasets": int(len(pivot)),
        }
    except Exception as e:
        print(f"autorank failed: {e}")
        return None


def pareto_scatter(pdm: pd.DataFrame, out_png: Path) -> None:
    if len(pdm) == 0:
        return
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, task in zip(axes, ["classification", "regression"]):
        sub = pdm[pdm["task"] == task]
        if len(sub) == 0:
            ax.set_title(f"{task} (no data)")
            continue
        for fw, grp in sub.groupby("framework"):
            ax.scatter(grp["wall_time_total"], grp["pct_improvement"], alpha=0.5, label=fw, s=18)
        ax.set_xscale("log")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("wall time (s, log scale)")
        ax.set_ylabel("pct_improvement over no-FE")
        ax.set_title(task)
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def win_matrix_plot(wm: pd.DataFrame, out_png: Path) -> None:
    if wm.empty:
        return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(wm)), max(4, 0.8 * len(wm))))
    mat = wm.astype(float).values
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(wm.columns)))
    ax.set_yticks(range(len(wm.index)))
    ax.set_xticklabels(wm.columns, rotation=45, ha="right")
    ax.set_yticklabels(wm.index)
    for i in range(len(wm.index)):
        for j in range(len(wm.columns)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black" if 0.25 <= v <= 0.75 else "white", fontsize=9)
    ax.set_title("Win rate (row beats column, per dataset)")
    fig.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def per_dataset_heatmap(pdm: pd.DataFrame, out_png: Path) -> None:
    if len(pdm) == 0:
        return
    import matplotlib.pyplot as plt
    pivot = pdm.pivot_table(index="dataset_id", columns="framework",
                            values="pct_improvement", aggfunc="mean")
    if pivot.empty:
        return
    pivot = pivot.sort_values(by=pivot.columns.tolist()[0], ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(pivot.columns)),
                                    max(5, 0.12 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r",
                   vmin=-np.nanmax(np.abs(pivot.values)) if np.isfinite(pivot.values).any() else None,
                   vmax=np.nanmax(np.abs(pivot.values)) if np.isfinite(pivot.values).any() else None)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks([])
    ax.set_ylabel(f"{len(pivot)} datasets")
    ax.set_title("pct_improvement over no-FE (per dataset × framework)")
    fig.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def build_report(master_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_master(master_path)
    pdm = per_dataset_means(df)

    summary = summary_by_framework(pdm)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    wm = win_matrix(pdm)
    if not wm.empty:
        wm.to_csv(out_dir / "win_matrix.csv")
        win_matrix_plot(wm, out_dir / "win_matrix.png")

    wilcoxon_rows = []
    for base in ["nofe", "openfe"]:
        if base in set(pdm.get("framework", [])):
            rows = wilcoxon_vs_baseline(pdm, base)
            if len(rows):
                wilcoxon_rows.append(rows)
    wilcoxon = pd.concat(wilcoxon_rows, ignore_index=True) if wilcoxon_rows else pd.DataFrame()
    if not wilcoxon.empty:
        wilcoxon.to_csv(out_dir / "wilcoxon.csv", index=False)

    cd_result = critical_difference(pdm, out_dir / "cd_plot.png")
    pareto_scatter(pdm, out_dir / "pareto_scatter.png")
    per_dataset_heatmap(pdm, out_dir / "per_dataset_heatmap.png")

    crash_by_fw = df.groupby("framework")["status"].apply(
        lambda s: (s != "ok").mean()
    ).round(3).to_dict()

    lines = []
    lines.append("# TabularAML Feature Generation Benchmark Report")
    lines.append("")
    lines.append(f"Source: `{master_path}`   Total rows: **{len(df)}**   OK rows: **{(df['status']=='ok').sum()}**")
    lines.append("")
    lines.append("## Summary (per framework)")
    lines.append("")
    lines.append(summary.to_markdown(index=False) if len(summary) else "_no data yet_")
    lines.append("")
    lines.append("### Crash / timeout rates")
    lines.append("")
    for fw, rate in crash_by_fw.items():
        lines.append(f"- **{fw}**: {rate:.1%} non-ok")
    lines.append("")
    lines.append("## Statistical tests")
    lines.append("")
    if not wilcoxon.empty:
        lines.append("### Wilcoxon signed-rank (one-sided, framework > baseline)")
        lines.append("")
        lines.append(wilcoxon.to_markdown(index=False))
        lines.append("")
    if cd_result:
        lines.append("### Friedman + Nemenyi critical-difference diagram")
        lines.append("")
        lines.append(f"- Omnibus: **{cd_result['omnibus']}**, p = {cd_result['pvalue']:.2e}")
        lines.append(f"- Post-hoc: **{cd_result['posthoc']}**")
        lines.append(f"- Critical difference: **{cd_result['cd']:.3f}** (n = {cd_result['n_datasets']})")
        lines.append("")
        lines.append("![CD plot](cd_plot.png)")
        lines.append("")
    lines.append("## Runtime vs. gain (Pareto)")
    lines.append("")
    lines.append("![Pareto scatter](pareto_scatter.png)")
    lines.append("")
    lines.append("## Win matrix")
    lines.append("")
    lines.append("![Win matrix](win_matrix.png)")
    lines.append("")
    lines.append("## Per-dataset heatmap")
    lines.append("")
    lines.append("![Heatmap](per_dataset_heatmap.png)")
    lines.append("")

    # Emit data-leakage caveat whenever OpenFE results are present.
    frameworks_present = set(df["framework"].dropna().unique()) if "framework" in df.columns else set()
    if "openfe" in frameworks_present:
        lines.append("## ⚠ OpenFE data-leakage caveat")
        lines.append("")
        lines.append(
            "OpenFE's `transform()` concatenates the training and test sets before "
            "computing `GroupByThenMean`-style aggregate features. As a result, aggregate "
            "columns in **both** outputs are influenced by test-set rows, introducing "
            "covariate-shift leakage that artificially inflates OpenFE's holdout scores "
            "on heavily-aggregated feature sets."
        )
        lines.append("")
        lines.append(
            "This benchmark deliberately uses the upstream package as distributed "
            "(`pip install openfe`) to reproduce what a reader would obtain. The companion "
            "test `tests/test_openfe_leakage_probe.py` quantifies the leakage and will "
            "fail CI if the upstream package ever makes it worse."
        )
        lines.append("")
        lines.append(
            "**Interpretation**: OpenFE's `pct_improvement` numbers should be treated as "
            "an upper bound on its real-world benefit. The other frameworks (TabularAML, "
            "AutoFeat, Featuretools) are not affected by this issue."
        )
        lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--master", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    build_report(Path(args.master), Path(args.out))
    print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
