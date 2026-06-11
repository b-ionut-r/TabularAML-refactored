#!/usr/bin/env python
"""Rigorous old-vs-new A/B benchmark for the genetic feature search.

Compares the pre-overhaul configuration ("old": mean-rule acceptance,
per-candidate FeatureBoost proxy, no confirmation / null-importance /
base expansion, old lite preset sizes) against the competition-grade
configuration ("new": current defaults + retuned lite preset sizes) at an
IDENTICAL time budget, baseline model and holdout split.

Per (dataset, seed, config) it records search CV gain, held-out test gain,
overfit gap (cv_gain - test_gain), features added, wall time and candidate
evaluations per minute. Results append to a CSV (resumable: completed combos
are skipped). The summary applies the pre-registered criteria:

  (a) win-rate >= 60% or Wilcoxon p < 0.10 in new's favor on held-out test gain
  (b) mean overfit gap strictly reduced
  (c) no dataset regresses on mean test gain by more than 1 std of the old
      config's seed noise on that dataset
  (d) candidate evals/min >= 1.5x old

Usage:
  python scripts/run_acceptance_ab_benchmark.py --seeds 42,43,44 \
      --time-budget 300 --parallel 2 --out reports/acceptance_ab.csv
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Real-world PMLB datasets (fetched via the Git-LFS media host — the only
# dataset source reachable under this environment's network policy; openml.org
# and UCI return 403). Balanced binary / multiclass / regression, small->mid.
DATASETS = {
    # binary
    "churn": "classification",        # 5000 x 20
    "spambase": "classification",     # 4601 x 57
    "hypothyroid": "classification",  # 3163 x 25
    "coil2000": "classification",     # 9822 x 85
    # multiclass
    "satimage": "classification",     # 6435 x 36, 6c
    "splice": "classification",       # 3188 x 60, 3c
    "optdigits": "classification",    # 5620 x 64, 10c
    "ann_thyroid": "classification",  # 7200 x 21, 3c
    # regression
    "503_wind": "regression",         # 6574 x 14
    "537_houses": "regression",       # 20640 x 8 (california housing)
    "573_cpu_act": "regression",      # 8192 x 21
    "4544_GeographicalOriginalofMusic": "regression",  # 1059 x 117
    # synthetic era showcase (CrunchDAO-style; handled specially in run_one)
    "synthetic_era": "era",
}

# Git-LFS media host (raw.githubusercontent.com serves only the LFS pointer)
PMLB_URL = "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/{name}/{name}.tsv.gz"
CACHE_DIR = REPO_ROOT / "cache" / "pmlb"

# Shared run frame (identical for both configs)
SHARED = dict(
    n_generations=12, n_parents=15, cv=4, ranking_method="multi_criteria",
    search_sample_size=10_000, cache_size_mb=500, meta_validation_frac=0.15,
    rotate_cv_folds=True, fold_rotation_period=4, proxy_top_pct=0.20,
    max_new_feats=0.6, early_stopping_iter=5, final_selection=True,
    use_gpu=False, log_file=None,
)

CONFIGS = {
    "old": dict(  # pre-overhaul behavior + old lite sizes
        acceptance="mean", confirmation_seeds=0, null_importance_selection=False,
        proxy_mode="featureboost", expand_datetime=False, expand_row_stats=False,
        cv_n_jobs=1, n_children=90, early_stopping_child_eval=25, min_pct_gain=0.003,
        proxy_row_cap=0,  # old behavior: full-row proxy
    ),
    "new": dict(  # current shipping defaults (throughput + gated expansion)
        acceptance="mean", confirmation_seeds=0, null_importance_selection=False,
        proxy_mode="batched", expand_datetime=True, expand_row_stats=True,
        cv_n_jobs="auto", n_children=160, early_stopping_child_eval=30, min_pct_gain=0.003,
    ),
    "full": dict(  # complete competition stack (opt-in trio enabled)
        acceptance="statistical", confirmation_seeds=1, null_importance_selection=True,
        proxy_mode="batched", expand_datetime=True, expand_row_stats=True,
        cv_n_jobs="auto", n_children=160, early_stopping_child_eval=30, min_pct_gain=0.002,
    ),
}

# Synthetic era showcase (CrunchDAO-style): cross-sectional signal whose
# coefficients drift across eras; evaluation on held-out FUTURE eras.
ERA_N_ERAS, ERA_PER_ERA, ERA_TRAIN_ERAS, ERA_DATA_SEED = 45, 250, 32, 777


def make_synthetic_era():
    rng = np.random.default_rng(ERA_DATA_SEED)  # data fixed across runs/configs
    frames = []
    for e in range(ERA_N_ERAS):
        Xe = rng.normal(size=(ERA_PER_ERA, 6))
        w = np.array([1.0, 0.6, 0.0, 0.0, 0.3, 0.0]) * (1 + 0.3 * np.sin(e / 5.0))
        signal = Xe @ w + 0.8 * np.cos(e / 7.0) * Xe[:, 2] * Xe[:, 3]
        y_e = signal + rng.normal(size=ERA_PER_ERA) * 1.2
        df = pd.DataFrame(Xe, columns=[f"f{i}" for i in range(6)])
        df["era"] = e
        df["target"] = y_e
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    y = df.pop("target")
    return df, y


def mean_per_era_spearman(y_true, y_pred, eras):
    from scipy.stats import spearmanr
    vals = []
    for e in pd.unique(eras):
        m = np.asarray(eras) == e
        if m.sum() < 3:
            continue
        c = spearmanr(np.asarray(y_true)[m], np.asarray(y_pred)[m])[0]
        if np.isfinite(c):
            vals.append(c)
    return float(np.mean(vals)) if vals else 0.0


def run_one_era(spec):
    """Era showcase: full config uses era_col + era_spearman; old has no era
    support (era column dropped). Both judged on held-out FUTURE eras by mean
    per-era Spearman of an identical downstream model. test_gain is the
    ABSOLUTE Spearman delta (relative gain is meaningless near 0)."""
    name, seed, config_name, time_budget, n_jobs = spec
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, n_jobs)))
    from tabularaml.generate.features import FeatureGenerator
    from tabularaml.eval.scorers import PREDEFINED_REG_SCORERS
    from tabularaml.eval.cv import sanitize_model_features

    X, y = make_synthetic_era()
    train_mask = X["era"] < ERA_TRAIN_ERAS
    X_tr, y_tr = X[train_mask].reset_index(drop=True), y[train_mask].reset_index(drop=True)
    X_te, y_te = X[~train_mask].reset_index(drop=True), y[~train_mask].reset_index(drop=True)
    eras_te = X_te["era"].values

    cfg = dict(CONFIGS[config_name])
    gen_kwargs = dict(task="regression", random_state=seed, n_jobs=n_jobs,
                      time_budget=time_budget, **SHARED, **cfg)
    if config_name == "old":
        # pre-overhaul framework had no era support: era column dropped
        X_tr_in, X_te_in = X_tr.drop(columns=["era"]), X_te.drop(columns=["era"])
        gen = FeatureGenerator(baseline_model=make_model("regression", seed, n_jobs),
                               **gen_kwargs)
    else:
        X_tr_in, X_te_in = X_tr, X_te.drop(columns=["era"])
        gen = FeatureGenerator(baseline_model=make_model("regression", seed, n_jobs),
                               era_col="era", scorer=PREDEFINED_REG_SCORERS["era_spearman"],
                               **gen_kwargs)

    def era_score(X_train_feats, X_test_feats):
        model = make_model("regression", seed, n_jobs)
        X_a = sanitize_model_features(X_train_feats)
        X_b = align_test_to_train(X_a, sanitize_model_features(X_test_feats))
        model.fit(X_a, y_tr)
        return mean_per_era_spearman(y_te, model.predict(X_b), eras_te)

    base_test = era_score(X_tr.drop(columns=["era"]).copy(), X_te.drop(columns=["era"]).copy())

    t0 = time.time()
    gen.search(X_tr_in.copy(), y_tr.copy())
    wall = time.time() - t0
    evals = int(sum(gen.adaptive_controller.op_usage.values()))

    gen.fit(X_tr_in.copy(), y_tr.copy())
    new_test = era_score(gen.transform(X_tr_in.copy()), gen.transform(X_te_in.copy()))

    return {
        "dataset": name, "seed": seed, "config": config_name, "task": "era",
        "cv_gain": float(getattr(gen, "pct_gain", 0.0)),
        "test_gain": float(new_test - base_test),  # ABSOLUTE Spearman delta
        "overfit_gap": float(getattr(gen, "pct_gain", 0.0)) - float(new_test - base_test),
        "base_test": float(base_test), "new_test": float(new_test),
        "n_new_feats": int(getattr(gen, "n_added_feats", 0)),
        "gens_completed": int(gen.state["counters"].get("current_gen", 0)) + 1,
        "wall_time_s": round(wall, 1),
        "evals": evals, "evals_per_min": round(evals / max(wall / 60.0, 1e-9), 1),
    }


def load_dataset(name):
    task = DATASETS[name]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{name}.tsv.gz"
    if not cache_file.exists():
        import urllib.request
        urllib.request.urlretrieve(PMLB_URL.format(name=name), cache_file)
    df = pd.read_csv(cache_file, sep="\t", compression="gzip")
    y = df["target"]
    X = df.drop(columns=["target"])
    if task == "classification":
        y = pd.Series(pd.factorize(y, sort=True)[0], index=y.index, name="target")
    else:
        y = pd.to_numeric(y, errors="coerce")
        keep = y.notna()
        X, y = X.loc[keep], y.loc[keep]
    return X.reset_index(drop=True), pd.Series(y).reset_index(drop=True), task


def make_model(task, seed, n_jobs):
    from lightgbm import LGBMRegressor, LGBMClassifier
    cls = LGBMRegressor if task == "regression" else LGBMClassifier
    return cls(n_jobs=n_jobs, verbose=-1, random_state=seed)


def pick_scorer(task, y):
    from tabularaml.eval.scorers import PREDEFINED_REG_SCORERS, PREDEFINED_CLS_SCORERS
    if task == "regression":
        return PREDEFINED_REG_SCORERS["rmse"]
    return (PREDEFINED_CLS_SCORERS["binary_crossentropy"] if y.nunique() == 2
            else PREDEFINED_CLS_SCORERS["categorical_crossentropy"])


def align_test_to_train(X_tr, X_te):
    """Match test schema/categories to the train frame for LightGBM."""
    X_te = X_te.reindex(columns=X_tr.columns)
    for col in X_tr.columns:
        if isinstance(X_tr[col].dtype, pd.CategoricalDtype):
            X_te[col] = pd.Categorical(X_te[col], categories=X_tr[col].cat.categories)
    return X_te


def holdout_score(model, scorer, X_tr, y_tr, X_te, y_te):
    from tabularaml.eval.cv import sanitize_model_features
    X_tr = sanitize_model_features(X_tr)
    X_te = align_test_to_train(X_tr, sanitize_model_features(X_te))
    model.fit(X_tr, y_tr)
    preds = model.predict_proba(X_te) if scorer.from_probs else model.predict(X_te)
    if scorer.name == "categorical_crossentropy":
        from sklearn.preprocessing import OneHotEncoder
        oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        oh.fit(np.asarray(y_tr).reshape(-1, 1))
        return scorer.score(oh.transform(np.asarray(y_te).reshape(-1, 1)), preds)
    return scorer.score(np.asarray(y_te), preds)


def run_one(spec):
    name, seed, config_name, time_budget, n_jobs = spec
    # Cap OpenMP before any lightgbm/xgboost load: multiple worker processes
    # with unconstrained OpenMP pools spin-wait each other to a standstill.
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, n_jobs)))
    if DATASETS.get(name) == "era":
        return run_one_era(spec)
    from sklearn.model_selection import train_test_split
    from tabularaml.generate.features import FeatureGenerator

    X, y, task = load_dataset(name)
    strat = y if task == "classification" else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed,
                                              stratify=strat)
    X_tr, X_te = X_tr.reset_index(drop=True), X_te.reset_index(drop=True)
    y_tr, y_te = y_tr.reset_index(drop=True), y_te.reset_index(drop=True)

    scorer = pick_scorer(task, y)
    sign = 1.0 if scorer.greater_is_better else -1.0

    # Baseline holdout score on ORIGINAL features (identical for both configs)
    base_test = holdout_score(make_model(task, seed, n_jobs), scorer,
                              X_tr.copy(), y_tr, X_te.copy(), y_te)

    gen = FeatureGenerator(task=task, scorer=scorer,
                           baseline_model=make_model(task, seed, n_jobs),
                           random_state=seed, n_jobs=n_jobs,
                           time_budget=time_budget,
                           **SHARED, **CONFIGS[config_name])
    t0 = time.time()
    gen.search(X_tr.copy(), y_tr.copy(), X_test=None)
    wall = time.time() - t0
    evals = int(sum(gen.adaptive_controller.op_usage.values()))

    # Transform both splits with the trained generator and score on holdout
    gen.fit(X_tr.copy(), y_tr.copy())
    X_tr_t = gen.transform(X_tr.copy())
    X_te_t = gen.transform(X_te.copy())
    new_test = holdout_score(make_model(task, seed, n_jobs), scorer,
                             X_tr_t, y_tr, X_te_t, y_te)

    test_gain = sign * (new_test - base_test) / (abs(base_test) + 1e-8)
    cv_gain = float(gen.pct_gain)
    return {
        "dataset": name, "seed": seed, "config": config_name, "task": task,
        "cv_gain": cv_gain, "test_gain": float(test_gain),
        "overfit_gap": cv_gain - float(test_gain),
        "base_test": float(base_test), "new_test": float(new_test),
        "n_new_feats": int(getattr(gen, "n_added_feats", 0)),
        "gens_completed": int(gen.state["counters"].get("current_gen", 0)) + 1,
        "wall_time_s": round(wall, 1),
        "evals": evals, "evals_per_min": round(evals / max(wall / 60.0, 1e-9), 1),
    }


def run_one_safe(spec):
    try:
        return run_one(spec)
    except Exception as e:
        traceback.print_exc()
        name, seed, config_name, *_ = spec
        return {"dataset": name, "seed": seed, "config": config_name,
                "error": f"{type(e).__name__}: {e}"}


def summarize(df, out_md):
    from scipy.stats import wilcoxon
    ok = df[df["error"].isna()] if "error" in df.columns else df
    if ok.empty or "test_gain" not in ok.columns:
        msg = "INSUFFICIENT DATA: no successful runs recorded."
        Path(out_md).write_text(msg)
        print(msg)
        return

    # Baseline: the literal pre-session code when present, else the flag arm.
    # Challenger: whichever non-baseline config is present.
    present = list(ok["config"].unique())
    base = "gitold" if "gitold" in present else "old"
    configs = [c for c in present if c != base]
    ch = configs[0] if configs else "new"

    # Era showcase rows use ABSOLUTE Spearman deltas — summarized separately
    era = ok[ok.get("task", pd.Series(dtype=str)) == "era"] if "task" in ok.columns else ok.iloc[0:0]
    ok = ok[ok.get("task", pd.Series(dtype=str)) != "era"] if "task" in ok.columns else ok

    piv = ok.pivot_table(index=["dataset", "seed"], columns="config",
                         values=["test_gain", "overfit_gap", "evals_per_min"])
    piv = piv.dropna()
    if piv.empty or len(piv) < 3:
        verdict = ["INSUFFICIENT DATA: fewer than 3 complete paired runs."]
        Path(out_md).write_text("\n".join(verdict))
        print("\n".join(verdict))
        return

    d = piv[("test_gain", ch)] - piv[("test_gain", base)]
    wins = int((d > 1e-12).sum()); losses = int((d < -1e-12).sum()); ties = len(d) - wins - losses
    win_rate = wins / max(1, wins + losses)
    try:
        stat, pval = wilcoxon(piv[("test_gain", ch)], piv[("test_gain", base)],
                              alternative="greater", zero_method="zsplit")
    except ValueError:
        stat, pval = np.nan, 1.0

    gap_old = piv[("overfit_gap", base)].mean()
    gap_new = piv[("overfit_gap", ch)].mean()
    epm_ratio = (piv[("evals_per_min", ch)] / piv[("evals_per_min", base)]).mean()

    # criterion (c): per-dataset regression check vs old's seed noise
    per_ds = ok.pivot_table(index="dataset", columns="config", values="test_gain", aggfunc=["mean", "std"])
    regressions = []
    for ds in per_ds.index:
        mean_old = per_ds.loc[ds, ("mean", base)]
        mean_ch = per_ds.loc[ds, ("mean", ch)]
        noise = per_ds.loc[ds, ("std", base)]
        noise = noise if np.isfinite(noise) and noise > 0 else 0.0
        if mean_ch < mean_old - max(noise, 1e-9):
            regressions.append((ds, float(mean_old), float(mean_ch), float(noise)))

    crit_a = (win_rate >= 0.60) or (pval < 0.10)
    crit_b = gap_new < gap_old
    crit_c = len(regressions) == 0
    crit_d = epm_ratio >= 1.5

    lines = []
    lines.append(f"# A/B benchmark — {base} vs {ch} genetic search\n")
    lines.append(f"Paired runs: {len(piv)} (dataset x seed)\n")
    lines.append("## Pre-registered criteria\n")
    lines.append(f"- (a) test-gain superiority: win/tie/loss = {wins}/{ties}/{losses}, "
                 f"win-rate={win_rate:.0%}, Wilcoxon one-sided p={pval:.4f} -> "
                 f"{'PASS' if crit_a else 'FAIL'}")
    lines.append(f"- (b) overfit gap reduced: {base}={gap_old:.4f}, {ch}={gap_new:.4f} -> "
                 f"{'PASS' if crit_b else 'FAIL'}")
    lines.append(f"- (c) no dataset regression beyond the baseline's seed noise: "
                 f"{'PASS' if crit_c else 'FAIL ' + str(regressions)}")
    lines.append(f"- (d) throughput >= 1.5x: mean evals/min ratio = {epm_ratio:.2f}x -> "
                 f"{'PASS' if crit_d else 'FAIL'}")
    lines.append(f"\n- mean test gain: {base}={piv[('test_gain', base)].mean():.4f}, "
                 f"{ch}={piv[('test_gain', ch)].mean():.4f}")
    if "gens_completed" in ok.columns:
        gens = ok.groupby("config")["gens_completed"].mean()
        lines.append(f"- mean generations completed: "
                     + ", ".join(f"{c}={gens[c]:.1f}" for c in gens.index))
    lines.append("\n## Verdict\n")
    if crit_a and crit_b and crit_c and crit_d:
        lines.append(f"ALL CRITERIA PASS: the {ch} configuration is confirmed superior; "
                     "apply the decision rule (defaults flip to the proven stack).")
    elif not crit_a:
        lines.append("TEST-GAIN CRITERION FAILED: per the decision rule, the strict acceptance "
                     "stack stays opt-in (defaults stay as shipped).")
    elif not crit_d:
        lines.append("THROUGHPUT CRITERION FAILED: keep current defaults but revert preset upsizing.")
    else:
        lines.append("PARTIAL PASS: see failed criteria above; apply the decision rule accordingly.")
    lines.append("\n## Per-dataset means (test gain)\n")
    lines.append(per_ds.round(4).to_string())

    if not era.empty:
        lines.append("\n## Era showcase (synthetic, held-out FUTURE eras, ABSOLUTE mean "
                     "per-era Spearman delta vs raw features)\n")
        era_piv = era.pivot_table(index="seed", columns="config",
                                  values=["base_test", "new_test", "test_gain"])
        lines.append(era_piv.round(4).to_string())

    report = "\n".join(lines)
    Path(out_md).write_text(report)
    print(report)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--configs", default="old,full",
                    help="comma list from {old,new,full}; first should be 'old'")
    ap.add_argument("--time-budget", type=int, default=300)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--out", default="reports/acceptance_ab.csv")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_md = out.with_suffix(".md")

    done = set()
    if out.exists():
        prev = pd.read_csv(out)
        done = {(r.dataset, int(r.seed), r.config) for r in prev.itertuples()
                if not (hasattr(r, "error") and isinstance(r.error, str) and r.error)}

    if not args.summary_only:
        names = [n.strip() for n in args.datasets.split(",") if n.strip()]
        seeds = [int(s) for s in args.seeds.split(",")]
        run_configs = [c.strip() for c in args.configs.split(",") if c.strip() in CONFIGS]
        n_jobs = max(1, (os.cpu_count() or 4) // max(1, args.parallel))
        specs = [(n, s, c, args.time_budget, n_jobs)
                 for n in names for s in seeds for c in run_configs
                 if (n, s, c) not in done]
        print(f"{len(specs)} runs to do ({len(done)} already complete), "
              f"parallel={args.parallel}, n_jobs per run={n_jobs}")

        def write_row(row):
            df_row = pd.DataFrame([row])
            header = not out.exists()
            df_row.to_csv(out, mode="a", header=header, index=False)
            print(f"  done: {row.get('dataset')}/{row.get('seed')}/{row.get('config')} "
                  f"test_gain={row.get('test_gain')} err={row.get('error', '')}")

        if args.parallel > 1 and len(specs) > 1:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            with ctx.Pool(args.parallel) as pool:
                for row in pool.imap_unordered(run_one_safe, specs):
                    write_row(row)
        else:
            for spec in specs:
                write_row(run_one_safe(spec))

    df = pd.read_csv(out)
    if "error" not in df.columns:
        df["error"] = np.nan
    summarize(df, out_md)


if __name__ == "__main__":
    main()
