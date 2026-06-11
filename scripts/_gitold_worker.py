#!/usr/bin/env python
"""Run ONE benchmark run on the PRE-SESSION code (git worktree of 85ba69b).

Executed as a subprocess so the old tabularaml package can be imported in
isolation. Prints a single 'GITOLD_RESULT {json}' line on stdout.

argv: dataset_name seed time_budget n_jobs old_repo_path cache_dir task
"""
import json
import os
import sys
import time
import traceback


def main():
    name, seed, time_budget, n_jobs, old_repo, cache_dir, task = sys.argv[1:8]
    seed, time_budget, n_jobs = int(seed), int(time_budget), int(n_jobs)
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, n_jobs)))
    sys.path.insert(0, old_repo)

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(os.path.join(cache_dir, f"{name}.tsv.gz"), sep="\t", compression="gzip")
    y = df["target"]
    X = df.drop(columns=["target"])
    if task == "classification":
        y = pd.Series(pd.factorize(y, sort=True)[0], name="target")
    else:
        y = pd.to_numeric(y, errors="coerce")
        keep = y.notna()
        X, y = X.loc[keep], y.loc[keep]
    X, y = X.reset_index(drop=True), pd.Series(y).reset_index(drop=True)

    strat = y if task == "classification" else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed,
                                              stratify=strat)
    X_tr, X_te = X_tr.reset_index(drop=True), X_te.reset_index(drop=True)
    y_tr, y_te = y_tr.reset_index(drop=True), y_te.reset_index(drop=True)

    # Old-repo imports (path inserted above shadows the installed/new package)
    from tabularaml.generate.features import FeatureGenerator
    from tabularaml.eval.scorers import PREDEFINED_REG_SCORERS, PREDEFINED_CLS_SCORERS
    from lightgbm import LGBMRegressor, LGBMClassifier

    assert old_repo in FeatureGenerator.__module__ or True  # path check below
    import tabularaml
    if not tabularaml.__file__.startswith(old_repo):
        raise RuntimeError(f"old repo not shadowing: {tabularaml.__file__}")

    if task == "regression":
        scorer = PREDEFINED_REG_SCORERS["rmse"]
    else:
        scorer = (PREDEFINED_CLS_SCORERS["binary_crossentropy"] if y.nunique() == 2
                  else PREDEFINED_CLS_SCORERS["categorical_crossentropy"])
    sign = 1.0 if scorer.greater_is_better else -1.0

    def make_model():
        cls = LGBMRegressor if task == "regression" else LGBMClassifier
        return cls(n_jobs=n_jobs, verbose=-1, random_state=seed)

    def holdout_score(X_a, y_a, X_b, y_b):
        from tabularaml.eval.cv import sanitize_model_features
        X_a = sanitize_model_features(X_a)
        X_b = sanitize_model_features(X_b)
        X_b = X_b.reindex(columns=X_a.columns)
        for col in X_a.columns:
            if isinstance(X_a[col].dtype, pd.CategoricalDtype):
                X_b[col] = pd.Categorical(X_b[col], categories=X_a[col].cat.categories)
        model = make_model()
        model.fit(X_a, y_a)
        preds = model.predict_proba(X_b) if scorer.from_probs else model.predict(X_b)
        if scorer.name == "categorical_crossentropy":
            from sklearn.preprocessing import OneHotEncoder
            oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            oh.fit(np.asarray(y_a).reshape(-1, 1))
            return scorer.score(oh.transform(np.asarray(y_b).reshape(-1, 1)), preds)
        return scorer.score(np.asarray(y_b), preds)

    base_test = holdout_score(X_tr.copy(), y_tr, X_te.copy(), y_te)

    gen = FeatureGenerator(task=task, scorer=scorer, baseline_model=make_model(),
                           random_state=seed, n_jobs=n_jobs, time_budget=time_budget,
                           n_generations=12, n_parents=15, n_children=90,
                           early_stopping_child_eval=25, early_stopping_iter=5,
                           min_pct_gain=0.003, cv=4, ranking_method="multi_criteria",
                           search_sample_size=10_000, cache_size_mb=500,
                           use_proxy_evaluation=True, proxy_top_pct=0.20,
                           meta_validation_frac=0.15, rotate_cv_folds=True,
                           fold_rotation_period=4, max_new_feats=0.6,
                           final_selection=True, use_gpu=False, log_file=None)
    t0 = time.time()
    gen.search(X_tr.copy(), y_tr.copy())
    wall = time.time() - t0
    evals = int(sum(gen.adaptive_controller.op_usage.values()))
    gens = int(gen.state["counters"].get("current_gen", 0)) + 1

    gen.fit(X_tr.copy(), y_tr.copy())
    X_tr_t = gen.transform(X_tr.copy())
    X_te_t = gen.transform(X_te.copy())
    new_test = holdout_score(X_tr_t, y_tr, X_te_t, y_te)

    test_gain = sign * (new_test - base_test) / (abs(base_test) + 1e-8)
    cv_gain = float(getattr(gen, "pct_gain", 0.0))
    print("GITOLD_RESULT " + json.dumps({
        "dataset": name, "seed": seed, "config": "gitold", "task": task,
        "cv_gain": cv_gain, "test_gain": float(test_gain),
        "overfit_gap": cv_gain - float(test_gain),
        "base_test": float(base_test), "new_test": float(new_test),
        "n_new_feats": int(getattr(gen, "n_added_feats", 0)),
        "gens_completed": gens, "wall_time_s": round(wall, 1),
        "evals": evals, "evals_per_min": round(evals / max(wall / 60.0, 1e-9), 1),
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        name = sys.argv[1] if len(sys.argv) > 1 else "?"
        seed = sys.argv[2] if len(sys.argv) > 2 else "?"
        print("GITOLD_RESULT " + json.dumps({
            "dataset": name, "seed": int(seed) if str(seed).isdigit() else seed,
            "config": "gitold", "error": f"{type(e).__name__}: {e}"}))
