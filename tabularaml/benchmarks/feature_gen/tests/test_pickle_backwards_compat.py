"""Old saved generators (without any of the new attributes) must load and run."""
import numpy as np
import pandas as pd
import pytest

from tabularaml.generate.features import FeatureGenerator, Feature, Interaction

NEW_ATTRS = [
    "acceptance", "acceptance_folds_frac", "confirmation_seeds",
    "null_importance_selection", "null_importance_n_perm", "null_importance_pct",
    "cv_n_jobs", "_cv_n_jobs_resolved", "_cv_epoch", "_best_fold_state",
    "_cv_int_hint", "proxy_mode", "proxy_ram_budget_mb", "proxy_halving",
    "expand_datetime", "expand_row_stats", "base_expander", "_priority_candidates",
    "era_col", "era_acceptance_frac",
    "adversarial_auc_warn", "adversarial_drop", "adversarial_report",
]


def _trained_micro_gen(tmp_path):
    rng = np.random.default_rng(0)
    n = 250
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                      "c": rng.normal(size=n)})
    y = pd.Series(X["a"] * X["b"] + rng.normal(size=n) * 0.1)
    gen = FeatureGenerator(n_generations=2, n_parents=5, n_children=16, cv=3,
                           log_file=None, use_proxy_evaluation=False,
                           final_selection=False, meta_validation_frac=0.0,
                           expand_datetime=False, expand_row_stats=False,
                           early_stopping_iter=10, time_budget=90)
    gen.search(X, y)
    return gen, X, y


def test_attr_stripped_state_loads_and_transforms(tmp_path):
    gen, X, y = _trained_micro_gen(tmp_path)

    # Simulate a pre-overhaul pickle: remove every new attribute
    for attr in NEW_ATTRS:
        gen.__dict__.pop(attr, None)
    for key in ("val_fold_scores", "fold_cv_epoch", "val_fold_per_era"):
        gen.state["best"].pop(key, None)

    path = str(tmp_path / "old_style.pkl")
    gen.save(path)
    loaded = FeatureGenerator.load(path)

    # _ensure_backwards_compat must have restored sane defaults
    assert loaded.acceptance == "statistical"
    assert loaded.confirmation_seeds == 1
    assert loaded.proxy_mode in ("batched", "none")
    assert loaded.base_expander is None
    assert loaded.state["best"]["val_fold_scores"] is None

    out = loaded.transform(X.iloc[:40].copy())
    assert len(out) == 40


def test_attr_stripped_interaction_tolerated(toy_reg):
    X, y = toy_reg
    gen = FeatureGenerator(log_file=None)
    gen._set_defaults(X, y)
    inter = Interaction(Feature("num_a", "num", 1.0), "square")
    for attr in ("is_global", "is_temporal"):
        inter.__dict__.pop(attr, None)
    pipe = gen._prepare_pipeline([inter])
    assert pipe.global_encoders == [] and pipe.temporal_encoders == []
    gen.adaptive_controller.update_operation_stats(inter, success=False, gain=-0.1)
