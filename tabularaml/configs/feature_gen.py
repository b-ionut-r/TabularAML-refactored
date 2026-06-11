_COMMON_COMPETITION_PARAMS = {
    # Shared preset upgrades validated by the A/B benchmark
    # (reports/acceptance_ab.md): two-stage batched proxy (2.24x candidate
    # throughput), parallel CV folds, gated base-table expansion. The strict
    # acceptance stack (statistical gate / confirmation / null importance)
    # did not prove paired test-gain superiority at small budgets, so presets
    # inherit the legacy acceptance defaults; opt in explicitly via
    # acceptance="statistical", confirmation_seeds>=1,
    # null_importance_selection=True (see docs/competition_mode.md).
    "proxy_mode": "batched",
    "cv_n_jobs": "auto",
    "expand_datetime": True,
    "expand_row_stats": True,
}

PRESET_PARAMS = {
    "lite": {
        # Faster exploration within tight budget
        "n_generations": 12,
        "n_parents": 15,
        "n_children": 160,
        "early_stopping_child_eval": 30,
        "early_stopping_iter": 5,
        "min_pct_gain": 0.003,
        "max_gen_new_feats_pct": 0.6,
        "time_budget": 5 * 60, # 5 minutes
        "cv": 4,
        "ranking_method": "multi_criteria",
        "search_sample_size": 10_000,
        "cache_size_mb": 500,
        "use_proxy_evaluation": True,
        "proxy_top_pct": 0.20,
        "meta_validation_frac": 0.15,
        "rotate_cv_folds": True,
        "fold_rotation_period": 4,
        **_COMMON_COMPETITION_PARAMS,
    },
    "medium": {
        # Better exploration/exploitation balance
        "n_generations": 25,
        "n_parents": 25,
        "n_children": 260,
        "early_stopping_child_eval": 45,
        "early_stopping_iter": 6,
        "min_pct_gain": 0.002,
        "max_gen_new_feats_pct": 1.2,
        "time_budget": 15 * 60, # 15 minutes
        "cv": 4,
        "ranking_method": "multi_criteria",
        "search_sample_size": 10_000,
        "cache_size_mb": 1000,
        "use_proxy_evaluation": True,
        "proxy_top_pct": 0.15,
        "meta_validation_frac": 0.15,
        "rotate_cv_folds": True,
        "fold_rotation_period": 5,
        **_COMMON_COMPETITION_PARAMS,
    },
    "best": {
        # Deeper search with controlled patience
        "n_generations": 45,
        "n_parents": 40,
        "n_children": 420,
        "early_stopping_child_eval": 70,
        "early_stopping_iter": 8,
        "min_pct_gain": 0.0015,
        "max_gen_new_feats_pct": 2.0,
        "time_budget": 60 * 60 * 1, # 1 hours
        "cv": 5,
        "ranking_method": "multi_criteria",
        "search_sample_size": 15_000,
        "cache_size_mb": 2000,
        "use_proxy_evaluation": True,
        "proxy_top_pct": 0.12,
        "meta_validation_frac": 0.15,
        "rotate_cv_folds": True,
        "fold_rotation_period": 5,
        **_COMMON_COMPETITION_PARAMS,
    },
    "extreme": {
        # Maximum thoroughness with efficiency guardrails
        "n_generations": 80,
        "n_parents": 60,
        "n_children": 640,
        "early_stopping_child_eval": 100,
        "early_stopping_iter": 12,
        "min_pct_gain": 0.001,
        "max_gen_new_feats_pct": 3.0,
        "time_budget": 60 * 60 * 4, # 4 hours
        "cv": 5,
        "ranking_method": "multi_criteria",
        "search_sample_size": 15_000,
        "cache_size_mb": 2000,
        "use_proxy_evaluation": True,
        "proxy_top_pct": 0.10,
        "meta_validation_frac": 0.15,
        "rotate_cv_folds": True,
        "fold_rotation_period": 5,
        **_COMMON_COMPETITION_PARAMS,
    }
}

# Pre-overhaul preset numbers, kept for exact A/B comparison runs.
LEGACY_PRESET_OVERRIDES = {
    "lite": {"n_children": 90, "early_stopping_child_eval": 25, "min_pct_gain": 0.003},
    "medium": {"n_children": 150, "early_stopping_child_eval": 35, "min_pct_gain": 0.002},
    "best": {"n_children": 240, "early_stopping_child_eval": 55, "min_pct_gain": 0.0015},
    "extreme": {"n_children": 360, "early_stopping_child_eval": 80, "min_pct_gain": 0.001},
}
