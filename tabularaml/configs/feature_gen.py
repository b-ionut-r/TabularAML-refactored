PRESET_PARAMS = {
    "lite": {
        # Faster exploration within tight budget
        "n_generations": 10, 
        "n_parents": 12,
        "n_children": 72,  
        "early_stopping_child_eval": 40,  
        "early_stopping_iter": 4,  
        "min_pct_gain": 0.004, 
        "max_gen_new_feats_pct": 0.5, 
        "time_budget": 5 * 60,
        "cv": 4,
        "ranking_method": "multi_criteria"
    },
    "medium": {
        # Better exploration/exploitation balance
        "n_generations": 18,
        "n_parents": 20,
        "n_children": 120, 
        "early_stopping_child_eval": 40, 
        "early_stopping_iter": 4,
        "min_pct_gain": 0.004,
        "max_gen_new_feats_pct": 1.0, 
        "time_budget": 15 * 60,
        "cv": 4,
        "ranking_method": "multi_criteria"
    },
    "best": {
        # Deeper search with controlled patience
        "n_generations": 36, 
        "n_parents": 35,
        "n_children": 210, 
        "early_stopping_child_eval": 60,  
        "early_stopping_iter": 6, 
        "min_pct_gain": 0.003, 
        "max_gen_new_feats_pct": 1.5,
        "time_budget": 60 * 60,
        "cv": 5,
        "ranking_method": "multi_criteria"
    },
    "extreme": {
        # Maximum thoroughness with efficiency guardrails
        "n_generations": 60, 
        "n_parents": 50,  
        "n_children": 300,
        "early_stopping_child_eval": 100, 
        "early_stopping_iter": 8, 
        "min_pct_gain": 0.0015, 
        "max_gen_new_feats_pct": 2.5,
        "time_budget": 240 * 60,
        "cv": 5,
        "ranking_method": "multi_criteria"
    }
}