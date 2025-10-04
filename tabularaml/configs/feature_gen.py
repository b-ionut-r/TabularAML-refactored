PRESET_PARAMS = {
    "lite": {
        # Faster exploration within tight budget
        "n_generations": 12, 
        "n_parents": 15,
        "n_children": 90,  
        "early_stopping_child_eval": 50,  
        "early_stopping_iter": 5,  
        "min_pct_gain": 0.003, 
        "max_gen_new_feats_pct": 0.6, 
        "time_budget": 5 * 60,
        "cv": 4,
        "ranking_method": "multi_criteria"
    },
    "medium": {
        # Better exploration/exploitation balance
        "n_generations": 25,
        "n_parents": 25,
        "n_children": 150, 
        "early_stopping_child_eval": 60, 
        "early_stopping_iter": 6,
        "min_pct_gain": 0.002,
        "max_gen_new_feats_pct": 1.2, 
        "time_budget": 15 * 60,
        "cv": 4,
        "ranking_method": "multi_criteria"
    },
    "best": {
        # Deeper search with controlled patience
        "n_generations": 45, 
        "n_parents": 40,
        "n_children": 240, 
        "early_stopping_child_eval": 80,  
        "early_stopping_iter": 8, 
        "min_pct_gain": 0.0015, 
        "max_gen_new_feats_pct": 2.0,
        "time_budget": 60 * 60,
        "cv": 5,
        "ranking_method": "multi_criteria"
    },
    "extreme": {
        # Maximum thoroughness with efficiency guardrails
        "n_generations": 80, 
        "n_parents": 60,  
        "n_children": 360,
        "early_stopping_child_eval": 120, 
        "early_stopping_iter": 12,
        "min_pct_gain": 0.001, 
        "max_gen_new_feats_pct": 3.0,
        "time_budget": 240 * 60,
        "cv": 5,
        "ranking_method": "multi_criteria"
    }
}