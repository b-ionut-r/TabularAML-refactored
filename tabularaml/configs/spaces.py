# Static configs
DEFAULT_CONFIG = {
    "XGB": {
        "learning_rate":     (0.01, 0.2, "float_log"),
        "n_estimators":      (100, 1000, "int"),
        "max_depth":         (3, 10, "int"),
        "min_child_weight":  (1, 10, "int"),
        "subsample":         (0.6, 0.9, "float"),
        "colsample_bytree":  (0.6, 0.9, "float"),
        "gamma":             (0.0, 5.0, "float"),
        "reg_alpha":         (0.0, 1.0, "float_log"),
        "reg_lambda":        (1.0, 10.0, "float_log"),
        "scale_pos_weight":  (1.0, 10.0, "float_log"),
        "verbosity": 0,
        "_priority": 120
    },
    "LGB": {
        "learning_rate":      (0.01, 0.2, "float_log"),
        "n_estimators":       (100, 1000, "int"),
        "max_depth":          (3, 15, "int"),
        "num_leaves":         (20, 100, "int"),
        "feature_fraction":   (0.6, 1.0, "float"),
        "bagging_fraction":   (0.6, 1.0, "float"),
        "lambda_l1":          (0.0, 1.0, "float_log"),
        "lambda_l2":          (0.0, 1.0, "float_log"),
        "min_child_samples":  (5, 50, "int"),
        "boost_from_average": ([True, False], "cat"),
        "verbosity": -1,
        "_priority": 110
    },
    "CAT": {
        "learning_rate":      (0.01, 0.3, "float_log"),
        "n_estimators":       (100, 1000, "int"),
        "max_depth":          (4, 10, "int"),
        "l2_leaf_reg":        (1.0, 10.0, "float_log"),
        "random_strength":    (1.0, 10.0, "float_log"),
        "bagging_temperature":(0.0, 1.0, "float"),
        "border_count":       (32, 255, "int"),
        "verbose": False,
        "_priority": 100
    },
    "RF": {
        "n_estimators":      (100, 500, "int"),
        "max_depth":         (5, 30, "int"),
        "min_samples_split": (2, 20, "int"),
        "min_samples_leaf":  (1, 10, "int"),
        "max_features":      (0.3, 0.8, "float"),
        "_priority": 70
    },
    "SGD_LINEAR": {
        "alpha":             (1e-5, 1e-1, "float_log"),
        "penalty":           (["l2","l1","elasticnet"], "cat"),
        "l1_ratio":          (0.0, 1.0, "float"),
        "learning_rate":     (["optimal","invscaling","adaptive"], "cat"),
        "eta0":              (1e-4, 1e-1, "float_log"),
        "power_t":           (0.1, 0.9, "float"),
        "max_iter":          (1000, 5000, "int"),
        "tol":               (1e-5, 1e-3, "float_log"),
        "_priority": 50
    }
}

LIGHT_CONFIG = {
    "XGB": {
        "learning_rate":     (0.01, 0.1, "float_log"),
        "n_estimators":      (100, 500, "int"),
        "max_depth":         (3, 8, "int"),
        "min_child_weight":  (1, 8, "int"),
        "subsample":         (0.6, 0.9, "float"),
        "colsample_bytree":  (0.6, 0.9, "float"),
        "gamma":             (0.0, 3.0, "float"),
        "reg_alpha":         (0.0, 1.0, "float_log"),
        "reg_lambda":        (1.0, 5.0, "float_log"),
        "scale_pos_weight":  (1.0, 10.0, "float_log"),
        "verbosity": 0,
        "_priority": 130
    },
    "LGB": {
        "learning_rate":      (0.01, 0.1, "float_log"),
        "n_estimators":       (100, 700, "int"),
        "max_depth":          (3, 12, "int"),
        "num_leaves":         (20, 80, "int"),
        "feature_fraction":   (0.6, 1.0, "float"),
        "bagging_fraction":   (0.6, 1.0, "float"),
        "lambda_l1":          (0.0, 1.0, "float_log"),
        "lambda_l2":          (0.0, 1.0, "float_log"),
        "min_child_samples":  (5, 50, "int"),
        "boost_from_average": ([True, False], "cat"),
        "verbosity": -1,
        "_priority": 120
    },
    "CAT": {
        "learning_rate":      (0.01, 0.2, "float_log"),
        "n_estimators":       (100, 500, "int"),
        "max_depth":          (4, 8, "int"),
        "l2_leaf_reg":        (1.0, 5.0, "float_log"),
        "random_strength":    (1.0, 5.0, "float_log"),
        "bagging_temperature":(0.0, 1.0, "float"),
        "border_count":       (32, 255, "int"),
        "verbose": False,
        "_priority": 110
    },
    "RF": {
        "n_estimators":      (100, 300, "int"),
        "max_depth":         (5, 20, "int"),
        "min_samples_split": (2, 10, "int"),
        "min_samples_leaf":  (1, 10, "int"),
        "max_features":      (0.3, 0.8, "float"),
        "_priority": 60
    },
    "SGD_LINEAR": {
        "alpha":             (1e-5, 1e-2, "float_log"),
        "penalty":           (["l2","l1","elasticnet"], "cat"),
        "l1_ratio":          (0.0, 1.0, "float"),
        "learning_rate":     (["optimal","invscaling"], "cat"),
        "eta0":              (1e-4, 1e-2, "float_log"),
        "power_t":           (0.1, 0.9, "float"),
        "max_iter":          (1000, 3000, "int"),
        "tol":               (1e-5, 1e-3, "float_log"),
        "_priority": 40
    }
}

INTENSE_CONFIG = {
    "XGB": {
        "learning_rate":     (0.001, 0.3, "float_log"),
        "n_estimators":      (200, 2000, "int"),
        "max_depth":         (3, 15, "int"),
        "min_child_weight":  (1, 20, "int"),
        "subsample":         (0.4, 1.0, "float"),
        "colsample_bytree":  (0.4, 1.0, "float"),
        "gamma":             (0.0, 10.0, "float"),
        "reg_alpha":         (0.0, 2.0, "float_log"),
        "reg_lambda":        (0.1, 20.0, "float_log"),
        "scale_pos_weight":  (0.5, 100.0, "float_log"),
        "verbosity": 0,
        "_priority": 140
    },
    "LGB": {
        "learning_rate":      (0.005, 0.2, "float_log"),
        "n_estimators":       (200, 1500, "int"),
        "max_depth":          (3, 20, "int"),
        "num_leaves":         (20, 300, "int"),
        "feature_fraction":   (0.4, 1.0, "float"),
        "bagging_fraction":   (0.4, 1.0, "float"),
        "lambda_l1":          (0.0, 2.0, "float_log"),
        "lambda_l2":          (0.0, 2.0, "float_log"),
        "min_child_samples":  (1, 100, "int"),
        "boost_from_average": ([True, False], "cat"),
        "verbosity": -1,
        "_priority": 135
    },
    "CAT": {
        "learning_rate":      (0.005, 0.3, "float_log"),
        "n_estimators":       (200, 1500, "int"),
        "max_depth":         (4, 12, "int"),
        "l2_leaf_reg":       (1.0, 20.0, "float_log"),
        "random_strength":   (1.0, 20.0, "float_log"),
        "bagging_temperature":(0.0, 2.0, "float"),
        "border_count":      (16, 512, "int"),
        "verbose": False,
        "_priority": 130
    },
    "RF": {
        "n_estimators":      (200, 1000, "int"),
        "max_depth":         (5, 50, "int"),
        "min_samples_split": (2, 50, "int"),
        "min_samples_leaf":  (1, 50, "int"),
        "max_features":      (0.2, 1.0, "float"),
        "_priority": 80
    },
    "SGD_LINEAR": {
        "alpha":             (1e-7, 1.0, "float_log"),
        "penalty":           (["l2","l1","elasticnet"], "cat"),
        "l1_ratio":          (0.0, 1.0, "float"),
        "learning_rate":     (["constant","optimal","invscaling","adaptive"], "cat"),
        "eta0":              (1e-5, 1.0, "float_log"),
        "power_t":           (0.01, 1.0, "float"),
        "max_iter":          (2000, 10000, "int"),
        "tol":               (1e-7, 1e-3, "float_log"),
        "_priority": 60
    }
}





