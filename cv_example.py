#!/usr/bin/env python3
"""
Modified for competition metric: exp(-RMSE / 100)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import optuna
from tqdm import tqdm

# Import your custom modules
from tabularaml.generate.features import FeatureGenerator


def competition_metric(rmse: float) -> float:
    """Competition metric: exp(-RMSE / 100)"""
    return np.exp(-rmse / 100)


def load_competition_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    feature_cols = ['latitude', 'longitude', 'day_of_year', 'day_of_week', 'hour', 'month']
    X_train = train_df[feature_cols]
    y_train = train_df['pollution_value']
    X_test = test_df[feature_cols]
    test_ids = test_df['id']

    return X_train, y_train, X_test, test_ids


def demonstrate_cv_splitter(X_train, y_train):
    print("=== Spatial-Temporal CV Splitter Demo ===\n")
    cv = SpatialTemporalKFold()

    split_info = cv.get_split_info(X_train, y_train)
    print("CV Splitter Information:")
    for key, value in split_info.items():
        print(f"  {key}: {value}")
    print()

    print("Fold Statistics:")
    fold_stats = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        train_lat_range = X_fold_train['latitude'].max() - X_fold_train['latitude'].min()
        val_lat_range = X_fold_val['latitude'].max() - X_fold_val['latitude'].min()
        train_time_range = X_fold_train['day_of_year'].max() - X_fold_train['day_of_year'].min()
        val_time_range = X_fold_val['day_of_year'].max() - X_fold_val['day_of_year'].min()

        train_mean, val_mean = y_fold_train.mean(), y_fold_val.mean()
        train_std, val_std = y_fold_train.std(), y_fold_val.std()

        stats = {
            'fold': fold_idx + 1,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'train_lat_range': train_lat_range,
            'val_lat_range': val_lat_range,
            'train_time_range': train_time_range,
            'val_time_range': val_time_range,
            'train_pollution_mean': train_mean,
            'val_pollution_mean': val_mean,
            'train_pollution_std': train_std,
            'val_pollution_std': val_std,
            'mean_diff': abs(train_mean - val_mean),
            'std_diff': abs(train_std - val_std)
        }
        fold_stats.append(stats)

        print(f"  Fold {fold_idx+1}: Train={len(train_idx)}, Val={len(val_idx)}, "
              f"Mean diff={abs(train_mean - val_mean):.3f}, "
              f"Spatial separation: {train_lat_range:.3f}/{val_lat_range:.3f}")

    return cv, fold_stats


def hyperparameter_optimization(X_enhanced, y_train, cv_splitter):
    print("\n=== Hyperparameter Optimization ===\n")

    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
            'max_depth': trial.suggest_int('max_depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'random_state': 42,
            'n_jobs': -1
        }

        cv_scores = []
        for train_idx, val_idx in cv_splitter.split(X_enhanced, y_train):
            X_fold_train, X_fold_val = X_enhanced.iloc[train_idx], X_enhanced.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = XGBRegressor(**params)
            model.fit(X_fold_train, y_fold_train)

            y_pred = model.predict(X_fold_val)
            rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
            cv_scores.append(rmse)

        mean_rmse = np.mean(cv_scores)
        score = competition_metric(mean_rmse)
        print(f"Trial RMSE: {mean_rmse:.4f} → Score: {score:.6f}")
        return mean_rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, timeout=3600)

    print(f"\nBest RMSE: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_params}")
    print(f"Best Competition Score: {competition_metric(study.best_value):.6f}")

    return study.best_params


def full_pipeline_example():
    print("=== Complete Competition Pipeline ===\n")

    X_train, y_train, X_test, test_ids = load_competition_data()
    feature_gen = FeatureGenerator()
    X_enhanced = feature_gen.fit_transform(X_train)

    # 1. CV Split
    cv_splitter, fold_stats = demonstrate_cv_splitter(X_train, y_train)

    # 2. Hyperparameter Tuning
    best_params = hyperparameter_optimization(X_enhanced, y_train, cv_splitter)

    # 3. Final CV Evaluation
    print("\n=== Final Model Evaluation ===\n")
    final_cv_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_enhanced, y_train)):
        X_fold_train, X_fold_val = X_enhanced.iloc[train_idx], X_enhanced.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        final_model = XGBRegressor(**best_params)
        final_model.fit(X_fold_train, y_fold_train)

        y_pred = final_model.predict(X_fold_val)
        rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
        score = competition_metric(rmse)
        final_cv_scores.append(rmse)

        print(f"  Fold {fold_idx+1}: RMSE = {rmse:.4f}, Score = {score:.6f}")

    mean_rmse = np.mean(final_cv_scores)
    std_rmse = np.std(final_cv_scores)
    final_score = competition_metric(mean_rmse)

    print(f"\nFinal CV Results:")
    print(f"  Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    print(f"  Competition Score: {final_score:.6f}")

    # 4. Train final model and predict
    print("\n=== Generating Final Predictions ===\n")
    final_model = XGBRegressor(**best_params)
    final_model.fit(X_enhanced, y_train)

    X_test_enhanced = feature_gen.transform(X_test)
    test_predictions = final_model.predict(X_test_enhanced)

    submission = pd.DataFrame({
        'id': test_ids,
        'pollution_value': test_predictions
    })

    filename = f"submission_{final_score:.6f}.csv"
    submission.to_csv(filename, index=False)

    print(f"Submission saved to '{filename}'")
    print(f"Test predictions range: {test_predictions.min():.2f} to {test_predictions.max():.2f}")

    return submission, final_model, feature_gen


if __name__ == "__main__":
    try:
        submission, model, feature_gen = full_pipeline_example()
        print("\n✅ Pipeline completed successfully!")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        print("\nRunning CV demonstration only...")

        X_train, y_train, *_ = load_competition_data()
        cv_splitter, fold_stats = demonstrate_cv_splitter(X_train, y_train)

        print("\nDetailed Fold Statistics:")
        df_stats = pd.DataFrame(fold_stats)
        print(df_stats.round(4))