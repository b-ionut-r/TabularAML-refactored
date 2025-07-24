import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import optuna
from tabularaml.generate.features import FeatureGenerator
from tabularaml.eval.scorers import Scorer
from spatial_temporal_cv import SpatialTemporalKFold
import gc
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# KEEP THE WINNING FORMULA - BUILD GRADUALLY
# ============================================================================

def rmse_exp(y_true, y_pred):
    """Competition metric"""
    return np.exp(-np.sqrt(mean_squared_error(y_true, y_pred))/100)

def create_cyclical_features(df):
    """Your exact cyclical features that worked"""
    df_copy = df.copy()
    
    # Cyclical Features
    if 'hour' in df_copy.columns:
        df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy['hour'] / 24.0)
        df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy['hour'] / 24.0)
    
    if 'day_of_week' in df_copy.columns:
        df_copy['dow_sin'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7.0)
        df_copy['dow_cos'] = np.cos(2 * np.pi * df_copy['day_of_week'] / 7.0)
    
    if 'day_of_year' in df_copy.columns:
        df_copy['doy_sin'] = np.sin(2 * np.pi * df_copy['day_of_year'] / 365.0)
        df_copy['doy_cos'] = np.cos(2 * np.pi * df_copy['day_of_year'] / 365.0)
    
    # Drop original columns
    columns_to_drop = ['hour', 'day_of_week', 'day_of_year', 'month']
    existing_columns_to_drop = [col for col in columns_to_drop if col in df_copy.columns]
    if existing_columns_to_drop:
        df_copy = df_copy.drop(columns=existing_columns_to_drop)
    
    return df_copy

def competition_score(y_true, y_pred):
    """Calculate competition score for validation"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return np.exp(-rmse / 100)

# ============================================================================
# ENHANCED FEATURE GENERATION - BUILDING ON YOUR SUCCESS
# ============================================================================

def multi_stage_feature_generation(X_train, y_train, save_dir="./features"):
    """
    Enhanced version of your feature generation approach
    Key improvements:
    1. Run longer (more generations)
    2. Two-stage approach
    3. Keep using SpatialTemporalKFold as you did
    """
    
    # Use your exact CV splitter for feature generation
    splitter = SpatialTemporalKFold(
        n_splits=5,
        spatial_clusters=30,  # You might have used different values
        temporal_clusters=10,
        random_state=42
    )
    
    # Your scorer
    rmse_exp_scorer = Scorer(
        name="rmse_exp",
        scorer=rmse_exp,
        greater_is_better=True,
        extra_params={},
        from_probs=False
    )
    
    print("Stage 1: Initial feature generation (matching your approach)...")
    
    # Stage 1 - Match your settings but run longer
    generator1 = FeatureGenerator(
        task="regression",
        scorer=rmse_exp_scorer,
        max_new_feats=2000,  # Increased from 1000
        cv=splitter,
        n_generations=2000,  # Increased from 1000
        save_path=f"{save_dir}/feature_generator_stage1.pkl",
    )
    
    results1 = generator1.search(X_train, y_train)
    X_stage1 = generator1.transform(X_train)
    
    print(f"Stage 1 complete: {X_stage1.shape[1]} features generated")
    
    # Stage 2 - Build on top of best features
    print("\nStage 2: Second pass on best features...")
    
    # Select top features based on importance
    feature_importance = get_feature_importance(X_stage1, y_train)
    top_features_idx = np.argsort(feature_importance)[-500:]  # Top 500 features
    X_selected = X_stage1.iloc[:, top_features_idx]
    
    generator2 = FeatureGenerator(
        task="regression",
        scorer=rmse_exp_scorer,
        max_new_feats=500,
        cv=splitter,
        n_generations=1000,
        save_path=f"{save_dir}/feature_generator_stage2.pkl",
    )
    
    results2 = generator2.search(X_selected, y_train)
    X_stage2 = generator2.transform(X_selected)
    
    # Combine both stages
    X_final = pd.concat([X_stage1, X_stage2], axis=1)
    
    # Remove duplicate columns if any
    X_final = X_final.loc[:, ~X_final.columns.duplicated()]
    
    print(f"Stage 2 complete: {X_final.shape[1]} total features")
    
    return X_final, generator1, generator2

def get_feature_importance(X, y):
    """Quick feature importance using LightGBM"""
    model = lgb.LGBMRegressor(
        n_estimators=100,
        num_leaves=31,
        learning_rate=0.1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        random_state=42,
        verbose=-1
    )
    model.fit(X, np.log1p(y))
    return model.feature_importances_

# ============================================================================
# ENHANCED HYPERPARAMETER OPTIMIZATION - MATCHING YOUR APPROACH
# ============================================================================

def enhanced_objective(trial, X_base, y_train, tss, n_clusters_range=(20, 50)):
    """
    Enhanced version of your objective function
    Key improvements:
    1. Wider hyperparameter ranges
    2. More sophisticated clustering
    3. Additional regularization options
    """
    
    # Extended hyperparameter search space
    params = {
        # Learning parameters
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
        
        # Tree parameters
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 30),
        'gamma': trial.suggest_float('gamma', 0, 20),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.3, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.3, 1.0),
        
        # Regularization parameters
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100.0, log=True),
        
        # Additional parameters
        'max_leaves': trial.suggest_int('max_leaves', 0, 2000),
        'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        'max_bin': trial.suggest_int('max_bin', 32, 512),
        
        # Fixed parameters
        'objective': 'reg:squarederror',
        'enable_categorical': True,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Clustering hyperparameter - expanded range
    n_clusters = trial.suggest_int('n_clusters', n_clusters_range[0], n_clusters_range[1])
    
    # Use your exact approach with FixedWindowTimeSeriesSplit
    X_base_cyclic = create_cyclical_features(X_base.copy())
    y = np.log1p(y_train.copy())  # Log transform as you did
    
    fold_scores = []
    
    for n_fold, (train_idx, valid_idx) in enumerate(tss.split(X_base_cyclic, y)):
        X_train_fold = X_base_cyclic.iloc[train_idx].copy()
        X_valid_fold = X_base_cyclic.iloc[valid_idx].copy()
        y_train_fold = y.iloc[train_idx]
        y_valid_fold = y.iloc[valid_idx]
        
        # Your exact clustering approach
        if 'latitude' in X_train_fold.columns and 'longitude' in X_train_fold.columns:
            lat_mean = X_train_fold['latitude'].mean()
            lon_mean = X_train_fold['longitude'].mean()
            
            train_coords_temp = X_train_fold[['latitude', 'longitude']].copy()
            valid_coords_temp = X_valid_fold[['latitude', 'longitude']].copy()
            
            train_coords_temp['latitude'].fillna(lat_mean, inplace=True)
            train_coords_temp['longitude'].fillna(lon_mean, inplace=True)
            valid_coords_temp['latitude'].fillna(lat_mean, inplace=True)
            valid_coords_temp['longitude'].fillna(lon_mean, inplace=True)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            kmeans.fit(train_coords_temp)
            
            X_train_fold['cluster'] = kmeans.predict(train_coords_temp)
            X_valid_fold['cluster'] = kmeans.predict(valid_coords_temp)
        
        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_valid_fold, y_valid_fold)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Predict and evaluate
        y_pred_fold = model.predict(X_valid_fold)
        y_pred_orig_scale = np.expm1(y_pred_fold)
        y_valid_orig_scale = np.expm1(y_valid_fold)
        
        exp_score = competition_score(y_valid_orig_scale, y_pred_orig_scale)
        fold_scores.append(exp_score)
    
    return np.mean(fold_scores)

# ============================================================================
# GRADUAL IMPROVEMENTS TO YOUR WINNING APPROACH
# ============================================================================

def create_enhanced_features(X_train, y_train, X_test):
    """
    Gradual enhancements to your feature engineering
    """
    # 1. Start with your exact approach
    X_train_enhanced = X_train.copy()
    X_test_enhanced = X_test.copy()
    
    # 2. Add a few careful enhancements that shouldn't hurt
    
    # Spatial density features (pollution often correlates with density)
    from sklearn.neighbors import KernelDensity
    
    coords_train = X_train[['latitude', 'longitude']].fillna(
        X_train[['latitude', 'longitude']].mean()
    )
    coords_test = X_test[['latitude', 'longitude']].fillna(
        X_train[['latitude', 'longitude']].mean()
    )
    
    kde = KernelDensity(bandwidth=2.0, kernel='gaussian')
    kde.fit(coords_train)
    
    X_train_enhanced['spatial_density'] = np.exp(kde.score_samples(coords_train))
    X_test_enhanced['spatial_density'] = np.exp(kde.score_samples(coords_test))
    
    # Hour-based features (building on your success with temporal features)
    X_train_enhanced['is_rush_hour'] = X_train['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    X_test_enhanced['is_rush_hour'] = X_test['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    
    X_train_enhanced['is_night'] = ((X_train['hour'] >= 22) | (X_train['hour'] <= 5)).astype(int)
    X_test_enhanced['is_night'] = ((X_test['hour'] >= 22) | (X_test['hour'] <= 5)).astype(int)
    
    # Interaction between location and time (pollution patterns vary by location and time)
    X_train_enhanced['lat_hour'] = X_train['latitude'] * X_train['hour']
    X_test_enhanced['lat_hour'] = X_test['latitude'] * X_test['hour']
    
    return X_train_enhanced, X_test_enhanced

def create_model_ensemble(X_train, y_train, X_test, best_params, tss):
    """
    Create a simple ensemble building on your XGBoost success
    """
    predictions = []
    
    # 1. Your winning XGBoost with best params
    print("Training XGBoost with your best parameters...")
    xgb_model = train_final_model(X_train, y_train, best_params, model_type='xgb')
    pred_xgb = xgb_model.predict(X_test)
    predictions.append(('xgb_best', pred_xgb, 0.7))  # High weight for proven winner
    
    # 2. LightGBM as a complement
    print("Training LightGBM...")
    lgb_params = convert_to_lgb_params(best_params)
    lgb_model = train_final_model(X_train, y_train, lgb_params, model_type='lgb')
    pred_lgb = lgb_model.predict(X_test)
    predictions.append(('lgb', pred_lgb, 0.2))
    
    # 3. XGBoost with slightly different params for diversity
    print("Training XGBoost variant...")
    xgb_variant_params = best_params.copy()
    xgb_variant_params['max_depth'] = min(best_params['max_depth'] + 2, 15)
    xgb_variant_params['learning_rate'] = best_params['learning_rate'] * 0.8
    xgb_variant_model = train_final_model(X_train, y_train, xgb_variant_params, model_type='xgb')
    pred_xgb_variant = xgb_variant_model.predict(X_test)
    predictions.append(('xgb_variant', pred_xgb_variant, 0.1))
    
    # Weighted average
    final_pred = np.zeros(len(X_test))
    for name, pred, weight in predictions:
        final_pred += weight * pred
    
    return final_pred, predictions

def train_final_model(X_train, y_train, params, model_type='xgb'):
    """
    Train final model with your exact approach
    """
    # Apply cyclical features
    X_train_cyclic = create_cyclical_features(X_train.copy())
    
    # Apply clustering
    n_clusters = params.get('n_clusters', 30)
    if 'latitude' in X_train_cyclic.columns and 'longitude' in X_train_cyclic.columns:
        lat_mean = X_train_cyclic['latitude'].mean()
        lon_mean = X_train_cyclic['longitude'].mean()
        
        coords_temp = X_train_cyclic[['latitude', 'longitude']].copy()
        coords_temp['latitude'].fillna(lat_mean, inplace=True)
        coords_temp['longitude'].fillna(lon_mean, inplace=True)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        kmeans.fit(coords_temp)
        X_train_cyclic['cluster'] = kmeans.predict(coords_temp)
    
    # Log transform target
    y_train_log = np.log1p(y_train)
    
    # Train model
    if model_type == 'xgb':
        model_params = {k: v for k, v in params.items() if k != 'n_clusters'}
        model = xgb.XGBRegressor(**model_params)
        model.fit(X_train_cyclic, y_train_log)
    elif model_type == 'lgb':
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train_cyclic, y_train_log, verbose=-1)
    
    # Store the kmeans for test set transformation
    model.kmeans_ = kmeans if 'kmeans' in locals() else None
    model.lat_mean_ = lat_mean if 'lat_mean' in locals() else None
    model.lon_mean_ = lon_mean if 'lon_mean' in locals() else None
    
    return model

def convert_to_lgb_params(xgb_params):
    """Convert XGBoost params to LightGBM equivalents"""
    lgb_params = {
        'n_estimators': xgb_params.get('n_estimators', 1000),
        'learning_rate': xgb_params.get('learning_rate', 0.05),
        'num_leaves': 2 ** xgb_params.get('max_depth', 6) - 1,
        'feature_fraction': xgb_params.get('colsample_bytree', 0.8),
        'bagging_fraction': xgb_params.get('subsample', 0.8),
        'bagging_freq': 1,
        'lambda_l1': xgb_params.get('reg_alpha', 0),
        'lambda_l2': xgb_params.get('reg_lambda', 1),
        'min_data_in_leaf': int(xgb_params.get('min_child_weight', 1)),
        'random_state': 42,
        'verbose': -1
    }
    return lgb_params

# ============================================================================
# MAIN PIPELINE - MATCHING YOUR APPROACH
# ============================================================================

def enhanced_pipeline(train_df, test_df, target_col='pollution_value', 
                     n_trials=1000, n_generations_features=3000):
    """
    Enhanced version of your winning approach
    Gradual improvements while keeping the core strategy
    """
    print("="*60)
    print("ENHANCED PIPELINE - BUILDING ON YOUR SUCCESS")
    print("="*60)
    
    # 1. Setup
    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]
    X_test = test_df.copy()
    
    # Save IDs
    test_ids = test_df['id'].values
    
    # 2. Feature Engineering Stage 1 - Your approach with enhancements
    print("\nApplying enhanced features...")
    X_train_enhanced, X_test_enhanced = create_enhanced_features(X_train, y_train, X_test)
    
    # 3. Genetic Algorithm Feature Generation - Your winning approach
    print("\nRunning genetic algorithm feature generation...")
    print("Using SpatialTemporalKFold as you did...")
    
    X_train_generated, gen1, gen2 = multi_stage_feature_generation(
        X_train_enhanced, 
        y_train,
        save_dir="./enhanced_features"
    )
    
    # Transform test set
    X_test_stage1 = gen1.transform(X_test_enhanced)
    
    # For stage 2, we need to select the same top features
    feature_importance = get_feature_importance(
        gen1.transform(X_train_enhanced), 
        y_train
    )
    top_features_idx = np.argsort(feature_importance)[-500:]
    X_test_selected = X_test_stage1.iloc[:, top_features_idx]
    X_test_stage2 = gen2.transform(X_test_selected)
    
    X_test_generated = pd.concat([X_test_stage1, X_test_stage2], axis=1)
    X_test_generated = X_test_generated.loc[:, ~X_test_generated.columns.duplicated()]
    
    print(f"Total features after generation: {X_train_generated.shape[1]}")
    
    # 4. Hyperparameter Optimization - Your exact approach
    print("\nOptimizing hyperparameters with FixedWindowTimeSeriesSplit...")
    
    from spatial_temporal_cv import FixedWindowTimeSeriesSplit
    
    tss = FixedWindowTimeSeriesSplit(
        n_splits=5,
        test_size=2700,  # Adjust based on your data
        gap=0,
        min_train_size=2700
    )
    
    # Create study
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        group=True,
        n_startup_trials=20,
        constant_liar=True,
        seed=42,
    )
    
    study = optuna.create_study(
        direction="maximize",
        study_name="xgboost_optimization_enhanced",
        sampler=sampler,
        storage="sqlite:///xgb_optuna_enhanced.db",
        load_if_exists=True,
    )
    
    # Optimize
    objective_func = lambda trial: enhanced_objective(
        trial, X_train_generated, y_train, tss, n_clusters_range=(20, 50)
    )
    
    study.optimize(objective_func, n_trials=n_trials)
    
    best_params = study.best_params.copy()
    print(f"\nBest score: {study.best_value:.4f}")
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    # 5. Generate final predictions - Following your OOF approach
    print("\nGenerating final predictions...")
    
    # Note: With FixedWindowTimeSeriesSplit, we can't get full OOF predictions
    # So we'll use the last fold's validation as a proxy for model performance
    
    # Train ensemble
    final_predictions, model_predictions = create_model_ensemble(
        X_train_generated, 
        y_train, 
        X_test_generated, 
        best_params, 
        tss
    )
    
    # 6. Post-processing
    print("\nApplying post-processing...")
    
    # Ensure non-negative
    final_predictions = np.maximum(final_predictions, 0)
    
    # Analyze prediction distribution
    print(f"\nPrediction statistics:")
    print(f"  Mean: {final_predictions.mean():.4f}")
    print(f"  Std:  {final_predictions.std():.4f}")
    print(f"  Min:  {final_predictions.min():.4f}")
    print(f"  Max:  {final_predictions.max():.4f}")
    
    # Compare to training distribution
    print(f"\nTraining target statistics:")
    print(f"  Mean: {y_train.mean():.4f}")
    print(f"  Std:  {y_train.std():.4f}")
    
    # 7. Create submission
    submission = pd.DataFrame({
        'id': test_ids,
        'pollution_value': final_predictions
    })
    
    return submission, study, best_params

# ============================================================================
# ADDITIONAL IMPROVEMENTS TO CONSIDER
# ============================================================================

def advanced_post_processing(train_df, y_train, test_df, predictions):
    """
    Advanced post-processing based on error analysis
    To be used after initial predictions
    """
    # 1. Quantile mapping
    train_quantiles = np.percentile(y_train, [10, 25, 50, 75, 90])
    pred_quantiles = np.percentile(predictions, [10, 25, 50, 75, 90])
    
    # Simple linear mapping if distributions are very different
    if abs(train_quantiles[2] - pred_quantiles[2]) > 0.1 * train_quantiles[2]:
        scale = train_quantiles[2] / pred_quantiles[2]
        predictions_adjusted = predictions * scale
    else:
        predictions_adjusted = predictions
    
    # 2. Boundary corrections
    # If test set has extreme locations not seen in training
    test_lat_min, test_lat_max = test_df['latitude'].min(), test_df['latitude'].max()
    train_lat_min, train_lat_max = train_df['latitude'].min(), train_df['latitude'].max()
    
    if test_lat_min < train_lat_min or test_lat_max > train_lat_max:
        # Apply conservative predictions for extrapolation
        extreme_mask = (
            (test_df['latitude'] < train_lat_min) | 
            (test_df['latitude'] > train_lat_max)
        )
        if extreme_mask.any():
            # Pull extreme predictions toward mean
            mean_pred = predictions_adjusted[~extreme_mask].mean()
            predictions_adjusted[extreme_mask] = (
                0.7 * predictions_adjusted[extreme_mask] + 0.3 * mean_pred
            )
    
    return predictions_adjusted

# Usage example:
if __name__ == "__main__":
    # This would be called with your actual data
    # submission, study, best_params = enhanced_pipeline(
    #     train_df, 
    #     test_df,
    #     n_trials=1000,  # or more
    #     n_generations_features=3000  # or more
    # )
    # submission.to_csv('submission_enhanced.csv', index=False)
    pass