# Competition Winning Strategy: Air Pollution Prediction with Severe Distribution Shifts

## Deep Strategic Analysis

### The Core Challenge
Your competition has **extreme distribution shifts** that make it exceptionally difficult:
- **Geographic concentration**: Test data clustered in Europe/Asia vs global training
- **January scarcity**: 330 training vs 2700 test January samples (8x imbalance)
- **Temporal concentration**: Test focused on weekday midday hours vs distributed training
- **Day-of-year restriction**: Test limited to late January vs full-year training

### Strategic Foundation: Maximum Data Utilization
The key insight is that with only 330 January training samples, **every data point is precious**. Traditional approaches waste this limited data through:
- Poor CV strategies that don't mirror test distribution
- Feature engineering that doesn't transfer cross-seasonally
- Models that overfit to limited January patterns

## Phase 1: Automated Feature Engineering Foundation (Deep Dive)

### 1.1 Base Feature Engineering Strategy
```python
# Create January-optimized base
fe_base = AdvancedSpatioTemporalFeatures(
    row_only=True,  # Safe for AFE - no train/val leakage
    january_bridge_features=True,  # Critical for cross-seasonal learning
    test_distribution=temporal_stats,  # Explicit test distribution matching
    use_distribution_matching=True  # Create test probability features
)
```

**Why this base is powerful:**
- **December-January bridges**: Allows 600+ December samples to inform January predictions
- **Universal geographic features**: Work across any geographic region
- **Test distribution features**: Explicit features showing likelihood under test distribution
- **Cyclical encoding**: Mathematical relationships preserve seasonal continuity

### 1.2 AFE with Domain-Aware CV
The critical insight is using **SpatioTemporalCV** for AFE, not standard CV:

```python
cv_afe = SpatioTemporalCV(
    spatial_weight=0.3,  # Emphasize temporal matching for January scarcity
    test_spatial_coords=test_coords,
    test_temporal_features=test_temporal
)

fg = FeatureGenerator(
    cv=cv_afe,  # This is game-changing
    n_generations=20,  # Extended search for quality
    n_parents=60,      # More diversity with richer base
    ranking_method="multi_criteria",  # Best for complex interactions
    min_pct_gain=0.0005,  # Lower threshold for January patterns
    time_budget=5400   # 1.5 hours for thorough search
)
```

**Why AFE succeeds here:**
- **Validates on test-like samples**: AFE learns features that work on January-heavy validation
- **Discovers January interactions**: Automatically finds cross-seasonal patterns
- **Geographic generalization**: Tests features across test-like spatial distribution
- **Higher feature density**: Each January sample generates multiple interaction features

### 1.3 Expected AFE Discoveries
Based on your data characteristics, AFE should discover:
- **Geographic-temporal interactions**: Latitude × winter intensity features
- **Hour-month combinations**: January midday patterns specific to test distribution
- **Cross-seasonal bridges**: Features connecting December patterns to January
- **Day-of-week amplifiers**: Weekday patterns enhanced for January context

## Phase 2: Enhanced CV Pipeline with Full Feature Engineering (Deep Dive)

### 2.1 The Critical CV Enhancement
The breakthrough strategy is using **full feature engineering inside CV loops**:

```python
def enhanced_cv_loop(X_train, y_train, test_features):
    cv = SpatioTemporalCV(spatial_weight=0.3)
    oof_predictions = np.zeros(len(X_train))
    feature_importance_aggregated = {}
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
        print(f"Fold {fold+1}: Processing with full pipeline...")
        
        # Critical: Apply AFE to both train and validation
        X_train_fold = afe_pipeline.transform(X_train.iloc[train_idx])
        X_val_fold = afe_pipeline.transform(X_train.iloc[val_idx])
        y_train_fold = y_train.iloc[train_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        # Fit full spatio-temporal features on TRAINING fold only
        fe_full = AdvancedSpatioTemporalFeatures(
            row_only=False,  # Now we can use clustering!
            n_spatial_clusters=30,
            n_temporal_clusters=15,
            january_bridge_features=True,
            test_distribution=temporal_stats
        )
        
        # Fit on train fold, transform both train and val
        X_train_enhanced = fe_full.fit_transform(X_train_fold, y_train_fold)
        X_val_enhanced = fe_full.transform(X_val_fold)  # Uses fitted clusterers
        
        # Domain adaptation weights for this fold
        weights_train = sample_weights[train_idx] if sample_weights is not None else None
        
        # Handle NaNs using train fold statistics only
        train_means = X_train_enhanced.mean()
        X_train_enhanced.fillna(train_means, inplace=True)
        X_val_enhanced.fillna(train_means, inplace=True)  # No leakage
        
        # Scale using train fold statistics
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_enhanced)
        X_val_scaled = scaler.transform(X_val_enhanced)
        
        # Train model with early stopping on validation
        model = xgb.XGBRegressor(
            n_estimators=2000,
            learning_rate=0.05,
            early_stopping_rounds=50,
            random_state=42
        )
        
        model.fit(
            X_train_scaled, y_train_fold,
            sample_weight=weights_train,
            eval_set=[(X_val_scaled, y_val_fold)],
            verbose=False
        )
        
        # Generate OOF predictions
        oof_predictions[val_idx] = model.predict(X_val_scaled)
        
        # Aggregate feature importance for January analysis
        if hasattr(model, 'feature_importances_'):
            fold_importance = dict(zip(X_train_enhanced.columns, model.feature_importances_))
            for feat, imp in fold_importance.items():
                feature_importance_aggregated[feat] = feature_importance_aggregated.get(feat, 0) + imp
    
    return oof_predictions, feature_importance_aggregated
```

### 2.2 Why This CV Strategy Wins

**Prevents Data Leakage While Maximizing Information:**
- **Clustering fitted on train fold**: Spatial/temporal clusters learn only from training data
- **Pattern statistics from train fold**: Hour-month patterns avoid future leakage
- **Scaling from train fold**: No validation statistics contamination
- **But uses full feature power**: Unlike row_only=True, gets clustering benefits

**Realistic Validation:**
- **Each fold mimics test distribution**: Via SpatioTemporalCV
- **Each fold uses full pipeline**: Same feature engineering as final model
- **Each fold handles January scarcity**: Via domain adaptation weights
- **Each fold validates generalization**: Geographic and temporal robustness

### 2.3 OOF Predictions for Meta-Learning

```python
# Analyze OOF performance patterns
january_mask = X_train['month'] == 1
january_oof_rmse = np.sqrt(mean_squared_error(y_train[january_mask], oof_predictions[january_mask]))
overall_oof_rmse = np.sqrt(mean_squared_error(y_train, oof_predictions))

print(f"January OOF RMSE: {january_oof_rmse:.4f}")
print(f"Overall OOF RMSE: {overall_oof_rmse:.4f}")
print(f"January penalty: {january_oof_rmse/overall_oof_rmse:.2f}x")

# Use OOF for ensemble weight optimization
def optimize_ensemble_weights(oof_predictions_dict, y_true):
    # Optimize weights specifically for January performance
    january_weights = optimize_weights(oof_predictions_dict, y_true[january_mask])
    overall_weights = optimize_weights(oof_predictions_dict, y_true)
    
    # Blend weights based on January importance
    final_weights = 0.7 * january_weights + 0.3 * overall_weights
    return final_weights
```

## Phase 3: Advanced Hyperparameter Optimization (Deep Dive)

### 3.1 Enhanced Optuna Integration with Persistent Storage

```python
def hyperparameter_optimization(X_train, y_train, X_test, temporal_stats, 
                               model_dir="model_spatiotemporal", 
                               n_trials=100, timeout=7200, 
                               study_path="optuna_study_spatiotemporal.db"):
    """
    Advanced hyperparameter optimization with January-specific parameters.
    Features:
    - Persistent SQLite storage for resume capability
    - TPE sampler with multivariate optimization
    - Detailed trial logging with score components
    - Competition metric optimization: exp(-RMSE/100)
    """
    
    def objective(trial):
        # Expanded parameter ranges for thorough exploration
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 2500),  # Wider range
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),  # Log scale
            'max_depth': trial.suggest_int('max_depth', 3, 15),  # Deeper trees allowed
            'subsample': trial.suggest_float('subsample', 0.40, 0.95),  # More aggressive subsampling
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.40, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 8.0, log=True),  # Log scale
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 8.0, log=True),  # Log scale
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 12),
            
            # January-specific domain adaptation
            'january_weight_boost': trial.suggest_float('january_weight_boost', 1.5, 4.0),
            'max_weight_clip': trial.suggest_float('max_weight_clip', 10.0, 20.0),
            
            # Clustering parameters for feature engineering
            'n_spatial_clusters': trial.suggest_int('n_spatial_clusters', 15, 35),
            'n_temporal_clusters': trial.suggest_int('n_temporal_clusters', 8, 18),
        }
        
        # Full cross-validation with feature engineering inside loop
        cv = SpatioTemporalCV(spatial_weight=0.3, random_state=42)
        scores = []
        january_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            # STEP 1: Apply AdvancedSpatioTemporalFeatures FIRST
            fe_trial = AdvancedSpatioTemporalFeatures(
                row_only=False,
                n_spatial_clusters=params['n_spatial_clusters'],
                n_temporal_clusters=params['n_temporal_clusters'],
                january_bridge_features=True,
                test_distribution=temporal_stats,
                use_distribution_matching=True
            )
            
            X_train_enhanced = fe_trial.fit_transform(X_train_fold, y_train_fold)
            X_val_enhanced = fe_trial.transform(X_val_fold)
            
            # STEP 2: Apply AFE pickles SECOND
            X_train_enhanced, X_val_enhanced, _ = load_and_apply_feature_generators(
                X_train_enhanced, X_val_enhanced, X_test.copy(), model_dir
            )
            
            # Standard preprocessing
            train_means = X_train_enhanced.mean()
            X_train_enhanced = X_train_enhanced.fillna(train_means)
            X_val_enhanced = X_val_enhanced.fillna(train_means)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_enhanced)
            X_val_scaled = scaler.transform(X_val_enhanced)
            
            # Apply optimized January boosting
            weights_optimized = base_sample_weights[train_idx].copy()
            january_train_mask = X_train.iloc[train_idx]['month'] == 1
            weights_optimized[january_train_mask] *= params['january_weight_boost']
            weights_optimized = np.clip(weights_optimized, 0.1, params['max_weight_clip'])
            
            # Train with optimized parameters
            model = xgb.XGBRegressor(
                n_estimators=params['n_estimators'],
                learning_rate=params['learning_rate'],
                max_depth=params['max_depth'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                reg_alpha=params['reg_alpha'],
                reg_lambda=params['reg_lambda'],
                min_child_weight=params['min_child_weight'],
                early_stopping_rounds=50,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train_fold, 
                     sample_weight=weights_optimized,
                     eval_set=[(X_val_scaled, y_val_fold)],
                     verbose=False)
            
            # Evaluate with detailed logging
            val_pred = model.predict(X_val_scaled)
            fold_rmse = np.sqrt(mean_squared_error(y_val_fold, val_pred))
            scores.append(fold_rmse)
            
            # Track January performance separately
            january_val_mask = X_train.iloc[val_idx]['month'] == 1
            if january_val_mask.sum() > 0:
                january_rmse = np.sqrt(mean_squared_error(
                    y_val_fold[january_val_mask], val_pred[january_val_mask]
                ))
                january_scores.append(january_rmse)
        
        # Calculate all score components
        overall_rmse = np.mean(scores)
        january_rmse = np.mean(january_scores) if january_scores else overall_rmse
        combined_rmse = 0.4 * overall_rmse + 0.6 * january_rmse
        
        # Convert to competition scores
        overall_competition_score = np.exp(-overall_rmse/100)
        january_competition_score = np.exp(-january_rmse/100)
        combined_competition_score = np.exp(-combined_rmse/100)
        
        # Store detailed results for logging
        trial.set_user_attr("overall_rmse", overall_rmse)
        trial.set_user_attr("january_rmse", january_rmse)
        trial.set_user_attr("combined_rmse", combined_rmse)
        trial.set_user_attr("overall_comp_score", overall_competition_score)
        trial.set_user_attr("january_comp_score", january_competition_score)
        trial.set_user_attr("combined_comp_score", combined_competition_score)
        
        # Print detailed trial results
        print(f"Trial {trial.number:3d}: Overall RMSE={overall_rmse:.4f} (Score={overall_competition_score:.6f}), "
              f"Jan RMSE={january_rmse:.4f} (Score={january_competition_score:.6f}), "
              f"Combined RMSE={combined_rmse:.4f} (Score={combined_competition_score:.6f})")
        
        return -combined_competition_score  # Minimize negative score = maximize score
    
    # Setup persistent storage with TPE sampler
    study_name = "spatiotemporal_competition_optimization"
    storage_url = f"sqlite:///{study_path}"
    n_startup_trials = max(1, int(n_trials * 0.1))  # 10% random exploration
    
    sampler = optuna.samplers.TPESampler(
        multivariate=True,  # Consider parameter correlations
        n_startup_trials=n_startup_trials,
        seed=42
    )
    
    # Create study with resume capability
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction='minimize',
        sampler=sampler,
        load_if_exists=True  # Resume from existing trials
    )
    
    # Run optimization
    existing_trials = len(study.trials)
    if existing_trials > 0:
        print(f"📁 Resumed study with {existing_trials} trials. Running {n_trials} more...")
    else:
        print(f"🆕 New study. Running {n_trials} trials...")
    
    print(f"⚙️  TPE: multivariate=True, startup_trials={n_startup_trials}")
    print(f"💾 Storage: {storage_url}")
    
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # Detailed results reporting
    best_trial = study.best_trial
    print(f"\n🏆 Best Competition Score: {-study.best_value:.6f}")
    print(f"📊 Best Components:")
    print(f"   Overall: {best_trial.user_attrs['overall_rmse']:.4f} RMSE → {best_trial.user_attrs['overall_comp_score']:.6f}")
    print(f"   January: {best_trial.user_attrs['january_rmse']:.4f} RMSE → {best_trial.user_attrs['january_comp_score']:.6f}")
    print(f"   Combined: {best_trial.user_attrs['combined_rmse']:.4f} RMSE → {best_trial.user_attrs['combined_comp_score']:.6f}")
    
    return study.best_params, best_trial.user_attrs['combined_rmse'], study

# Usage with persistent storage
best_params, best_rmse, study = hyperparameter_optimization(
    X_train, y_train, X_test, temporal_stats,
    n_trials=200, timeout=4*3600,  # 4 hours
    study_path="competition_optimization_v1.db"
)
```

### 3.2 Key Optimization Features

#### Enhanced Parameter Ranges
The updated optimization explores much wider parameter spaces:

- **`n_estimators`**: 50-2500 (vs 800-2500) - includes fast models for ensemble diversity
- **`learning_rate`**: 0.01-0.3 with log scale - better exploration of learning rates
- **`max_depth`**: 3-15 (vs 4-10) - allows both shallow and very deep trees
- **`subsample/colsample_bytree`**: 0.40-0.95 (vs 0.65-0.95) - more aggressive regularization
- **Regularization**: Log-scale sampling for `reg_alpha` and `reg_lambda`

#### Persistent Storage Benefits
```python
# Resume optimization from previous session
study = hyperparameter_optimization(
    X_train, y_train, X_test, temporal_stats,
    n_trials=50,  # Add 50 more trials
    study_path="my_optimization.db"  # Same database
)

# The study automatically:
# 1. Loads existing 200 trials from database
# 2. Continues TPE optimization with learned parameter relationships  
# 3. Runs 50 additional trials
# 4. Returns best result from all 250 trials
```

#### Detailed Trial Monitoring
Each trial now logs complete score breakdown:
```
Trial  42: Overall RMSE=51.2341 (Score=0.599845), Jan RMSE=67.8234 (Score=0.507421), Combined RMSE=61.0156 (Score=0.543210)
Trial  43: Overall RMSE=49.8765 (Score=0.607123), Jan RMSE=65.4321 (Score=0.519876), Combined RMSE=59.1234 (Score=0.553987)
```

This enables:
- **Real-time monitoring** of January vs overall performance
- **Early identification** of promising parameter regions
- **Competition score tracking** instead of just RMSE

#### TPE Multivariate Optimization
The TPE sampler with `multivariate=True` discovers parameter correlations like:
- High `january_weight_boost` works best with moderate `reg_alpha`
- Deep trees (`max_depth` > 10) benefit from lower `learning_rate`
- Aggressive subsampling pairs well with higher `n_estimators`

### 3.3 Multi-Model Architecture Strategy

```python
# Create different optimization runs for ensemble diversity
ensemble_configs = {
    'conservative': {
        'reg_alpha_range': (1.0, 8.0),
        'reg_lambda_range': (1.0, 8.0), 
        'january_boost_range': (1.5, 2.5),
        'study_path': 'conservative_model.db'
    },
    'aggressive': {
        'reg_alpha_range': (0.1, 2.0),
        'reg_lambda_range': (0.1, 2.0),
        'january_boost_range': (2.5, 4.0),
        'study_path': 'aggressive_model.db'
    },
    'balanced': {
        'reg_alpha_range': (0.5, 4.0),
        'reg_lambda_range': (0.5, 4.0),
        'january_boost_range': (2.0, 3.0),
        'study_path': 'balanced_model.db'
    }
}

# Train different model types for ensemble
ensemble_models = {}
for config_name, config in ensemble_configs.items():
    print(f"Optimizing {config_name} model...")
    best_params, _, study = hyperparameter_optimization(
        X_train, y_train, X_test, temporal_stats,
        n_trials=100,
        study_path=config['study_path']
    )
    ensemble_models[config_name] = best_params
```

## Phase 4: Ensemble Strategy with Distribution Matching (Deep Dive)

### 4.1 Multi-Stage Ensemble Architecture

```python
class CompetitionEnsemble:
    def __init__(self, optimized_models, oof_predictions):
        self.models = optimized_models
        self.oof_predictions = oof_predictions
        self.ensemble_weights = None
        
    def fit_stage1_ensemble(self, X_train, y_train):
        """Train individual optimized models"""
        self.stage1_models = {}
        
        for model_name, params in self.models.items():
            # Train with full pipeline and domain adaptation
            model = self._create_model(model_name, params)
            
            # Use full dataset with enhanced features
            X_enhanced = self._apply_full_pipeline(X_train, fit=True)
            weights = self._get_optimized_weights(X_train)
            
            model.fit(X_enhanced, y_train, sample_weight=weights)
            self.stage1_models[model_name] = model
    
    def fit_stage2_metalearner(self, X_train, y_train):
        """Meta-learner using OOF predictions"""
        # Create meta-features from OOF predictions
        meta_features = np.column_stack([
            self.oof_predictions[name] for name in self.models.keys()
        ])
        
        # Add January indicator as meta-feature
        january_indicator = (X_train['month'] == 1).astype(int).values.reshape(-1, 1)
        meta_features_enhanced = np.concatenate([meta_features, january_indicator], axis=1)
        
        # Simple linear meta-learner with January-aware weights
        from sklearn.linear_model import Ridge
        self.meta_learner = Ridge(alpha=1.0)
        
        # Weight January samples more in meta-learning
        january_mask = X_train['month'] == 1
        meta_weights = np.where(january_mask, 3.0, 1.0)  # 3x weight for January
        
        self.meta_learner.fit(meta_features_enhanced, y_train, sample_weight=meta_weights)
    
    def fit_stage3_final_weights(self, X_train, y_train):
        """Optimize final ensemble weights for competition metric"""
        def weight_objective(weights):
            weights = np.array(weights) / np.sum(weights)  # Normalize
            
            ensemble_pred = np.zeros(len(y_train))
            for i, (name, model) in enumerate(self.stage1_models.items()):
                ensemble_pred += weights[i] * self.oof_predictions[name]
            
            # Use competition metric: exp(-RMSE/100)
            rmse = np.sqrt(mean_squared_error(y_train, ensemble_pred))
            competition_score = np.exp(-rmse/100)
            return -competition_score  # Minimize negative score
        
        # Optimize weights with constraints
        from scipy.optimize import minimize
        n_models = len(self.stage1_models)
        initial_weights = np.ones(n_models) / n_models
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        result = minimize(weight_objective, initial_weights, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        self.final_weights = result.x
```

### 4.2 Distribution-Aware Prediction

```python
def predict_with_distribution_awareness(self, X_test):
    """Final prediction with geographic and temporal adaptation"""
    
    # Apply same full pipeline as training
    X_test_enhanced = self._apply_full_pipeline(X_test, fit=False)
    
    # Get base predictions from each model
    base_predictions = {}
    for name, model in self.stage1_models.items():
        base_predictions[name] = model.predict(X_test_enhanced)
    
    # Stage 2: Meta-learner predictions
    meta_features = np.column_stack([base_predictions[name] for name in self.models.keys()])
    january_test_indicator = (X_test['month'] == 1).astype(int).values.reshape(-1, 1)
    meta_features_enhanced = np.concatenate([meta_features, january_test_indicator], axis=1)
    
    meta_predictions = self.meta_learner.predict(meta_features_enhanced)
    
    # Stage 3: Final weighted ensemble
    weighted_predictions = np.sum([
        self.final_weights[i] * base_predictions[name] 
        for i, name in enumerate(self.models.keys())
    ], axis=0)
    
    # Blend meta-learner and weighted ensemble
    # More weight to meta-learner for January samples
    january_test_mask = X_test['month'] == 1
    final_predictions = np.where(
        january_test_mask,
        0.6 * meta_predictions + 0.4 * weighted_predictions,  # January: favor meta
        0.3 * meta_predictions + 0.7 * weighted_predictions   # Others: favor weighted
    )
    
    # Apply final post-processing
    final_predictions = np.maximum(final_predictions, 0)  # Non-negative
    final_predictions = np.minimum(final_predictions, 1000)  # Reasonable upper bound
    
    return final_predictions
```

## Phase 5: Final Validation and Submission (Deep Dive)

### 5.1 Adversarial Validation for Robustness

```python
def adversarial_validation_analysis():
    """Identify train samples most similar to test distribution"""
    
    # Combine train and test for adversarial validation
    X_combined = pd.concat([
        X_train[['latitude', 'longitude', 'hour', 'day_of_week', 'month', 'day_of_year']],
        X_test[['latitude', 'longitude', 'hour', 'day_of_week', 'month', 'day_of_year']]
    ])
    
    # Create target: 0 for train, 1 for test
    y_adversarial = np.concatenate([
        np.zeros(len(X_train)),
        np.ones(len(X_test))
    ])
    
    # Train classifier to distinguish train vs test
    from sklearn.ensemble import RandomForestClassifier
    adversarial_model = RandomForestClassifier(n_estimators=100, random_state=42)
    adversarial_model.fit(X_combined, y_adversarial)
    
    # Get prediction probabilities for train samples
    train_test_probs = adversarial_model.predict_proba(X_combined[:len(X_train)])[:, 1]
    
    # Identify most test-like training samples
    test_like_threshold = np.percentile(train_test_probs, 90)  # Top 10% most test-like
    test_like_mask = train_test_probs > test_like_threshold
    
    print(f"Found {test_like_mask.sum()} test-like training samples")
    print(f"January samples in test-like: {X_train[test_like_mask]['month'].eq(1).sum()}")
    
    # Validate model performance on test-like samples
    test_like_predictions = ensemble.predict(X_train[test_like_mask])
    test_like_rmse = np.sqrt(mean_squared_error(
        y_train[test_like_mask], 
        test_like_predictions
    ))
    
    print(f"Test-like samples RMSE: {test_like_rmse:.4f}")
    return test_like_mask, test_like_rmse
```

### 5.2 Multi-Submission Strategy

```python
def create_submission_variants():
    """Create multiple submission variants for robustness"""
    
    submissions = {}
    
    # Variant 1: Conservative (higher regularization, lower January boost)
    conservative_params = {**optimized_params}
    conservative_params.update({
        'reg_alpha': optimized_params['reg_alpha'] * 1.5,
        'reg_lambda': optimized_params['reg_lambda'] * 1.5,
        'january_weight_boost': optimized_params['january_weight_boost'] * 0.8
    })
    submissions['conservative'] = train_and_predict(conservative_params)
    
    # Variant 2: Aggressive (optimized parameters as-is)
    submissions['aggressive'] = train_and_predict(optimized_params)
    
    # Variant 3: January-focused (extra January emphasis)
    january_params = {**optimized_params}
    january_params.update({
        'january_weight_boost': optimized_params['january_weight_boost'] * 1.3,
        'max_weight_clip': optimized_params['max_weight_clip'] * 1.2
    })
    submissions['january_focused'] = train_and_predict(january_params)
    
    # Variant 4: Ensemble blend
    submissions['ensemble_blend'] = (
        0.3 * submissions['conservative'] +
        0.4 * submissions['aggressive'] +
        0.3 * submissions['january_focused']
    )
    
    return submissions

def final_submission_selection(submissions):
    """Select best submission based on validation metrics"""
    
    # Evaluate each variant on test-like training samples
    results = {}
    for name, predictions in submissions.items():
        # Cross-validation score
        cv_score = evaluate_with_spatiotemporal_cv(predictions)
        
        # Adversarial validation score
        adv_score = evaluate_on_test_like_samples(predictions)
        
        # Combined score weighted by importance
        combined_score = 0.6 * cv_score + 0.4 * adv_score
        
        results[name] = {
            'cv_score': cv_score,
            'adv_score': adv_score,
            'combined_score': combined_score
        }
        
        print(f"{name}: CV={cv_score:.4f}, Adv={adv_score:.4f}, Combined={combined_score:.4f}")
    
    # Select best performing variant
    best_variant = min(results.keys(), key=lambda x: results[x]['combined_score'])
    print(f"Selected variant: {best_variant}")
    
    return submissions[best_variant], results
```

## Expected Performance Timeline

### Baseline Performance
- **Standard CV**: ~55 RMSE (overly optimistic)
- **SpatioTemporalCV**: ~58 RMSE (more realistic)
- **January samples**: ~75 RMSE (severe challenge)

### Phase-by-Phase Improvements
1. **AFE Foundation**: 58 → 53 RMSE (8% improvement)
2. **Enhanced CV Pipeline**: 53 → 48 RMSE (10% improvement)  
3. **Hyperparameter Optimization**: 48 → 45 RMSE (6% improvement)
4. **Advanced Ensemble**: 45 → 42 RMSE (7% improvement)
5. **Final Optimization**: 42 → 40 RMSE (5% improvement)

### Final Expected Performance
- **Overall RMSE**: ~40-42 (25-30% improvement over baseline)
- **January RMSE**: ~55-60 (25% improvement over January baseline)
- **Competition Score**: exp(-41/100) = 0.664 (vs baseline 0.577 = 15% improvement)

## Critical Success Factors

### Data Efficiency
- **Every January sample counts**: Domain adaptation weights, cross-seasonal features
- **Maximum information extraction**: Full pipeline in CV, OOF predictions
- **Realistic validation**: SpatioTemporalCV prevents overfitting

### Generalization
- **Geographic robustness**: Universal features, adaptive clustering
- **Temporal robustness**: Cyclical encoding, distribution matching
- **Pattern transfer**: December→January bridges, meta-learning

### Technical Excellence
- **No data leakage**: Proper fit/transform splits in CV
- **Memory efficiency**: Selective feature engineering, batch processing
- **Reproducibility**: Fixed random seeds, saved model states

This strategy leverages every component of spatio_temporal.py while addressing the unique challenges of severe distribution shifts and January scarcity. The multi-phase approach ensures robust validation while maximizing the limited training signal.