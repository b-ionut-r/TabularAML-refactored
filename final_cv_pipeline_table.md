# 🏆 Final CV Strategy Pipeline - Complete Implementation Guide

## 📋 Master Strategy Table

| **Pipeline Phase** | **CV Strategy** | **Configuration** | **Expected Time** | **Key Instructions** |
|-------------------|-----------------|-------------------|-------------------|----------------------|
| **Phase 1: Feature Engineering** | `HybridSpatialTemporalCV` | • n_spatial_folds=3<br>• n_january_folds=2<br>• n_clusters=9<br>• Total: 5 folds | ~2-3 days<br>(1000+ features) | • Use for ALL genetic programming fitness evaluations<br>• Set n_estimators=100 for speed<br>• Track January-fold performance separately<br>• Expect val scores ~0.45-0.50 |
| **Phase 2: Feature Selection** | `HybridSpatialTemporalCV` | Same as Phase 1 | ~4-6 hours | • Use to select top 50-100 features<br>• Run with n_estimators=200<br>• Keep features that perform well on BOTH spatial and January folds |
| **Phase 3: Model Selection** | `TemporalJanuaryPriority` | • n_splits=5<br>• First fold is 100% January | ~2-3 hours | • Compare XGBoost vs LightGBM vs CatBoost<br>• **Critical**: Check fold 0 (January-only) score separately<br>• Pick model with best January performance |
| **Phase 4: Hyperparameter Tuning** | `WeightedEnsembleCV` | • Strategies:<br>&nbsp;&nbsp;- HybridSpatialTemporalCV (50%)<br>&nbsp;&nbsp;- TemporalJanuaryPriority (30%)<br>&nbsp;&nbsp;- JanuaryOnlySpatialCV (20%)<br>• Total: ~13 folds | ~1-2 days<br>(200 Optuna trials) | • Use Optuna with TPESampler<br>• Track individual CV strategy scores<br>• Prune trials with poor January performance<br>• Focus on robust hyperparameters |
| **Phase 5: Final Training** | **No CV - Full Data** | • Train 3 models:<br>&nbsp;&nbsp;1. Balanced (40%)<br>&nbsp;&nbsp;2. January-weighted (35%)<br>&nbsp;&nbsp;3. Spatial-weighted (25%) | ~2-4 hours | • Model 1: Regular training<br>• Model 2: sample_weight = 5.0 for January<br>• Model 3: sample_weight = 1/sqrt(location_frequency)<br>• Ensemble with weights shown |
| **Phase 6: Pseudo-labeling** | **No CV - Confidence Based** | • Use prediction variance<br>• Take top 20% confident | ~1-2 hours | • Only if confident predictions > 500 samples<br>• Downweight pseudo-labels to 0.3<br>• Retrain final model |
| **Phase 7: Post-processing** | **No CV - Direct Validation** | • Scale to January mean<br>• Clip outliers | ~30 minutes | • Ensure mean matches train January mean<br>• Check prediction distribution<br>• Submit! |

## 🔧 Implementation Code Templates

### Phase 1-2: Feature Engineering & Selection
```python
from cv_splitters import HybridSpatialTemporalCV
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

# Initialize CV (use for both Phase 1 and 2)
cv = HybridSpatialTemporalCV(
    n_spatial_folds=3,
    n_january_folds=2,
    n_clusters=9,
    random_state=42
)

# Genetic programming fitness function
def evaluate_features(feature_set):
    model = XGBRegressor(
        n_estimators=100,  # Fast for feature search
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    scores = cross_val_score(
        model, 
        X_train[feature_set], 
        y_train,
        cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    rmse = np.sqrt(-scores.mean())
    return np.exp(-rmse / 100)  # Competition metric
```

### Phase 3: Model Selection
```python
from cv_splitters import TemporalJanuaryPriority

cv_temporal = TemporalJanuaryPriority(n_splits=5)

models = {
    'xgboost': XGBRegressor(n_estimators=300),
    'lightgbm': LGBMRegressor(n_estimators=300),
    'catboost': CatBoostRegressor(iterations=300, verbose=0)
}

for name, model in models.items():
    scores = cross_val_score(
        model, X_best_features, y_train,
        cv=cv_temporal, 
        scoring='neg_mean_squared_error'
    )
    
    # CRITICAL: Check January performance
    january_score = np.exp(-np.sqrt(-scores[0])/100)  # First fold is January-only
    overall_score = np.exp(-np.sqrt(-scores.mean())/100)
    
    print(f"{name}: January={january_score:.4f}, Overall={overall_score:.4f}")
    # Pick model with best January score!
```

### Phase 4: Hyperparameter Tuning
```python
from cv_splitters import WeightedEnsembleCV, HybridSpatialTemporalCV, TemporalJanuaryPriority, JanuaryOnlySpatialCV
import optuna

# Create weighted ensemble
cv_strategies = [
    (HybridSpatialTemporalCV(n_spatial_folds=3, n_january_folds=2), 0.5),
    (TemporalJanuaryPriority(n_splits=5), 0.3),
    (JanuaryOnlySpatialCV(n_splits=3), 0.2)
]
weighted_cv = WeightedEnsembleCV(cv_strategies)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0)
    }
    
    model = XGBRegressor(**params, random_state=42)
    score, details = weighted_cv.get_competition_score(model, X_train, y_train)
    
    # Track individual strategy performance
    for strategy_name, metrics in details.items():
        trial.set_user_attr(f"{strategy_name}_rmse", metrics['mean'])
    
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200, n_jobs=1)
```

### Phase 5: Final Training
```python
best_params = study.best_params

# Model 1: Balanced (40% weight)
model_balanced = XGBRegressor(**best_params)
model_balanced.fit(X_train, y_train)

# Model 2: January-focused (35% weight)
january_weights = X_train['month'].map({1: 5.0}).fillna(1.0)
model_january = XGBRegressor(**best_params)
model_january.fit(X_train, y_train, sample_weight=january_weights)

# Model 3: Spatial diversity (25% weight)
location_counts = X_train.groupby(['latitude', 'longitude']).size()
spatial_weights = X_train.apply(
    lambda row: 1.0 / np.sqrt(location_counts[(row['latitude'], row['longitude'])]), 
    axis=1
)
model_spatial = XGBRegressor(**best_params)
model_spatial.fit(X_train, y_train, sample_weight=spatial_weights)

# Ensemble predictions
predictions = (
    0.40 * model_balanced.predict(X_test) +
    0.35 * model_january.predict(X_test) +
    0.25 * model_spatial.predict(X_test)
)
```

## ⚠️ Critical Success Factors

1. **Speed Matters in Phase 1**: With genetic programming, you'll evaluate 1000+ feature sets. Keep models light (100 estimators).

2. **January Fold is Truth**: In Phase 3, the model that performs best on the January-only fold (fold 0 of TemporalJanuaryPriority) is likely your winner.

3. **Don't Skip Weighted Ensemble**: Phase 4 is slower but critical. The weighted ensemble ensures your hyperparameters work across all distribution shifts.

4. **Monitor Validation-LB Gap**: If your validation scores don't correlate with LB, increase January representation (adjust n_january_folds in HybridSpatialTemporalCV).

5. **Trust Your Test Results**: You proved HybridSpatialTemporalCV works (0.4621 vs 0.3709). Stick with it.

## 📈 Expected Progression

| Phase | Validation Score | LB Score (Expected) | Notes |
|-------|-----------------|---------------------|--------|
| Baseline | 0.3709 | 0.65 | Original spatial CV |
| After Phase 1-2 | 0.46-0.48 | 0.78-0.82 | Better features with Hybrid CV |
| After Phase 3-4 | 0.48-0.50 | 0.82-0.85 | Optimized model & hyperparameters |
| After Phase 5-7 | 0.50-0.52 | 0.85-0.88 | Ensemble & post-processing |

## 🏁 Final Checklist

- [ ] Use HybridSpatialTemporalCV for ALL feature engineering
- [ ] Select model based on January-only fold performance
- [ ] Run 200+ Optuna trials with WeightedEnsembleCV
- [ ] Train 3 specialized models with different weights
- [ ] Apply pseudo-labeling if confidence is high
- [ ] Match prediction distribution to January training data
- [ ] Submit and monitor LB correlation!

**Remember**: This strategy is proven by your tests. The key is disciplined execution across all phases.