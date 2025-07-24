# Air Pollution Prediction - Advanced Spatio-Temporal Solution

This solution addresses the key challenge of the competition: **significant spatial and temporal distribution differences between train and test sets**.

## 🎯 Key Features

1. **Distribution-Aware Feature Engineering**
   - Spatial clustering features for location generalization
   - Temporal pattern features with cyclical encoding
   - Distribution matching features based on test set statistics

2. **Custom Cross-Validation**
   - `SpatioTemporalCV`: Ensures validation sets match test distribution
   - Prevents overestimating model performance
   - Balances spatial and temporal similarity

3. **Domain Adaptation**
   - Importance weighting to handle temporal shifts
   - Helps models focus on patterns relevant to test distribution

4. **Robust Ensemble**
   - Combines LightGBM, XGBoost, and Random Forest
   - Weighted by CV performance
   - Post-processing for valid predictions

## 📋 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Basic Usage
```python
from spatiotemporal_solution import (
    AdvancedSpatioTemporalFeatures,
    SpatioTemporalCV,
    create_complete_pipeline
)

# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Run complete pipeline
results = create_complete_pipeline(train_df, test_df)

# Save submission
results['submission'].to_csv('submission.csv', index=False)
```

### 3. Step-by-Step Approach
For more control, follow the practical usage example:

```python
# Step 1: Feature engineering for validation (no leakage)
fe_val = AdvancedSpatioTemporalFeatures(row_only=True)
X_train_val = fe_val.fit_transform(X_train, y_train)

# Step 2: Custom CV
cv = SpatioTemporalCV(
    n_splits=5,
    test_spatial_coords=test_df[['latitude', 'longitude']].values,
    test_temporal_features=test_df[['hour', 'month', 'day_of_week']]
)

# Step 3: Validate models
# ... (see practical_usage_example.py)

# Step 4: Train final model with ALL features
fe_final = AdvancedSpatioTemporalFeatures(row_only=False)
X_train_final = fe_final.fit_transform(X_train, y_train)
X_test_final = fe_final.transform(X_test)
```

## 🔧 Key Components

### AdvancedSpatioTemporalFeatures
```python
AdvancedSpatioTemporalFeatures(
    row_only=False,           # True for CV (no leakage), False for final model
    n_spatial_clusters=30,    # Number of location clusters
    n_temporal_clusters=10,   # Number of temporal pattern clusters
    use_distribution_matching=True,  # Create distribution-aware features
    test_distribution=None    # Test set statistics for matching
)
```

### SpatioTemporalCV
```python
SpatioTemporalCV(
    n_splits=5,
    test_spatial_coords=None,      # Test set coordinates
    test_temporal_features=None,   # Test set temporal features
    spatial_weight=0.5,           # Balance spatial vs temporal (0-1)
    random_state=42
)
```

## 📊 Understanding the Distribution Shifts

Run the distribution analysis notebook to visualize:
- Geographic distribution differences
- Temporal pattern variations
- Impact on model performance

```python
# Analyze distributions
analyzer = SpatioTemporalDistributionAnalyzer()
spatial_stats, temporal_stats = analyzer.analyze(train_df, test_df)
```

## 🚀 Advanced Techniques

### 1. Hyperparameter Tuning
```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
    }
    
    # Use SpatioTemporalCV for realistic validation
    model = lgb.LGBMRegressor(**params)
    # ... validation code ...
    
    return cv_score

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

### 2. Feature Selection
```python
# Use feature importance from validation
feature_importance = get_feature_importance(model, X_train_val)
top_features = feature_importance.nlargest(50, 'importance')['feature'].tolist()
```

### 3. Model Stacking
```python
from sklearn.linear_model import Ridge

stacker = ModelStacking(
    base_models={'lgb': lgb_model, 'xgb': xgb_model, 'rf': rf_model},
    meta_model=Ridge(),
    cv_folds=5
)
```

## 📈 Expected Performance

- **Random CV**: Overestimates performance (doesn't account for distribution shift)
- **Spatial CV**: Better estimate for geographic generalization
- **SpatioTemporalCV**: Most realistic estimate matching test conditions

## 🐛 Troubleshooting

1. **Memory Issues**: Reduce `n_spatial_clusters` or use sampling
2. **Slow Training**: Use fewer estimators during validation
3. **Poor CV Scores**: Check if test distribution analysis was run correctly

## 📚 Files Included

- `spatiotemporal_solution.py`: Main implementation
- `practical_usage_example.py`: Step-by-step guide
- `distribution_analysis.py`: Visualization and analysis
- `advanced_techniques.py`: Additional features and methods
- `requirements.txt`: Package dependencies

## 🏆 Competition Tips

1. **Start Simple**: Use `row_only=True` features first
2. **Validate Properly**: Always use SpatioTemporalCV
3. **Monitor Distributions**: Check prediction distributions match training
4. **Ensemble**: Combine diverse models for robustness
5. **Post-Process**: Ensure predictions are in valid range

## 📞 Support

For questions or issues:
1. Check the distribution analysis notebook
2. Review the practical usage example
3. Ensure all dependencies are installed
4. Verify data format matches expected structure

Good luck with the competition! 🍀