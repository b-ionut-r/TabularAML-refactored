import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

class PollutionEnsemble:
    def __init__(self):
        self.models = {
            'lgb': lgb.LGBMRegressor(
                num_leaves=80, learning_rate=0.03, n_estimators=1000,
                feature_fraction=0.7, bagging_fraction=0.7, verbose=-1
            ),
            'xgb': xgb.XGBRegressor(
                n_estimators=1000, learning_rate=0.03, max_depth=6,
                subsample=0.7, colsample_bytree=0.7, verbosity=0
            ),
            'catboost': cb.CatBoostRegressor(
                iterations=1000, learning_rate=0.03, depth=6,
                verbose=False
            )
        }
        self.meta_model = Ridge(alpha=1.0)
        self.scalers = {}
        
    def create_features(self, df):
        df = df.copy()
        
        # All previous feature engineering
        # Cyclical encoding
        for col, period in [('hour', 24), ('day_of_week', 7), 
                           ('day_of_year', 365), ('month', 12)]:
            df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
            df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)
        
        # Spatial features
        df['coord_sum'] = df['latitude'] + df['longitude']
        df['coord_diff'] = df['latitude'] - df['longitude']
        df['coord_product'] = df['latitude'] * df['longitude']
        df['coord_ratio'] = df['latitude'] / (df['longitude'] + 1e-5)
        
        # Temporal interactions
        df['hour_month'] = df['hour'] * df['month']
        df['hour_dayofweek'] = df['hour'] * df['day_of_week']
        
        # Pollution-prone hours/days
        df['is_rush_hour'] = df['hour'].isin([7,8,9,17,18,19]).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        
        return df
    
    def train_ensemble(self, X, y, n_folds=5):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # First level: train base models
        oof_preds = {name: np.zeros(len(X)) for name in self.models.keys()}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Scale features for this model
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[name] = scaler
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Clone model for this fold
                if name == 'lgb':
                    fold_model = lgb.LGBMRegressor(**self.models[name].get_params())
                    fold_model.fit(X_train, y_train,
                                 eval_set=[(X_val, y_val)],
                                 eval_metric='rmse',
                                 callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                elif name == 'xgb':
                    fold_model = xgb.XGBRegressor(**self.models[name].get_params())
                    fold_model.fit(X_train, y_train,
                                 eval_set=[(X_val, y_val)],
                                 early_stopping_rounds=50,
                                 verbose=False)
                else:  # catboost
                    fold_model = cb.CatBoostRegressor(**self.models[name].get_params())
                    fold_model.fit(X_train, y_train,
                                 eval_set=(X_val, y_val),
                                 early_stopping_rounds=50,
                                 verbose=False)
                
                oof_preds[name][val_idx] = fold_model.predict(X_val)
            
            # Train on full data for final model
            self.models[name].fit(X_scaled, y)
            
            rmse = np.sqrt(np.mean((oof_preds[name] - y)**2))
            print(f"{name} CV RMSE: {rmse:.4f}")
        
        # Second level: train meta-model
        meta_features = np.column_stack(list(oof_preds.values()))
        self.meta_model.fit(meta_features, y)
        
        # Calculate ensemble score
        ensemble_pred = self.meta_model.predict(meta_features)
        ensemble_rmse = np.sqrt(np.mean((ensemble_pred - y)**2))
        print(f"Ensemble CV RMSE: {ensemble_rmse:.4f}")
        
        return oof_preds
    
    def predict(self, X):
        predictions = {}
        
        for name, model in self.models.items():
            X_scaled = self.scalers[name].transform(X)
            predictions[name] = model.predict(X_scaled)
        
        meta_features = np.column_stack(list(predictions.values()))
        final_pred = self.meta_model.predict(meta_features)
        
        return final_pred

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Initialize ensemble
ensemble = PollutionEnsemble()

# Create features
train_fe = ensemble.create_features(train)
test_fe = ensemble.create_features(test)

# Prepare data
feature_cols = [col for col in train_fe.columns 
                if col not in ['id', 'pollution_value']]
X = train_fe[feature_cols].values
y = train_fe['pollution_value']
X_test = test_fe[feature_cols].values

# Train ensemble
oof_predictions = ensemble.train_ensemble(X, y)

# Make predictions
test_predictions = ensemble.predict(X_test)

# Post-processing: clip predictions to reasonable range
train_min = y.min()
train_max = y.max()
test_predictions = np.clip(test_predictions, 
                          train_min * 0.9, 
                          train_max * 1.1)

# Create submission
submission = pd.read_csv('submission_example.csv')
submission['pollution_value'] = test_predictions
submission.to_csv('submission_ensemble.csv', index=False)
print("Ensemble submission created!")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': ensemble.models['lgb'].feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))