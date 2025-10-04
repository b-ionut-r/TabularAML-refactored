"""
Domain Adaptation Techniques for Tabular Data
Handles temporal and geographic distribution shifts between train and test datasets.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DomainAdapter:
    """
    Comprehensive domain adaptation for tabular data with temporal and geographic shifts.
    
    Handles:
    - Temporal distribution shifts (seasonal, monthly, daily patterns)
    - Geographic distribution shifts (latitude/longitude differences)  
    - Combined adaptation strategies
    - Ensemble methods for robust performance
    """
    
    def __init__(self, date_col='date', geo_cols=['latitude', 'longitude'], random_state=42):
        """
        Initialize DomainAdapter.
        
        Args:
            date_col (str): Name of date column
            geo_cols (list): Names of geographic coordinate columns
            random_state (int): Random seed for reproducibility
        """
        self.date_col = date_col
        self.geo_cols = geo_cols
        self.random_state = random_state
        
        # Storage for computed weights and models
        self.temporal_weights = None
        self.geographic_weights = None
        self.combined_weights = None
        self.models = {}
        self.ensemble_weights = None
        
    def create_temporal_features(self, df):
        """
        Create cyclical temporal features that handle seasonal continuity.
        
        Args:
            df (pd.DataFrame): DataFrame with date column
            
        Returns:
            pd.DataFrame: DataFrame with added temporal features
        """
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        # Extract basic temporal components
        df['month'] = df[self.date_col].dt.month
        df['day_of_year'] = df[self.date_col].dt.dayofyear
        df['day_of_week'] = df[self.date_col].dt.dayofweek
        df['hour'] = df[self.date_col].dt.hour
        
        # Cyclical encoding for continuity (December -> January)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Season indicators
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)
        
        # Special indicators for extreme shifts
        df['is_january'] = (df['month'] == 1).astype(int)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df
    
    def compute_temporal_weights(self, train_df, test_df, temporal_col='month'):
        """
        Compute importance weights to match test temporal distribution.
        
        Args:
            train_df (pd.DataFrame): Training data
            test_df (pd.DataFrame): Test data  
            temporal_col (str): Column to compute distribution over
            
        Returns:
            np.array: Importance weights for training samples
        """
        # Ensure temporal features exist
        if temporal_col not in train_df.columns:
            train_df = self.create_temporal_features(train_df)
            test_df = self.create_temporal_features(test_df)
        
        # Compute distributions
        test_dist = test_df[temporal_col].value_counts(normalize=True)
        train_dist = train_df[temporal_col].value_counts(normalize=True)
        
        # Calculate importance weights
        weights = []
        for value in train_df[temporal_col]:
            if value in test_dist.index and value in train_dist.index:
                weight = test_dist[value] / train_dist[value]
            else:
                weight = 0.01  # Minimal weight for unseen values
            weights.append(weight)
        
        weights = np.array(weights)
        # Normalize weights
        weights = weights / weights.sum() * len(weights)
        
        self.temporal_weights = weights
        return weights
    
    def compute_geographic_weights(self, train_df, test_df, method='proximity'):
        """
        Compute weights based on geographic distribution similarity.
        
        Args:
            train_df (pd.DataFrame): Training data
            test_df (pd.DataFrame): Test data
            method (str): 'proximity' or 'density'
            
        Returns:
            np.array: Geographic importance weights
        """
        if not all(col in train_df.columns for col in self.geo_cols):
            raise ValueError(f"Geographic columns {self.geo_cols} not found in data")
        
        # Handle missing values in geographic columns
        train_geo_data = train_df[self.geo_cols].copy()
        test_geo_data = test_df[self.geo_cols].copy()
        
        # Check for missing values
        train_has_nan = train_geo_data.isnull().any().any()
        test_has_nan = test_geo_data.isnull().any().any()
        
        if train_has_nan or test_has_nan:
            print(f"Warning: Found NaN values in geographic columns. Imputing with median values.")
            
            # Simple median imputation
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            
            # Fit on training data and transform both
            train_geo_data = pd.DataFrame(
                imputer.fit_transform(train_geo_data),
                columns=self.geo_cols,
                index=train_geo_data.index
            )
            test_geo_data = pd.DataFrame(
                imputer.transform(test_geo_data),
                columns=self.geo_cols,
                index=test_geo_data.index
            )
        
        if method == 'proximity':
            # Weight by proximity to test distribution
            nbrs = NearestNeighbors(n_neighbors=min(10, len(test_geo_data)))
            nbrs.fit(test_geo_data)
            distances, _ = nbrs.kneighbors(train_geo_data)
            
            # Convert distances to weights (closer = higher weight)
            weights = 1 / (1 + distances.mean(axis=1))
            
        elif method == 'density':
            # Weight by density matching using clustering
            n_clusters = min(20, len(test_geo_data) // 5)
            
            # Fit clusters on test data
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
            test_clusters = kmeans.fit_predict(test_geo_data)
            train_clusters = kmeans.predict(train_geo_data)
            
            # Compute cluster densities
            test_cluster_counts = pd.Series(test_clusters).value_counts(normalize=True)
            train_cluster_counts = pd.Series(train_clusters).value_counts(normalize=True)
            
            # Weight by density ratio
            weights = []
            for cluster in train_clusters:
                if cluster in test_cluster_counts.index:
                    weight = test_cluster_counts[cluster] / train_cluster_counts.get(cluster, 0.01)
                else:
                    weight = 0.01
                weights.append(weight)
            weights = np.array(weights)
        
        # Normalize weights
        weights = weights / weights.sum() * len(weights)
        
        self.geographic_weights = weights
        return weights
    
    def compute_combined_weights(self, temporal_weight=0.7, geographic_weight=0.3):
        """
        Combine temporal and geographic weights.
        
        Args:
            temporal_weight (float): Weight for temporal component
            geographic_weight (float): Weight for geographic component
            
        Returns:
            np.array: Combined importance weights
        """
        if self.temporal_weights is None or self.geographic_weights is None:
            raise ValueError("Must compute temporal and geographic weights first")
        
        # Normalize input weights
        total = temporal_weight + geographic_weight
        temporal_weight /= total
        geographic_weight /= total
        
        # Combine weights
        combined = (temporal_weight * self.temporal_weights + 
                   geographic_weight * self.geographic_weights)
        
        # Normalize final weights
        combined = combined / combined.sum() * len(combined)
        
        self.combined_weights = combined
        return combined
    
    def _fix_weight_length(self, weights, target_length):
        """Fix weight array length to match target length."""
        if len(weights) > target_length:
            # Truncate
            return weights[:target_length]
        elif len(weights) < target_length:
            # Pad with mean weight
            mean_weight = weights.mean()
            padding = np.full(target_length - len(weights), mean_weight)
            return np.concatenate([weights, padding])
        else:
            return weights
    
    def fit_adaptation_models(self, X_train, y_train):
        """
        Train multiple models with different adaptation strategies.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training targets
        """
        if any(w is None for w in [self.temporal_weights, self.geographic_weights, self.combined_weights]):
            raise ValueError("Must compute weights before fitting models")
        
        # Store original data for weight length checking
        original_length = len(X_train)
        
        # Handle missing values in features and targets simultaneously
        X_train_clean = X_train.copy()
        y_train_clean = y_train.copy()
        
        # Find samples that have NaN in either features or target
        feature_nan_mask = X_train_clean.isnull().any(axis=1)
        target_nan_mask = y_train_clean.isnull()
        combined_nan_mask = feature_nan_mask | target_nan_mask
        
        if combined_nan_mask.any():
            print(f"Warning: Found {combined_nan_mask.sum()} samples with NaN values. Removing these samples.")
            
            # Remove samples with any NaN values
            valid_mask = ~combined_nan_mask
            X_train_clean = X_train_clean[valid_mask]
            y_train_clean = y_train_clean[valid_mask]
            
            # Update weights to match the cleaned dataset
            if len(self.temporal_weights) == original_length:
                self.temporal_weights = self.temporal_weights[valid_mask]
                self.geographic_weights = self.geographic_weights[valid_mask]
                self.combined_weights = self.combined_weights[valid_mask]
            else:
                print(f"Warning: Weight length {len(self.temporal_weights)} doesn't match original data length {original_length}")
        
        # Handle any remaining NaN values in features (should be rare after above)
        if X_train_clean.isnull().any().any():
            print("Warning: Found remaining NaN values in training features. Imputing with median values.")
            
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            
            # Store column names and index
            columns = X_train_clean.columns
            index = X_train_clean.index
            
            # Impute missing values
            X_train_clean = pd.DataFrame(
                imputer.fit_transform(X_train_clean),
                columns=columns,
                index=index
            )
            
            # Store imputer for later use in predictions
            self.feature_imputer = imputer
        else:
            self.feature_imputer = None
        
        
        # Ensure weights match the final cleaned dataset length
        final_length = len(X_train_clean)
        if len(self.temporal_weights) != final_length:
            print(f"Warning: Adjusting weight lengths from {len(self.temporal_weights)} to {final_length}")
            # Truncate or pad weights to match
            if len(self.temporal_weights) > final_length:
                self.temporal_weights = self.temporal_weights[:final_length]
                self.geographic_weights = self.geographic_weights[:final_length]
                self.combined_weights = self.combined_weights[:final_length]
            else:
                # This shouldn't happen, but handle it just in case
                missing_count = final_length - len(self.temporal_weights)
                mean_temporal = self.temporal_weights.mean()
                mean_geographic = self.geographic_weights.mean()
                mean_combined = self.combined_weights.mean()
                
                self.temporal_weights = np.concatenate([self.temporal_weights, np.full(missing_count, mean_temporal)])
                self.geographic_weights = np.concatenate([self.geographic_weights, np.full(missing_count, mean_geographic)])
                self.combined_weights = np.concatenate([self.combined_weights, np.full(missing_count, mean_combined)])
        
        # Baseline model (no adaptation)
        self.models['baseline'] = RandomForestRegressor(
            n_estimators=100, 
            random_state=self.random_state
        )
        self.models['baseline'].fit(X_train_clean, y_train_clean)
        
        # Debug info
        print(f"Debug: X_train_clean shape: {X_train_clean.shape}")
        print(f"Debug: y_train_clean shape: {y_train_clean.shape}")
        print(f"Debug: temporal_weights shape: {self.temporal_weights.shape}")
        print(f"Debug: geographic_weights shape: {self.geographic_weights.shape}")
        print(f"Debug: combined_weights shape: {self.combined_weights.shape}")
        
        # Force weights to match training data length
        n_samples = len(X_train_clean)
        if len(self.temporal_weights) != n_samples:
            print(f"ERROR: Weights length mismatch! Fixing by truncating/padding to {n_samples}")
            self.temporal_weights = self._fix_weight_length(self.temporal_weights, n_samples)
            self.geographic_weights = self._fix_weight_length(self.geographic_weights, n_samples)
            self.combined_weights = self._fix_weight_length(self.combined_weights, n_samples)
        
        # Temporal adaptation model
        self.models['temporal'] = RandomForestRegressor(
            n_estimators=100, 
            random_state=self.random_state + 1
        )
        self.models['temporal'].fit(X_train_clean, y_train_clean, sample_weight=self.temporal_weights)
        
        # Geographic adaptation model  
        self.models['geographic'] = RandomForestRegressor(
            n_estimators=100, 
            random_state=self.random_state + 2
        )
        self.models['geographic'].fit(X_train_clean, y_train_clean, sample_weight=self.geographic_weights)
        
        # Combined adaptation model
        self.models['combined'] = RandomForestRegressor(
            n_estimators=100, 
            random_state=self.random_state + 3
        )
        self.models['combined'].fit(X_train_clean, y_train_clean, sample_weight=self.combined_weights)
    
    def optimize_ensemble_weights(self, X_val, y_val, n_trials=100):
        """
        Optimize ensemble weights using validation data to minimize RMSE.
        
        Args:
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation targets
            n_trials (int): Number of optimization trials
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            def objective(trial):
                w_baseline = trial.suggest_float('baseline', 0.0, 1.0)
                w_temporal = trial.suggest_float('temporal', 0.0, 1.0) 
                w_geographic = trial.suggest_float('geographic', 0.0, 1.0)
                w_combined = trial.suggest_float('combined', 0.0, 1.0)
                
                # Normalize weights
                total = w_baseline + w_temporal + w_geographic + w_combined
                if total == 0:
                    return float('inf')
                
                weights = [w_baseline/total, w_temporal/total, w_geographic/total, w_combined/total]
                
                # Compute ensemble prediction
                pred = np.zeros(len(X_val))
                for i, (name, model) in enumerate(self.models.items()):
                    pred += weights[i] * model.predict(X_val)
                
                # Return RMSE (square root of MSE)
                return np.sqrt(mean_squared_error(y_val, pred))
            
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            
            best_params = study.best_params
            total = sum(best_params.values())
            self.ensemble_weights = [v/total for v in best_params.values()]
            
        except ImportError:
            # Fallback to simple grid search if optuna not available
            print("Optuna not available, using simple validation approach")
            best_mse = float('inf')
            best_weights = None
            
            # Simple grid search optimizing RMSE
            for w1 in [0.1, 0.3, 0.5, 0.7]:
                for w2 in [0.1, 0.3, 0.5, 0.7]:
                    for w3 in [0.1, 0.3, 0.5, 0.7]:
                        w4 = 1.0 - w1 - w2 - w3
                        if w4 < 0:
                            continue
                        
                        weights = [w1, w2, w3, w4]
                        pred = np.zeros(len(X_val))
                        for i, (name, model) in enumerate(self.models.items()):
                            pred += weights[i] * model.predict(X_val)
                        
                        rmse = np.sqrt(mean_squared_error(y_val, pred))
                        if rmse < best_mse:
                            best_mse = rmse
                            best_weights = weights
            
            self.ensemble_weights = best_weights
    
    def predict(self, X_test, method='ensemble'):
        """
        Make predictions using specified adaptation method.
        
        Args:
            X_test (pd.DataFrame): Test features
            method (str): 'baseline', 'temporal', 'geographic', 'combined', or 'ensemble'
            
        Returns:
            np.array: Predictions
        """
        # Handle missing values in test features
        X_test_clean = X_test.copy()
        
        if X_test_clean.isnull().any().any():
            if hasattr(self, 'feature_imputer') and self.feature_imputer is not None:
                print("Warning: Found NaN values in test features. Using fitted imputer.")
                
                # Store column names and index
                columns = X_test_clean.columns
                index = X_test_clean.index
                
                # Apply the same imputation as used during training
                X_test_clean = pd.DataFrame(
                    self.feature_imputer.transform(X_test_clean),
                    columns=columns,
                    index=index
                )
            else:
                print("Warning: Found NaN values in test features but no imputer available. Using median imputation.")
                
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                
                # Store column names and index
                columns = X_test_clean.columns
                index = X_test_clean.index
                
                # Impute missing values
                X_test_clean = pd.DataFrame(
                    imputer.fit_transform(X_test_clean),
                    columns=columns,
                    index=index
                )
        
        if method == 'ensemble':
            if self.ensemble_weights is None:
                raise ValueError("Must optimize ensemble weights before ensemble prediction")
            
            pred = np.zeros(len(X_test_clean))
            for i, (name, model) in enumerate(self.models.items()):
                pred += self.ensemble_weights[i] * model.predict(X_test_clean)
            return pred
        
        elif method in self.models:
            return self.models[method].predict(X_test_clean)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def validate_adaptation(self, train_df, test_df, target_col, val_split=0.2):
        """
        Validate domain adaptation effectiveness using temporal split.
        
        Args:
            train_df (pd.DataFrame): Training data with features and target
            test_df (pd.DataFrame): Test data for distribution reference
            target_col (str): Name of target column
            val_split (float): Fraction of data for validation
            
        Returns:
            dict: Validation results comparing different methods
        """
        # Create temporal validation split
        train_df = train_df.sort_values(self.date_col)
        split_idx = int(len(train_df) * (1 - val_split))
        
        train_split = train_df.iloc[:split_idx]
        val_split_df = train_df.iloc[split_idx:]
        
        # Prepare features
        feature_cols = [col for col in train_df.columns if col not in [target_col, self.date_col]]
        
        X_train = train_split[feature_cols]
        y_train = train_split[target_col]
        X_val = val_split_df[feature_cols]
        y_val = val_split_df[target_col]
        
        # Compute weights
        self.compute_temporal_weights(train_split, test_df)
        self.compute_geographic_weights(train_split, test_df)
        self.compute_combined_weights()
        
        # Fit models
        self.fit_adaptation_models(X_train, y_train)
        
        # Optimize ensemble
        self.optimize_ensemble_weights(X_val, y_val)
        
        # Evaluate all methods using RMSE
        results = {}
        for method in ['baseline', 'temporal', 'geographic', 'combined', 'ensemble']:
            pred = self.predict(X_val, method=method)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            results[f'{method}_rmse'] = rmse
            
            # Calculate improvement over baseline
            if method != 'baseline':
                improvement = (results['baseline_rmse'] - rmse) / results['baseline_rmse'] * 100
                results[f'{method}_improvement_pct'] = improvement
        
        return results
    
    def get_adaptation_summary(self):
        """
        Get summary of computed weights and adaptation strategies.
        
        Returns:
            dict: Summary statistics
        """
        summary = {}
        
        if self.temporal_weights is not None:
            summary['temporal_weights'] = {
                'mean': self.temporal_weights.mean(),
                'std': self.temporal_weights.std(),
                'min': self.temporal_weights.min(),
                'max': self.temporal_weights.max()
            }
        
        if self.geographic_weights is not None:
            summary['geographic_weights'] = {
                'mean': self.geographic_weights.mean(),
                'std': self.geographic_weights.std(),
                'min': self.geographic_weights.min(),
                'max': self.geographic_weights.max()
            }
        
        if self.ensemble_weights is not None:
            summary['ensemble_weights'] = {
                'baseline': self.ensemble_weights[0],
                'temporal': self.ensemble_weights[1], 
                'geographic': self.ensemble_weights[2],
                'combined': self.ensemble_weights[3]
            }
        
        return summary


# Example usage
if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    
    # Create sample data with distribution shift
    n_train, n_test = 1000, 200
    
    # Training data (full year)
    train_dates = pd.date_range('2023-01-01', '2023-12-31', periods=n_train)
    train_data = {
        'date': train_dates,
        'latitude': np.random.normal(35, 5, n_train),
        'longitude': np.random.normal(-100, 10, n_train),
        'feature1': np.random.normal(0, 1, n_train),
        'feature2': np.random.normal(0, 1, n_train),
    }
    
    # Test data (mostly January, different location)
    test_dates = pd.date_range('2024-01-01', '2024-01-31', periods=n_test)
    test_data = {
        'date': test_dates,
        'latitude': np.random.normal(40, 3, n_test),  # Different location
        'longitude': np.random.normal(-90, 8, n_test),
        'feature1': np.random.normal(0.2, 1, n_test),
        'feature2': np.random.normal(-0.1, 1, n_test),
    }
    
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    # Add synthetic target (with seasonal/geographic effects)
    train_df['target'] = (
        np.sin(2 * np.pi * pd.to_datetime(train_df['date']).dt.dayofyear / 365) +
        (train_df['latitude'] - 35) * 0.1 +
        train_df['feature1'] * 2 +
        train_df['feature2'] * 1.5 +
        np.random.normal(0, 0.5, n_train)
    )
    
    # Initialize and run domain adaptation
    adapter = DomainAdapter()
    
    # Add temporal features
    train_df = adapter.create_temporal_features(train_df)
    test_df = adapter.create_temporal_features(test_df)
    
    # Validate adaptation
    results = adapter.validate_adaptation(train_df, test_df, 'target')
    
    print("Domain Adaptation Validation Results:")
    print("="*50)
    for method, value in results.items():
        if 'improvement' in method:
            print(f"{method}: {value:.2f}%")
        elif 'mse' in method:
            print(f"{method}: {value:.4f}")
    
    print("\nWeight Summary:")
    print("="*50)
    summary = adapter.get_adaptation_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")