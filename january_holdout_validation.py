import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import warnings

class JanuaryHoldoutValidator:
    """
    Specialized holdout validation focused on January patterns for the air pollution competition.
    
    Creates dedicated holdout sets that:
    1. Contain primarily January data (matching test set bias)
    2. Are spatially and temporally separated from training data
    3. Provide reliable estimates of test set performance
    4. Include multiple holdout strategies for robust validation
    """
    
    def __init__(self,
                 holdout_size: float = 0.15,           # 15% of data for holdout
                 january_focus_ratio: float = 0.85,    # 85% of holdout should be January
                 spatial_buffer: float = 0.15,         # Spatial buffer in degrees
                 temporal_buffer: int = 45,             # Temporal buffer in days
                 min_holdout_size: int = 100,           # Minimum holdout size
                 random_state: Optional[int] = None):
        
        self.holdout_size = holdout_size
        self.january_focus_ratio = january_focus_ratio
        self.spatial_buffer = spatial_buffer
        self.temporal_buffer = temporal_buffer
        self.min_holdout_size = min_holdout_size
        self.random_state = random_state
        
        # Validation
        if not (0 < holdout_size < 1):
            raise ValueError("holdout_size must be between 0 and 1")
        if not (0 <= january_focus_ratio <= 1):
            raise ValueError("january_focus_ratio must be between 0 and 1")
    
    def create_january_focused_holdout(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a holdout set focused on January data to match test distribution.
        
        Returns
        -------
        train_idx : ndarray
            Training set indices
        holdout_idx : ndarray
            Holdout set indices (January-focused)
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        required_cols = ['month', 'latitude', 'longitude', 'day_of_year']
        missing_cols = [col for col in required_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Calculate target holdout size
        target_holdout_size = max(self.min_holdout_size, int(len(X) * self.holdout_size))
        target_january_size = int(target_holdout_size * self.january_focus_ratio)
        target_other_size = target_holdout_size - target_january_size
        
        # Get January and non-January data
        january_mask = X['month'] == 1
        january_indices = X[january_mask].index.values
        other_indices = X[~january_mask].index.values
        
        # Check data availability
        if len(january_indices) < target_january_size:
            warnings.warn(f"Insufficient January data. Available: {len(january_indices)}, "
                         f"needed: {target_january_size}. Using all available January data.")
            target_january_size = len(january_indices)
            target_other_size = max(0, target_holdout_size - target_january_size)
        
        if len(other_indices) < target_other_size:
            warnings.warn(f"Insufficient non-January data. Available: {len(other_indices)}, "
                         f"needed: {target_other_size}. Using all available non-January data.")
            target_other_size = len(other_indices)
        
        # Sample holdout data
        np.random.seed(self.random_state)
        
        # Sample January data for holdout
        if target_january_size > 0:
            holdout_january = np.random.choice(january_indices, target_january_size, replace=False)
        else:
            holdout_january = np.array([])
        
        # Sample other months for holdout (to maintain some diversity)
        if target_other_size > 0 and len(other_indices) > 0:
            holdout_other = np.random.choice(other_indices, target_other_size, replace=False)
        else:
            holdout_other = np.array([])
        
        # Combine holdout indices
        holdout_idx = np.concatenate([holdout_january, holdout_other])
        
        # Apply temporal buffer
        buffered_indices = self._apply_temporal_buffer(X, holdout_idx)
        
        # Apply spatial buffer
        if self.spatial_buffer > 0:
            buffered_indices = self._apply_spatial_buffer(X, holdout_idx, buffered_indices)
        
        # Create training set (exclude holdout and buffer)
        train_idx = np.setdiff1d(X.index.values, buffered_indices)
        
        return train_idx, holdout_idx
    
    def create_multiple_january_holdouts(self, X: pd.DataFrame, y: pd.Series, 
                                       n_holdouts: int = 3) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Create multiple January-focused holdout sets for robust validation.
        
        Parameters
        ----------
        n_holdouts : int
            Number of different holdout sets to create
            
        Returns
        -------
        dict
            Dictionary with holdout names as keys and (train_idx, holdout_idx) as values
        """
        holdouts = {}
        
        # Create different types of January holdouts
        strategies = [
            ('high_january', 0.95),     # 95% January
            ('medium_january', 0.85),   # 85% January  
            ('balanced_january', 0.70), # 70% January
        ]
        
        for i, (name, jan_ratio) in enumerate(strategies[:n_holdouts]):
            # Temporarily modify january_focus_ratio
            original_ratio = self.january_focus_ratio
            self.january_focus_ratio = jan_ratio
            
            # Use different random seed for each holdout
            original_seed = self.random_state
            if self.random_state is not None:
                self.random_state = self.random_state + i * 1000
            
            try:
                train_idx, holdout_idx = self.create_january_focused_holdout(X, y)
                holdouts[name] = (train_idx, holdout_idx)
            finally:
                # Restore original values
                self.january_focus_ratio = original_ratio
                self.random_state = original_seed
        
        return holdouts
    
    def create_test_distribution_holdout(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create holdout that exactly matches test set distribution (99.96% January).
        """
        target_holdout_size = max(self.min_holdout_size, int(len(X) * self.holdout_size))
        
        # Calculate exact test distribution
        target_january_size = int(target_holdout_size * 0.9996)
        target_february_size = max(1, int(target_holdout_size * 0.0004))
        
        # Adjust if needed
        if target_january_size + target_february_size > target_holdout_size:
            target_february_size = target_holdout_size - target_january_size
        
        # Sample data
        january_indices = X[X['month'] == 1].index.values
        february_indices = X[X['month'] == 2].index.values
        
        np.random.seed(self.random_state)
        
        # Sample January data
        if len(january_indices) >= target_january_size:
            holdout_january = np.random.choice(january_indices, target_january_size, replace=False)
        else:
            warnings.warn("Insufficient January data for exact test distribution matching")
            holdout_january = january_indices
        
        # Sample February data  
        if len(february_indices) >= target_february_size and target_february_size > 0:
            holdout_february = np.random.choice(february_indices, target_february_size, replace=False)
        else:
            holdout_february = np.array([])
        
        # Combine holdout indices
        holdout_idx = np.concatenate([holdout_january, holdout_february])
        
        # Apply buffers
        buffered_indices = self._apply_temporal_buffer(X, holdout_idx)
        if self.spatial_buffer > 0:
            buffered_indices = self._apply_spatial_buffer(X, holdout_idx, buffered_indices)
        
        # Create training set
        train_idx = np.setdiff1d(X.index.values, buffered_indices)
        
        return train_idx, holdout_idx
    
    def _apply_temporal_buffer(self, X: pd.DataFrame, holdout_idx: np.ndarray) -> np.ndarray:
        """Apply temporal buffer around holdout set."""
        holdout_days = X.loc[holdout_idx, 'day_of_year'].values
        buffer_indices = set(holdout_idx)
        
        for day in holdout_days:
            # Create buffer range handling year wraparound
            buffer_range = []
            for d in range(-self.temporal_buffer, self.temporal_buffer + 1):
                buffered_day = day + d
                if buffered_day <= 0:
                    buffered_day += 365
                elif buffered_day > 365:
                    buffered_day -= 365
                buffer_range.append(buffered_day)
            
            # Add indices within buffer range
            day_mask = X['day_of_year'].isin(buffer_range)
            buffer_indices.update(X[day_mask].index.tolist())
        
        return np.array(list(buffer_indices))
    
    def _apply_spatial_buffer(self, X: pd.DataFrame, holdout_idx: np.ndarray, 
                            current_buffer: np.ndarray) -> np.ndarray:
        """Apply spatial buffer around holdout set."""
        from scipy.spatial.distance import cdist
        
        holdout_coords = X.loc[holdout_idx, ['latitude', 'longitude']].values
        all_coords = X[['latitude', 'longitude']].values
        
        # Calculate distances
        distances = cdist(all_coords, holdout_coords)
        min_distances = np.min(distances, axis=1)
        
        # Points within buffer distance
        spatial_buffer_mask = min_distances <= self.spatial_buffer
        spatial_buffer_indices = X.index[spatial_buffer_mask].values
        
        # Combine with existing buffer
        return np.union1d(current_buffer, spatial_buffer_indices)
    
    def validate_holdout_quality(self, X: pd.DataFrame, holdout_idx: np.ndarray) -> Dict:
        """
        Validate the quality of a holdout set.
        
        Returns
        -------
        dict
            Quality metrics for the holdout set
        """
        if len(holdout_idx) == 0:
            return {'quality': 'Poor', 'reason': 'Empty holdout set'}
        
        holdout_data = X.loc[holdout_idx]
        
        # Calculate distributions
        january_ratio = (holdout_data['month'] == 1).mean()
        month_distribution = holdout_data['month'].value_counts(normalize=True).to_dict()
        
        # Spatial coverage
        lat_range = holdout_data['latitude'].max() - holdout_data['latitude'].min()
        lon_range = holdout_data['longitude'].max() - holdout_data['longitude'].min()
        
        # Temporal coverage
        day_range = holdout_data['day_of_year'].max() - holdout_data['day_of_year'].min()
        
        # Quality assessment
        quality_score = 0
        reasons = []
        
        # January focus (higher score for more January data, matching test set)
        if january_ratio >= 0.95:
            quality_score += 30
            reasons.append("Excellent January focus")
        elif january_ratio >= 0.80:
            quality_score += 20
            reasons.append("Good January focus")
        elif january_ratio >= 0.60:
            quality_score += 10
            reasons.append("Moderate January focus")
        else:
            reasons.append("Poor January focus")
        
        # Size adequacy
        if len(holdout_idx) >= self.min_holdout_size:
            quality_score += 20
            reasons.append("Adequate size")
        else:
            reasons.append("Small holdout size")
        
        # Spatial diversity
        if lat_range > 10 and lon_range > 10:
            quality_score += 20
            reasons.append("Good spatial diversity")
        elif lat_range > 5 and lon_range > 5:
            quality_score += 10
            reasons.append("Moderate spatial diversity")
        else:
            reasons.append("Limited spatial diversity")
        
        # Temporal focus (January should be concentrated)
        if day_range <= 60:  # January-focused
            quality_score += 20
            reasons.append("Good temporal focus")
        elif day_range <= 120:
            quality_score += 10
            reasons.append("Moderate temporal focus")
        else:
            reasons.append("Poor temporal focus")
        
        # Overall quality
        if quality_score >= 80:
            overall_quality = "Excellent"
        elif quality_score >= 60:
            overall_quality = "Good"
        elif quality_score >= 40:
            overall_quality = "Fair"
        else:
            overall_quality = "Poor"
        
        return {
            'quality': overall_quality,
            'score': quality_score,
            'holdout_size': len(holdout_idx),
            'january_ratio': january_ratio,
            'month_distribution': month_distribution,
            'spatial_range': {'latitude': lat_range, 'longitude': lon_range},
            'temporal_range': day_range,
            'reasons': reasons
        }
    
    def evaluate_model_on_holdouts(self, model, X: pd.DataFrame, y: pd.Series, 
                                 holdouts: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict:
        """
        Evaluate a trained model on multiple holdout sets.
        
        Parameters
        ----------
        model : sklearn-compatible model
            Trained model to evaluate
        X : DataFrame
            Feature data
        y : Series
            Target data
        holdouts : dict
            Dictionary of holdout sets from create_multiple_january_holdouts
            
        Returns
        -------
        dict
            Evaluation results for each holdout set
        """
        results = {}
        
        for holdout_name, (train_idx, holdout_idx) in holdouts.items():
            # Get holdout data
            X_holdout = X.iloc[holdout_idx]
            y_holdout = y.iloc[holdout_idx]
            
            try:
                # Make predictions
                y_pred = model.predict(X_holdout)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_holdout, y_pred))
                mae = np.mean(np.abs(y_holdout - y_pred))
                
                # Calculate January-specific metrics
                january_mask = X_holdout['month'] == 1
                if january_mask.any():
                    y_jan_true = y_holdout[january_mask]
                    y_jan_pred = y_pred[january_mask]
                    january_rmse = np.sqrt(mean_squared_error(y_jan_true, y_jan_pred))
                else:
                    january_rmse = None
                
                results[holdout_name] = {
                    'rmse': rmse,
                    'mae': mae,
                    'january_rmse': january_rmse,
                    'holdout_size': len(holdout_idx),
                    'january_samples': january_mask.sum(),
                    'january_ratio': january_mask.mean()
                }
                
            except Exception as e:
                results[holdout_name] = {
                    'error': str(e),
                    'holdout_size': len(holdout_idx)
                }
        
        return results


# Convenience function
def create_january_holdout_validator(holdout_size: float = 0.15, 
                                   random_state: int = 42) -> JanuaryHoldoutValidator:
    """
    Create a January holdout validator optimized for the air pollution competition.
    """
    return JanuaryHoldoutValidator(
        holdout_size=holdout_size,
        january_focus_ratio=0.85,      # 85% January data
        spatial_buffer=0.15,           # 15% of coordinate range
        temporal_buffer=45,            # 45-day buffer
        min_holdout_size=100,
        random_state=random_state
    )

# Example usage
if __name__ == "__main__":
    """
    # Example usage
    train_df = pd.read_csv('train.csv')
    X = train_df[['latitude', 'longitude', 'day_of_year', 'month', 'hour']]
    y = train_df['pollution_value']
    
    # Create holdout validator
    validator = create_january_holdout_validator(holdout_size=0.15, random_state=42)
    
    # Create multiple holdout sets
    holdouts = validator.create_multiple_january_holdouts(X, y, n_holdouts=3)
    
    # Validate quality of each holdout
    for name, (train_idx, holdout_idx) in holdouts.items():
        quality = validator.validate_holdout_quality(X, holdout_idx)
        print(f"{name}: {quality['quality']} (January ratio: {quality['january_ratio']:.3f})")
    
    # Use one holdout for model evaluation
    train_idx, holdout_idx = holdouts['high_january']
    X_train, X_holdout = X.iloc[train_idx], X.iloc[holdout_idx]
    y_train, y_holdout = y.iloc[train_idx], y.iloc[holdout_idx]
    
    # Train your model on X_train, y_train
    # Evaluate on X_holdout, y_holdout
    """
    pass