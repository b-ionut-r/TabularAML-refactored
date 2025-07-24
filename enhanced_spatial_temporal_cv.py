import numpy as np
import pandas as pd
from typing import Generator, Tuple, Optional, Union
from spatial_temporal_cv import SpatialTemporalKFold
import warnings

class EnhancedSpatialTemporalKFold(SpatialTemporalKFold):
    """
    Enhanced version of SpatialTemporalKFold specifically optimized for the 
    air pollution competition with extreme January bias in test set.
    
    Key improvements:
    - January-aware temporal buffering
    - Adaptive clustering based on data distribution
    - Enhanced validation metrics
    - Competition-specific parameter optimization
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 spatial_clusters: int = 12,  # Reduced for high overlap data
                 temporal_clusters: int = 4,   # Reduced for January concentration
                 lat_col: str = 'latitude',
                 lon_col: str = 'longitude',
                 time_cols: Optional[dict] = None,
                 stratify: bool = True,
                 n_quantiles: int = 3,        # Simplified stratification
                 buffer_distance: float = 0.2, # Increased for better separation
                 temporal_buffer: int = 60,    # 2-month buffer for seasonal patterns
                 january_aware: bool = True,   # New: January-specific handling
                 january_buffer: int = 90,     # New: Larger buffer around January
                 exclude_adjacent_months: bool = True, # New: Exclude Nov-Feb when validating January
                 adaptive_clustering: bool = True,     # New: Adapt clusters to data
                 random_state: Optional[int] = None,
                 shuffle: bool = True):
        
        # Initialize parent class with optimized parameters
        super().__init__(
            n_splits=n_splits,
            spatial_clusters=spatial_clusters,
            temporal_clusters=temporal_clusters,
            lat_col=lat_col,
            lon_col=lon_col,
            time_cols=time_cols,
            stratify=stratify,
            n_quantiles=n_quantiles,
            buffer_distance=buffer_distance,
            temporal_buffer=temporal_buffer,
            random_state=random_state,
            shuffle=shuffle
        )
        
        # New competition-specific parameters
        self.january_aware = january_aware
        self.january_buffer = january_buffer
        self.exclude_adjacent_months = exclude_adjacent_months
        self.adaptive_clustering = adaptive_clustering
        
    def _analyze_data_distribution(self, X: pd.DataFrame) -> dict:
        """
        Analyze data distribution to optimize CV parameters.
        """
        analysis = {
            'total_samples': len(X),
            'january_ratio': (X['month'] == 1).mean() if 'month' in X.columns else 0,
            'spatial_range': {
                'lat_range': X[self.lat_col].max() - X[self.lat_col].min(),
                'lon_range': X[self.lon_col].max() - X[self.lon_col].min()
            },
            'temporal_distribution': X.groupby('month').size().to_dict() if 'month' in X.columns else {},
            'unique_spatial_locations': len(X[[self.lat_col, self.lon_col]].drop_duplicates())
        }
        
        return analysis
    
    def _adapt_clustering_parameters(self, X: pd.DataFrame) -> Tuple[int, int]:
        """
        Adapt clustering parameters based on data characteristics.
        """
        if not self.adaptive_clustering:
            return self.spatial_clusters, self.temporal_clusters
            
        analysis = self._analyze_data_distribution(X)
        
        # Adapt spatial clusters based on unique locations and sample size
        unique_locations = analysis['unique_spatial_locations']
        adapted_spatial = min(
            self.spatial_clusters,
            max(self.n_splits, unique_locations // 10),  # At least 10 samples per cluster
            len(X) // (self.n_splits * 5)  # At least 5 samples per split per cluster
        )
        
        # Adapt temporal clusters based on January concentration
        january_ratio = analysis['january_ratio']
        if january_ratio > 0.5:  # High January concentration
            adapted_temporal = max(2, self.temporal_clusters // 2)
        else:
            adapted_temporal = self.temporal_clusters
            
        if adapted_spatial != self.spatial_clusters or adapted_temporal != self.temporal_clusters:
            warnings.warn(f"Adapted clustering parameters: spatial {self.spatial_clusters}->{adapted_spatial}, "
                         f"temporal {self.temporal_clusters}->{adapted_temporal}")
        
        return adapted_spatial, adapted_temporal
    
    def _create_january_aware_temporal_buffer(self, X: pd.DataFrame, val_indices: np.ndarray) -> np.ndarray:
        """
        Create January-aware temporal buffer that accounts for year wraparound.
        """
        if not self.january_aware or 'month' not in X.columns:
            return self._create_standard_temporal_buffer(X, val_indices)
        
        val_data = X.iloc[val_indices]
        buffer_indices = []
        
        # Check if validation set contains January data
        has_january = (val_data['month'] == 1).any()
        
        if has_january and self.exclude_adjacent_months:
            # Exclude November, December, January, February from training
            # when validating on January data
            adjacent_months = [11, 12, 1, 2]
            buffer_mask = X['month'].isin(adjacent_months)
            buffer_indices.extend(X[buffer_mask].index.tolist())
        
        # Add standard temporal buffer
        standard_buffer = self._create_standard_temporal_buffer(X, val_indices)
        buffer_indices.extend(standard_buffer.tolist())
        
        return np.unique(buffer_indices)
    
    def _create_standard_temporal_buffer(self, X: pd.DataFrame, val_indices: np.ndarray) -> np.ndarray:
        """
        Create standard temporal buffer around validation indices.
        """
        if 'day_of_year' not in X.columns:
            return np.array([])
            
        val_days = X.iloc[val_indices]['day_of_year'].values
        buffer_indices = []
        
        # Use January-specific buffer if applicable
        buffer_days = self.january_buffer if self.january_aware and (X.iloc[val_indices]['month'] == 1).any() else self.temporal_buffer
        
        for day in val_days:
            # Handle year wraparound (day 1 is close to day 365)
            buffer_range = []
            for d in range(-buffer_days, buffer_days + 1):
                buffered_day = day + d
                if buffered_day <= 0:
                    buffered_day += 365
                elif buffered_day > 365:
                    buffered_day -= 365
                buffer_range.append(buffered_day)
            
            day_mask = X['day_of_year'].isin(buffer_range)
            buffer_indices.extend(X[day_mask].index.tolist())
        
        return np.unique(buffer_indices)
    
    def split(self, X: Union[pd.DataFrame, np.ndarray], 
              y: Optional[Union[pd.Series, np.ndarray]] = None, 
              groups: Optional[np.ndarray] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Enhanced split method with competition-specific optimizations.
        """
        X = self._validate_data(X, y)
        
        # Adapt clustering parameters based on data
        original_spatial, original_temporal = self.spatial_clusters, self.temporal_clusters
        self.spatial_clusters, self.temporal_clusters = self._adapt_clustering_parameters(X)
        
        try:
            # Get base splits from parent class
            for fold_idx, (train_idx, val_idx) in enumerate(super().split(X, y, groups)):
                
                if self.january_aware:
                    # Apply January-aware temporal buffering
                    temporal_buffer_idx = self._create_january_aware_temporal_buffer(X, val_idx)
                    
                    # Remove buffer indices from training set
                    train_idx = np.setdiff1d(train_idx, temporal_buffer_idx)
                    
                    # Validate split quality
                    if len(train_idx) < len(X) * 0.1:  # Less than 10% training data
                        warnings.warn(f"Fold {fold_idx} has very small training set ({len(train_idx)} samples)")
                        continue
                        
                    if len(val_idx) < len(X) * 0.05:  # Less than 5% validation data
                        warnings.warn(f"Fold {fold_idx} has very small validation set ({len(val_idx)} samples)")
                        continue
                
                yield train_idx, val_idx
                
        finally:
            # Restore original parameters
            self.spatial_clusters, self.temporal_clusters = original_spatial, original_temporal
    
    def get_enhanced_split_info(self, X: Union[pd.DataFrame, np.ndarray], 
                               y: Optional[Union[pd.Series, np.ndarray]] = None) -> dict:
        """
        Get detailed information about splits with competition-specific metrics.
        """
        base_info = self.get_split_info(X, y)
        X = self._validate_data(X, y)
        
        # Add competition-specific analysis
        enhanced_info = {
            **base_info,
            'data_analysis': self._analyze_data_distribution(X),
            'january_aware_settings': {
                'january_aware': self.january_aware,
                'january_buffer': self.january_buffer,
                'exclude_adjacent_months': self.exclude_adjacent_months,
                'adaptive_clustering': self.adaptive_clustering
            }
        }
        
        # Analyze January distribution across folds
        january_stats = []
        for train_idx, val_idx in self.split(X, y):
            train_january_ratio = (X.iloc[train_idx]['month'] == 1).mean() if 'month' in X.columns else 0
            val_january_ratio = (X.iloc[val_idx]['month'] == 1).mean() if 'month' in X.columns else 0
            
            january_stats.append({
                'train_january_ratio': train_january_ratio,
                'val_january_ratio': val_january_ratio,
                'train_size': len(train_idx),
                'val_size': len(val_idx)
            })
        
        enhanced_info['january_distribution'] = {
            'fold_stats': january_stats,
            'avg_train_january_ratio': np.mean([s['train_january_ratio'] for s in january_stats]),
            'avg_val_january_ratio': np.mean([s['val_january_ratio'] for s in january_stats]),
            'january_separation_quality': self._assess_january_separation_quality(january_stats)
        }
        
        return enhanced_info
    
    def _assess_january_separation_quality(self, january_stats: list) -> str:
        """
        Assess the quality of January temporal separation.
        """
        avg_train_jan = np.mean([s['train_january_ratio'] for s in january_stats])
        avg_val_jan = np.mean([s['val_january_ratio'] for s in january_stats])
        
        # Good separation if validation has more January data than training
        # (simulates test set bias toward January)
        if avg_val_jan > avg_train_jan * 1.5:
            return "Good - Validation biased toward January (matches test distribution)"
        elif avg_val_jan > avg_train_jan:
            return "Fair - Some January bias in validation"
        else:
            return "Poor - No January bias in validation (may not represent test set)"

# Competition-optimized configuration
def create_competition_optimized_cv(n_splits: int = 5, random_state: int = 42) -> EnhancedSpatialTemporalKFold:
    """
    Create competition-optimized SpatialTemporalKFold configuration.
    """
    return EnhancedSpatialTemporalKFold(
        n_splits=n_splits,
        spatial_clusters=12,           # Optimized for competition data
        temporal_clusters=4,           # Reduced for January concentration
        buffer_distance=0.2,           # 20% of coordinate range
        temporal_buffer=60,            # 2-month standard buffer
        january_aware=True,            # Enable January-specific handling
        january_buffer=90,             # 3-month buffer around January
        exclude_adjacent_months=True,   # Exclude Nov-Feb when validating January
        adaptive_clustering=True,       # Adapt to data characteristics
        stratify=True,
        n_quantiles=3,                 # Simplified stratification
        random_state=random_state
    )

# Example usage and validation
if __name__ == "__main__":
    # Example of how to use the enhanced CV
    """
    # Load your data
    train_df = pd.read_csv('train.csv')
    X = train_df[['latitude', 'longitude', 'day_of_year', 'day_of_week', 'hour', 'month']]
    y = train_df['pollution_value']
    
    # Create enhanced CV
    cv = create_competition_optimized_cv(n_splits=5, random_state=42)
    
    # Get detailed information
    info = cv.get_enhanced_split_info(X, y)
    print("CV Strategy Information:")
    print(f"January separation quality: {info['january_distribution']['january_separation_quality']}")
    print(f"Average validation January ratio: {info['january_distribution']['avg_val_january_ratio']:.3f}")
    
    # Use in model validation
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train your model here
        print(f"Fold {fold_idx}: Train size: {len(train_idx)}, Val size: {len(val_idx)}")
    """
    pass