import numpy as np
import pandas as pd
from typing import Generator, Tuple, List, Dict
from sklearn.model_selection import StratifiedKFold
from spatial_temporal_cv import SpatialTemporalKFold
import warnings

class ImprovedCompetitionCV:
    """
    Enhanced CV strategy specifically designed for the air pollution competition
    with extreme January bias in test set (99.96% January data).
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 random_state: int = 42,
                 use_january_holdout: bool = True,
                 january_holdout_size: float = 0.15):
        
        self.n_splits = n_splits
        self.random_state = random_state
        self.use_january_holdout = use_january_holdout
        self.january_holdout_size = january_holdout_size
        
        # Initialize multiple CV strategies
        self.spatial_temporal_cv = SpatialTemporalKFold(
            n_splits=n_splits,
            spatial_clusters=12,      # Reduced due to high spatial overlap
            temporal_clusters=4,      # Reduced due to January concentration  
            buffer_distance=0.2,      # 20% of coordinate range for strong separation
            temporal_buffer=60,       # 2-month buffer to avoid seasonal leakage
            stratify=True,
            n_quantiles=3,           # Simpler stratification
            random_state=random_state
        )
        
    def create_january_focused_cv(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create CV splits that mirror test set's January bias (99.96% January).
        """
        january_mask = X['month'] == 1
        january_indices = X[january_mask].index.values
        other_indices = X[~january_mask].index.values
        
        splits = []
        
        for fold in range(self.n_splits):
            # Split January data for validation (mirrors test distribution)
            np.random.seed(self.random_state + fold)
            
            # Take 20% of January data for validation
            n_jan_val = len(january_indices) // self.n_splits
            start_idx = fold * n_jan_val
            end_idx = (fold + 1) * n_jan_val if fold < self.n_splits - 1 else len(january_indices)
            
            val_january = january_indices[start_idx:end_idx]
            
            # Add tiny fraction of other months (0.04% like test set)
            n_other_val = max(1, len(val_january) // 250)  # 0.4% ratio
            if len(other_indices) >= n_other_val:
                val_other = np.random.choice(other_indices, n_other_val, replace=False)
                val_idx = np.concatenate([val_january, val_other])
            else:
                val_idx = val_january
            
            # Training set excludes validation and nearby temporal data
            buffer_mask = self._create_temporal_buffer_mask(X, val_idx, buffer_days=45)
            train_idx = X.index.difference(pd.Index(val_idx).union(pd.Index(buffer_mask))).values
            
            splits.append((train_idx, val_idx))
            
        return splits
    
    def create_january_holdout(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a dedicated January holdout set for final validation.
        """
        january_mask = X['month'] == 1
        january_indices = X[january_mask].index.values
        
        np.random.seed(self.random_state)
        n_holdout = int(len(january_indices) * self.january_holdout_size)
        holdout_idx = np.random.choice(january_indices, n_holdout, replace=False)
        
        # Training set excludes holdout and temporal buffer
        buffer_mask = self._create_temporal_buffer_mask(X, holdout_idx, buffer_days=30)
        train_idx = X.index.difference(pd.Index(holdout_idx).union(pd.Index(buffer_mask))).values
        
        return train_idx, holdout_idx
    
    def _create_temporal_buffer_mask(self, X: pd.DataFrame, target_indices: np.ndarray, buffer_days: int) -> np.ndarray:
        """
        Create mask for temporal buffer around target indices.
        """
        target_days = X.loc[target_indices, 'day_of_year'].values
        buffer_mask = []
        
        for day in target_days:
            # Handle year wraparound
            buffer_range = []
            for d in range(-buffer_days, buffer_days + 1):
                buffered_day = day + d
                if buffered_day <= 0:
                    buffered_day += 365
                elif buffered_day > 365:
                    buffered_day -= 365
                buffer_range.append(buffered_day)
            
            day_mask = X['day_of_year'].isin(buffer_range)
            buffer_mask.extend(X[day_mask].index.tolist())
        
        return np.unique(buffer_mask)
    
    def create_geographic_stratified_cv(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create geographically stratified CV splits.
        """
        # Create geographic regions based on lat/lon
        lat_bins = pd.cut(X['latitude'], bins=4, labels=['South', 'South-Mid', 'North-Mid', 'North'])
        lon_bins = pd.cut(X['longitude'], bins=4, labels=['West', 'West-Mid', 'East-Mid', 'East'])
        geo_strata = lat_bins.astype(str) + '_' + lon_bins.astype(str)
        
        # Use stratified CV on geographic regions
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        splits = []
        for train_idx, val_idx in skf.split(X, geo_strata):
            splits.append((train_idx, val_idx))
        
        return splits
    
    def get_comprehensive_cv_strategy(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Get all CV strategies for comprehensive model validation.
        """
        strategies = {}
        
        # 1. Enhanced Spatial-Temporal CV
        try:
            strategies['spatial_temporal'] = list(self.spatial_temporal_cv.split(X, y))
        except Exception as e:
            warnings.warn(f"Spatial-temporal CV failed: {e}")
            strategies['spatial_temporal'] = []
        
        # 2. January-focused CV (mirrors test distribution)
        strategies['january_focused'] = self.create_january_focused_cv(X, y)
        
        # 3. Geographic stratified CV
        strategies['geographic_stratified'] = self.create_geographic_stratified_cv(X, y)
        
        # 4. January holdout for final validation
        if self.use_january_holdout:
            train_idx, holdout_idx = self.create_january_holdout(X, y)
            strategies['january_holdout'] = [(train_idx, holdout_idx)]
        
        return strategies
    
    def recommend_primary_cv(self, X: pd.DataFrame, y: pd.Series) -> str:
        """
        Recommend the primary CV strategy based on data characteristics.
        """
        january_ratio = (X['month'] == 1).mean()
        
        if january_ratio > 0.4:  # High January concentration
            return 'january_focused'
        elif len(X) > 5000:  # Large dataset
            return 'spatial_temporal' 
        else:
            return 'geographic_stratified'

# Example usage
def setup_competition_cv(X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Setup the complete CV strategy for the air pollution competition.
    """
    cv_manager = ImprovedCompetitionCV(
        n_splits=5,
        random_state=42,
        use_january_holdout=True
    )
    
    # Get all CV strategies
    cv_strategies = cv_manager.get_comprehensive_cv_strategy(X, y)
    
    # Get recommendation
    primary_strategy = cv_manager.recommend_primary_cv(X, y)
    
    return {
        'cv_strategies': cv_strategies,
        'primary_strategy': primary_strategy,
        'cv_manager': cv_manager
    }

# Validation function
def validate_cv_quality(cv_strategies: Dict, X: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Validate the quality of CV strategies.
    """
    results = {}
    
    for strategy_name, splits in cv_strategies.items():
        if not splits:
            continue
            
        fold_stats = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            # Calculate statistics for this fold
            train_january_ratio = (X.loc[train_idx, 'month'] == 1).mean()
            val_january_ratio = (X.loc[val_idx, 'month'] == 1).mean()
            
            spatial_separation = calculate_spatial_separation(
                X.loc[train_idx, ['latitude', 'longitude']],
                X.loc[val_idx, ['latitude', 'longitude']]
            )
            
            fold_stats.append({
                'fold': fold_idx,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'train_january_ratio': train_january_ratio,
                'val_january_ratio': val_january_ratio,
                'spatial_separation': spatial_separation
            })
        
        results[strategy_name] = {
            'n_folds': len(splits),
            'fold_stats': fold_stats,
            'avg_val_january_ratio': np.mean([s['val_january_ratio'] for s in fold_stats]),
            'avg_spatial_separation': np.mean([s['spatial_separation'] for s in fold_stats])
        }
    
    return results

def calculate_spatial_separation(train_coords: pd.DataFrame, val_coords: pd.DataFrame) -> float:
    """
    Calculate minimum spatial separation between train and validation sets.
    """
    from scipy.spatial.distance import cdist
    
    if len(train_coords) == 0 or len(val_coords) == 0:
        return 0.0
    
    # Sample for efficiency if datasets are large
    if len(train_coords) > 1000:
        train_coords = train_coords.sample(1000, random_state=42)
    if len(val_coords) > 1000:
        val_coords = val_coords.sample(1000, random_state=42)
    
    distances = cdist(train_coords.values, val_coords.values)
    return np.min(distances)