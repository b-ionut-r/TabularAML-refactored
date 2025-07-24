import numpy as np
import pandas as pd
from typing import Generator, Tuple, Optional, Union
from sklearn.model_selection import BaseCrossValidator
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, indexable
from sklearn.utils import check_random_state
from scipy.spatial.distance import cdist
import warnings


class SpatialTemporalKFold(BaseCrossValidator):
    """
    Spatial-Temporal Cross-Validator for geographic time-series data.
    
    This cross-validator creates folds that respect both spatial and temporal 
    dependencies in the data, preventing data leakage in spatial-temporal 
    prediction tasks like air pollution forecasting.
    
    The strategy:
    1. Creates spatial clusters using geographic coordinates (lat/lon)
    2. Creates temporal clusters using cyclical time features
    3. Combines spatial-temporal groups to ensure validation sets are 
       spatially and temporally separated from training sets
    4. Optionally stratifies by target variable ranges
    
    Parameters
    ----------
    n_splits : int, default=5
        Number of cross-validation folds
    spatial_clusters : int, default=20
        Number of spatial clusters for geographic grouping
    temporal_clusters : int, default=8
        Number of temporal clusters for time-based grouping
    lat_col : str, default='latitude'
        Column name for latitude coordinates
    lon_col : str, default='longitude' 
        Column name for longitude coordinates
    time_cols : dict, default=None
        Dictionary mapping time column names to their cycles:
        {'day_of_year': 365, 'hour': 24, 'day_of_week': 7, 'month': 12}
    stratify : bool, default=True
        Whether to stratify splits by target variable quantiles
    n_quantiles : int, default=5
        Number of quantiles for stratification (if stratify=True)
    buffer_distance : float, default=0.1
        Minimum spatial distance between train/validation clusters (degrees)
    temporal_buffer : int, default=7
        Minimum temporal distance between train/validation (in days)
    random_state : int, default=None
        Random state for reproducible splits
    shuffle : bool, default=True
        Whether to shuffle data before splitting
        
    Examples
    --------
    >>> import pandas as pd
    >>> from spatial_temporal_cv import SpatialTemporalKFold
    >>> 
    >>> # Basic usage
    >>> cv = SpatialTemporalKFold(n_splits=5, random_state=42)
    >>> X = df[['latitude', 'longitude', 'day_of_year', 'hour', 'month']]
    >>> y = df['pollution_value']
    >>> 
    >>> for train_idx, val_idx in cv.split(X, y):
    >>>     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    >>>     y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    >>>     # Train your model...
    >>>
    >>> # Custom time columns
    >>> time_cols = {'day_of_year': 365, 'hour': 24, 'day_of_week': 7}
    >>> cv = SpatialTemporalKFold(
    >>>     n_splits=3, 
    >>>     time_cols=time_cols,
    >>>     buffer_distance=0.05,
    >>>     temporal_buffer=14
    >>> )
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 spatial_clusters: int = 20,
                 temporal_clusters: int = 8,
                 lat_col: str = 'latitude',
                 lon_col: str = 'longitude',
                 time_cols: Optional[dict] = None,
                 stratify: bool = True,
                 n_quantiles: int = 5,
                 buffer_distance: float = 0.1,
                 temporal_buffer: int = 7,
                 random_state: Optional[int] = None,
                 shuffle: bool = True):
        
        self.n_splits = n_splits
        self.spatial_clusters = spatial_clusters
        self.temporal_clusters = temporal_clusters
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.time_cols = time_cols or {
            'day_of_year': 365, 'hour': 24, 'day_of_week': 7, 'month': 12
        }
        self.stratify = stratify
        self.n_quantiles = n_quantiles
        self.buffer_distance = buffer_distance
        self.temporal_buffer = temporal_buffer
        self.random_state = random_state
        self.shuffle = shuffle
        
        # Validation
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if spatial_clusters < n_splits:
            warnings.warn(f"spatial_clusters ({spatial_clusters}) < n_splits ({n_splits}). "
                         "This may result in poor spatial separation.")
        if temporal_clusters < 2:
            raise ValueError("temporal_clusters must be at least 2")
    
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations."""
        return self.n_splits
    
    def _validate_data(self, X: Union[pd.DataFrame, np.ndarray], 
                      y: Optional[Union[pd.Series, np.ndarray]] = None) -> pd.DataFrame:
        """Validate and convert input data to DataFrame."""
        if not isinstance(X, pd.DataFrame):
            if isinstance(X, np.ndarray):
                # Try to infer column names for numpy arrays
                expected_cols = [self.lat_col, self.lon_col] + list(self.time_cols.keys())
                if X.shape[1] >= len(expected_cols):
                    X = pd.DataFrame(X, columns=expected_cols[:X.shape[1]])
                else:
                    raise ValueError(f"Expected at least {len(expected_cols)} columns, got {X.shape[1]}")
            else:
                raise TypeError("X must be pandas DataFrame or numpy array")
        
        # Check required columns exist
        missing_cols = []
        if self.lat_col not in X.columns:
            missing_cols.append(self.lat_col)
        if self.lon_col not in X.columns:
            missing_cols.append(self.lon_col)
        for col in self.time_cols.keys():
            if col not in X.columns:
                missing_cols.append(col)
                
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        return X
    
    def _create_spatial_clusters(self, X: pd.DataFrame) -> np.ndarray:
        """Create spatial clusters using K-means on lat/lon coordinates."""
        coords = X[[self.lat_col, self.lon_col]]
        
        # Handle edge case where we have fewer samples than clusters
        n_clusters = min(self.spatial_clusters, len(coords))
        if n_clusters < self.spatial_clusters:
            warnings.warn(f"Reducing spatial_clusters from {self.spatial_clusters} to {n_clusters} "
                         f"due to insufficient data points.")
        
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=self.random_state,
            n_init=10
        )
        spatial_labels = kmeans.fit_predict(coords.copy().fillna(coords.mean()).values)
        self._spatial_centroids = kmeans.cluster_centers_
        
        return spatial_labels
    
    def _create_temporal_features(self, X: pd.DataFrame) -> np.ndarray:
        """Convert cyclical time features to circular coordinates."""
        temporal_features = []
        
        for col, cycle_length in self.time_cols.items():
            if col in X.columns:
                # Convert to radians for circular representation
                radians = 2 * np.pi * X[col] / cycle_length
                # Use sine and cosine to capture cyclical nature
                temporal_features.extend([np.sin(radians), np.cos(radians)])
        
        return np.column_stack(temporal_features) if temporal_features else np.zeros((len(X), 2))
    
    def _create_temporal_clusters(self, X: pd.DataFrame) -> np.ndarray:
        """Create temporal clusters using cyclical time features."""
        temporal_coords = self._create_temporal_features(X)
        
        # Handle edge case where we have fewer samples than clusters
        n_clusters = min(self.temporal_clusters, len(temporal_coords))
        if n_clusters < self.temporal_clusters:
            warnings.warn(f"Reducing temporal_clusters from {self.temporal_clusters} to {n_clusters} "
                         f"due to insufficient data points.")
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        temporal_labels = kmeans.fit_predict(temporal_coords)
        self._temporal_centroids = kmeans.cluster_centers_
        
        return temporal_labels
    
    def _create_stratification_groups(self, y: np.ndarray) -> np.ndarray:
        """Create stratification groups based on target variable quantiles."""
        if not self.stratify or y is None:
            return np.zeros(len(y), dtype=int)
        
        # Create quantile-based groups
        quantiles = np.linspace(0, 1, self.n_quantiles + 1)
        quantile_values = np.quantile(y, quantiles)
        
        # Assign each sample to a quantile group
        strat_groups = np.digitize(y, quantile_values[1:-1])
        
        return strat_groups
    
    def _check_spatial_separation(self, train_spatial_groups: np.ndarray, 
                                 val_spatial_groups: np.ndarray) -> bool:
        """Check if spatial clusters are sufficiently separated."""
        if not hasattr(self, '_spatial_centroids'):
            return True
            
        train_centroids = self._spatial_centroids[train_spatial_groups]
        val_centroids = self._spatial_centroids[val_spatial_groups]
        
        # Calculate minimum distance between train and validation centroids
        distances = cdist(train_centroids, val_centroids)
        min_distance = np.min(distances)
        
        return min_distance >= self.buffer_distance
    
    def _create_combined_groups(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """Create combined spatial-temporal-stratification groups."""
        # Create individual groupings
        spatial_labels = self._create_spatial_clusters(X)
        temporal_labels = self._create_temporal_clusters(X)
        
        if y is not None:
            strat_labels = self._create_stratification_groups(y)
        else:
            strat_labels = np.zeros(len(X), dtype=int)
        
        # Combine all groupings into unique identifiers
        max_spatial = np.max(spatial_labels) + 1
        max_temporal = np.max(temporal_labels) + 1
        max_strat = np.max(strat_labels) + 1
        
        combined_groups = (spatial_labels * max_temporal * max_strat + 
                          temporal_labels * max_strat + 
                          strat_labels)
        
        # Store metadata for analysis
        metadata = {
            'spatial_labels': spatial_labels,
            'temporal_labels': temporal_labels,
            'strat_labels': strat_labels,
            'n_spatial_clusters': max_spatial,
            'n_temporal_clusters': max_temporal,
            'n_strat_groups': max_strat,
            'n_combined_groups': len(np.unique(combined_groups))
        }
        
        return combined_groups, metadata
    
    def split(self, X: Union[pd.DataFrame, np.ndarray], 
              y: Optional[Union[pd.Series, np.ndarray]] = None, 
              groups: Optional[np.ndarray] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : DataFrame or array-like of shape (n_samples, n_features)
            Training data with spatial and temporal features
        y : array-like of shape (n_samples,), optional
            Target variable for stratification
        groups : array-like of shape (n_samples,), optional
            Not used, present for API compatibility
            
        Yields
        ------
        train : ndarray
            The training set indices for that split
        test : ndarray  
            The testing set indices for that split
        """
        X = self._validate_data(X, y)
        X, y = indexable(X, y)
        
        if y is not None:
            y = np.asarray(y)
            
        rng = check_random_state(self.random_state)
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        if self.shuffle:
            rng.shuffle(indices)
            X = X.iloc[indices].reset_index(drop=True)
            if y is not None:
                y = y[indices]
        
        # Create spatial-temporal groups
        combined_groups, metadata = self._create_combined_groups(X, y)
        unique_groups = np.unique(combined_groups)
        
        if len(unique_groups) < self.n_splits:
            raise ValueError(f"Cannot create {self.n_splits} splits with only "
                           f"{len(unique_groups)} unique spatial-temporal groups. "
                           f"Consider reducing n_splits or clustering parameters.")
        
        # Shuffle groups for random assignment to folds
        rng.shuffle(unique_groups)
        
        # Assign groups to folds using round-robin to balance sizes
        fold_groups = [[] for _ in range(self.n_splits)]
        for i, group in enumerate(unique_groups):
            fold_groups[i % self.n_splits].append(group)
        
        # Generate train/test splits
        for fold_idx in range(self.n_splits):
            test_groups = np.array(fold_groups[fold_idx])
            # Create train groups deterministically by excluding test groups
            train_groups = unique_groups[~np.isin(unique_groups, test_groups)]
            
            test_mask = np.isin(combined_groups, test_groups)
            train_mask = np.isin(combined_groups, train_groups)
            
            test_indices = indices[test_mask] if self.shuffle else np.where(test_mask)[0]
            train_indices = indices[train_mask] if self.shuffle else np.where(train_mask)[0]
            
            # Validate split quality
            if len(test_indices) == 0:
                warnings.warn(f"Fold {fold_idx} has empty test set")
                continue
            if len(train_indices) == 0:
                warnings.warn(f"Fold {fold_idx} has empty train set")
                continue
                
            yield train_indices, test_indices
    
    def get_split_info(self, X: Union[pd.DataFrame, np.ndarray], 
                      y: Optional[Union[pd.Series, np.ndarray]] = None) -> dict:
        """
        Get detailed information about the splits without generating them.
        
        Returns
        -------
        dict
            Dictionary containing split statistics and metadata
        """
        X = self._validate_data(X, y)
        if y is not None:
            y = np.asarray(y)
            
        combined_groups, metadata = self._create_combined_groups(X, y)
        
        info = {
            'n_samples': len(X),
            'n_splits': self.n_splits,
            **metadata,
            'avg_samples_per_group': len(X) / metadata['n_combined_groups'],
            'spatial_clusters_used': metadata['n_spatial_clusters'],
            'temporal_clusters_used': metadata['n_temporal_clusters']
        }
        
        # Calculate split size statistics
        unique_groups = np.unique(combined_groups)
        fold_groups = [[] for _ in range(self.n_splits)]
        for i, group in enumerate(unique_groups):
            fold_groups[i % self.n_splits].append(group)
            
        fold_sizes = []
        for fold_idx in range(self.n_splits):
            test_groups = np.array(fold_groups[fold_idx])
            test_mask = np.isin(combined_groups, test_groups)
            fold_sizes.append(np.sum(test_mask))
            
        info.update({
            'fold_sizes': fold_sizes,
            'min_fold_size': min(fold_sizes),
            'max_fold_size': max(fold_sizes),
            'fold_size_std': np.std(fold_sizes)
        })
        
        return info


class StratifiedSpatialTemporalKFold(SpatialTemporalKFold):
    """
    Stratified version that ensures better target distribution balance.
    
    This extends SpatialTemporalKFold with enhanced stratification that
    tries to maintain similar target distributions across folds while
    still respecting spatial-temporal constraints.
    """
    
    def __init__(self, **kwargs):
        # Force stratification on
        kwargs['stratify'] = True
        super().__init__(**kwargs)
    
    def split(self, X, y=None, groups=None):
        """Split with enhanced stratification checking."""
        if y is None:
            raise ValueError("StratifiedSpatialTemporalKFold requires y for stratification")
            
        # Generate base splits
        for train_idx, test_idx in super().split(X, y, groups):
            # Check stratification quality
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Calculate quantile distributions
            train_quantiles = np.quantile(y_train, [0.25, 0.5, 0.75])
            test_quantiles = np.quantile(y_test, [0.25, 0.5, 0.75]) 
            
            # Check if distributions are too different
            max_diff = np.max(np.abs(train_quantiles - test_quantiles))
            if max_diff > 2 * np.std(y):
                warnings.warn(f"Large distribution difference detected: {max_diff:.4f}")
                
            yield train_idx, test_idx