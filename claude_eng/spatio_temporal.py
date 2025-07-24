# spatiotemporal_solution.py
"""
Complete modular implementation for the air pollution prediction competition.
Handles both spatial and temporal distribution shifts between train and test sets.

Usage:
    from spatiotemporal_solution import (
        AdvancedSpatioTemporalFeatures,
        SpatioTemporalCV,
        create_complete_pipeline
    )
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import BaseCrossValidator
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance, ks_2samp, skew, kurtosis
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DISTRIBUTION ANALYSIS
# ============================================================================

class SpatioTemporalDistributionAnalyzer:
    """Analyze distribution differences between train and test sets."""
    
    def __init__(self):
        self.spatial_stats = {}
        self.temporal_stats = {}
        
    def analyze(self, train_df, test_df, verbose=True):
        """
        Analyze spatial and temporal distribution differences.
        
        Returns:
            tuple: (spatial_stats, temporal_stats) dictionaries
        """
        if verbose:
            print("Analyzing distribution differences...")
        
        # Spatial analysis
        lat_ks = ks_2samp(train_df['latitude'].dropna(), test_df['latitude'].dropna())
        lon_ks = ks_2samp(train_df['longitude'].dropna(), test_df['longitude'].dropna())
        
        self.spatial_stats = {
            'latitude_ks': lat_ks,
            'longitude_ks': lon_ks,
            'significant_shift': (lat_ks.pvalue < 0.05) or (lon_ks.pvalue < 0.05)
        }
        
        # Temporal analysis
        temporal_features = ['hour', 'day_of_week', 'month', 'day_of_year']
        
        for feature in temporal_features:
            # Calculate distributions
            train_dist = train_df[feature].value_counts(normalize=True).sort_index()
            test_dist = test_df[feature].value_counts(normalize=True).sort_index()
            
            # Align distributions
            all_values = sorted(set(train_dist.index) | set(test_dist.index))
            train_dist = train_dist.reindex(all_values, fill_value=0)
            test_dist = test_dist.reindex(all_values, fill_value=0)
            
            # Calculate metrics
            w_dist = wasserstein_distance(all_values, all_values, train_dist, test_dist)
            ks_stat = ks_2samp(train_df[feature].dropna(), test_df[feature].dropna())
            
            self.temporal_stats[feature] = {
                'wasserstein_distance': w_dist,
                'ks_statistic': ks_stat.statistic,
                'ks_pvalue': ks_stat.pvalue,
                'train_distribution': train_dist,
                'test_distribution': test_dist,
                'significant_shift': ks_stat.pvalue < 0.05
            }
        
        if verbose:
            self._print_summary()
        
        return self.spatial_stats, self.temporal_stats
    
    def _print_summary(self):
        """Print analysis summary."""
        print("\nSpatial Distribution:")
        print(f"  Latitude KS: {self.spatial_stats['latitude_ks'].statistic:.4f} "
              f"(p={self.spatial_stats['latitude_ks'].pvalue:.2e})")
        print(f"  Longitude KS: {self.spatial_stats['longitude_ks'].statistic:.4f} "
              f"(p={self.spatial_stats['longitude_ks'].pvalue:.2e})")
        
        print("\nTemporal Distribution Shifts:")
        for feature, stats in self.temporal_stats.items():
            if stats['significant_shift']:
                print(f"  {feature}: Wasserstein={stats['wasserstein_distance']:.4f} "
                      f"(significant, p={stats['ks_pvalue']:.2e})")


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class AdvancedSpatioTemporalFeatures(BaseEstimator, TransformerMixin):
    """
    Advanced feature engineering with distribution shift awareness.
    
    Parameters:
    -----------
    row_only : bool, default=False
        If True, only row-wise operations (no leakage risk for CV)
    n_spatial_clusters : int, default=30
        Number of spatial clusters for location features
    n_temporal_clusters : int, default=10
        Number of temporal pattern clusters
    use_distribution_matching : bool, default=True
        Whether to create features that help with distribution shift
    test_distribution : dict, optional
        Test set distribution statistics for creating matching features
    january_bridge_features : bool, default=True
        Whether to include January-specific features for winter analysis,
        to compensate for scarcity of January data in training.
    """
    
    def __init__(self, row_only=False, n_spatial_clusters=30, n_temporal_clusters=10,
                 use_distribution_matching=True, test_distribution=None, january_bridge_features=True):
        self.row_only = row_only
        self.n_spatial_clusters = n_spatial_clusters
        self.n_temporal_clusters = n_temporal_clusters
        self.use_distribution_matching = use_distribution_matching
        self.test_distribution = test_distribution
        self.january_bridge_features = january_bridge_features  # Default to True for winter analysis
        
        # Fitted attributes
        self.spatial_kmeans_ = None
        self.temporal_kmeans_ = None
        self.spatial_stats_ = {}
        self.temporal_stats_ = {}
        self.pollution_by_pattern_ = {}
        self.spatial_means_ = {}
        self.temporal_means_ = {}
        
    def fit(self, X, y=None):
        """Fit the transformer to learn spatial and temporal patterns."""
        X_fit = X.copy()
        
        if not self.row_only and y is not None:
            # --- SPATIAL CLUSTERING ---
            spatial_cols = ['latitude', 'longitude']
            spatial_df = X_fit[spatial_cols].copy()

            # Learn and store means from training data to handle potential NaNs
            for col in spatial_cols:
                self.spatial_means_[col] = spatial_df[col].mean()

            # Fill NaNs with learned means for KMeans fitting on a temporary copy
            for col in spatial_cols:
                spatial_df[col].fillna(self.spatial_means_[col], inplace=True)

            self.spatial_kmeans_ = KMeans(
                n_clusters=self.n_spatial_clusters, 
                random_state=42,
                n_init=10
            )
            spatial_clusters = self.spatial_kmeans_.fit_predict(spatial_df.values)
            
            # --- TEMPORAL CLUSTERING ---
            temporal_cols = ['hour', 'day_of_week', 'month', 'day_of_year']
            temporal_df = X_fit[temporal_cols].copy()

            # Learn and store means
            for col in temporal_cols:
                self.temporal_means_[col] = temporal_df[col].mean()

            # Fill NaNs for KMeans fitting on a temporary copy
            for col in temporal_cols:
                temporal_df[col].fillna(self.temporal_means_[col], inplace=True)
                
            temporal_features = self._create_temporal_features(temporal_df)
            self.temporal_kmeans_ = KMeans(
                n_clusters=self.n_temporal_clusters, 
                random_state=42,
                n_init=10
            )
            temporal_clusters = self.temporal_kmeans_.fit_predict(temporal_features)
            
            # Calculate statistics per cluster
            for i in range(self.n_spatial_clusters):
                mask = spatial_clusters == i
                if mask.sum() > 0:
                    cluster_y = y[mask]
                    self.spatial_stats_[i] = {
                        'mean': np.mean(cluster_y),
                        'std': np.std(cluster_y),
                        'median': np.median(cluster_y),
                        'q25': np.percentile(cluster_y, 25),
                        'q75': np.percentile(cluster_y, 75),
                        'skew': skew(cluster_y) if len(cluster_y) > 2 else 0,
                        'count': len(cluster_y)
                    }
            
            # Temporal pattern statistics
            for i in range(self.n_temporal_clusters):
                mask = temporal_clusters == i
                if mask.sum() > 0:
                    pattern_y = y[mask]
                    self.temporal_stats_[i] = {
                        'mean': np.mean(pattern_y),
                        'std': np.std(pattern_y),
                        'median': np.median(pattern_y),
                        'count': len(pattern_y)
                    }
            
            # Hour-month pattern statistics
            for hour in range(24):
                for month in range(1, 13):
                    mask = (X_fit['hour'] == hour) & (X_fit['month'] == month)
                    if mask.sum() > 0:
                        self.pollution_by_pattern_[f'h{hour}_m{month}'] = {
                            'mean': np.mean(y[mask]),
                            'std': np.std(y[mask]) if mask.sum() > 1 else 0
                        }
        
        return self
    
    def transform(self, X):
        """Transform with engineered features."""
        X_new = X.copy()
        
        # 1. Cyclical encoding (always safe)
        X_new['hour_sin'] = np.sin(2 * np.pi * X['hour'] / 24)
        X_new['hour_cos'] = np.cos(2 * np.pi * X['hour'] / 24)
        X_new['month_sin'] = np.sin(2 * np.pi * X['month'] / 12)
        X_new['month_cos'] = np.cos(2 * np.pi * X['month'] / 12)
        X_new['dow_sin'] = np.sin(2 * np.pi * X['day_of_week'] / 7)
        X_new['dow_cos'] = np.cos(2 * np.pi * X['day_of_week'] / 7)
        X_new['doy_sin'] = np.sin(2 * np.pi * X['day_of_year'] / 365)
        X_new['doy_cos'] = np.cos(2 * np.pi * X['day_of_year'] / 365)
        
        # 2. Temporal features
        X_new['hour_month_interaction'] = X['hour'] * X['month']
        X_new['hour_dow_interaction'] = X['hour'] * X['day_of_week']
        X_new['month_dow_interaction'] = X['month'] * X['day_of_week']
        X_new['is_weekend'] = (X['day_of_week'] >= 5).astype(int)
        X_new['is_weekday'] = (X['day_of_week'] < 5).astype(int)
        X_new['is_rush_hour'] = X['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        X_new['is_night'] = ((X['hour'] >= 22) | (X['hour'] <= 5)).astype(int)
        X_new['is_business_hours'] = (
            (X['hour'].between(9, 17)) & (X['day_of_week'] < 5)
        ).astype(int)
        
        # Season features
        X_new['season'] = pd.cut(X['month'], bins=[0, 3, 6, 9, 12], labels=[0, 1, 2, 3]).astype(int)
        X_new['is_summer'] = X['month'].isin([6, 7, 8]).astype(int)
        X_new['is_winter'] = X['month'].isin([12, 1, 2]).astype(int)
        X_new['quarter'] = (X['month'] - 1) // 3 + 1
        X_new['week_of_year'] = (X['day_of_year'] - 1) // 7 + 1
        
        # 3. Spatial features (always included)
        X_new['lat_lon_interaction'] = X['latitude'] * X['longitude']
        X_new['distance_from_equator'] = np.abs(X['latitude'])
        X_new['distance_from_prime_meridian'] = np.abs(X['longitude'])
        X_new['lat_squared'] = X['latitude'] ** 2
        X_new['lon_squared'] = X['longitude'] ** 2
        X_new['lat_cubed'] = X['latitude'] ** 3
        X_new['lon_cubed'] = X['longitude'] ** 3
        X_new['northern_hemisphere'] = (X['latitude'] > 0).astype(int)
        X_new['eastern_hemisphere'] = (X['longitude'] > 0).astype(int)
        
        # 4. Distribution matching features
        if self.use_distribution_matching and self.test_distribution is not None:
            for feature in ['hour', 'month', 'day_of_week']:
                if feature in self.test_distribution:
                    test_dist = self.test_distribution[feature]['test_distribution']
                    X_new[f'{feature}_test_prob'] = X[feature].map(test_dist).fillna(0)
        
        # 5. Advanced features (if not row_only)
        if not self.row_only:
            self._add_cluster_features(X, X_new)
            self._add_pattern_features(X, X_new)
        
        # 6. Complex interactions
        X_new['spatial_temporal_interaction'] = (
            X_new['distance_from_equator'] * X_new['month_sin']
        )
        X_new['lat_hour_interaction'] = X['latitude'] * X['hour']
        X_new['lon_hour_interaction'] = X['longitude'] * X['hour']
        X_new['lat_month_interaction'] = X['latitude'] * X['month']
        X_new['lon_month_interaction'] = X['longitude'] * X['month']

        # ============================================================================
        # JANUARY/WINTER SPECIFIC INTERACTIONS
        # ============================================================================
        # 1. December-January Continuity Bridge
        if self.january_bridge_features:
            X_new['dec_jan_bridge'] = ((X['month'] == 12) | (X['month'] == 1)).astype(int)
            X_new['dec_jan_transition'] = np.where(
                X['month'] == 1, 1.0,  # January = +1
                np.where(X['month'] == 12, -1.0, 0.0)  # December = -1, others = 0
            )
            # 2. Winter Intensity Features
            X_new['winter_depth'] = np.where(
                X['month'] == 1, 1.0,        # Peak winter
                np.where(X['month'].isin([12, 2]), 0.7, 0.0)  # Shoulder winter
            )
            X_new['days_from_jan_1'] = np.where(
                X['month'] == 1, X['day_of_year'] - 1,        # Days into January
                np.where(X['month'] == 12, 31 - (X['day_of_year'] - 334), 365)  # Days to January
            )
            # 3. Winter Geographic Interactions
            X_new['winter_northern_hemisphere'] = X_new['winter_depth'] * X_new['northern_hemisphere']
            X_new['winter_latitude_intensity'] = X_new['winter_depth'] * X_new['distance_from_equator']
            X_new['winter_high_latitude'] = X_new['winter_depth'] * (X['latitude'] > 45).astype(int)
            # 4. Winter Temporal Patterns
            X_new['winter_heating_hours'] = X_new['winter_depth'] * X['hour'].isin([6, 7, 8, 17, 18, 19, 20]).astype(int)
            X_new['winter_business_hours'] = X_new['winter_depth'] * X_new['is_business_hours']
            X_new['winter_weekend'] = X_new['winter_depth'] * X_new['is_weekend']
            X_new['winter_night'] = X_new['winter_depth'] * X_new['is_night']
            # 5. January-Specific Hour Patterns
            jan_mask = (X['month'] == 1)
            X_new['jan_morning_rush'] = jan_mask.astype(int) * X['hour'].isin([7, 8, 9]).astype(int)
            X_new['jan_evening_rush'] = jan_mask.astype(int) * X['hour'].isin([17, 18, 19]).astype(int)
            X_new['jan_midday'] = jan_mask.astype(int) * X['hour'].isin([11, 12, 13]).astype(int)
            X_new['jan_early_morning'] = jan_mask.astype(int) * X['hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
            # 6. Winter Day-of-Week Interactions
            X_new['winter_monday'] = X_new['winter_depth'] * (X['day_of_week'] == 0).astype(int)
            X_new['winter_friday'] = X_new['winter_depth'] * (X['day_of_week'] == 4).astype(int)
            X_new['winter_workday_pattern'] = X_new['winter_depth'] * X['day_of_week'] * X_new['is_weekday']
            # 7. Cross-Seasonal Learning Features
            # These help December patterns inform January predictions
            X_new['seasonal_boundary'] = np.cos(2 * np.pi * (X['day_of_year'] - 1) / 365)  # Peaks at Jan 1
            X_new['winter_solstice_distance'] = np.abs(X['day_of_year'] - 355)  # Distance from Dec 21
            X_new['year_end_proximity'] = np.where(
                X['day_of_year'] >= 300, (365 - X['day_of_year']) / 65,  # Last 65 days of year
                np.where(X['day_of_year'] <= 65, (65 - X['day_of_year']) / 65, 0)  # First 65 days
            )
            # 8. January Temperature Proxy Features (using latitude as proxy)
            # Higher latitudes = colder in January
            X_new['jan_cold_proxy'] = jan_mask.astype(int) * np.maximum(0, X['latitude'] - 30) / 60
            X_new['jan_heating_demand'] = X_new['jan_cold_proxy'] * X_new['jan_morning_rush']
            X_new['jan_evening_heating'] = X_new['jan_cold_proxy'] * X_new['jan_evening_rush']
            # 9. January-December Pattern Transfer
            # Create features that allow December patterns to inform January
            month_diff = np.where(X['month'] == 1, 12, X['month'])  # Treat Jan as month 13
            X_new['winter_month_continuous'] = np.where(
                X['month'].isin([12, 1]),
                month_diff + (X['day_of_year'] - 1) / 365,  # Continuous across Dec-Jan boundary
                X['month']
            )
            # 10. Enhanced Winter Cyclical Features
            # More granular winter encoding
            winter_day = np.where(
                X['month'] == 12, X['day_of_year'] - 334,  # Dec days: 0-30
                np.where(X['month'] == 1, X['day_of_year'] + 31, -1)  # Jan days: 32-62
            )
            valid_winter = winter_day >= 0
            X_new['winter_day_sin'] = np.where(valid_winter, np.sin(2 * np.pi * winter_day / 62), 0)
            X_new['winter_day_cos'] = np.where(valid_winter, np.cos(2 * np.pi * winter_day / 62), 0)
            # 11. January Pollution Context Features
            # These help model understand January's unique pollution profile
            X_new['jan_inversion_risk'] = jan_mask.astype(int) * X_new['jan_early_morning'] * (X['latitude'] > 40).astype(int)
            X_new['jan_urban_heating'] = jan_mask.astype(int) * X_new['jan_heating_demand'] * X_new['is_business_hours']
            X_new['jan_weekend_heating'] = jan_mask.astype(int) * X_new['is_weekend'] * X_new['jan_cold_proxy']

        # Ensure all features are numeric    
        # Drop id column if present
        if 'id' in X_new.columns:
            X_new = X_new.drop('id', axis=1)
        
        return X_new
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _create_temporal_features(self, X):
        """Create temporal feature matrix for clustering."""
        features = []
        features.extend([
            np.sin(2 * np.pi * X['hour'] / 24),
            np.cos(2 * np.pi * X['hour'] / 24),
            np.sin(2 * np.pi * X['month'] / 12),
            np.cos(2 * np.pi * X['month'] / 12),
            np.sin(2 * np.pi * X['day_of_week'] / 7),
            np.cos(2 * np.pi * X['day_of_week'] / 7),
            np.sin(2 * np.pi * X['day_of_year'] / 365),
            np.cos(2 * np.pi * X['day_of_year'] / 365)
        ])
        return np.column_stack(features)
    
    def _add_cluster_features(self, X, X_new):
        """Add spatial and temporal cluster features."""
        # --- SPATIAL ---
        # Make a copy to fill NaNs for prediction without altering original X
        spatial_pred_df = X[['latitude', 'longitude']].copy()
        for col, mean_val in self.spatial_means_.items():
            spatial_pred_df[col].fillna(mean_val, inplace=True)
        coords = spatial_pred_df.values

        # Spatial clustering
        if self.spatial_kmeans_ is not None:
            X_new['spatial_cluster'] = self.spatial_kmeans_.predict(coords)
            
            # Distance features
            distances = cdist(coords, self.spatial_kmeans_.cluster_centers_)
            X_new['min_spatial_cluster_dist'] = distances.min(axis=1)
            X_new['mean_spatial_cluster_dist'] = distances.mean(axis=1)
            
            # Nearest clusters
            nearest_clusters = np.argsort(distances, axis=1)[:, :3]
            for i in range(3):
                X_new[f'nearest_spatial_cluster_{i}'] = nearest_clusters[:, i]
                X_new[f'dist_to_spatial_cluster_{i}'] = distances[
                    np.arange(len(distances)), nearest_clusters[:, i]
                ]
            
            # Cluster statistics
            if self.spatial_stats_:
                self._add_cluster_statistics(X_new, 'spatial')
        
        # --- TEMPORAL ---
        if self.temporal_kmeans_ is not None:
            # Make a copy to fill NaNs for prediction
            temporal_cols = ['hour', 'day_of_week', 'month', 'day_of_year']
            temporal_pred_df = X[temporal_cols].copy()
            for col, mean_val in self.temporal_means_.items():
                if col in temporal_pred_df.columns:
                    temporal_pred_df[col].fillna(mean_val, inplace=True)

            temporal_features = self._create_temporal_features(temporal_pred_df)
            X_new['temporal_cluster'] = self.temporal_kmeans_.predict(temporal_features)
            
            # Distance features
            temp_distances = cdist(
                temporal_features, 
                self.temporal_kmeans_.cluster_centers_
            )
            X_new['min_temporal_cluster_dist'] = temp_distances.min(axis=1)
            
            # Cluster statistics
            if self.temporal_stats_:
                self._add_cluster_statistics(X_new, 'temporal')
    
    def _add_cluster_statistics(self, X_new, cluster_type):
        """Add cluster-based pollution statistics."""
        if cluster_type == 'spatial':
            stats_dict = self.spatial_stats_
            cluster_col = 'spatial_cluster'
            prefix = 'spatial_cluster_pollution'
        else:
            stats_dict = self.temporal_stats_
            cluster_col = 'temporal_cluster'
            prefix = 'temporal_cluster_pollution'
        
        means = []
        stds = []
        
        for idx in range(len(X_new)):
            cluster_id = X_new.iloc[idx][cluster_col]
            if cluster_id in stats_dict:
                means.append(stats_dict[cluster_id]['mean'])
                stds.append(stats_dict[cluster_id]['std'])
            else:
                means.append(np.nan)
                stds.append(np.nan)
        
        X_new[f'{prefix}_mean'] = means
        X_new[f'{prefix}_std'] = stds
        
        # Fill NaN values that arose from this step
        X_new[f'{prefix}_mean'].fillna(X_new[f'{prefix}_mean'].mean(), inplace=True)
        X_new[f'{prefix}_std'].fillna(X_new[f'{prefix}_std'].mean(), inplace=True)
    
    def _add_pattern_features(self, X, X_new):
        """Add hour-month pattern features."""
        if self.pollution_by_pattern_:
            pattern_means = []
            pattern_stds = []
            
            for idx in range(len(X)):
                pattern_key = f"h{X.iloc[idx]['hour']}_m{X.iloc[idx]['month']}"
                if pattern_key in self.pollution_by_pattern_:
                    pattern_means.append(self.pollution_by_pattern_[pattern_key]['mean'])
                    pattern_stds.append(self.pollution_by_pattern_[pattern_key]['std'])
                else:
                    pattern_means.append(np.nan)
                    pattern_stds.append(np.nan)
            
            X_new['hour_month_pattern_mean'] = pattern_means
            X_new['hour_month_pattern_std'] = pattern_stds
            
            # Fill NaN values that arose from this step
            X_new['hour_month_pattern_mean'].fillna(
                X_new['hour_month_pattern_mean'].mean(), inplace=True
            )
            X_new['hour_month_pattern_std'].fillna(
                X_new['hour_month_pattern_std'].mean(), inplace=True
            )


# ============================================================================
# CUSTOM CROSS-VALIDATION
# ============================================================================

class SpatioTemporalCV:
    """
    Custom CV that matches both spatial AND temporal distributions of test set.
    
    Parameters:
    -----------
    n_splits : int
        Number of CV folds
    test_spatial_coords : array-like
        Test set coordinates for spatial matching
    test_temporal_features : DataFrame
        Test set temporal features for distribution matching.
    spatial_weight : float, default=0.5
        Weight for spatial distribution matching (vs temporal)
    random_state : int
        Random seed
    """
    
    def __init__(self, n_splits=5, test_spatial_coords=None, 
                 test_temporal_features=None, spatial_weight=0.5, 
                 random_state=42):
        self.n_splits = n_splits
        self.test_spatial_coords = test_spatial_coords
        self.test_temporal_features = test_temporal_features
        self.spatial_weight = spatial_weight
        self.temporal_weight = 1 - spatial_weight
        self.random_state = random_state

    def get_n_splits(self):
        return self.n_splits
        
    def split(self, X, y=None, groups=None):
        """Generate train/validation splits matching test distribution."""
        np.random.seed(self.random_state)
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate similarity scores
        spatial_scores = self._calculate_spatial_scores(X)
        temporal_scores = self._calculate_temporal_scores(X)
        
        # Combined score
        combined_scores = (
            self.spatial_weight * spatial_scores + 
            self.temporal_weight * temporal_scores
        )
        
        # Create stratified folds
        sorted_indices = np.argsort(combined_scores)[::-1]
        fold_size = n_samples // self.n_splits
        
        for fold in range(self.n_splits):
            val_indices = []
            
            # Sample from different similarity buckets
            n_buckets = 10
            bucket_size = n_samples // n_buckets
            samples_per_bucket = fold_size // n_buckets
            
            for bucket in range(n_buckets):
                start_idx = bucket * bucket_size
                end_idx = min((bucket + 1) * bucket_size, n_samples)
                bucket_indices = sorted_indices[start_idx:end_idx]
                
                n_select = min(samples_per_bucket, len(bucket_indices))
                if n_select > 0:
                    selected = np.random.choice(
                        bucket_indices, n_select, replace=False
                    )
                    val_indices.extend(selected)
            
            val_indices = np.array(val_indices)
            train_indices = np.setdiff1d(indices, val_indices)
            
            yield train_indices, val_indices
    
    def _calculate_spatial_scores(self, X):
        """Calculate spatial similarity to test set."""
        if self.test_spatial_coords is None:
            return np.ones(len(X))
        
        from sklearn.neighbors import KernelDensity
        
        train_coords = X[['latitude', 'longitude']].fillna(X[['latitude', 'longitude']].mean())
        
        # Fit KDE on test coordinates
        kde = KernelDensity(bandwidth=2.0, kernel='gaussian')
        kde.fit(self.test_spatial_coords)
        
        # Score train coordinates
        log_density = kde.score_samples(train_coords)
        scores = np.exp(log_density)
        
        # Normalize
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return scores
    
    def _calculate_temporal_scores(self, X):
        """Calculate temporal similarity to test set."""
        if self.test_temporal_features is None:
            return np.ones(len(X))
        
        scores = np.zeros(len(X))
        temporal_features = ['hour', 'month', 'day_of_week', 'day_of_year']
        
        for feature in temporal_features:
            if feature in self.test_temporal_features.columns:
                # Get test distribution
                test_dist = self.test_temporal_features[feature].value_counts(
                    normalize=True
                )
                
                # Score each sample
                feature_scores = X[feature].map(test_dist).fillna(0).values
                scores += feature_scores
        
        # Normalize
        scores = scores / len(temporal_features)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return scores


# ============================================================================
# DOMAIN ADAPTATION
# ============================================================================

class TemporalDomainAdaptation:
    """Apply domain adaptation techniques to handle temporal distribution shift."""
    
    def __init__(self, method='importance_weighting'):
        self.method = method
        self.weights_ = None
        
    def fit(self, X_train, X_test):
        """Calculate importance weights based on temporal distributions."""
        if self.method == 'importance_weighting':
            from sklearn.neighbors import KernelDensity
            
            # Focus on temporal features
            temporal_features = ['hour', 'month', 'day_of_week', 'day_of_year']
            
            # Create copies and fill NaNs with mean for KDE
            X_train_temp = X_train[temporal_features].copy()
            X_test_temp = X_test[temporal_features].copy()
            for col in temporal_features:
                mean_val = X_train_temp[col].mean()
                X_train_temp[col].fillna(mean_val, inplace=True)
                X_test_temp[col].fillna(mean_val, inplace=True)
            
            # Encode cyclically
            train_temporal = self._encode_temporal(X_train_temp)
            test_temporal = self._encode_temporal(X_test_temp)
            
            # Fit KDE
            kde_train = KernelDensity(bandwidth=0.5)
            kde_test = KernelDensity(bandwidth=0.5)
            
            kde_train.fit(train_temporal)
            kde_test.fit(test_temporal)
            
            # Calculate density ratio
            log_dens_train = kde_train.score_samples(train_temporal)
            log_dens_test = kde_test.score_samples(train_temporal)
            
            # Importance weights
            self.weights_ = np.exp(log_dens_test - log_dens_train)
            
            # Clip and normalize
            self.weights_ = np.clip(self.weights_, 0.1, 10.0)
            self.weights_ = self.weights_ / self.weights_.mean()
            
        return self
    
    def _encode_temporal(self, temporal_df):
        """Cyclically encode temporal features."""
        encoded = []
        
        if 'hour' in temporal_df.columns:
            encoded.extend([
                np.sin(2 * np.pi * temporal_df['hour'] / 24),
                np.cos(2 * np.pi * temporal_df['hour'] / 24)
            ])
        
        if 'month' in temporal_df.columns:
            encoded.extend([
                np.sin(2 * np.pi * temporal_df['month'] / 12),
                np.cos(2 * np.pi * temporal_df['month'] / 12)
            ])
            
        if 'day_of_week' in temporal_df.columns:
            encoded.extend([
                np.sin(2 * np.pi * temporal_df['day_of_week'] / 7),
                np.cos(2 * np.pi * temporal_df['day_of_week'] / 7)
            ])
            
        if 'day_of_year' in temporal_df.columns:
            encoded.extend([
                np.sin(2 * np.pi * temporal_df['day_of_year'] / 365),
                np.cos(2 * np.pi * temporal_df['day_of_year'] / 365)
            ])
        
        return np.column_stack(encoded)
    
    def get_weights(self):
        """Return the calculated importance weights."""
        return self.weights_


# ============================================================================
# COMPLETE PIPELINE
# ============================================================================

def create_complete_pipeline(train_df, test_df, target_col='pollution_value',
                           use_domain_adaptation=True, verbose=True):
    """
    Complete pipeline handling both spatial and temporal distribution shifts.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data with features and target
    test_df : pd.DataFrame
        Test data with features only
    target_col : str
        Name of the target column
    use_domain_adaptation : bool
        Whether to use domain adaptation weights
    verbose : bool
        Whether to print progress
    
    Returns:
    --------
    dict : Dictionary containing:
        - cv_scores: Cross-validation scores for each model
        - distribution_stats: Distribution analysis results
        - predictions: Final predictions
        - submission: Submission DataFrame
        - feature_names: List of feature names used
    """
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import RandomForestRegressor
    
    # 1. Analyze distributions
    analyzer = SpatioTemporalDistributionAnalyzer()
    spatial_stats, temporal_stats = analyzer.analyze(train_df, test_df, verbose)
    
    # 2. Prepare data
    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]
    X_test = test_df.copy()
    
    # 3. Domain adaptation
    sample_weights = None
    if use_domain_adaptation:
        if verbose:
            print("\nCalculating domain adaptation weights...")
        domain_adapter = TemporalDomainAdaptation()
        domain_adapter.fit(X_train, X_test)
        sample_weights = domain_adapter.get_weights()
        if verbose:
            print(f"Weight range: [{sample_weights.min():.2f}, "
                  f"{sample_weights.max():.2f}]")
    
    # 4. Feature engineering
    if verbose:
        print("\nEngineering features...")
    
    # For validation
    fe_val = AdvancedSpatioTemporalFeatures(
        row_only=True,
        test_distribution=temporal_stats
    )
    
    # For final model
    fe_final = AdvancedSpatioTemporalFeatures(
        row_only=False,
        n_spatial_clusters=30,
        n_temporal_clusters=15,
        test_distribution=temporal_stats
    )
    
    # 5. Custom CV
    cv = SpatioTemporalCV(
        n_splits=5,
        test_spatial_coords=test_df[['latitude', 'longitude']].values,
        test_temporal_features=test_df[['hour', 'month', 'day_of_week', 'day_of_year']],
        spatial_weight=0.5
    )
    
    # 6. Transform for validation
    X_train_fe_val = fe_val.fit_transform(X_train, y_train)
    
    # 7. Models
    models = {
        'lgb': lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            random_state=42,
            verbose=-1
        ),
        'xgb': xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ),
        'rf': RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
    }
    
    # 8. Validate
    if verbose:
        print("\nValidating with spatio-temporal aware CV...")
    
    cv_scores = {}
    for name, model in models.items():
        if verbose:
            print(f"\nValidating {name}...")
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_fe_val, y_train)):
            X_tr, X_val = X_train_fe_val.iloc[train_idx], X_train_fe_val.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # --- FIX: Handle remaining NaNs before scaling ---
            # Calculate mean from the training part of the fold
            train_means = X_tr.mean()
            # Fill NaNs in both train and validation parts using train_means
            X_tr.fillna(train_means, inplace=True)
            X_val.fillna(train_means, inplace=True) # Use train_means to avoid leakage

            # Get weights for this fold
            weights_tr = sample_weights[train_idx] if sample_weights is not None else None
            
            # Scale
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_val_scaled = scaler.transform(X_val)
            
            # Train
            fit_params = {}
            if name in ['lgb', 'xgb']:
                fit_params['eval_set'] = [(X_val_scaled, y_val)]
                fit_params['callbacks'] = [lgb.early_stopping(50, verbose=False)] if name == 'lgb' else []

            if name == 'xgb':
                 model.fit(X_tr_scaled, y_tr, sample_weight=weights_tr, eval_set=[(X_val_scaled, y_val)], early_stopping_rounds=50, verbose=False)
            elif name == 'lgb':
                 model.fit(X_tr_scaled, y_tr, sample_weight=weights_tr, eval_set=[(X_val_scaled, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
            else:
                model.fit(X_tr_scaled, y_tr, sample_weight=weights_tr)

            # Evaluate
            val_pred = model.predict(X_val_scaled)
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            scores.append(rmse)
            
            if verbose:
                print(f"  Fold {fold + 1}: RMSE = {rmse:.4f}")
        
        cv_scores[name] = np.mean(scores)
        if verbose:
            print(f"  Average RMSE: {cv_scores[name]:.4f}")
    
    # 9. Train final model
    if verbose:
        print("\nTraining final model with all features...")
    
    X_train_fe_final = fe_final.fit_transform(X_train, y_train)
    X_test_fe_final = fe_final.transform(X_test)
    
    # --- FIX: Handle remaining NaNs before final scaling ---
    # Calculate means on the full training set
    final_train_means = X_train_fe_final.mean()
    # Fill NaNs in both final train and test sets
    X_train_fe_final.fillna(final_train_means, inplace=True)
    X_test_fe_final.fillna(final_train_means, inplace=True)
    
    if verbose:
        print(f"Final feature count: {X_train_fe_final.shape[1]}")
    
    # Scale
    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train_fe_final)
    X_test_scaled = scaler_final.transform(X_test_fe_final)
    
    # 10. Ensemble
    best_models = sorted(cv_scores.items(), key=lambda x: x[1])[:3]
    ensemble_preds = []
    
    for name, _ in best_models:
        if verbose:
            print(f"Training {name} for ensemble...")
        # Re-initialize the model to use its best iteration if found via early stopping
        model = models[name]

        model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        
        preds = model.predict(X_test_scaled)
        ensemble_preds.append(preds)
    
    # Weighted average
    weights = [1/cv_scores[name] for name, _ in best_models]
    weights = np.array(weights) / sum(weights)
    
    final_predictions = np.average(ensemble_preds, axis=0, weights=weights)
    
    # Post-process
    final_predictions = np.maximum(final_predictions, 0)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'pollution_value': final_predictions
    })
    
    return {
        'cv_scores': cv_scores,
        'distribution_stats': {
            'spatial': spatial_stats,
            'temporal': temporal_stats
        },
        'predictions': final_predictions,
        'submission': submission,
        'feature_names': X_train_fe_final.columns.tolist(),
        'feature_engineer': fe_final,
        'scaler': scaler_final
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage with dummy data
    print("Creating dummy data...")
    # Create a more realistic dummy dataset
    n_train = 1000
    n_test = 500
    train_data = {
        'id': range(n_train),
        'latitude': np.random.uniform(30, 50, n_train),
        'longitude': np.random.uniform(-120, -70, n_train),
        'hour': np.random.randint(0, 24, n_train),
        'day_of_week': np.random.randint(0, 7, n_train),
        'month': np.random.randint(1, 13, n_train),
        'day_of_year': np.random.randint(1, 366, n_train),
        'feature1': np.random.randn(n_train) * 10,
        'pollution_value': np.random.rand(n_train) * 100
    }
    # Introduce some NaNs
    train_data['feature1'][::10] = np.nan
    train_data['latitude'][::15] = np.nan

    test_data = {
        'id': range(n_test),
        'latitude': np.random.uniform(32, 52, n_test),      # Slight shift
        'longitude': np.random.uniform(-118, -68, n_test),   # Slight shift
        'hour': np.random.randint(6, 20, n_test),           # Shifted hours
        'day_of_week': np.random.randint(0, 5, n_test),     # Shifted day of week
        'month': np.random.randint(3, 10, n_test),          # Shifted month
        'day_of_year': np.random.randint(60, 270, n_test),  # Shifted day of year
        'feature1': np.random.randn(n_test) * 12
    }
    
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    print("Running complete pipeline...")
    results = create_complete_pipeline(train_df, test_df)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"CV scores: {results['cv_scores']}")
    print(f"Number of features: {len(results['feature_names'])}")
    
    # Save submission
    results['submission'].to_csv('submission.csv', index=False)
    print("\nSubmission saved to submission.csv")