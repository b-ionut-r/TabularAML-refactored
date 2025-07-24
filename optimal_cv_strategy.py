"""
OPTIMAL CROSS-VALIDATION STRATEGY FOR AIR POLLUTION PREDICTION COMPETITION

Based on comprehensive spatial-temporal distribution analysis, this module provides
the optimal CV strategy for genetic feature engineering in this specific competition.

Key Findings:
- EXTREME temporal shift: January 4.4% → 100.0% (191.2% total shift)
- EXTREME spatial shift: 95.0% novel locations in test set
- Test set represents extreme out-of-distribution scenario

Optimal Strategy: HybridSpatialTemporalCV
- 3 spatial folds (60%): Geographic extrapolation testing
- 2 January-only folds (40%): Temporal distribution matching
- 42.8% January representation in validation (vs 100% in test)
- Computationally efficient for genetic programming
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.cluster import KMeans
from typing import Iterator, Tuple
import warnings


class OptimalCompetitionCV(BaseCrossValidator):
    """
    Optimal Cross-Validator for Air Pollution Prediction Competition
    
    This is the recommended CV strategy based on comprehensive analysis of
    spatial-temporal distribution shifts. It addresses both major challenges:
    1. Temporal shift (January 4.4% → 100.0%)
    2. Spatial shift (95% novel locations)
    
    The strategy combines:
    - Spatial folds for geographic extrapolation
    - January-only folds for temporal distribution matching
    - Optimal balance of validation rigor and computational efficiency
    
    Parameters
    ----------
    n_spatial_folds : int, default=3
        Number of spatial folds for geographic extrapolation testing
    n_january_folds : int, default=2
        Number of January-only folds for temporal distribution matching
    n_clusters : int, default=10
        Number of spatial clusters for spatial folding
    min_january_samples : int, default=20
        Minimum January samples required per fold
    random_state : int, default=42
        Random state for reproducible clustering
    
    Examples
    --------
    >>> from tabularaml.generate.features import FeatureGenerator
    >>> cv = OptimalCompetitionCV()
    >>> generator = FeatureGenerator(cv=cv, task="regression")
    >>> results = generator.search(X_train, y_train)
    """
    
    def __init__(self, n_spatial_folds=3, n_january_folds=2, n_clusters=10,
                 min_january_samples=20, random_state=42):
        self.n_spatial_folds = n_spatial_folds
        self.n_january_folds = n_january_folds
        self.n_clusters = n_clusters
        self.min_january_samples = min_january_samples
        self.random_state = random_state
        self.n_splits = n_spatial_folds + n_january_folds
        
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator"""
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : DataFrame
            Training data, must contain 'latitude', 'longitude', and 'month' columns
        y : array-like, default=None
            Target variable (unused but included for compatibility)
        groups : array-like, default=None
            Group labels (unused)
            
        Yields
        ------
        train : ndarray
            The training set indices for that split
        test : ndarray
            The testing set indices for that split
        """
        X = self._validate_data(X)
        indices = np.arange(len(X))
        
        # PART 1: SPATIAL FOLDS (60% of total folds)
        # These test geographic extrapolation capability
        if self.n_spatial_folds > 0:
            yield from self._generate_spatial_folds(X, indices)
        
        # PART 2: JANUARY-ONLY FOLDS (40% of total folds)
        # These test temporal distribution matching
        if self.n_january_folds > 0:
            yield from self._generate_january_folds(X, indices)
    
    def _generate_spatial_folds(self, X, indices):
        """Generate spatial folds for geographic extrapolation testing"""
        # Extract unique locations
        locations = X[['latitude', 'longitude']].drop_duplicates()
        
        # Cluster locations spatially
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        locations_clean = locations.fillna(locations.mean())
        location_clusters = kmeans.fit_predict(locations_clean.values)
        
        # Map clusters back to full dataset
        location_to_cluster = {}
        for i, (_, row) in enumerate(locations.iterrows()):
            key = f"{row['latitude']},{row['longitude']}"
            location_to_cluster[key] = location_clusters[i]
        
        # Assign cluster to each sample
        sample_clusters = X.apply(
            lambda row: location_to_cluster.get(f"{row['latitude']},{row['longitude']}", -1), 
            axis=1
        ).values
        
        # Create spatial folds
        clusters_per_fold = self.n_clusters // self.n_spatial_folds
        
        for i in range(self.n_spatial_folds):
            # Determine validation clusters for this fold
            start_cluster = i * clusters_per_fold
            end_cluster = (i + 1) * clusters_per_fold if i < self.n_spatial_folds - 1 else self.n_clusters
            val_clusters = list(range(start_cluster, end_cluster))
            
            # Create masks
            val_mask = np.isin(sample_clusters, val_clusters)
            train_idx = indices[~val_mask]
            val_idx = indices[val_mask]
            
            # Ensure we have sufficient validation samples
            if len(val_idx) > 0:
                yield train_idx, val_idx
    
    def _generate_january_folds(self, X, indices):
        """Generate January-only folds for temporal distribution matching"""
        # Filter to January data only
        january_mask = X['month'] == 1
        january_indices = indices[january_mask]
        january_data = X[january_mask]
        
        if len(january_data) < self.n_january_folds * self.min_january_samples:
            warnings.warn(
                f"Insufficient January data for {self.n_january_folds} folds. "
                f"Need {self.n_january_folds * self.min_january_samples} samples, "
                f"got {len(january_data)}."
            )
            return
        
        # Get unique January locations
        january_locations = january_data.apply(
            lambda row: f"{row['latitude']},{row['longitude']}", axis=1
        ).values
        unique_locations = list(set(january_locations))
        
        # Shuffle locations for random distribution
        rng = np.random.RandomState(self.random_state)
        rng.shuffle(unique_locations)
        
        # Create January folds
        locs_per_fold = len(unique_locations) // self.n_january_folds
        
        for i in range(self.n_january_folds):
            # Select validation locations for this fold
            start_idx = i * locs_per_fold
            end_idx = (i + 1) * locs_per_fold if i < self.n_january_folds - 1 else len(unique_locations)
            val_locations = set(unique_locations[start_idx:end_idx])
            
            # Create validation mask for January data
            val_mask = np.isin(january_locations, list(val_locations))
            val_idx = january_indices[val_mask]
            
            # Training includes remaining January data + ALL non-January data
            train_january_idx = january_indices[~val_mask]
            non_january_indices = indices[~january_mask]
            train_idx = np.concatenate([train_january_idx, non_january_indices])
            
            # Ensure minimum validation size
            if len(val_idx) >= self.min_january_samples:
                yield train_idx, val_idx
    
    def _validate_data(self, X):
        """Validate and convert input data"""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        required_cols = ['latitude', 'longitude', 'month']
        missing_cols = [col for col in required_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"X must contain columns: {missing_cols}")
        
        return X
    
    def get_fold_info(self, X):
        """
        Get information about fold composition for analysis
        
        Returns
        -------
        fold_info : list of dict
            Information about each fold including size, type, and January representation
        """
        fold_info = []
        
        for i, (train_idx, val_idx) in enumerate(self.split(X)):
            val_data = X.iloc[val_idx]
            january_count = (val_data['month'] == 1).sum()
            january_pct = january_count / len(val_idx) * 100
            
            # Determine fold type
            if january_pct == 100:
                fold_type = "January-Only"
            elif january_pct > 50:
                fold_type = "January-Heavy"
            else:
                fold_type = "Spatial"
            
            fold_info.append({
                'fold_id': i + 1,
                'fold_type': fold_type,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'january_count': january_count,
                'january_pct': january_pct,
                'unique_locations': len(val_data[['latitude', 'longitude']].drop_duplicates())
            })
        
        return fold_info


# Convenience aliases for backward compatibility
HybridSpatialTemporalCV = OptimalCompetitionCV
CompetitionOptimalCV = OptimalCompetitionCV


def demonstrate_optimal_cv(X_train, y_train=None):
    """
    Demonstrate the optimal CV strategy on your data
    
    Parameters
    ----------
    X_train : DataFrame
        Training features
    y_train : array-like, optional
        Training target (for display purposes)
    """
    print("🚀 DEMONSTRATING OPTIMAL CV STRATEGY FOR COMPETITION")
    print("=" * 60)
    
    # Create and test the optimal CV strategy
    cv = OptimalCompetitionCV(n_spatial_folds=3, n_january_folds=2)
    
    print(f"\n📊 CV Configuration:")
    print(f"   - Total folds: {cv.get_n_splits()}")
    print(f"   - Spatial folds: {cv.n_spatial_folds} (geographic extrapolation)")
    print(f"   - January folds: {cv.n_january_folds} (temporal distribution matching)")
    print(f"   - Spatial clusters: {cv.n_clusters}")
    
    # Analyze fold composition
    fold_info = cv.get_fold_info(X_train)
    
    print(f"\n🔍 Fold Analysis:")
    print("Fold | Type          | Train | Val | Jan% | Locations")
    print("-----|---------------|-------|-----|------|----------")
    
    total_january_pct = 0
    for info in fold_info:
        total_january_pct += info['january_pct']
        print(f"{info['fold_id']:4d} | {info['fold_type']:13s} | "
              f"{info['train_size']:5d} | {info['val_size']:3d} | "
              f"{info['january_pct']:4.1f}% | {info['unique_locations']:8d}")
    
    avg_january_pct = total_january_pct / len(fold_info)
    print(f"\n📈 Average January representation: {avg_january_pct:.1f}%")
    print(f"   - Training data: 4.4% January")
    print(f"   - Validation folds: {avg_january_pct:.1f}% January")
    print(f"   - Test data: 100.0% January")
    print(f"   - Improvement: {avg_january_pct/4.4:.1f}x better representation!")
    
    print(f"\n✅ SUCCESS METRICS:")
    print(f"   - Temporal matching: {avg_january_pct:.1f}% January (vs 4.4% in standard CV)")
    print(f"   - Spatial testing: {cv.n_spatial_folds} folds test geographic extrapolation")
    print(f"   - Efficiency: {cv.get_n_splits()} folds (fast enough for genetic programming)")
    print(f"   - Reliability: Addresses both major distribution shifts")
    
    print(f"\n🎯 INTEGRATION WITH GENETIC PROGRAMMING:")
    print(f"   generator = FeatureGenerator(cv=OptimalCompetitionCV(), ...)")
    print(f"   # Features will be evaluated with January-aware validation")
    print(f"   # Spatial extrapolation will be tested automatically")
    print(f"   # Feature ranking will be competition-specific")
    
    return cv


if __name__ == "__main__":
    # Example usage
    print("🏆 OPTIMAL CV STRATEGY FOR AIR POLLUTION PREDICTION COMPETITION")
    print("=" * 70)
    print("\nThis strategy addresses the extreme distribution shifts found in this competition:")
    print("- TEMPORAL: January 4.4% → 100.0% (191.2% total shift)")
    print("- SPATIAL: 95.0% novel locations in test set")
    print("\nUse OptimalCompetitionCV for maximum CV-LB correlation!")