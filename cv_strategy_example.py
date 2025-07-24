"""
Complete example demonstrating the improved CV strategies for the air pollution competition.
This script shows how to use all the enhanced CV approaches together for robust model validation.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings

# Import our enhanced CV strategies
from enhanced_spatial_temporal_cv import create_competition_optimized_cv
from improved_cv_strategy import setup_competition_cv
from test_distribution_cv import create_test_distribution_cv
from january_holdout_validation import create_january_holdout_validator

def load_and_prepare_data():
    """Load and prepare the competition data."""
    try:
        # Load data
        train_df = pd.read_csv('train.csv')
        
        # Prepare features and target
        feature_cols = ['latitude', 'longitude', 'day_of_year', 'day_of_week', 'hour', 'month']
        X = train_df[feature_cols].copy()
        y = train_df['pollution_value'].copy()
        
        print(f"Data loaded: {len(X)} samples, {len(feature_cols)} features")
        print(f"January ratio in training data: {(X['month'] == 1).mean():.3f}")
        
        return X, y
        
    except FileNotFoundError:
        print("train.csv not found. Creating synthetic data for demonstration.")
        return create_synthetic_data()

def create_synthetic_data(n_samples=5000):
    """Create synthetic data that mimics the competition structure."""
    np.random.seed(42)
    
    # Create synthetic features
    data = {
        'latitude': np.random.uniform(-60, 70, n_samples),
        'longitude': np.random.uniform(-180, 180, n_samples),
        'day_of_year': np.random.randint(1, 366, n_samples),
        'day_of_week': np.random.randint(0, 7, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'month': np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
                                 n_samples, 
                                 p=[0.15, 0.08, 0.08, 0.08, 0.12, 0.08, 0.08, 0.12, 0.08, 0.08, 0.08, 0.07])
    }
    
    X = pd.DataFrame(data)
    
    # Create synthetic target with some patterns
    y = (
        X['latitude'] * 0.5 +  # Latitude effect
        np.sin(2 * np.pi * X['day_of_year'] / 365) * 10 +  # Seasonal effect
        np.sin(2 * np.pi * X['hour'] / 24) * 5 +  # Daily cycle
        np.random.normal(0, 5, n_samples)  # Noise
    )
    
    return X, pd.Series(y)

def demonstrate_enhanced_spatial_temporal_cv(X, y):
    """Demonstrate the enhanced SpatialTemporalKFold."""
    print("\n" + "="*60)
    print("1. ENHANCED SPATIAL-TEMPORAL CV")
    print("="*60)
    
    # Create enhanced CV
    cv = create_competition_optimized_cv(n_splits=5, random_state=42)
    
    # Get detailed information
    info = cv.get_enhanced_split_info(X, y)
    
    print(f"CV Configuration:")
    print(f"  - Spatial clusters: {cv.spatial_clusters}")
    print(f"  - Temporal clusters: {cv.temporal_clusters}")
    print(f"  - Buffer distance: {cv.buffer_distance}")
    print(f"  - Temporal buffer: {cv.temporal_buffer} days")
    print(f"  - January-aware: {cv.january_aware}")
    
    print(f"\nData Analysis:")
    data_analysis = info['data_analysis']
    print(f"  - Total samples: {data_analysis['total_samples']}")
    print(f"  - January ratio: {data_analysis['january_ratio']:.3f}")
    print(f"  - Unique locations: {data_analysis['unique_spatial_locations']}")
    
    print(f"\nJanuary Distribution Quality:")
    jan_dist = info['january_distribution']
    print(f"  - Average train January ratio: {jan_dist['avg_train_january_ratio']:.3f}")
    print(f"  - Average val January ratio: {jan_dist['avg_val_january_ratio']:.3f}")
    print(f"  - Separation quality: {jan_dist['january_separation_quality']}")
    
    # Demonstrate CV splits
    scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Train a simple model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        
        # Evaluate
        y_pred = model.predict(X.iloc[val_idx])
        rmse = np.sqrt(mean_squared_error(y.iloc[val_idx], y_pred))
        scores.append(rmse)
        
        # Calculate January ratio in validation
        val_jan_ratio = (X.iloc[val_idx]['month'] == 1).mean()
        
        print(f"  Fold {fold_idx}: RMSE={rmse:.3f}, Val size={len(val_idx)}, Jan ratio={val_jan_ratio:.3f}")
    
    print(f"  Mean CV RMSE: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

def demonstrate_test_distribution_cv(X, y):
    """Demonstrate the test distribution matching CV."""
    print("\n" + "="*60)
    print("2. TEST DISTRIBUTION MATCHING CV")
    print("="*60)
    
    # Create test distribution CV
    cv = create_test_distribution_cv(n_splits=5, random_state=42)
    
    # Get distribution information
    info = cv.get_distribution_info(X)
    
    print(f"Target vs Achieved Distribution:")
    print(f"  - Target January ratio: {info['target_january_ratio']:.4f}")
    print(f"  - Achieved January ratio: {info['achieved_january_ratio']:.4f}")
    print(f"  - January error: {info['january_ratio_error']:.4f}")
    print(f"  - Distribution quality: {info['distribution_quality']}")
    
    # Demonstrate CV splits
    scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Train model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        
        # Evaluate
        y_pred = model.predict(X.iloc[val_idx])
        rmse = np.sqrt(mean_squared_error(y.iloc[val_idx], y_pred))
        scores.append(rmse)
        
        # Analyze validation set
        val_data = X.iloc[val_idx]
        jan_ratio = (val_data['month'] == 1).mean()
        feb_ratio = (val_data['month'] == 2).mean()
        
        print(f"  Fold {fold_idx}: RMSE={rmse:.3f}, Size={len(val_idx)}, Jan={jan_ratio:.4f}, Feb={feb_ratio:.4f}")
    
    print(f"  Mean CV RMSE: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

def demonstrate_january_holdout_validation(X, y):
    """Demonstrate the January holdout validation."""
    print("\n" + "="*60)
    print("3. JANUARY HOLDOUT VALIDATION")
    print("="*60)
    
    # Create holdout validator
    validator = create_january_holdout_validator(holdout_size=0.15, random_state=42)
    
    # Create multiple holdout sets
    holdouts = validator.create_multiple_january_holdouts(X, y, n_holdouts=3)
    
    print(f"Created {len(holdouts)} holdout sets:")
    
    # Validate quality of each holdout
    for name, (train_idx, holdout_idx) in holdouts.items():
        quality = validator.validate_holdout_quality(X, holdout_idx)
        
        print(f"\n  {name.upper()}:")
        print(f"    - Quality: {quality['quality']} (score: {quality['score']}/100)")
        print(f"    - Size: {quality['holdout_size']}")
        print(f"    - January ratio: {quality['january_ratio']:.3f}")
        print(f"    - Spatial range: {quality['spatial_range']['latitude']:.1f}° lat, {quality['spatial_range']['longitude']:.1f}° lon")
        print(f"    - Reasons: {', '.join(quality['reasons'][:2])}")
    
    # Demonstrate model evaluation on holdouts
    print(f"\nModel Performance on Holdouts:")
    
    # Train a model on the training data from the best holdout
    best_holdout = 'high_january'  # Use the high January ratio holdout
    train_idx, holdout_idx = holdouts[best_holdout]
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    
    # Evaluate on all holdouts
    results = validator.evaluate_model_on_holdouts(model, X, y, holdouts)
    
    for name, result in results.items():
        if 'error' not in result:
            print(f"  {name}: RMSE={result['rmse']:.3f}, January RMSE={result['january_rmse']:.3f if result['january_rmse'] else 'N/A'}")
        else:
            print(f"  {name}: Error - {result['error']}")

def demonstrate_comprehensive_strategy(X, y):
    """Demonstrate the comprehensive CV strategy."""
    print("\n" + "="*60)
    print("4. COMPREHENSIVE CV STRATEGY")
    print("="*60)
    
    # Setup comprehensive CV
    cv_setup = setup_competition_cv(X, y)
    cv_strategies = cv_setup['cv_strategies']
    primary_strategy = cv_setup['primary_strategy']
    
    print(f"Primary recommended strategy: {primary_strategy}")
    print(f"Available strategies: {list(cv_strategies.keys())}")
    
    # Evaluate each strategy
    strategy_scores = {}
    
    for strategy_name, splits in cv_strategies.items():
        if not splits:
            print(f"\n  {strategy_name}: No splits available")
            continue
            
        print(f"\n  {strategy_name.upper()}:")
        scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            # Train model
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            
            # Evaluate
            y_pred = model.predict(X.iloc[val_idx])
            rmse = np.sqrt(mean_squared_error(y.iloc[val_idx], y_pred))
            scores.append(rmse)
            
            # Get validation set info
            val_jan_ratio = (X.iloc[val_idx]['month'] == 1).mean()
            
            if fold_idx < 3:  # Show first 3 folds
                print(f"    Fold {fold_idx}: RMSE={rmse:.3f}, Val size={len(val_idx)}, Jan ratio={val_jan_ratio:.3f}")
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        strategy_scores[strategy_name] = mean_score
        
        print(f"    Mean RMSE: {mean_score:.3f} ± {std_score:.3f}")
    
    # Compare strategies
    print(f"\nStrategy Comparison (RMSE):")
    sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1])
    for strategy, score in sorted_strategies:
        print(f"  {strategy}: {score:.3f}")

def main():
    """Main demonstration function."""
    print("Air Pollution Competition - Enhanced CV Strategies Demo")
    print("="*60)
    
    # Load data
    X, y = load_and_prepare_data()
    
    # Demonstrate each CV strategy
    try:
        demonstrate_enhanced_spatial_temporal_cv(X, y)
    except Exception as e:
        print(f"Error in enhanced spatial-temporal CV: {e}")
    
    try:
        demonstrate_test_distribution_cv(X, y)
    except Exception as e:
        print(f"Error in test distribution CV: {e}")
    
    try:
        demonstrate_january_holdout_validation(X, y)
    except Exception as e:
        print(f"Error in January holdout validation: {e}")
    
    try:
        demonstrate_comprehensive_strategy(X, y)
    except Exception as e:
        print(f"Error in comprehensive strategy: {e}")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR COMPETITION")
    print("="*60)
    
    print("""
1. PRIMARY VALIDATION: Use Enhanced SpatialTemporalKFold
   - Optimized parameters for competition data
   - January-aware temporal buffering
   - Adaptive clustering

2. SECONDARY VALIDATION: Test Distribution Matching CV
   - Mirrors exact test set distribution (99.96% January)
   - Best estimate of final test performance
   - Use for final model selection

3. FINAL VALIDATION: January Holdout
   - Hold out 15% of data with high January concentration
   - Use for final model confidence assessment
   - Multiple holdout sets for robustness

4. ENSEMBLE APPROACH: 
   - Track performance across all CV strategies
   - Models that perform well across all strategies are most robust
   - Weight predictions based on CV performance correlation

5. FEATURE ENGINEERING PRIORITIES:
   - Strong seasonal features (month interactions)
   - Geographic clustering features
   - Temporal lag features for January patterns
   - Spatial-temporal interaction features
    """)

if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    main()