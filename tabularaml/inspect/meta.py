import pandas as pd
import numpy as np
from scipy import stats
import warnings
from collections import OrderedDict
from typing import Dict, Optional, List, Union, Any, OrderedDict as OrderedDictType

class DatasetAnalyzer:
    """
    Optimized analyzer for tabular datasets that extracts meaningful metadata and statistics
    without using computationally expensive libraries like PyMFE.
    """
    def __init__(self, verbose: bool = False, max_classes_threshold: int = 25):
        self.verbose = verbose
        self.max_classes_threshold = max_classes_threshold
        self._reset_state()
        
        # Define metrics with their importance scores (higher = more important)
        # Core metrics applicable to all tasks, ordered by importance
        self.important_metrics = OrderedDict([
            ('task_type', 100),            # Most important - determines analysis type
            ('num_instances', 95),         # Sample size is critical
            ('num_features', 90),          # Feature count is fundamental
            ('missing_ratio', 85),         # Missing data affects all analyses
            ('outlier_ratio', 80),         # Outliers impact model performance
            ('attr_to_inst', 75),          # Feature-to-sample ratio (risk of overfitting)
            ('correlation_mean', 70),      # Feature interdependence
            ('entropy_mean', 65),          # Information content
            ('skewness_mean', 60),         # Distribution asymmetry
            ('kurtosis_mean', 55),         # Tail heaviness
            ('mad_mean', 50),              # Variability measure
            ('cat_cardinality_mean', 45),  # Average number of unique values per categorical feature
            ('cat_entropy_mean', 40),      # Information content of categorical features
            ('cat_mode_freq_mean', 35),    # Mode dominance in categorical features
        ])
        
        # Classification-specific metrics, ordered by importance
        self.clf_metrics = OrderedDict([
            ('n_classes', 90),             # Number of classes is fundamental for classification
            ('class_balance', 85),         # Class imbalance strongly affects performance
            ('majority_class_ratio', 80),  # Dominant class proportion
        ])
        
        # Regression-specific metrics, ordered by importance
        self.reg_metrics = OrderedDict([
            ('y_std', 90),                # Target variability is crucial for regression
            ('y_skew', 85),               # Target distribution skewness
            ('y_min', 75),                # Target range minimum
            ('y_max', 75),                # Target range maximum
        ])

    def _reset_state(self):
        self.meta_features: OrderedDictType[str, float] = OrderedDict()
        self.X: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.task_type: Optional[str] = None
        self.is_analyzed: bool = False

    def _analyze_missing_values(self, X: pd.DataFrame) -> Dict[str, float]:
        total_cells = X.size
        missing_cells = X.isnull().sum().sum()
        cols_with_missing = (X.isnull().sum() > 0).sum()
        missing_ratio = missing_cells / total_cells if total_cells else 0.0

        return {
            'missing_ratio': float(missing_ratio),
            'num_missing_cols': float(cols_with_missing),
            'total_missing': float(missing_cells),
        }

    def _impute_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        X_imp = X.copy()
        for col in X_imp.columns:
            if pd.api.types.is_numeric_dtype(X_imp[col]):
                if X_imp[col].isnull().any():
                    X_imp[col] = X_imp[col].fillna(X_imp[col].median())
            else:
                if X_imp[col].isnull().any():
                    mode = X_imp[col].mode()
                    fill = mode.iloc[0] if not mode.empty else 'missing'
                    X_imp[col] = X_imp[col].fillna(fill)
        return X_imp

    def _analyze_target(self, y: Optional[pd.Series]) -> Dict[str, float]:
        if y is None:
            return {'task_type': 'unsup'}

        uniq = int(y.nunique())
        is_num = pd.api.types.is_numeric_dtype(y)
        is_int = pd.api.types.is_integer_dtype(y)

        if (not is_num) or (is_int and uniq <= self.max_classes_threshold):
            freqs = y.value_counts() / len(y)
            return {
                'task_type': 'clf',
                'n_classes': float(uniq),
                'class_balance': float(freqs.std()),
                'majority_class_ratio': float(freqs.iloc[0]),
                'minority_class_ratio': float(freqs.iloc[-1]) if len(freqs)>1 else 0.0
            }
        else:
            # For regression tasks
            return {
                'task_type': 'reg',
                'y_mean': float(y.mean()),
                'y_std': float(y.std()),
                'y_min': float(y.min()),
                'y_max': float(y.max()),
                'y_skew': float(y.skew()),
            }

    def _compute_dataset_stats(self, X: pd.DataFrame) -> Dict[str, float]:
        stats_dict = {
            'num_instances': float(X.shape[0]),
            'num_features': float(X.shape[1]),
            'num_numeric_features': float(len(X.select_dtypes(include=[np.number]).columns)),
            'num_categorical_features': float(len(X.select_dtypes(exclude=[np.number]).columns)),
            'attr_to_inst': float(X.shape[1]) / X.shape[0] if X.shape[0] else np.nan
        }

        num = X.select_dtypes(include=[np.number])
        if not num.empty:
            total = num.size
            missing = num.isna().sum().sum()
            stats_dict['data_density'] = 1 - missing/total

        return stats_dict

    def _compute_outlier_metrics(self, X: pd.DataFrame) -> Dict[str, float]:
        num = X.select_dtypes(include=[np.number])
        if num.empty:
            return {'outlier_ratio': 0.0}

        Q1, Q3 = num.quantile(0.25), num.quantile(0.75)
        IQR = Q3 - Q1
        mask = ((num < (Q1 - 1.5 * IQR)) | (num > (Q3 + 1.5 * IQR)))
        ratio = mask.sum().sum() / num.size
        return {'outlier_ratio': float(ratio)}
    
    def _compute_manual_statistical_metrics(self, X: pd.DataFrame) -> Dict[str, float]:
        """Manually compute statistical measures instead of using PyMFE"""
        num_data = X.select_dtypes(include=[np.number])
        if num_data.empty:
            return {}
        
        result = {}
        
        # Calculate mean correlation (only for numeric features)
        corr_matrix = num_data.corr()
        # Extract upper triangle without diagonal
        upper_tri = np.triu(corr_matrix, k=1)
        corr_values = upper_tri[upper_tri != 0]
        if len(corr_values) > 0:
            result['correlation_mean'] = float(np.mean(np.abs(corr_values)))
        else:
            result['correlation_mean'] = 0.0
            
        # Calculate mean skewness for numeric features
        skewness_values = num_data.skew()
        result['skewness_mean'] = float(np.mean(np.abs(skewness_values)))
        
        # Calculate mean kurtosis for numeric features
        kurtosis_values = num_data.kurtosis()
        result['kurtosis_mean'] = float(np.mean(np.abs(kurtosis_values)))
        
        # Calculate mean absolute deviation
        mad_values = []
        for col in num_data.columns:
            median = num_data[col].median()
            mad = np.mean(np.abs(num_data[col] - median))
            mad_values.append(mad)
        result['mad_mean'] = float(np.mean(mad_values))
        
        # Calculate entropy for numeric features (binned)
        entropy_values = []
        for col in num_data.columns:
            # Bin the data into 10 bins for entropy calculation
            hist, _ = np.histogram(num_data[col].dropna(), bins=10)
            if hist.sum() > 0:
                hist = hist / hist.sum()
                # Calculate entropy only for non-zero probabilities
                entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
                entropy_values.append(entropy)
        
        if entropy_values:
            result['entropy_mean'] = float(np.mean(entropy_values))
        else:
            result['entropy_mean'] = 0.0
            
        return result
    
    def _compute_categorical_metrics(self, X: pd.DataFrame) -> Dict[str, float]:
        """Compute metrics specific to categorical features"""
        cat_data = X.select_dtypes(exclude=[np.number])
        if cat_data.empty:
            return {
                'cat_cardinality_mean': 0.0,
                'cat_entropy_mean': 0.0,
                'cat_mode_freq_mean': 0.0,
                'cat_cardinality_ratio_mean': 0.0
            }
        
        cardinality_values = []
        entropy_values = []
        mode_freq_values = []
        cardinality_ratio_values = []
        
        for col in cat_data.columns:
            # Calculate cardinality (number of unique values)
            unique_values = cat_data[col].nunique()
            total_values = len(cat_data[col])
            cardinality_values.append(unique_values)
            
            # Calculate cardinality ratio (unique values / total values)
            cardinality_ratio = unique_values / total_values if total_values > 0 else 0
            cardinality_ratio_values.append(cardinality_ratio)
            
            # Calculate value distribution and entropy
            value_counts = cat_data[col].value_counts(normalize=True)
            
            # Calculate entropy
            entropy = -np.sum(value_counts * np.log2(value_counts))
            entropy_values.append(entropy)
            
            # Calculate mode frequency (frequency of most common value)
            mode_freq = value_counts.iloc[0] if not value_counts.empty else 0
            mode_freq_values.append(mode_freq)
        
        result = {
            'cat_cardinality_mean': float(np.mean(cardinality_values)),
            'cat_entropy_mean': float(np.mean(entropy_values)),
            'cat_mode_freq_mean': float(np.mean(mode_freq_values)),
            'cat_cardinality_ratio_mean': float(np.mean(cardinality_ratio_values))
        }
        
        # Additional advanced metrics
        if len(cardinality_values) > 0:
            result['cat_cardinality_max'] = float(max(cardinality_values))
            result['cat_cardinality_min'] = float(min(cardinality_values))
            result['cat_entropy_std'] = float(np.std(entropy_values)) if len(entropy_values) > 1 else 0.0
        
        return result
    
    def _filter_important_metrics(self, all_metrics: Dict[str, float]) -> OrderedDictType[str, float]:
        """Filter and order metrics by their importance score"""
        # Create an OrderedDict to store metrics ordered by importance
        metrics_with_importance = {}
        
        # Add core metrics with their importance scores
        for key, importance in self.important_metrics.items():
            if key in all_metrics:
                metrics_with_importance[key] = (all_metrics[key], importance)
        
        # Add task-specific metrics with their importance scores
        if self.task_type == 'clf':
            for key, importance in self.clf_metrics.items():
                if key in all_metrics:
                    metrics_with_importance[key] = (all_metrics[key], importance)
        elif self.task_type == 'reg':
            for key, importance in self.reg_metrics.items():
                if key in all_metrics:
                    metrics_with_importance[key] = (all_metrics[key], importance)
        
        # Sort metrics by importance score (descending)
        sorted_metrics = sorted(metrics_with_importance.items(), 
                               key=lambda x: x[1][1], reverse=True)
        
        # Return as OrderedDict with just the values (not the importance scores)
        return OrderedDict([(k, v[0]) for k, v in sorted_metrics])

    def analyze(self, X: pd.DataFrame, y: Optional[pd.Series] = None, task: Optional[str] = None) -> OrderedDictType[str, float]:
        self._reset_state()
        self.X, self.y = X.copy(), (y.copy() if y is not None else None)
        all_results = {}

        if task in ('clf', 'reg', 'unsup'):
            self.task_type = task
            if task == 'clf' and self.y is not None:
                tgt = self._analyze_target(self.y)
                all_results.update(tgt)
            elif task == 'reg' and self.y is not None:
                tgt = self._analyze_target(self.y)
                all_results.update(tgt)
            else:
                all_results['task_type'] = 'unsup'
        else:
            tgt = self._analyze_target(self.y)
            self.task_type = tgt['task_type']
            all_results.update(tgt)

        all_results.update(self._compute_dataset_stats(self.X))
        all_results.update(self._analyze_missing_values(self.X))
        
        X_imp = self._impute_missing_values(self.X)
        all_results.update(self._compute_outlier_metrics(X_imp))
        
        # Replace PyMFE with manual calculations
        manual_stats = self._compute_manual_statistical_metrics(X_imp)
        all_results.update(manual_stats)

        # Compute categorical metrics
        cat_metrics = self._compute_categorical_metrics(X_imp)
        all_results.update(cat_metrics)

        # Order metrics by importance
        ordered_results = self._filter_important_metrics(all_results)
        self.meta_features = ordered_results
        self.is_analyzed = True

        if self.verbose:
            print(f"Extracted {len(ordered_results)} metrics, ordered by importance")
            if self.verbose and len(ordered_results) > 0:
                print("Top 5 most important metrics:")
                for i, (key, value) in enumerate(list(ordered_results.items())[:5]):
                    print(f"  {i+1}. {key}: {value}")

        return ordered_results

    def get_meta_features(self) -> OrderedDictType[str, float]:
        """Return the meta-features as an OrderedDict, sorted by importance"""
        if not self.is_analyzed:
            raise RuntimeError("Call .analyze() first")
        return self.meta_features
        
    def get_top_n_features(self, n: int = 5) -> OrderedDictType[str, float]:
        """Return the top N most important meta-features"""
        if not self.is_analyzed:
            raise RuntimeError("Call .analyze() first")
        return OrderedDict(list(self.meta_features.items())[:n])

# # Example usage:
# if __name__ == "__main__":
#     # Sample data
#     X = pd.DataFrame({
#         'A': [1, 2, np.nan, 4, 5],
#         'B': ['x', 'y', None, 'y', 'z'],
#         'C': [1.1, np.nan, 2.2, 3.3, np.nan]
#     })
#     y_clf = pd.Series([0, 1, 0, 1, 0])  # Classification
#     y_reg = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5])  # Regression
    
#     analyzer = DatasetAnalyzer(verbose=False)
    
#     print("\nClassification Dataset:")
#     metrics_clf = analyzer.analyze(X, y_clf)
#     print(metrics_clf)
    
#     print("\nRegression Dataset:")
#     metrics_reg = analyzer.analyze(X, y_reg)
#     print(metrics_reg)
    
#     print("\nUnsupervised Dataset:")
#     metrics_unsup = analyzer.analyze(X)
#     print(metrics_unsup)