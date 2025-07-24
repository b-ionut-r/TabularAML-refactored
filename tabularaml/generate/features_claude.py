import os, random, time, math
import hashlib
import threading
from pathlib import Path
from collections import Counter, defaultdict, deque
from datetime import datetime
from itertools import combinations, permutations
from enum import Enum
from dataclasses import dataclass, field
from typing import Union, List, Optional, Callable, Literal, Dict, Any, Tuple, Set
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.metrics import pairwise_distances
from tqdm.auto import tqdm
from xgboost import XGBClassifier, XGBRegressor
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

# Assuming the helper modules (tabularaml.*) are in the correct path
from tabularaml.eval.cv import cross_val_score
from tabularaml.eval.scorers import PREDEFINED_REG_SCORERS, PREDEFINED_CLS_SCORERS, PREDEFINED_SCORERS, Scorer
from tabularaml.generate.ops import OPS, ALL_OPS_LAMBDAS
from tabularaml.inspect.importance import FeatureImportanceAnalyzer
from tabularaml.preprocessing.encoders import CategoricalEncoder
from tabularaml.preprocessing.imputers import SimpleImputer
from tabularaml.preprocessing.pipeline import PipelineWrapper
from tabularaml.configs.feature_gen import PRESET_PARAMS
from tabularaml.utils import is_gpu_available

# Enhanced Operation Registry - Only Row-wise Operations to Prevent Data Leakage
class OperationRegistry:
    """Plugin-based system for feature operations - all operations are row-wise to prevent data leakage"""
    
    def __init__(self):
        # Only include row-wise operations that don't use global statistics
        self.operations = {
            "numeric": {
                "unary": {
                    # Mathematical transformations (row-wise only)
                    "log": lambda x: np.log(np.abs(x) + 1e-8),
                    "sqrt": lambda x: np.sqrt(np.abs(x)),
                    "square": lambda x: x ** 2,
                    "cube": lambda x: x ** 3,
                    "inverse": lambda x: 1 / (x + 1e-8),
                    "abs": lambda x: np.abs(x),
                    "sign": lambda x: np.sign(x),
                    "neg": lambda x: -x,
                    "exp": lambda x: np.where(np.abs(x) <= 50, np.exp(x), np.nan),
                    
                    # Trigonometric (row-wise)
                    "sin": lambda x: np.sin(x),
                    "cos": lambda x: np.cos(x),
                    "tan": lambda x: np.where(np.abs(np.tan(x)) < 1e10, np.tan(x), np.nan),
                    "arcsin": lambda x: np.arcsin(np.clip(x, -1, 1)),
                    "arccos": lambda x: np.arccos(np.clip(x, -1, 1)),
                    "arctan": lambda x: np.arctan(x),
                    
                    # Advanced transformations (row-wise)
                    "sigmoid": lambda x: np.where(np.abs(x) <= 50, 1 / (1 + np.exp(-x)), np.where(x > 50, 1.0, 0.0)),
                    "tanh": lambda x: np.tanh(x),
                    "log10": lambda x: np.log10(np.abs(x) + 1e-8),
                    "log2": lambda x: np.log2(np.abs(x) + 1e-8),
                    "cbrt": lambda x: np.cbrt(x),
                    "floor": lambda x: np.floor(x),
                    "ceil": lambda x: np.ceil(x),
                    "round": lambda x: np.round(x),
                    
                    # Box-Cox like transformations (row-wise, fixed parameters)
                    "yeo_johnson": lambda x: self._yeo_johnson_transform(x),
                    "log1p": lambda x: np.log1p(np.abs(x)),
                    "power_2": lambda x: np.sign(x) * (np.abs(x) ** 0.5),
                    "power_3": lambda x: np.sign(x) * (np.abs(x) ** (1/3)),
                },
                "binary": {
                    # Basic arithmetic (row-wise)
                    "add": lambda x, y: x + y,
                    "subtract": lambda x, y: x - y,
                    "multiply": lambda x, y: np.where((np.abs(x * y)) < 1e15, x * y, np.nan),
                    "divide": lambda x, y: np.where(np.abs(y) > 1e-15, np.where(np.abs(x / y) < 1e15, x / y, np.nan), np.nan),
                    "power": lambda x, y: self._safe_power(x, y),
                    "modulo": lambda x, y: np.where(np.abs(y) > 1e-15, np.mod(x, y), np.nan),
                    
                    # Statistical combinations (row-wise)
                    "max": lambda x, y: np.maximum(x, y),
                    "min": lambda x, y: np.minimum(x, y),
                    "mean": lambda x, y: (x + y) / 2,
                    "geometric_mean": lambda x, y: np.where((x > 0) & (y > 0), np.sqrt(x * y), np.nan),
                    "harmonic_mean": lambda x, y: np.where((x > 0) & (y > 0), 2 / (1/(x + 1e-8) + 1/(y + 1e-8)), np.nan),
                    
                    # Distance metrics (row-wise)
                    "euclidean": lambda x, y: np.sqrt((x - y) ** 2),
                    "manhattan": lambda x, y: np.abs(x - y),
                    "relative_diff": lambda x, y: np.where(np.abs(y) > 1e-15, (x - y) / np.abs(y), np.nan),
                    "percent_change": lambda x, y: np.where(np.abs(y) > 1e-15, 100 * (x - y) / np.abs(y), np.nan),
                    "diff_ratio": lambda x, y: np.where((x + y) != 0, (x - y) / (np.abs(x + y) + 1e-15), np.nan),
                    
                    # Logical operations (row-wise)
                    "greater": lambda x, y: (x > y).astype(float),
                    "less": lambda x, y: (x < y).astype(float),
                    "equal": lambda x, y: (np.abs(x - y) < 1e-6).astype(float),
                    "not_equal": lambda x, y: (np.abs(x - y) >= 1e-6).astype(float),
                    
                    # Weighted combinations (row-wise)
                    "weighted_sum_07_03": lambda x, y: 0.7 * x + 0.3 * y,
                    "weighted_sum_06_04": lambda x, y: 0.6 * x + 0.4 * y,
                    "weighted_diff": lambda x, y: 0.7 * x - 0.3 * y,
                }
            },
            "categorical": {
                "unary": {
                    # These require pipeline to avoid data leakage
                    "count": "pipeline_required",  # Count encoding
                    "freq": "pipeline_required",   # Frequency encoding
                    "target": "pipeline_required", # Target encoding
                },
                "binary": {
                    "concat": lambda x, y: x.astype(str) + "_" + y.astype(str),
                    "match": lambda x, y: (x == y).astype(float),
                    "mismatch": lambda x, y: (x != y).astype(float),
                }
            },
            "temporal": {
                "unary": {
                    # Row-wise temporal extractions (no global stats)
                    "year": lambda x: pd.to_datetime(x).dt.year,
                    "month": lambda x: pd.to_datetime(x).dt.month,
                    "day": lambda x: pd.to_datetime(x).dt.day,
                    "hour": lambda x: pd.to_datetime(x).dt.hour,
                    "minute": lambda x: pd.to_datetime(x).dt.minute,
                    "dayofweek": lambda x: pd.to_datetime(x).dt.dayofweek,
                    "dayofyear": lambda x: pd.to_datetime(x).dt.dayofyear,
                    "quarter": lambda x: pd.to_datetime(x).dt.quarter,
                    "is_weekend": lambda x: (pd.to_datetime(x).dt.dayofweek >= 5).astype(float),
                    "is_month_end": lambda x: pd.to_datetime(x).dt.is_month_end.astype(float),
                    "is_month_start": lambda x: pd.to_datetime(x).dt.is_month_start.astype(float),
                    "week_of_year": lambda x: pd.to_datetime(x).dt.isocalendar().week.astype(float),
                },
                "binary": {
                    "days_diff": lambda x, y: (pd.to_datetime(x) - pd.to_datetime(y)).dt.days,
                    "years_diff": lambda x, y: (pd.to_datetime(x) - pd.to_datetime(y)).dt.days / 365.25,
                    "hours_diff": lambda x, y: (pd.to_datetime(x) - pd.to_datetime(y)).dt.total_seconds() / 3600,
                }
            }
        }
    
    def _yeo_johnson_transform(self, x):
        """Yeo-Johnson transformation with fixed lambda to avoid data leakage"""
        lam = 0.5  # Fixed lambda to avoid using data statistics
        return np.where(x >= 0, 
                       (np.power(np.abs(x) + 1, lam) - 1) / lam if lam != 0 else np.log(np.abs(x) + 1),
                       -(np.power(-x + 1, 2 - lam) - 1) / (2 - lam) if lam != 2 else -np.log(-x + 1))
    
    def _safe_power(self, base, exp):
        """Safe power operation"""
        # Limit exponent to prevent overflow
        exp_limited = np.clip(exp, -10, 10)
        base_safe = np.where(np.abs(base) < 1e5, base, np.sign(base) * 1e5)
        
        # Handle negative base with fractional exponent
        result = np.where(
            (base_safe >= 0) | (np.abs(exp_limited - np.round(exp_limited)) < 1e-10),
            np.power(np.abs(base_safe), exp_limited) * np.where(
                (base_safe < 0) & (np.round(exp_limited) % 2 == 1), -1, 1
            ),
            np.nan
        )
        
        return np.where(np.abs(result) < 1e15, result, np.nan)
    
    def register_operation(self, dtype: str, op_type: str, name: str, func: Callable):
        """Register new operation"""
        if dtype not in self.operations:
            self.operations[dtype] = {}
        if op_type not in self.operations[dtype]:
            self.operations[dtype][op_type] = {}
        self.operations[dtype][op_type][name] = func
    
    def get_operations(self, dtype: str, op_type: str) -> Dict[str, Callable]:
        """Get operations for specific dtype and type"""
        return self.operations.get(dtype, {}).get(op_type, {})

# Feature Semantics Tracking
@dataclass
class FeatureSemantics:
    """Track semantic information about features"""
    name: str
    dtype: str
    domain: str  # e.g., "financial", "temporal", "categorical"
    meaning: str  # Human-readable description
    derivation_path: List[str] = field(default_factory=list)
    statistical_properties: Dict[str, float] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    
    def __hash__(self):
        return hash((self.name, self.dtype, self.domain))
    
    def get_semantic_distance(self, other: 'FeatureSemantics') -> float:
        """Calculate semantic distance between features"""
        domain_match = 1.0 if self.domain == other.domain else 0.0
        dtype_match = 1.0 if self.dtype == other.dtype else 0.0
        path_similarity = len(set(self.derivation_path) & set(other.derivation_path)) / max(len(self.derivation_path), len(other.derivation_path), 1)
        
        return (domain_match + dtype_match + path_similarity) / 3.0

# Enhanced Feature Class
class EnhancedFeature:
    """Enhanced feature with semantic information and hierarchical construction"""
    
    def __init__(self, name: str, dtype: Literal["num", "cat", "temporal"], 
                 weight: float = 1.0, depth: int = 0, 
                 semantics: Optional[FeatureSemantics] = None,
                 require_pipeline: bool = False):
        self.name = name
        self.dtype = dtype
        self.weight = weight
        self.depth = depth
        self.semantics = semantics or FeatureSemantics(name, dtype, "unknown", "Base feature")
        self.require_pipeline = require_pipeline
        self.generating_interaction = None
        self.performance_history = []
        self.usage_count = 0
        self.creation_time = time.time()
        self.fitness_scores = {}
        
    def update_weight(self, new_weight: float):
        """Update feature weight"""
        self.weight = new_weight
        
    def update_performance(self, score: float, context: str = ""):
        """Update performance history"""
        self.performance_history.append({
            'score': score,
            'context': context,
            'timestamp': time.time()
        })
        
    def get_average_performance(self) -> float:
        """Get average performance across history"""
        if not self.performance_history:
            return 0.0
        return np.mean([p['score'] for p in self.performance_history])
    
    def set_generating_interaction(self, interaction: 'AdvancedInteraction'):
        """Set the interaction that generated this feature"""
        self.generating_interaction = interaction
        if interaction:
            self.semantics.derivation_path.extend(interaction.get_derivation_path())
            self.semantics.dependencies.update(interaction.get_dependencies())
    
    def get_complexity_score(self) -> float:
        """Calculate complexity score based on depth and dependencies"""
        return self.depth + len(self.semantics.dependencies) * 0.1
    
    def is_functionally_equivalent(self, other: 'EnhancedFeature', threshold: float = 0.99) -> bool:
        """Check if two features are functionally equivalent"""
        if self.dtype != other.dtype:
            return False
        
        semantic_sim = self.semantics.get_semantic_distance(other.semantics)
        return semantic_sim > threshold

# Advanced Interaction Class
class AdvancedInteraction:
    """Advanced interaction supporting hierarchical composition and semantic tracking"""
    
    def __init__(self, 
                 components: List[Union[EnhancedFeature, 'AdvancedInteraction']], 
                 operation: str,
                 operation_type: str = "binary",
                 metadata: Optional[Dict[str, Any]] = None):
        self.components = components
        self.operation = operation
        self.operation_type = operation_type
        self.metadata = metadata or {}
        
        # Infer properties from components
        self.dtype = self._infer_dtype()
        self.depth = self._calculate_depth()
        self.weight = self._calculate_weight()
        self.name = self._generate_name()
        self.require_pipeline = self._check_pipeline_requirement()
        
        # Semantic information
        self.semantics = self._derive_semantics()
        
        # Performance tracking
        self.success_rate = 0.5
        self.average_gain = 0.0
        self.evaluation_count = 0
        
    def _infer_dtype(self) -> str:
        """Infer data type from components"""
        dtypes = [comp.dtype if isinstance(comp, EnhancedFeature) else comp.dtype 
                 for comp in self.components]
        
        if all(dt == "num" for dt in dtypes):
            return "num"
        elif all(dt == "cat" for dt in dtypes):
            return "cat"
        elif "temporal" in dtypes:
            return "temporal"
        else:
            return "cat"  # Mixed types default to categorical
    
    def _calculate_depth(self) -> int:
        """Calculate depth based on components"""
        max_depth = 0
        for comp in self.components:
            if isinstance(comp, EnhancedFeature):
                max_depth = max(max_depth, comp.depth)
            else:
                max_depth = max(max_depth, comp.depth)
        return max_depth + 1
    
    def _calculate_weight(self) -> float:
        """Calculate weight based on components"""
        weights = []
        for comp in self.components:
            if isinstance(comp, EnhancedFeature):
                weights.append(comp.weight)
            else:
                weights.append(comp.weight)
        return np.mean(weights) if weights else 1.0
    
    def _generate_name(self) -> str:
        """Generate name for the interaction"""
        comp_names = []
        for comp in self.components:
            if isinstance(comp, EnhancedFeature):
                comp_names.append(comp.name)
            else:
                comp_names.append(comp.name)
        
        if len(comp_names) == 1:
            return f"{comp_names[0]}_{self.operation}"
        else:
            return f"{comp_names[0]}_{self.operation}_{'_'.join(comp_names[1:])}"
    
    def _check_pipeline_requirement(self) -> bool:
        """Check if pipeline is required"""
        return self.operation in ["target", "count", "freq"] or any(
            (isinstance(comp, EnhancedFeature) and comp.require_pipeline) or
            (isinstance(comp, AdvancedInteraction) and comp.require_pipeline)
            for comp in self.components
        )
    
    def _derive_semantics(self) -> FeatureSemantics:
        """Derive semantic information from components"""
        # Combine semantic information from components
        domains = set()
        meanings = []
        derivation_paths = []
        dependencies = set()
        
        for comp in self.components:
            if isinstance(comp, EnhancedFeature):
                domains.add(comp.semantics.domain)
                meanings.append(comp.semantics.meaning)
                derivation_paths.extend(comp.semantics.derivation_path)
                dependencies.update(comp.semantics.dependencies)
                dependencies.add(comp.name)
            else:
                domains.add(comp.semantics.domain)
                meanings.append(comp.semantics.meaning)
                derivation_paths.extend(comp.semantics.derivation_path)
                dependencies.update(comp.semantics.dependencies)
        
        # Create combined semantics
        combined_domain = list(domains)[0] if len(domains) == 1 else "mixed"
        combined_meaning = f"{self.operation}({', '.join(meanings)})"
        
        return FeatureSemantics(
            name=self.name,
            dtype=self.dtype,
            domain=combined_domain,
            meaning=combined_meaning,
            derivation_path=derivation_paths + [self.operation],
            dependencies=dependencies
        )
    
    def get_derivation_path(self) -> List[str]:
        """Get the full derivation path"""
        return self.semantics.derivation_path
    
    def get_dependencies(self) -> Set[str]:
        """Get all feature dependencies"""
        return self.semantics.dependencies
    
    def _map_dtype_to_operation_category(self, dtype: str) -> str:
        """Map internal dtype to operation registry category"""
        if dtype == "num":
            return "numeric"
        elif dtype == "cat":
            return "categorical"
        elif dtype == "temporal":
            return "temporal"
        else:
            return "numeric"  # Default fallback
    
    def generate(self, X: pd.DataFrame, y: pd.Series = None, 
                operation_registry: OperationRegistry = None, feature_cache: Dict[str, np.ndarray] = None) -> Tuple[str, np.ndarray]:
        """Generate feature values"""
        if operation_registry is None:
            operation_registry = OperationRegistry()
        
        if feature_cache is None:
            feature_cache = {}
        
        # If this feature has already been computed and cached in this run, return it.
        if self.name in feature_cache:
            return self.name, feature_cache[self.name]

        if self.require_pipeline:
            raise Exception("Can't generate feature using lambdas. Requires pipeline to avoid data leakage.")
        
        # Get component values, generating them recursively if they are not available.
        component_values = []
        for comp in self.components:
            # Check cache first
            if comp.name in feature_cache:
                component_values.append(feature_cache[comp.name])
                continue

            # Check if it's a base feature in the provided DataFrame
            if comp.name in X.columns:
                values = X[comp.name].values
                component_values.append(values)
                feature_cache[comp.name] = values  # Cache for this run
                continue
            
            # If not found, it's a derived feature that must be generated.
            # Find the interaction definition needed to generate the component.
            interaction_to_generate = None
            if isinstance(comp, AdvancedInteraction):
                interaction_to_generate = comp
            elif isinstance(comp, EnhancedFeature) and comp.generating_interaction:
                interaction_to_generate = comp.generating_interaction
            
            if interaction_to_generate:
                # Recursively call generate. The result will be cached within that call.
                _, values = interaction_to_generate.generate(X, y, operation_registry, feature_cache)
                component_values.append(values)
            else:
                # If we can't find it and can't generate it, we must raise an error.
                raise ValueError(
                    f"Feature '{comp.name}' not found in DataFrame or feature cache, "
                    f"and its generation method is not defined."
                )
        
        # Apply operation
        if self.operation_type == "unary" and len(component_values) == 1:
            # Map dtype to operation registry category
            op_category = self._map_dtype_to_operation_category(self.dtype)
            op_func = operation_registry.get_operations(op_category, "unary").get(self.operation)
            if op_func and op_func != "pipeline_required":
                result = op_func(component_values[0])
            else:
                raise ValueError(f"Unknown or pipeline-required unary operation: {self.operation}")
        elif self.operation_type == "binary" and len(component_values) == 2:
            # Map dtype to operation registry category  
            op_category = self._map_dtype_to_operation_category(self.dtype)
            op_func = operation_registry.get_operations(op_category, "binary").get(self.operation)
            if op_func and op_func != "pipeline_required":
                result = op_func(component_values[0], component_values[1])
            else:
                raise ValueError(f"Unknown or pipeline-required binary operation: {self.operation}")
        else:
            raise ValueError(f"Unsupported operation type: {self.operation_type}")
        
        # Handle different result types
        if isinstance(result, pd.Series):
            result = result.values
        elif not isinstance(result, np.ndarray):
            result = np.array(result)
        
        # Cache the result of this generation
        if feature_cache is not None:
            feature_cache[self.name] = result
        
        return self.name, result
    
    def get_new_feature_instance(self) -> EnhancedFeature:
        """Create a new EnhancedFeature instance from this interaction"""
        feature = EnhancedFeature(
            name=self.name,
            dtype=self.dtype,
            weight=self.weight,
            depth=self.depth,
            semantics=self.semantics,
            require_pipeline=self.require_pipeline
        )
        feature.set_generating_interaction(self)
        return feature
    
    def update_performance(self, success: bool, gain: float = 0.0):
        """Update performance statistics"""
        self.evaluation_count += 1
        alpha = 0.1  # Learning rate
        
        if success:
            self.success_rate = (1 - alpha) * self.success_rate + alpha * 1.0
            self.average_gain = (1 - alpha) * self.average_gain + alpha * gain
        else:
            self.success_rate = (1 - alpha) * self.success_rate + alpha * 0.0
    
    def get_priority_score(self) -> float:
        """Calculate priority score for selection"""
        return 0.7 * self.success_rate + 0.3 * min(1.0, self.average_gain * 10)

# Multi-Population Evolution System
class Population:
    """Individual population with specific evolution strategy"""
    
    def __init__(self, population_id: str, strategy: str, size: int = 100):
        self.population_id = population_id
        self.strategy = strategy
        self.size = size
        self.generation = []
        self.fitness_history = []
        self.diversity_metrics = {}
        self.age = 0
        
    def calculate_diversity(self) -> float:
        """Calculate population diversity"""
        if len(self.generation) < 2:
            return 1.0
        
        # Semantic diversity
        semantic_distances = []
        for i in range(len(self.generation)):
            for j in range(i + 1, len(self.generation)):
                feat1, feat2 = self.generation[i], self.generation[j]
                dist = feat1.semantics.get_semantic_distance(feat2.semantics)
                semantic_distances.append(dist)
        
        return np.mean(semantic_distances) if semantic_distances else 1.0
    
    def get_best_features(self, n: int = 10) -> List[EnhancedFeature]:
        """Get best features from population"""
        sorted_features = sorted(self.generation, key=lambda f: f.weight, reverse=True)
        return sorted_features[:n]
    
    def add_feature(self, feature: EnhancedFeature):
        """Add feature to population"""
        self.generation.append(feature)
        
        # Maintain population size
        if len(self.generation) > self.size:
            # Remove worst features
            self.generation.sort(key=lambda f: f.weight, reverse=True)
            self.generation = self.generation[:self.size]
    
    def evolve(self, operation_registry: OperationRegistry, 
               mutation_rate: float = 0.1) -> List[AdvancedInteraction]:
        """Evolve population according to strategy"""
        candidates = []
        
        if self.strategy == "exploitation":
            # Focus on best features
            best_features = self.get_best_features(min(20, len(self.generation)))
            candidates.extend(self._generate_exploitation_candidates(best_features, operation_registry))
        
        elif self.strategy == "exploration":
            # Focus on diverse features
            diverse_features = self._select_diverse_features(min(20, len(self.generation)))
            candidates.extend(self._generate_exploration_candidates(diverse_features, operation_registry))
        
        elif self.strategy == "balanced":
            # Mix of exploitation and exploration
            best_features = self.get_best_features(min(10, len(self.generation)))
            diverse_features = self._select_diverse_features(min(10, len(self.generation)))
            candidates.extend(self._generate_exploitation_candidates(best_features, operation_registry))
            candidates.extend(self._generate_exploration_candidates(diverse_features, operation_registry))
        
        self.age += 1
        return candidates
    
    def _select_diverse_features(self, n: int) -> List[EnhancedFeature]:
        """Select diverse features from population"""
        if len(self.generation) <= n:
            return self.generation[:]
        
        selected = [random.choice(self.generation)]
        
        for _ in range(n - 1):
            best_candidate = None
            best_min_distance = -1
            
            for candidate in self.generation:
                if candidate in selected:
                    continue
                
                min_distance = float('inf')
                for selected_feat in selected:
                    dist = candidate.semantics.get_semantic_distance(selected_feat.semantics)
                    min_distance = min(min_distance, dist)
                
                if min_distance > best_min_distance:
                    best_min_distance = min_distance
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
        
        return selected
    
    def _generate_exploitation_candidates(self, features: List[EnhancedFeature], 
                                        operation_registry: OperationRegistry) -> List[AdvancedInteraction]:
        """Generate candidates focused on exploitation"""
        candidates = []
        
        # Unary operations on best features
        for feat in features[:10]:
            for op in operation_registry.get_operations(feat.dtype, "unary"):
                candidates.append(AdvancedInteraction([feat], op, "unary"))
        
        # Binary operations between best features
        for i in range(len(features)):
            for j in range(i + 1, min(i + 5, len(features))):
                feat1, feat2 = features[i], features[j]
                op_dtype = "numeric" if feat1.dtype == feat2.dtype == "num" else "categorical"
                for op in list(operation_registry.get_operations(op_dtype, "binary").keys())[:3]:
                    candidates.append(AdvancedInteraction([feat1, feat2], op, "binary"))
        
        return candidates
    
    def _generate_exploration_candidates(self, features: List[EnhancedFeature], 
                                       operation_registry: OperationRegistry) -> List[AdvancedInteraction]:
        """Generate candidates focused on exploration"""
        candidates = []
        
        # Random operations
        for _ in range(50):
            if random.random() < 0.3:  # Unary
                feat = random.choice(features)
                ops = list(operation_registry.get_operations(feat.dtype, "unary").keys())
                if ops:
                    op = random.choice(ops)
                    candidates.append(AdvancedInteraction([feat], op, "unary"))
            else:  # Binary
                feat1, feat2 = random.sample(features, 2)
                op_dtype = "numeric" if feat1.dtype == feat2.dtype == "num" else "categorical"
                ops = list(operation_registry.get_operations(op_dtype, "binary").keys())
                if ops:
                    op = random.choice(ops)
                    candidates.append(AdvancedInteraction([feat1, feat2], op, "binary"))
        
        return candidates

# Multi-Objective Feature Evaluator
class MultiObjectiveEvaluator:
    """Evaluate features on multiple objectives"""
    
    def __init__(self, objectives: List[str] = None):
        self.objectives = objectives or ["accuracy", "complexity", "diversity"]
        self.pareto_front = []
        
    def evaluate_feature(self, feature: EnhancedFeature, X: pd.DataFrame, y: pd.Series,
                        baseline_score: float, model, scorer, cv) -> Dict[str, float]:
        """Evaluate feature on multiple objectives"""
        scores = {}
        
        # Accuracy objective
        if "accuracy" in self.objectives:
            X_with_feat = X.copy()
            if feature.name in X_with_feat.columns:
                try:
                    cv_result = cross_val_score(model, X_with_feat, y, scorer, cv=cv, return_dict=True)
                    improvement = cv_result["mean_val_score"] - baseline_score
                    scores["accuracy"] = improvement
                except:
                    scores["accuracy"] = 0.0
            else:
                scores["accuracy"] = 0.0
        
        # Complexity objective (minimize)
        if "complexity" in self.objectives:
            scores["complexity"] = -feature.get_complexity_score()
        
        # Diversity objective
        if "diversity" in self.objectives:
            diversity_score = self._calculate_diversity_score(feature, X)
            scores["diversity"] = diversity_score
        
        # Interpretability objective
        if "interpretability" in self.objectives:
            scores["interpretability"] = self._calculate_interpretability_score(feature)
        
        return scores
    
    def _calculate_diversity_score(self, feature: EnhancedFeature, X: pd.DataFrame) -> float:
        """Calculate diversity score based on correlation with existing features"""
        if feature.name not in X.columns:
            return 1.0
        
        try:
            feat_values = X[feature.name].values
            correlations = []
            
            for col in X.columns:
                if col != feature.name:
                    corr = np.corrcoef(feat_values, X[col].values)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            if correlations:
                return 1.0 - np.max(correlations)  # Lower correlation = higher diversity
            else:
                return 1.0
        except:
            return 1.0
    
    def _calculate_interpretability_score(self, feature: EnhancedFeature) -> float:
        """Calculate interpretability score"""
        # Simple heuristic: fewer operations = more interpretable
        base_score = 1.0
        complexity_penalty = feature.depth * 0.1
        dependency_penalty = len(feature.semantics.dependencies) * 0.05
        
        return max(0.1, base_score - complexity_penalty - dependency_penalty)
    
    def is_pareto_optimal(self, scores: Dict[str, float]) -> bool:
        """Check if scores represent a Pareto optimal solution"""
        for front_scores, front_feature in self.pareto_front:
            dominates = True
            for obj in self.objectives:
                if scores[obj] < front_scores[obj]:
                    dominates = False
                    break
            if dominates:
                return False
        return True
    
    def update_pareto_front(self, scores: Dict[str, float], feature: EnhancedFeature):
        """Update Pareto front with new solution"""
        if self.is_pareto_optimal(scores):
            # Remove dominated solutions
            self.pareto_front = [
                (s, f) for s, f in self.pareto_front
                if not all(scores[obj] >= s[obj] for obj in self.objectives)
            ]
            # Add new solution
            self.pareto_front.append((scores, feature))

# Meta-Learning Knowledge Base
class MetaLearningKnowledgeBase:
    """Knowledge base for meta-learning across datasets"""
    
    def __init__(self):
        self.successful_patterns = defaultdict(list)
        self.failed_patterns = defaultdict(list)
        self.dataset_characteristics = {}
        
    def record_success(self, dataset_id: str, pattern: str, context: Dict[str, Any]):
        """Record successful pattern"""
        self.successful_patterns[dataset_id].append({
            'pattern': pattern,
            'context': context,
            'timestamp': time.time()
        })
    
    def record_failure(self, dataset_id: str, pattern: str, context: Dict[str, Any]):
        """Record failed pattern"""
        self.failed_patterns[dataset_id].append({
            'pattern': pattern,
            'context': context,
            'timestamp': time.time()
        })
    
    def get_dataset_characteristics(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Extract dataset characteristics"""
        characteristics = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_numeric': len(X.select_dtypes(include=[np.number]).columns),
            'n_categorical': len(X.select_dtypes(include=['object', 'category']).columns),
            'target_type': 'regression' if type_of_target(y) == 'continuous' else 'classification',
            'class_balance': len(np.unique(y)) if type_of_target(y) != 'continuous' else 1,
            'missing_ratio': X.isnull().sum().sum() / (X.shape[0] * X.shape[1]),
            'feature_correlation_avg': np.mean(np.abs(X.corr().values[np.triu_indices(X.shape[1], k=1)]))
        }
        return characteristics
    
    def suggest_operations(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Suggest operations based on similar datasets"""
        current_chars = self.get_dataset_characteristics(X, y)
        
        # Find similar datasets
        similar_datasets = []
        for dataset_id, chars in self.dataset_characteristics.items():
            similarity = self._calculate_similarity(current_chars, chars)
            if similarity > 0.7:  # Threshold for similarity
                similar_datasets.append(dataset_id)
        
        # Extract successful patterns
        suggested_ops = []
        for dataset_id in similar_datasets:
            for success in self.successful_patterns[dataset_id]:
                suggested_ops.append(success['pattern'])
        
        return list(set(suggested_ops))
    
    def _calculate_similarity(self, chars1: Dict[str, Any], chars2: Dict[str, Any]) -> float:
        """Calculate similarity between dataset characteristics"""
        similarities = []
        
        # Numeric features
        for key in ['n_samples', 'n_features', 'n_numeric', 'n_categorical']:
            if key in chars1 and key in chars2:
                val1, val2 = chars1[key], chars2[key]
                sim = 1 - abs(val1 - val2) / max(val1, val2, 1)
                similarities.append(sim)
        
        # Categorical features
        for key in ['target_type']:
            if key in chars1 and key in chars2:
                sim = 1.0 if chars1[key] == chars2[key] else 0.0
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0

# Enhanced Adaptive Controller
class EnhancedAdaptiveController:
    """Enhanced adaptive controller with reinforcement learning"""
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.operation_q_values = defaultdict(lambda: defaultdict(float))
        self.operation_counts = defaultdict(lambda: defaultdict(int))
        self.state_history = []
        self.reward_history = []
        
    def get_operation_priority(self, state: Dict[str, Any], 
                              available_ops: List[str]) -> List[str]:
        """Get operation priority using Q-learning"""
        state_key = self._encode_state(state)
        
        # Epsilon-greedy selection
        epsilon = 0.1
        if random.random() < epsilon:
            return random.sample(available_ops, len(available_ops))
        
        # Sort by Q-values
        op_values = [(op, self.operation_q_values[state_key][op]) for op in available_ops]
        op_values.sort(key=lambda x: x[1], reverse=True)
        
        return [op for op, _ in op_values]
    
    def update_q_values(self, state: Dict[str, Any], action: str, reward: float, 
                       next_state: Dict[str, Any]):
        """Update Q-values using Q-learning"""
        state_key = self._encode_state(state)
        next_state_key = self._encode_state(next_state)
        
        # Q-learning update
        current_q = self.operation_q_values[state_key][action]
        max_next_q = max(self.operation_q_values[next_state_key].values(), default=0.0)
        
        new_q = current_q + self.learning_rate * (reward + 0.9 * max_next_q - current_q)
        self.operation_q_values[state_key][action] = new_q
        
        # Update counts
        self.operation_counts[state_key][action] += 1
    
    def _encode_state(self, state: Dict[str, Any]) -> str:
        """Encode state to string for indexing"""
        return f"gen_{state.get('generation', 0)}_stag_{state.get('stagnation', 0)}_div_{state.get('diversity', 0):.2f}"

# Main Enhanced Feature Generator
class EnhancedFeatureGenerator:
    """Enhanced feature generator with all improvements"""
    
    def __init__(self,
                 baseline_model=None,
                 model_fit_kwargs: dict = None,
                 task: Optional[Literal["regression", "classification"]] = None,
                 scorer: Optional[Scorer] = None,
                 n_generations: int = 50,
                 n_populations: int = 5,
                 population_size: int = 100,
                 n_children: int = 250,
                 cv: Union[int, BaseCrossValidator] = 5,
                 use_gpu: bool = True,
                 enable_meta_learning: bool = True,
                 enable_multi_objective: bool = True,
                 enable_distributed: bool = True,
                 n_workers: int = 2,
                 log_file: Union[str, Path] = None,
                 time_budget: Optional[int] = 7200,
                 save_path: Optional[Union[str, Path]] = "./feature_gen_checkpoints"):
        
        # Core parameters
        self.baseline_model = baseline_model
        self.model_fit_kwargs = model_fit_kwargs or {}
        self.task = task
        self.scorer = scorer
        self.n_generations = n_generations
        self.n_populations = n_populations
        self.population_size = population_size
        self.n_children = n_children
        self.cv = cv
        self.use_gpu = use_gpu
        self.enable_meta_learning = enable_meta_learning
        self.enable_multi_objective = enable_multi_objective
        self.enable_distributed = enable_distributed
        self.n_workers = n_workers
        self.log_file = log_file
        self.time_budget = time_budget
        self.save_path = save_path
        
        # Initialize components
        self.operation_registry = OperationRegistry()
        self.populations = []
        self.multi_objective_evaluator = MultiObjectiveEvaluator() if enable_multi_objective else None
        self.meta_learning_kb = MetaLearningKnowledgeBase() if enable_meta_learning else None
        self.adaptive_controller = EnhancedAdaptiveController()
        
        # Initialize device
        self.device = "cuda" if is_gpu_available() and use_gpu else "cpu"
        
        # Initialize pipeline
        self.pipeline = PipelineWrapper(
            imputer=None, 
            scaler=None, 
            encoder=CategoricalEncoder()
        )
        
        # State tracking
        self.state = {
            'generation': 0,
            'best_score': float('-inf'),
            'best_features': [],
            'stagnation_count': 0,
            'diversity_history': [],
            'feature_history': []
        }
        
        # Performance metrics
        self.metrics = {
            'generation_times': [],
            'feature_counts': [],
            'score_improvements': [],
            'population_diversities': []
        }
        
        # Setup logging
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
    def _log(self, message: str):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        print(full_message)
        
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(full_message + "\n")
    
    def _setup_task_components(self, X: pd.DataFrame, y: pd.Series):
        """Setup task-specific components"""
        # Infer task type
        if self.task is None:
            self.task = "regression" if type_of_target(y) == "continuous" else "classification"
        
        # Setup model
        if self.baseline_model is None:
            if self.task == "regression":
                self.baseline_model = XGBRegressor(
                    device=self.device, 
                    enable_categorical=True, 
                    verbosity=0
                )
            else:
                self.baseline_model = XGBClassifier(
                    device=self.device, 
                    enable_categorical=True, 
                    verbosity=0
                )
        
        # Setup scorer
        if self.scorer is None:
            if self.task == "regression":
                self.scorer = PREDEFINED_REG_SCORERS["rmse"]
            else:
                if len(np.unique(y)) == 2:
                    self.scorer = PREDEFINED_CLS_SCORERS["binary_crossentropy"]
                else:
                    self.scorer = PREDEFINED_CLS_SCORERS["categorical_crossentropy"]
    
    def _initialize_populations(self, X: pd.DataFrame, y: pd.Series):
        """Initialize multiple populations with different strategies"""
        # Get initial feature importance
        analyzer = FeatureImportanceAnalyzer(
            task_type=self.task,
            pipeline=self.pipeline,
            cv=self.cv,
            use_gpu=(self.device == "cuda")
        )
        analyzer.fit(X, y)
        importance_df = analyzer.get_importance(normalize=True)
        
        # Create base features
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        base_features = []
        for col in X.columns:
            dtype = "num" if col in num_cols else "cat"
            weight = importance_df.loc[col, "weighted_importance"] if col in importance_df.index else 0.1
            
            # Create semantic information
            semantics = FeatureSemantics(
                name=col,
                dtype=dtype,
                domain=self._infer_domain(col, X[col]),
                meaning=f"Base {dtype} feature: {col}"
            )
            
            feature = EnhancedFeature(
                name=col,
                dtype=dtype,
                weight=weight,
                semantics=semantics
            )
            base_features.append(feature)
        
        # Create populations with different strategies
        strategies = ["exploitation", "exploration", "balanced"]
        for i in range(self.n_populations):
            strategy = strategies[i % len(strategies)]
            population = Population(
                population_id=f"pop_{i}",
                strategy=strategy,
                size=self.population_size
            )
            
            # Add base features to each population
            for feature in base_features:
                population.add_feature(deepcopy(feature))
            
            self.populations.append(population)
    
    def _infer_domain(self, col_name: str, col_data: pd.Series) -> str:
        """Infer domain from column name and data"""
        name_lower = col_name.lower()
        
        # Financial keywords
        if any(keyword in name_lower for keyword in ['price', 'cost', 'amount', 'value', 'revenue', 'profit']):
            return "financial"
        
        # Temporal keywords
        if any(keyword in name_lower for keyword in ['date', 'time', 'year', 'month', 'day']):
            return "temporal"
        
        # Geographical keywords
        if any(keyword in name_lower for keyword in ['location', 'address', 'city', 'country', 'region']):
            return "geographical"
        
        # Demographic keywords
        if any(keyword in name_lower for keyword in ['age', 'gender', 'education', 'income']):
            return "demographic"
        
        return "general"
    
    def _evaluate_candidates_parallel(self, candidates: List[AdvancedInteraction], 
                                    X: pd.DataFrame, y: pd.Series) -> List[Tuple[AdvancedInteraction, float]]:
        """Evaluate candidates in parallel with a nested progress bar."""
        if not self.enable_distributed:
            return self._evaluate_candidates_sequential(candidates, X, y)
        
        results = []
        
        base_feature_cache = {col: X[col].values for col in X.columns}
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_candidate = {
                executor.submit(self._evaluate_single_candidate, candidate, X, y, base_feature_cache.copy()): candidate
                for candidate in candidates
            }
            
            # FIXED: Wrap as_completed with tqdm for a nested progress bar
            pbar_candidates = tqdm(as_completed(future_to_candidate), total=len(candidates), desc="  Evaluating Candidates", leave=False)
            
            for future in pbar_candidates:
                candidate = future_to_candidate[future]
                try:
                    score = future.result()
                    results.append((candidate, score))
                except Exception as e:
                    self._log(f"Error evaluating candidate {candidate.name}: {e}")
                    results.append((candidate, 0.0))
        
        return results

    def _evaluate_candidates_sequential(self, candidates: List[AdvancedInteraction], 
                                      X: pd.DataFrame, y: pd.Series) -> List[Tuple[AdvancedInteraction, float]]:
        """Evaluate candidates sequentially with a nested progress bar."""
        results = []
        
        feature_cache = {col: X[col].values for col in X.columns}
        
        # FIXED: Use a consistent, nested-style progress bar
        pbar_candidates = tqdm(candidates, desc="  Evaluating Candidates", leave=False)
        
        for candidate in pbar_candidates:
            try:
                score = self._evaluate_single_candidate(candidate, X, y, feature_cache)
                results.append((candidate, score))
            except Exception as e:
                self._log(f"Error evaluating candidate {candidate.name}: {e}")
                results.append((candidate, 0.0))
        
        return results
    
    def _evaluate_single_candidate(self, candidate: AdvancedInteraction, 
                                  X: pd.DataFrame, y: pd.Series, feature_cache: Dict[str, np.ndarray] = None) -> float:
        """Evaluate single candidate"""
        try:
            if feature_cache is None:
                feature_cache = {}
            
            # Generate feature
            feature_name, feature_values = candidate.generate(X, y, self.operation_registry, feature_cache)
            
            # Validate feature values
            if np.isnan(feature_values).all() or np.isinf(feature_values).all():
                return 0.0
            
            # Handle NaN/inf values
            if np.any(np.isnan(feature_values)) or np.any(np.isinf(feature_values)):
                feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Check for zero variance
            if np.var(feature_values) < 1e-10:
                return 0.0
            
            # Add to dataframe
            X_with_feature = X.copy()
            X_with_feature[feature_name] = feature_values
            
            # Evaluate with cross-validation
            cv_result = cross_val_score(
                self.baseline_model, 
                X_with_feature, 
                y, 
                self.scorer,
                cv=self.cv,
                return_dict=True,
                model_fit_kwargs=self.model_fit_kwargs
            )
            
            return cv_result["mean_val_score"]
            
        except Exception as e:
            self._log(f"Error in candidate evaluation: {e}")
            return 0.0
    
    def _select_best_candidates(self, evaluated_candidates: List[Tuple[AdvancedInteraction, float]], 
                              X: pd.DataFrame, y: pd.Series, n_select: int = 10) -> List[AdvancedInteraction]:
        """Select best candidates using multi-objective optimization"""
        if not self.enable_multi_objective:
            # Simple selection by score - filter for actual improvements
            baseline_score = self.state['best_score']
            
            # Filter candidates that actually improve the score
            improving_candidates = []
            for candidate, score in evaluated_candidates:
                if self.scorer.greater_is_better:
                    if score > baseline_score + 0.0001:  # Small threshold for numerical stability
                        improving_candidates.append((candidate, score))
                else:
                    if score < baseline_score - 0.0001:
                        improving_candidates.append((candidate, score))
            
            # If we have improving candidates, select the best ones
            if improving_candidates:
                improving_candidates.sort(key=lambda x: x[1], reverse=self.scorer.greater_is_better)
                return [candidate for candidate, _ in improving_candidates[:n_select]]
            else:
                # Fallback to top candidates even if no clear improvement
                evaluated_candidates.sort(key=lambda x: x[1], reverse=self.scorer.greater_is_better)
                return [candidate for candidate, _ in evaluated_candidates[:min(n_select//2, 10)]]
        
        # Multi-objective selection
        baseline_score = self.state['best_score']
        selected_candidates = []
        
        for candidate, score in evaluated_candidates:
            # Create feature instance
            feature = candidate.get_new_feature_instance()
            
            # Evaluate on multiple objectives
            objectives = self.multi_objective_evaluator.evaluate_feature(
                feature, X, y, baseline_score, self.baseline_model, self.scorer, self.cv
            )
            
            # Check if Pareto optimal
            if self.multi_objective_evaluator.is_pareto_optimal(objectives):
                selected_candidates.append(candidate)
                self.multi_objective_evaluator.update_pareto_front(objectives, feature)
        
        # If we have too many Pareto optimal solutions, select by primary objective
        if len(selected_candidates) > n_select:
            candidate_scores = [(c, s) for c, s in evaluated_candidates if c in selected_candidates]
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            selected_candidates = [c for c, _ in candidate_scores[:n_select]]
        
        return selected_candidates
    
    def _update_populations(self, selected_candidates: List[AdvancedInteraction]):
        """Update populations with selected candidates"""
        for candidate in selected_candidates:
            # Create feature instance
            feature = candidate.get_new_feature_instance()
            
            # Add to appropriate population based on strategy
            if candidate.success_rate > 0.7:
                # High success rate - add to exploitation population
                exploitation_pops = [p for p in self.populations if p.strategy == "exploitation"]
                if exploitation_pops:
                    exploitation_pops[0].add_feature(feature)
            elif candidate.success_rate < 0.3:
                # Low success rate - add to exploration population
                exploration_pops = [p for p in self.populations if p.strategy == "exploration"]
                if exploration_pops:
                    exploration_pops[0].add_feature(feature)
            else:
                # Medium success rate - add to balanced population
                balanced_pops = [p for p in self.populations if p.strategy == "balanced"]
                if balanced_pops:
                    balanced_pops[0].add_feature(feature)
                else:
                    # Fallback to first population
                    self.populations[0].add_feature(feature)
    
    def _update_base_features_in_populations(self, X_current: pd.DataFrame):
        """Update all populations with current feature set as base features"""
        # Create features for all columns in X_current
        num_cols = X_current.select_dtypes(include=[np.number]).columns.tolist()
        
        for pop in self.populations:
            # Add any new features from X_current that aren't already in the population
            existing_feature_names = {f.name for f in pop.generation}
            
            for col in X_current.columns:
                if col not in existing_feature_names:
                    dtype = "num" if col in num_cols else "cat"
                    
                    # Create semantic information
                    semantics = FeatureSemantics(
                        name=col,
                        dtype=dtype,
                        domain=self._infer_domain(col, X_current[col]),
                        meaning=f"Generated {dtype} feature: {col}" if col not in self.state.get('original_columns', []) else f"Base {dtype} feature: {col}"
                    )
                    
                    feature = EnhancedFeature(
                        name=col,
                        dtype=dtype,
                        weight=1.0,
                        semantics=semantics
                    )
                    pop.add_feature(feature)
    
    def _calculate_population_diversity(self) -> float:
        """Calculate diversity across all populations"""
        all_features = []
        for pop in self.populations:
            all_features.extend(pop.generation)
        
        if len(all_features) < 2:
            return 1.0
        
        # Calculate semantic diversity
        semantic_distances = []
        for i in range(len(all_features)):
            for j in range(i + 1, len(all_features)):
                dist = all_features[i].semantics.get_semantic_distance(all_features[j].semantics)
                semantic_distances.append(dist)
        
        return np.mean(semantic_distances) if semantic_distances else 1.0
    
    def _migration_step(self):
        """Perform migration between populations"""
        if len(self.populations) < 2:
            return
        
        migration_rate = 0.1
        
        for i, pop in enumerate(self.populations):
            n_migrants = int(len(pop.generation) * migration_rate)
            if n_migrants == 0:
                continue
            
            # Select migrants (best features)
            migrants = pop.get_best_features(n_migrants)
            
            # Send to other populations
            for j, other_pop in enumerate(self.populations):
                if i != j:
                    for migrant in migrants:
                        # Create copy and add to other population
                        migrant_copy = deepcopy(migrant)
                        other_pop.add_feature(migrant_copy)
    
    def _update_adaptive_controller(self, generation: int, selected_candidates: List[AdvancedInteraction]):
        """Update adaptive controller with generation results"""
        # Calculate state
        diversity = self._calculate_population_diversity()
        stagnation = self.state['stagnation_count']
        
        current_state = {
            'generation': generation,
            'diversity': diversity,
            'stagnation': stagnation
        }
        
        # Update Q-values for successful operations
        for candidate in selected_candidates:
            reward = candidate.success_rate
            self.adaptive_controller.update_q_values(
                current_state, 
                candidate.operation, 
                reward,
                current_state  # Simplified - same state
            )
    
    def _record_meta_learning_patterns(self, X: pd.DataFrame, y: pd.Series, 
                                     selected_candidates: List[AdvancedInteraction]):
        """Record patterns for meta-learning"""
        if not self.enable_meta_learning:
            return
        
        # Create dataset ID
        dataset_id = hashlib.md5(str(X.shape).encode()).hexdigest()[:8]
        
        # Record dataset characteristics
        self.meta_learning_kb.dataset_characteristics[dataset_id] = \
            self.meta_learning_kb.get_dataset_characteristics(X, y)
        
        # Record successful patterns
        for candidate in selected_candidates:
            if candidate.success_rate > 0.5:
                pattern = f"{candidate.operation}_{candidate.dtype}"
                context = {
                    'depth': candidate.depth,
                    'component_types': [comp.dtype for comp in candidate.components]
                }
                self.meta_learning_kb.record_success(dataset_id, pattern, context)
    
    def search(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, PipelineWrapper, List[EnhancedFeature]]:
        """Enhanced search with multi-population evolution"""
        start_time = time.time()
        
        # Setup
        self._setup_task_components(X, y)
        self._initialize_populations(X, y)
        
        # Get baseline score
        baseline_result = cross_val_score(
            self.baseline_model, X, y, self.scorer, 
            cv=self.cv, return_dict=True, model_fit_kwargs=self.model_fit_kwargs
        )
        baseline_score = baseline_result["mean_val_score"]
        self.state['best_score'] = baseline_score
        
        self._log(f"Starting enhanced feature generation")
        self._log(f"Task: {self.task}, Baseline score: {baseline_score:.5f}")
        self._log(f"Populations: {self.n_populations}, Generations: {self.n_generations}")
        
        # Get meta-learning suggestions
        suggested_ops = []
        if self.enable_meta_learning:
            suggested_ops = self.meta_learning_kb.suggest_operations(X, y)
            self._log(f"Meta-learning suggestions: {suggested_ops}")
        
        # Main evolution loop
        X_current = X.copy()
        self.state['original_columns'] = list(X.columns)  # Track original columns
        
        with tqdm(total=self.n_generations, desc="Generations") as pbar:
            for generation in range(self.n_generations):
                gen_start_time = time.time()
                
                # Check time budget
                if self.time_budget and (time.time() - start_time) > self.time_budget:
                    self._log(f"Time budget exceeded. Stopping at generation {generation}")
                    break
                
                # Generate candidates from all populations
                all_candidates = []
                for pop in self.populations:
                    candidates = pop.evolve(self.operation_registry)
                    all_candidates.extend(candidates)
                
                # Add meta-learning guided candidates
                if suggested_ops and generation < self.n_generations // 2:
                    meta_candidates = self._generate_meta_learning_candidates(suggested_ops)
                    all_candidates.extend(meta_candidates)
                
                # Limit candidates to manageable number
                if len(all_candidates) > self.n_children:
                    all_candidates = random.sample(all_candidates, self.n_children)
                
                self._log(f"Generation {generation + 1}: Evaluating {len(all_candidates)} candidates")
                
                # Evaluate candidates
                evaluated_candidates = self._evaluate_candidates_parallel(all_candidates, X_current, y)
                
                # Select best candidates - be more aggressive
                n_select = min(50, len(evaluated_candidates))  # Increased from 20
                selected_candidates = self._select_best_candidates(evaluated_candidates, X_current, y, n_select)
                
                # Debug logging
                if evaluated_candidates:
                    best_score = max(evaluated_candidates, key=lambda x: x[1] if self.scorer.greater_is_better else -x[1])[1]
                    self._log(f"  Best candidate score: {best_score:.5f} (baseline: {self.state['best_score']:.5f})")
                    self._log(f"  Selected {len(selected_candidates)} candidates for evaluation")
                
                # Update populations
                self._update_populations(selected_candidates)
                
                # Apply best features to current dataset
                gen_best_score = self.state['best_score']
                gen_best_X = X_current.copy()
                
                feature_cache = {col: X_current[col].values for col in X_current.columns}
                
                for candidate in selected_candidates:
                    try:
                        feature_name, feature_values = candidate.generate(X_current, y, self.operation_registry, feature_cache)
                        X_test = X_current.copy()
                        X_test[feature_name] = feature_values
                        
                        test_result = cross_val_score(
                            self.baseline_model, X_test, y, self.scorer,
                            cv=self.cv, return_dict=True, model_fit_kwargs=self.model_fit_kwargs
                        )
                        test_score = test_result["mean_val_score"]
                        
                        improvement_threshold = 0.0001  # More sensitive threshold
                        if self.scorer.greater_is_better:
                            is_improvement = test_score > gen_best_score + improvement_threshold
                        else:
                            is_improvement = test_score < gen_best_score - improvement_threshold

                        if is_improvement:
                            gen_best_score = test_score
                            gen_best_X = X_test
                            candidate.update_performance(True, test_score - self.state['best_score'])
                            self._log(f"  Added feature: {feature_name}, New Best Score: {test_score:.5f}")
                        else:
                            candidate.update_performance(False, 0)
                            
                    except Exception as e:
                        self._log(f"Error applying candidate: {e}")
                        candidate.update_performance(False, -0.001)
                
                # Update state
                if self.scorer.greater_is_better:
                    improvement = gen_best_score - self.state['best_score']
                else:
                    improvement = self.state['best_score'] - gen_best_score
                
                if improvement > 0:
                    self.state['best_score'] = gen_best_score
                    self.state['stagnation_count'] = 0
                    X_current = gen_best_X
                    self._update_base_features_in_populations(X_current)
                else:
                    self.state['stagnation_count'] += 1
                
                # Migration step
                if generation % 5 == 0 and generation > 0:
                    self._migration_step()
                
                # Update adaptive controller
                self._update_adaptive_controller(generation, selected_candidates)
                
                # Record meta-learning patterns
                self._record_meta_learning_patterns(X_current, y, selected_candidates)
                
                diversity = self._calculate_population_diversity()
                gen_time = time.time() - gen_start_time
                self.metrics['generation_times'].append(gen_time)
                self.metrics['feature_counts'].append(X_current.shape[1])
                self.metrics['score_improvements'].append(improvement)
                self.metrics['population_diversities'].append(diversity)
                
                # FIXED: Log the overall best score from self.state['best_score'] for clarity
                self._log(f"Generation {generation + 1}: Score: {self.state['best_score']:.5f} "
                         f"({'+' if improvement > 0 else ''}{improvement:.5f}), Features: {X_current.shape[1]}, "
                         f"Diversity: {diversity:.3f}, Time: {gen_time:.2f}s")
                
                pbar.set_postfix({
                    'Score': f"{self.state['best_score']:.5f}",
                    'Features': X_current.shape[1],
                    'Diversity': f"{diversity:.3f}"
                })
                pbar.update(1)
                
                if self.state['stagnation_count'] >= 10:
                    self._log(f"Early stopping due to stagnation")
                    break
                
                if self.save_path and generation % 5 == 0:
                    self._save_checkpoint(X_current, generation)
        
        total_time = time.time() - start_time
        
        final_features = []
        for col in X_current.columns:
            is_original = col in X.columns
            if is_original:
                dtype = "num" if col in X.select_dtypes(include=[np.number]).columns else "cat"
                feature = EnhancedFeature(name=col, dtype=dtype, weight=1.0)
                final_features.append(feature)
            else:
                for pop in self.populations:
                    found = False
                    for feat in pop.generation:
                        if feat.name == col:
                            final_features.append(feat)
                            found = True
                            break
                    if found:
                        break

        self._log(f"Enhanced feature generation completed in {total_time:.2f}s")
        self._log(f"Final score: {self.state['best_score']:.5f}")
        self._log(f"Features generated: {X_current.shape[1] - X.shape[1]}")
        self._log(f"Average generation time: {np.mean(self.metrics['generation_times']):.2f}s")
        self._log(f"Average diversity: {np.mean(self.metrics['population_diversities']):.3f}")
        
        return X_current, self.pipeline, final_features
    
    def _generate_meta_learning_candidates(self, suggested_ops: List[str]) -> List[AdvancedInteraction]:
        """Generate candidates based on meta-learning suggestions"""
        candidates = []
        
        for op in suggested_ops:
            # Parse operation suggestion
            parts = op.split('_')
            if len(parts) >= 2:
                operation = parts[0]
                dtype = parts[1]
                
                # Find suitable features
                suitable_features = []
                for pop in self.populations:
                    for feat in pop.generation:
                        if feat.dtype == dtype:
                            suitable_features.append(feat)
                
                if suitable_features:
                    # Create unary candidates
                    for feat in suitable_features[:5]:  # Limit to avoid explosion
                        candidates.append(AdvancedInteraction([feat], operation, "unary"))
                    
                    # Create binary candidates if suitable
                    if len(suitable_features) > 1:
                        for i in range(min(3, len(suitable_features))):
                            for j in range(i + 1, min(i + 3, len(suitable_features))):
                                candidates.append(AdvancedInteraction(
                                    [suitable_features[i], suitable_features[j]], 
                                    operation, 
                                    "binary"
                                ))
        
        return candidates
    
    def _save_checkpoint(self, X: pd.DataFrame, generation: int):
        """Save checkpoint"""
        checkpoint_path = Path(self.save_path) / f"checkpoint_gen_{generation}.pkl"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint_data = {
            'X': X,
            'generation': generation,
            'state': self.state,
            'populations': self.populations,
            'metrics': self.metrics,
            'meta_learning_kb': self.meta_learning_kb
        }
        
        try:
            import cloudpickle
            with open(checkpoint_path, 'wb') as f:
                cloudpickle.dump(checkpoint_data, f)
            self._log(f"Checkpoint saved: {checkpoint_path}")
        except ImportError:
            self._log("cloudpickle required for checkpointing")
    
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit the feature generator"""
        self.X_transformed, self.pipeline, self.final_features = self.search(X, y)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using generated features"""
        if not hasattr(self, 'final_features'):
            raise ValueError("Must call fit() before transform()")
        
        X_transformed = X.copy()
        
        # Create feature cache for transform
        feature_cache = {}
        
        # Apply generated features
        for feature in self.final_features:
            if hasattr(feature, 'generating_interaction') and feature.generating_interaction:
                try:
                    feature_name, feature_values = feature.generating_interaction.generate(
                        X_transformed, None, self.operation_registry, feature_cache
                    )
                    X_transformed[feature_name] = feature_values
                except Exception as e:
                    self._log(f"Error generating feature {feature.name}: {e}")
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)

# Convenience aliases
FeatureGenerator = EnhancedFeatureGenerator
Feature = EnhancedFeature
Interaction = AdvancedInteraction

# Convenience function
def generate_features(X: pd.DataFrame, y: pd.Series, **kwargs) -> Tuple[pd.DataFrame, List[EnhancedFeature]]:
    """Convenience function for feature generation"""
    generator = EnhancedFeatureGenerator(**kwargs)
    X_transformed, pipeline, features = generator.search(X, y)
    return X_transformed, features