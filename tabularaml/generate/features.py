import hashlib
import io
import os, random, time
import pickle
import sys
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Union, List, Optional, Callable, Literal, Dict, Set, Tuple
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.multiclass import type_of_target
from tqdm.auto import tqdm

from tabularaml.eval.cv import cross_val_score, make_cv_splitter, sanitize_model_features
from tabularaml.eval.scorers import PREDEFINED_REG_SCORERS, PREDEFINED_CLS_SCORERS, PREDEFINED_SCORERS, Scorer
from tabularaml.eval.splitters import RotatedGroupKFold, normalize_rotatable_splitter
from tabularaml.generate.ops import OPS, ALL_OPS_LAMBDAS, AGG_OPS, TEMPORAL_OPS, build_temporal_ops
from tabularaml.inspect.importance import FeatureImportanceAnalyzer
from tabularaml.preprocessing.encoders import CategoricalEncoder, GroupByEncoder, TemporalEncoder
from tabularaml.preprocessing.imputers import SimpleImputer
from tabularaml.preprocessing.pipeline import PipelineWrapper
from tabularaml.configs.feature_gen import PRESET_PARAMS
from tabularaml.utils import is_gpu_available

_FEATURE_GENERATOR_SAVE_FORMAT = "tabularaml.feature_generator"
_FEATURE_GENERATOR_SAVE_VERSION = 2
_FEATURE_GENERATOR_PICKLE_PROTOCOL = 4
_PICKLE_MODULE_ALIASES = {
    "numpy._core": "numpy.core",
    "numpy.core": "numpy._core",
}


def _iter_compatible_module_names(module_name: str):
    """Yield module name candidates for cross-version pickle compatibility."""
    yield module_name
    for source_prefix, target_prefix in _PICKLE_MODULE_ALIASES.items():
        if module_name == source_prefix or module_name.startswith(f"{source_prefix}."):
            yield f"{target_prefix}{module_name[len(source_prefix):]}"


class _CompatibleUnpickler(pickle.Unpickler):
    """Unpickler that remaps known module paths that changed across versions."""

    def find_class(self, module, name):
        last_exc = None
        tried = set()

        for candidate in _iter_compatible_module_names(module):
            if candidate in tried:
                continue
            tried.add(candidate)
            try:
                return super().find_class(candidate, name)
            except (ModuleNotFoundError, ImportError, AttributeError) as exc:
                last_exc = exc

        if last_exc is not None:
            raise last_exc
        return super().find_class(module, name)


def _compatible_pickle_load(file_obj):
    """Load pickle bytes using compatibility module remapping."""
    return _CompatibleUnpickler(file_obj).load()


def _compatible_pickle_loads(payload: bytes):
    """Load pickled bytes using compatibility module remapping."""
    return _compatible_pickle_load(io.BytesIO(payload))


def _restore_missing_pipeline_columns(
    X: pd.DataFrame,
    pipeline,
    log_fn: Optional[Callable[[str], None]] = None,
) -> pd.DataFrame:
    """Backfill columns expected by a fitted sklearn pipeline.

    Some generated features only materialize on the training fold. When the
    held-out fold is missing those inputs, restore them as NaN so the fitted
    imputer/encoder stack can handle them instead of crashing on a schema
    mismatch.
    """
    if not isinstance(X, pd.DataFrame) or pipeline is None or not hasattr(pipeline, "named_steps"):
        return X

    scaler_step = pipeline.named_steps.get("scaling_encoding")
    required_cols = list(getattr(scaler_step, "feature_names_in_", []))
    if not required_cols:
        return X

    missing = [col for col in required_cols if col not in X.columns]
    if not missing:
        return X

    restored = X.copy()
    imputer = pipeline.named_steps.get("imputing")
    numeric_cols = set(getattr(imputer, "numerical_columns_", [])) if imputer is not None else set()
    categorical_cols = set(getattr(imputer, "categorical_columns_", [])) if imputer is not None else set()

    for col in missing:
        if col in categorical_cols and col not in numeric_cols:
            restored[col] = pd.Series(
                [pd.NA] * len(restored),
                index=restored.index,
                dtype="category",
            )
        else:
            restored[col] = np.nan

    if log_fn is not None:
        preview = ", ".join(map(str, missing[:10]))
        suffix = " ..." if len(missing) > 10 else ""
        log_fn(
            f"Warning: restoring {len(missing)} missing pipeline column(s) at transform time "
            f"as missing-value placeholders: {preview}{suffix}"
        )

    return restored

class Feature:
    """Feature with name, dtype, weight, depth, and pipeline requirements."""
    def __init__(self, name: str, dtype: Literal["num", "cat"], weight: float, 
                 depth: Optional[int] = None, require_pipeline: Optional[bool] = False):
        self.name, self.dtype, self.weight = name, dtype, weight
        self.depth = depth if depth is not None else self.get_feature_depth()
        self.require_pipeline = require_pipeline
        self.generating_interaction = None
        
    def get_feature_depth(self):
        n = 0
        for ops in OPS["num"]["unary"] + OPS["cat"]["unary"]:
            n += self.name.count(f"_{ops}")
        for ops in OPS["num"]["binary"] + OPS["cat"]["binary"]:
            n += self.name.count(f"_{ops}_")
        return n

    def get_col_from_df(self, X: pd.DataFrame): 
        return X[self.name].values
    
    def update_weight(self, new_weight: float): 
        self.weight = new_weight

    def set_generating_interaction(self, interaction: 'Interaction'): 
        self.generating_interaction = interaction

class Interaction:
    """Feature interactions for engineering new features via unary/binary operations."""
    def __init__(self, feature_1: Feature, op: str, feature_2: Optional[Feature] = None):
        self.feature_1, self.op, self.feature_2 = feature_1, op, feature_2
        
        # Determine if this is an aggregation operation
        self.is_agg = op in AGG_OPS
        self.is_temporal = op in TEMPORAL_OPS
        
        if self.is_temporal:
            # Temporal: feature_1 is the numeric column, feature_2 unused (unary-style)
            self.type = "unary"
            self.dtype = "num"
            self.depth = feature_1.depth + 1
            self.weight = feature_1.weight
            self.require_pipeline = True  # Must go through pipeline
            self.name = f"{op}_{feature_1.name}"
        elif self.is_agg:
            # Aggregation: feature_1 is categorical key, feature_2 is numeric column
            self.type = "binary"
            self.dtype = "num"  # Aggregation result is always numeric
            self.depth = max(feature_1.depth, feature_2.depth) + 1 if feature_2 else feature_1.depth + 1
            self.weight = (feature_1.weight + feature_2.weight) / 2 if feature_2 else feature_1.weight
            self.require_pipeline = True  # Must go through pipeline to prevent leakage
            agg_name = op.replace("groupby_", "")
            self.name = f"groupby_{agg_name}_{feature_1.name}_{feature_2.name}" if feature_2 else f"groupby_{agg_name}_{feature_1.name}"
        else:
            self.type = "unary" if feature_2 is None else "binary"
            self.dtype = (feature_1.dtype if feature_2 is None else 
                        "num" if feature_1.dtype == feature_2.dtype == "num" else "cat")
            self.depth = (feature_1.depth + 1 if feature_2 is None else 
                        max(feature_1.depth, feature_2.depth) + 1)
            self.weight = feature_1.weight if feature_2 is None else (feature_1.weight + feature_2.weight) / 2
            self.require_pipeline = feature_2 is None and op in ["target", "count", "freq"]
            self.name = f"{feature_1.name}_{op}" if self.type == "unary" else f"{feature_1.name}_{op}_{feature_2.name}"
         
    def generate(self, X, y = None):
        if not self.require_pipeline:
            try:
                if self.type == "unary":
                    return ALL_OPS_LAMBDAS[self.op](X, self.feature_1.name)[1]
                elif self.type == "binary":
                    # Check for column shape issues
                    if X[self.feature_1.name].ndim > 1 or X[self.feature_2.name].ndim > 1:
                        raise ValueError(f"Multi-dimensional columns detected: {self.feature_1.name} shape={X[self.feature_1.name].shape}, {self.feature_2.name} shape={X[self.feature_2.name].shape}")
                    return ALL_OPS_LAMBDAS[self.op](X, self.feature_1.name, self.feature_2.name)[1]
            except Exception as e:
                raise Exception(f"Error generating {self.name}: {str(e)}")
        raise Exception("Can't generate feature using lambdas. Requires pipeline to avoid data leakage.")
    
    def get_new_feature_instance(self):
        return Feature(name=self.name, dtype=self.dtype, weight=self.weight, depth=self.depth, require_pipeline=self.require_pipeline)


class FeatureCache:
    """Hash-based cache for computed feature values to avoid redundant computation."""
    def __init__(self, max_size_mb=2000):
        self._cache = {}
        self._max_bytes = max_size_mb * 1024 * 1024
        self._current_bytes = 0
        self.hits = 0
        self.misses = 0

    def _key(self, parent_names, op_name):
        # Preserve order: sub(a,b) and sub(b,a) must have different cache keys
        return hashlib.md5(f"{list(parent_names)}_{op_name}".encode()).hexdigest()

    def get_or_compute(self, parent_names, op_name, compute_fn):
        """Return cached result or compute and cache it."""
        key = self._key(parent_names, op_name)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        result = compute_fn()
        nbytes = result[1].nbytes if hasattr(result[1], 'nbytes') else 0
        if self._current_bytes + nbytes < self._max_bytes:
            self._cache[key] = result
            self._current_bytes += nbytes
        return result

    def clear(self):
        """Clear the cache completely."""
        self._cache.clear()
        self._current_bytes = 0

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class StagnationLevel(Enum):
    NONE, MILD, MODERATE, SEVERE, CRITICAL = 0, 1, 2, 3, 4

@dataclass
class AdaptiveState:
    """Enhanced container for adaptive system state"""
    stagnation_level: StagnationLevel = StagnationLevel.NONE
    generations_without_features: int = 0
    generations_without_improvement: int = 0
    consecutive_successful_generations: int = 0
    exploration_intensity: float = 0.0
    min_gain_reduction_factor: float = 1.0
    feature_weights_modified: bool = False
    failed_strategies_count: Dict[str, int] = field(default_factory=dict)
    last_restart_gen: int = -1
    total_restarts: int = 0


class ImprovedAdaptiveController:
    """Enhanced adaptive controller with better stagnation handling."""
    
    def __init__(self, original_min_pct_gain: float = 0.005, exploration_factor: float = 0.2):
        self.original_min_pct_gain = original_min_pct_gain
        self.exploration_factor = exploration_factor
        self.state = AdaptiveState()
        
        # Operation tracking
        self.op_stats = {"num": {"unary": {}, "binary": {}}, "cat": {"unary": {}, "binary": {}}}
        self.op_usage = Counter()
        self.op_success = Counter()
        self.op_combinations_tried = defaultdict(set)  # Track which combinations have been tried
        
        # Feature tracking
        self.feature_usage = Counter()
        self.feature_as_parent_success = Counter()  # Track success when used as parent
        self.feature_as_parent_attempts = Counter()
        self.failed_interactions = Counter()
        self.successful_children = defaultdict(list)  # Track which features produced good children
        
        # Strategy tracking
        self.strategy_success = {"hopeful_monster": 0, "normal": 0}
        self.strategy_attempts = {"hopeful_monster": 0, "normal": 0}

        # Memory of what worked
        self.successful_patterns = []  # List of (parent_features, operation, gain) tuples
        self.weight_modifications = {}

    def initialize_operations(self, ops):
        """Initialize operation statistics with diversity bias."""
        for dtype in ops:
            # Dynamically create op_stats entry for new op categories (agg, temporal, etc.)
            if dtype not in self.op_stats:
                self.op_stats[dtype] = {}
            for op_type in ops[dtype]:
                if op_type not in self.op_stats[dtype]:
                    self.op_stats[dtype][op_type] = {}
                for op in ops[dtype][op_type]:
                    if op not in self.op_stats[dtype][op_type]:
                        # Start with higher scores for rarely used operations
                        initial_score = 0.7 if self.op_usage[op] < 5 else 0.5
                        self.op_stats[dtype][op_type][op] = {
                            "success_rate": initial_score, 
                            "avg_gain": 0.0, 
                            "priority_score": initial_score,
                            "consecutive_failures": 0
                        }

    def assess_stagnation(self, no_features_count: int, no_improvement_count: int) -> None:
        """Enhanced stagnation assessment with CRITICAL level."""
        self.state.generations_without_features = no_features_count
        self.state.generations_without_improvement = no_improvement_count
        
        # Track consecutive successful generations
        if no_features_count == 0 and no_improvement_count == 0:
            self.state.consecutive_successful_generations += 1
        else:
            self.state.consecutive_successful_generations = 0
        
        # Determine stagnation level with new CRITICAL level
        current_level = (
            StagnationLevel.CRITICAL if no_features_count >= 8 or no_improvement_count >= 12
            else StagnationLevel.SEVERE if no_features_count >= 4 or no_improvement_count >= 6
            else StagnationLevel.MODERATE if no_features_count >= 2 or no_improvement_count >= 4
            else StagnationLevel.MILD if no_features_count >= 1 or no_improvement_count >= 2
            else StagnationLevel.NONE
        )
        
        # Update stagnation level
        if current_level.value > self.state.stagnation_level.value:
            self.state.stagnation_level = current_level
        elif (current_level.value < self.state.stagnation_level.value and 
              self.state.consecutive_successful_generations >= 2):
            # Gradual recovery
            self.state.stagnation_level = StagnationLevel(max(0, self.state.stagnation_level.value - 1))
        
        # Set exploration intensity with more aggressive values
        self.state.exploration_intensity = {
            StagnationLevel.CRITICAL: 1.5,  # Over 100% for extreme measures
            StagnationLevel.SEVERE: 1.0,
            StagnationLevel.MODERATE: 0.6,
            StagnationLevel.MILD: 0.3,
            StagnationLevel.NONE: 0.0
        }[self.state.stagnation_level]
    
    def get_adaptive_min_gain(self) -> float:
        """Get adaptively adjusted minimum gain threshold with more aggressive reduction."""
        self.state.min_gain_reduction_factor = {
            StagnationLevel.CRITICAL: 0.1,  # Accept almost any improvement
            StagnationLevel.SEVERE: 0.25,
            StagnationLevel.MODERATE: 0.5,
            StagnationLevel.MILD: 0.75,
            StagnationLevel.NONE: 1.0
        }[self.state.stagnation_level]
        return self.original_min_pct_gain * self.state.min_gain_reduction_factor
    
    def update_operation_stats(self, interaction: 'Interaction', success: bool, gain: float = 0.0):
        """Enhanced operation tracking with pattern memory."""
        op = interaction.op
        # Agg/temporal ops are stored under their own dtype key, not "num"/"cat"
        if getattr(interaction, 'is_agg', False):
            dtype, op_type = "agg", "binary"
        elif getattr(interaction, 'is_temporal', False):
            dtype, op_type = "temporal", "unary"
        else:
            dtype = interaction.dtype
            op_type = interaction.type
        
        # Update usage and success counters
        self.op_usage[op] += 1
        if success:
            self.op_success[op] += 1
            # Remember successful patterns
            pattern = (interaction.feature_1.name, op, 
                      interaction.feature_2.name if interaction.feature_2 else None)
            self.successful_patterns.append((pattern, gain))
            # Track which features produce good children
            self.successful_children[interaction.feature_1.name].append(interaction.name)
            if interaction.feature_2:
                self.successful_children[interaction.feature_2.name].append(interaction.name)
        else:
            # Track failed combinations more granularly
            combo_key = f"{interaction.feature_1.name}|{op}"
            if interaction.feature_2:
                combo_key += f"|{interaction.feature_2.name}"
            self.failed_interactions[combo_key] += 1
        
        # Track parent success rate
        self.feature_as_parent_attempts[interaction.feature_1.name] += 1
        if success:
            self.feature_as_parent_success[interaction.feature_1.name] += 1
        if interaction.feature_2:
            self.feature_as_parent_attempts[interaction.feature_2.name] += 1
            if success:
                self.feature_as_parent_success[interaction.feature_2.name] += 1
        
        # Update operation stats with consecutive failure tracking
        if op not in self.op_stats[dtype][op_type]:
            self.op_stats[dtype][op_type][op] = {
                "success_rate": 0.5, "avg_gain": 0.0, 
                "priority_score": 0.5, "consecutive_failures": 0
            }
        
        stats = self.op_stats[dtype][op_type][op]
        decay = 0.9
        
        if success:
            stats["consecutive_failures"] = 0
            stats["success_rate"] = stats["success_rate"] * decay + 1.0 * (1 - decay)
            stats["avg_gain"] = stats["avg_gain"] * decay + gain * (1 - decay)
        else:
            stats["consecutive_failures"] += 1
            stats["success_rate"] = stats["success_rate"] * decay + 0.0 * (1 - decay)
        
        # Penalize operations with many consecutive failures
        penalty = 1.0 - min(0.5, stats["consecutive_failures"] * 0.1)
        stats["priority_score"] = (0.7 * stats["success_rate"] + 
                                  0.3 * min(1.0, stats["avg_gain"] * 10)) * penalty
    
    def get_creative_operations(self, dtype: str, op_type: str, n: int = 5) -> List[str]:
        """Get operations that haven't been tried much or have been forgotten."""
        if dtype not in self.op_stats or op_type not in self.op_stats[dtype]:
            return []
        
        ops_with_scores = []
        for op, stats in self.op_stats[dtype][op_type].items():
            # Boost score for rarely used operations
            usage_boost = 1.0 / (1.0 + self.op_usage[op] * 0.05)
            # Boost score for operations that haven't failed recently
            failure_boost = 1.0 if stats["consecutive_failures"] < 3 else 0.5
            
            creativity_score = usage_boost * failure_boost * (1.0 - stats["priority_score"])
            ops_with_scores.append((op, creativity_score))
        
        # Return top n most "creative" (underused but not terrible) operations
        ops_with_scores.sort(key=lambda x: x[1], reverse=True)
        return [op for op, _ in ops_with_scores[:n]]
    
    def get_parent_quality_score(self, feature_name: str) -> float:
        """Calculate how good a feature has been as a parent."""
        attempts = self.feature_as_parent_attempts.get(feature_name, 0)
        if attempts == 0:
            return 0.5  # Neutral score for untested features
        
        success_rate = self.feature_as_parent_success.get(feature_name, 0) / attempts
        # Bonus for features that have produced multiple successful children
        diversity_bonus = min(0.2, len(self.successful_children.get(feature_name, [])) * 0.05)
        
        return min(1.0, success_rate + diversity_bonus)
    
    def should_restart(self, generation_num: int) -> bool:
        """Determine if a partial restart would be beneficial."""
        if self.state.stagnation_level != StagnationLevel.CRITICAL:
            return False
        
        # Don't restart too frequently
        if generation_num - self.state.last_restart_gen < 20:
            return False
        
        # Check if strategies are consistently failing
        for strategy, attempts in self.strategy_attempts.items():
            if attempts > 5:
                success_rate = self.strategy_success[strategy] / attempts
                if success_rate < 0.1:  # Less than 10% success rate
                    return True
        
        return self.state.generations_without_improvement > 15
    
    def get_restart_features(self, all_features: List[Feature], n: int = 10, 
                           current_columns: Optional[List[str]] = None) -> List[Feature]:
        """Select best features to keep after restart based on their history."""
        # Filter to only features that actually exist
        if current_columns is not None:
            all_features = [f for f in all_features if f.name in current_columns]
        
        feature_scores = []
        
        for feat in all_features:
            # Score based on multiple factors
            parent_score = self.get_parent_quality_score(feat.name)
            weight_score = feat.weight
            children_score = len(self.successful_children.get(feat.name, [])) * 0.1
            
            # Bonus for original features
            is_original = feat.depth == 0
            original_bonus = 0.2 if is_original else 0
            
            total_score = (0.3 * parent_score + 0.4 * weight_score + 
                          0.2 * children_score + 0.1 * original_bonus)
            feature_scores.append((feat, total_score))
        
        # Keep top n features
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        return [feat for feat, _ in feature_scores[:min(n, len(feature_scores))]]
    
    def update_strategy_stats(self, strategy: str, success: bool):
        """Track success rates of different strategies."""
        self.strategy_attempts[strategy] = self.strategy_attempts.get(strategy, 0) + 1
        if success:
            self.strategy_success[strategy] = self.strategy_success.get(strategy, 0) + 1
    
    def get_strategy_success_rate(self, strategy: str) -> float:
        """Get success rate of a strategy."""
        attempts = self.strategy_attempts.get(strategy, 0)
        if attempts == 0:
            return 0.5  # Neutral score
        return self.strategy_success.get(strategy, 0) / attempts
    
    def rank_candidates_with_memory(self, batch: List['Interaction'], 
                                   X: pd.DataFrame, y: pd.Series) -> List['Interaction']:
        """Rank candidates considering past success patterns."""
        if not batch:
            return []
        
        # Calculate base scores
        candidate_scores = []
        
        for interaction in batch:
            # Base weight score
            weight_score = interaction.weight
            
            # Operation success score
            op_score = self._get_op_priority_score(interaction.op)
            
            # Parent quality scores
            parent1_score = self.get_parent_quality_score(interaction.feature_1.name)
            parent2_score = (self.get_parent_quality_score(interaction.feature_2.name) 
                           if interaction.feature_2 else parent1_score)
            parent_score = (parent1_score + parent2_score) / 2
            
            # Novelty score - prefer combinations we haven't tried
            combo_key = f"{interaction.feature_1.name}|{interaction.op}"
            if interaction.feature_2:
                combo_key += f"|{interaction.feature_2.name}"
            
            times_failed = self.failed_interactions.get(combo_key, 0)
            novelty_score = 1.0 / (1.0 + times_failed)
            
            # Pattern similarity score - boost if similar to successful patterns
            pattern_score = self._get_pattern_similarity_score(interaction)
            
            # Complexity penalty
            complexity_penalty = (interaction.depth / 5.0) ** 2 if interaction.depth > 3 else 0
            
            # Adaptive weighting based on stagnation
            if self.state.stagnation_level.value >= StagnationLevel.SEVERE.value:
                # During severe stagnation, heavily weight novelty and pattern similarity
                total_score = (0.15 * weight_score + 0.15 * op_score + 0.15 * parent_score +
                             0.35 * novelty_score + 0.25 * pattern_score - 0.05 * complexity_penalty)
            else:
                # Normal weighting
                total_score = (0.3 * weight_score + 0.2 * op_score + 0.2 * parent_score +
                             0.15 * novelty_score + 0.1 * pattern_score - 0.05 * complexity_penalty)
            
            candidate_scores.append((interaction, total_score))
        
        # Sort by score
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        return [interaction for interaction, _ in candidate_scores]
    
    def _get_pattern_similarity_score(self, interaction: 'Interaction') -> float:
        """Score based on similarity to successful patterns."""
        if not self.successful_patterns:
            return 0.5
        
        pattern = (interaction.feature_1.name, interaction.op,
                  interaction.feature_2.name if interaction.feature_2 else None)
        
        # Check exact matches
        for past_pattern, gain in self.successful_patterns[-20:]:  # Look at recent successes
            if pattern == past_pattern:
                return min(1.0, 0.7 + gain * 10)  # High score for exact match
        
        # Check partial matches (same operation with similar features)
        partial_matches = 0
        for past_pattern, gain in self.successful_patterns[-20:]:
            if past_pattern[1] == pattern[1]:  # Same operation
                partial_matches += 1
        
        return min(0.7, 0.3 + partial_matches * 0.1)
    
    def _get_op_priority_score(self, op: str) -> float:
        """Get priority score for an operation."""
        for dtype in self.op_stats:
            for op_type in self.op_stats[dtype]:
                if op in self.op_stats[dtype][op_type]:
                    return self.op_stats[dtype][op_type][op]["priority_score"]
        return 0.5
    
    def reset_for_new_run(self):
        """Reset state for a new feature generation run."""
        self.state = AdaptiveState()
        self.weight_modifications.clear()
        # Keep some learned knowledge
        self.op_usage.clear()
        self.feature_usage.clear()
        self.failed_interactions.clear()
        self.successful_children.clear()
        self.feature_as_parent_attempts.clear()
        self.feature_as_parent_success.clear()
        self.strategy_success = {"hopeful_monster": 0, "normal": 0}
        self.strategy_attempts = {"hopeful_monster": 0, "normal": 0}

    def get_status_summary(self) -> dict:
        """Get summary of current adaptive state."""
        hopeful_rate = self.get_strategy_success_rate("hopeful_monster")
        normal_rate = self.get_strategy_success_rate("normal")

        return {
            "stagnation_level": self.state.stagnation_level.name,
            "exploration_intensity": f"{self.state.exploration_intensity:.2f}",
            "min_gain_factor": f"{self.state.min_gain_reduction_factor:.2f}",
            "consecutive_success": self.state.consecutive_successful_generations,
            "weights_modified": self.state.feature_weights_modified,
            "features_modified": len(self.weight_modifications),
            "strategy_success": f"HM:{hopeful_rate:.2f}, N:{normal_rate:.2f}",
            "total_restarts": self.state.total_restarts
        }


class FeatureGenerator:
    """
    Enhanced Feature Generator with improved stagnation handling.
    """
    def __init__(self,
                 baseline_model = None,
                 model_fit_kwargs: dict = {},
                 task: Optional[Literal["regression", "classification"]] = None,
                 scorer: Optional[Scorer] = None,
                 logging_scorers: Optional[List[Scorer]] = None,
                 mode: Optional[str] = None,
                 n_generations: int = 15,
                 n_parents: int = 40,
                 n_children: int = 200,
                 ranking_method: Literal["multi_criteria", "shap", "none"] = "multi_criteria",
                 min_pct_gain: float = 0.001,
                 imp_weights=None,
                 max_new_feats=None,
                 early_stopping_iter: Union[float, int, bool] = 0.4,
                 early_stopping_child_eval: Union[float, int, bool] = 0.3,
                 ops=None,
                 cv: Union[int, BaseCrossValidator] = 5,
                 groups=None,
                 use_gpu: bool = True,
                 log_file: Union[str, Path] = "cache/logs/feat_gen_log.txt",
                 adaptive: bool = True,
                 time_budget=None,
                 search_sample_size: Optional[int] = None,
                 max_ops_per_generation=None,
                 exploration_factor: float = 0.2,
                 save_path=None,
                 save_each_trial: bool = False,
                 cache_size_mb: int = 2000,
                 use_proxy_evaluation: bool = True,
                 proxy_top_pct: float = 0.15,
                 meta_validation_frac: float = 0.15,
                 rotate_cv_folds: bool = True,
                 fold_rotation_period: int = 5,
                 final_selection: bool = True,
                 time_col: Optional[str] = None,
                 id_col: Optional[str] = None,
                 temporal_windows: Optional[list] = None,
                 target_encoding_strategy: Literal["mean", "smoothed", "catboost"] = "smoothed",
                 te_smoothing: float = 10.0,
                 use_adversarial_validation: bool = False,
                 adv_drift_weight: float = 0.5,
                 adv_drift_max: float = 0.1,
                 seed_templates: bool = True,
                 seed_top_k: int = 15,
                 seed_max_candidates: int = 500,
                 redundancy_prune: bool = True,
                 redundancy_corr_threshold: float = 0.95,
                 batch_evaluation: bool = False,
                 batch_size: int = 5,
                 random_state: int = 42,
                 n_jobs: int = -1):

        # Capture provided parameters
        provided_params = locals().copy()
        provided_params.pop('self')
        
        self.mode = mode
        self.random_state = random_state
        self.n_jobs = n_jobs

        # Always set params from constructor args first (preserves explicit UI overrides)
        self.n_generations = n_generations
        self.n_parents = n_parents
        self.n_children = n_children
        self.ranking_method = ranking_method
        self.min_pct_gain = min_pct_gain
        self.early_stopping_iter = early_stopping_iter
        self.early_stopping_child_eval = early_stopping_child_eval
        self.time_budget = time_budget
        self.search_sample_size = search_sample_size

        # Mode overrides only params still at their constructor defaults
        if mode:
            self._set_params_from_mode(provided_params)

        # Core parameters (always set normally)
        self.baseline_model = baseline_model
        self.model_fit_kwargs = model_fit_kwargs
        self.task = task
        self.scorer = scorer
        self.logging_scorers = logging_scorers or []
        self.infer_task = any(p is None for p in (baseline_model, task, scorer))
        self.imp_weights = imp_weights
        self.max_new_feats = max_new_feats
        self.adaptive = adaptive
        self.save_path = save_path
        self.save_each_trial = save_each_trial

        # Convert early_stopping_iter from fraction to absolute generation count if float
        if isinstance(self.early_stopping_iter, float):
            self.early_stopping_iter = int(self.early_stopping_iter * self.n_generations)
        elif not isinstance(self.early_stopping_iter, int):
            self.early_stopping_iter = float('inf')
        
        # Technical setup
        self.adaptive_controller = ImprovedAdaptiveController(
            original_min_pct_gain=min_pct_gain, 
            exploration_factor=exploration_factor
        )
        self.ops = ops if ops is not None else OPS
        self.cv = cv
        self.groups = groups
        self._groups_active = groups
        self.device = "cuda" if is_gpu_available() and use_gpu else "cpu"

        # Target-encoding strategy (set before any CategoricalEncoder is built below)
        self.target_encoding_strategy = target_encoding_strategy
        self.te_smoothing = te_smoothing

        self.pipeline = PipelineWrapper(imputer=None, scaler=None, encoder=self._make_cat_encoder())
        
        # Feature value cache
        self._feature_cache = FeatureCache(max_size_mb=cache_size_mb)
        
        # Proxy evaluation settings
        self.use_proxy_evaluation = use_proxy_evaluation
        self.proxy_top_pct = proxy_top_pct
        self._lgb_available = None  # Lazy check
        
        # CV bias fix settings
        self.meta_validation_frac = meta_validation_frac
        self.rotate_cv_folds = rotate_cv_folds
        self.fold_rotation_period = fold_rotation_period
        
        # Regularized post-selection
        self.final_selection = final_selection
        
        # Temporal operator settings
        self.time_col = time_col
        self.id_col = id_col
        self.temporal_windows = temporal_windows

        # Adversarial validation (train/test distribution-shift aware feature pruning)
        self.use_adversarial_validation = use_adversarial_validation
        self.adv_drift_weight = adv_drift_weight
        self.adv_drift_max = adv_drift_max
        self.X_test = None            # optional unlabeled test features, set via search()
        self._adv_drift_scores = {}   # per-feature drift score cache for current generation

        # Template seeding (deterministic 2nd-order + groupby-cross coverage)
        self.seed_templates = seed_templates
        self.seed_top_k = seed_top_k
        self.seed_max_candidates = seed_max_candidates

        # Joint selection / redundancy pruning
        self.redundancy_prune = redundancy_prune
        self.redundancy_corr_threshold = redundancy_corr_threshold
        self.batch_evaluation = batch_evaluation
        self.batch_size = batch_size
        
        # Rebuild temporal ops with custom windows if provided
        if temporal_windows is not None:
            custom_temporal = build_temporal_ops(temporal_windows)
            # Update the module-level dicts so Interaction can reference them
            TEMPORAL_OPS.clear()
            TEMPORAL_OPS.update(custom_temporal)
            OPS["temporal"] = {"unary": list(TEMPORAL_OPS.keys())}
        
        # Legacy compatibility
        self.max_ops_per_generation = max_ops_per_generation
        self.exploration_factor = exploration_factor
        
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log_file = log_file

        # Ensure these exist before search() / _set_defaults() is called so that
        # early-exit paths (e.g. fit/transform without generate) never hit AttributeError.
        self.interactions: list = []
        self.generation: list = []

    def _make_cat_encoder(self, target_enc_cols=None, count_enc_cols=None, freq_enc_cols=None):
        """Build a CategoricalEncoder using the generator's configured target-encoding strategy."""
        return CategoricalEncoder(
            target_enc_cols=target_enc_cols,
            count_enc_cols=count_enc_cols,
            freq_enc_cols=freq_enc_cols,
            target_encoding_strategy=getattr(self, "target_encoding_strategy", "smoothed"),
            te_smoothing=getattr(self, "te_smoothing", 10.0),
        )

    def _ensure_no_duplicates(self, X: pd.DataFrame, context: str = "") -> pd.DataFrame:
        """Ensure DataFrame has no duplicate columns."""
        if X.columns.duplicated().any():
            duplicates = X.columns[X.columns.duplicated()].tolist()
            self._log(f"Warning: Found duplicate columns {context}: {duplicates}")
            return X.loc[:, ~X.columns.duplicated(keep='first')]
        return X
    
    def _set_params_from_mode(self, provided_params):
        """Set instance parameters from mode preset, only for parameters not explicitly provided."""
        mode_dict = PRESET_PARAMS.get(self.mode)
        if mode_dict:
            # Get function signature to compare with defaults
            import inspect
            sig = inspect.signature(FeatureGenerator.__init__)
            
            for param, value in mode_dict.items():
                # Only set if parameter wasn't explicitly provided (equals default)
                if param in provided_params and param in sig.parameters:
                    default_value = sig.parameters[param].default
                    if provided_params[param] == default_value:
                        setattr(self, param, value)
                    # If explicitly provided, use that value
                    else:
                        setattr(self, param, provided_params[param])
                else:
                    # Parameter not in constructor, set mode value
                    setattr(self, param, value)
        else:
            raise Exception(f"{self.mode.upper()} mode undefined. Use: 'lite', 'medium', 'best', 'extreme'")
        
    def _log(self, message):
        """Log message to terminal and file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        tqdm.write(formatted_message)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(f"{formatted_message}\n")

    def _get_num_cat_cols(self, X: pd.DataFrame) -> tuple[list, list]:
        return (X.select_dtypes(include=['number']).columns.tolist(),
                X.select_dtypes(include=['object', 'category']).columns.tolist())

    def _create_search_subsample(self, X: pd.DataFrame, y: pd.Series, sample_size: int, groups=None) -> tuple:
        """Create a stratified subsample for the search phase, respecting groups/time if provided."""
        if len(X) <= sample_size:
            return X, y, groups

        from sklearn.model_selection import StratifiedShuffleSplit
        groups_arr = np.asarray(groups) if groups is not None else None

        try:
            if groups_arr is not None:
                frac = sample_size / len(X)
                df_temp = pd.DataFrame({'idx': np.arange(len(X)), 'g': groups_arr})
                sampled = df_temp.groupby('g', group_keys=False).sample(frac=frac, random_state=self.random_state)
                indices = sampled['idx'].values
                
                if len(indices) > sample_size:
                    indices = np.random.RandomState(self.random_state).choice(indices, sample_size, replace=False)
                elif len(indices) < sample_size:
                    remaining = np.setdiff1d(np.arange(len(X)), indices)
                    extra = np.random.RandomState(self.random_state).choice(remaining, sample_size - len(indices), replace=False)
                    indices = np.concatenate([indices, extra])
                indices = np.sort(indices)
            else:
                if self.task != "regression":
                    stratify_labels = y
                else:
                    n_bins = min(10, len(y.unique()))
                    stratify_labels = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
                sfss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=self.random_state)
                indices, _ = next(sfss.split(X, stratify_labels))
        except Exception:
            indices = np.random.RandomState(self.random_state).choice(len(X), size=sample_size, replace=False)
            if groups_arr is not None:
                indices = np.sort(indices)

        groups_sub = groups_arr[indices] if groups_arr is not None else None
        return X.iloc[indices].copy(), y.iloc[indices].copy(), groups_sub

    def _get_top_k_features(self, X: pd.DataFrame, y: pd.Series, k: int = 50, pipeline=None) -> pd.DataFrame:
        """Get top k features by importance."""
        pipeline.imputer = SimpleImputer()
        analyzer = FeatureImportanceAnalyzer(
            task_type=self.task, weights=self.imp_weights, preferred_gbm="xgboost" if self.device == "cuda" else "lightgbm",
            pipeline=pipeline, cv=self.cv, use_gpu=(self.device == "cuda"))
        analyzer.fit(X, y, groups=self._groups_active)
        pipeline.imputer = None 
        imp_df = analyzer.get_importance(normalize=False)[["weighted_importance"]]
        imp_df.sort_values(by="weighted_importance", axis=0, ascending=False, inplace=True)
        return imp_df if k == -1 else imp_df[:k]

    def _eval_baseline(self, X: pd.DataFrame, y: pd.Series, pipeline=None, groups=None) -> tuple[float, float]:
        """Evaluate baseline model performance."""
        pipeline = pipeline.get_pipeline(X, y) if pipeline is not None else pipeline
        eval_groups = self._groups_active if groups is None else groups
        cv_dict = cross_val_score(self.baseline_model, X, y, self.scorer, cv=self.cv,
                                 return_dict=True, pipeline=pipeline, model_fit_kwargs=self.model_fit_kwargs,
                                 groups=eval_groups)
        return cv_dict["mean_train_score"], cv_dict["mean_val_score"]

    def _eval_logging_scorers(self, X: pd.DataFrame, y: pd.Series, pipeline=None) -> Dict[str, Tuple[float, float]]:
        """Evaluate all logging scorers and return dict of {scorer_name: (train_score, val_score)}."""
        if not self.logging_scorers:
            return {}

        results = {}
        pipeline_obj = pipeline.get_pipeline(X, y) if pipeline is not None else pipeline

        for scorer in self.logging_scorers:
            try:
                cv_dict = cross_val_score(self.baseline_model, X, y, scorer, cv=self.cv,
                                         return_dict=True, pipeline=pipeline_obj,
                                         model_fit_kwargs=self.model_fit_kwargs,
                                         groups=self._groups_active)
                results[scorer.name] = (cv_dict["mean_train_score"], cv_dict["mean_val_score"])
            except Exception as e:
                self._log(f"Warning: Failed to evaluate logging scorer {scorer.name}: {str(e)}")

        return results

    def _format_logging_scores(self, scores_dict: Dict[str, Tuple[float, float]]) -> str:
        """Format logging scorer results for display."""
        if not scores_dict:
            return ""
        parts = []
        for name, (train, val) in scores_dict.items():
            parts.append(f"{name}: Train={train:.5f}, Val={val:.5f}")
        return " | ".join(parts)

    def _check_lgb_available(self):
        """Lazily check if LightGBM is available."""
        if self._lgb_available is None:
            try:
                import lightgbm
                self._lgb_available = True
            except ImportError:
                self._lgb_available = False
        return self._lgb_available

    def _get_lgb_objective(self):
        """Get the LightGBM objective string from the current task."""
        if self.task == "regression":
            return "regression"
        else:
            n_classes = len(np.unique(getattr(self, '_current_y', [0, 1])))
            return "binary" if n_classes <= 2 else "multiclass"

    def _train_base_model_and_get_residuals(self, X, y, cv):
        """Train base model on current features, return OOF predictions."""
        import lightgbm as lgb
        objective = self._get_lgb_objective()
        oof_preds = np.zeros(len(y))
        params = {"objective": objective, "verbosity": -1,
                  "learning_rate": 0.1, "num_leaves": 31,
                  "n_jobs": self.n_jobs, "random_state": self.random_state}
        if objective == "multiclass":
            n_classes = len(np.unique(y))
            params["num_class"] = n_classes
            oof_preds = np.zeros((len(y), n_classes))

        for train_idx, val_idx in cv.split(X, y, groups=self._groups_active):
            X_train = sanitize_model_features(X.iloc[train_idx].copy())
            X_val = sanitize_model_features(X.iloc[val_idx].copy())
            for col in X_train.select_dtypes(include=['object']).columns:
                X_train[col] = X_train[col].astype('category')
                X_val[col] = pd.Categorical(X_val[col], categories=X_train[col].cat.categories)
            dtrain = lgb.Dataset(X_train, y.iloc[train_idx])
            model = lgb.train(params, dtrain, num_boost_round=200)
            # Must use raw margins for init_score
            oof_preds[val_idx] = model.predict(X_val, raw_score=True)
        return oof_preds

    def _featureboost_score(self, candidate_series, y, oof_preds, cv):
        """Score a single candidate feature via residual-based incremental training.
        
        Uses the OpenFE 'FeatureBoost' trick: train a tiny single-feature LightGBM
        with init_score set to the base model's OOF predictions.
        """
        import lightgbm as lgb
        objective = self._get_lgb_objective()
        
        # Prepare candidate values
        if hasattr(candidate_series, 'values'):
            cand_vals = candidate_series.values
        else:
            cand_vals = np.asarray(candidate_series)
        
        if cand_vals.ndim == 1:
            cand_vals = cand_vals.reshape(-1, 1)
        
        # Skip if candidate has too many NaN/inf
        finite_mask = np.isfinite(cand_vals.ravel())
        if finite_mask.mean() < 0.5:
            return -np.inf
        
        scores = []
        params = {"objective": objective, "num_leaves": 16,
                  "verbosity": -1, "n_jobs": self.n_jobs, "learning_rate": 0.1,
                  "random_state": self.random_state}
        if objective == "multiclass":
            params["num_class"] = len(np.unique(y))

        for train_idx, val_idx in cv.split(cand_vals, y, groups=self._groups_active):
            try:
                train_cand = cand_vals[train_idx].copy()
                val_cand = cand_vals[val_idx].copy()
                
                # Replace non-finite with 0 for LGB
                train_cand = np.nan_to_num(train_cand, nan=0.0, posinf=0.0, neginf=0.0)
                val_cand = np.nan_to_num(val_cand, nan=0.0, posinf=0.0, neginf=0.0)
                
                init_train = oof_preds[train_idx]
                init_val = oof_preds[val_idx]
                
                dtrain = lgb.Dataset(
                    train_cand, y.iloc[train_idx],
                    init_score=init_train
                )
                dval = lgb.Dataset(
                    val_cand, y.iloc[val_idx],
                    init_score=init_val,
                    reference=dtrain
                )
                model = lgb.train(
                    params, dtrain, num_boost_round=50,
                    valid_sets=[dval],
                    callbacks=[lgb.early_stopping(10, verbose=False),
                              lgb.log_evaluation(period=0)]
                )
                
                # Score improvement: compare base predictions vs base + residual model
                # init_val is raw margin
                tree_margin = model.predict(val_cand, raw_score=True)
                new_preds_margin = init_val + tree_margin
                
                if objective == "binary":
                    import scipy.special
                    base_preds = scipy.special.expit(init_val)
                    new_preds = scipy.special.expit(new_preds_margin)
                elif objective == "multiclass":
                    import scipy.special
                    base_preds = scipy.special.softmax(init_val, axis=1)
                    new_preds = scipy.special.softmax(new_preds_margin, axis=1)
                else:
                    base_preds = init_val
                    new_preds = new_preds_margin

                base_score = self.scorer.score(y.iloc[val_idx], base_preds)
                new_score = self.scorer.score(y.iloc[val_idx], new_preds)
                
                if self.scorer.greater_is_better:
                    scores.append(new_score - base_score)
                else:
                    scores.append(base_score - new_score)  # Lower is better, so improvement = base - new
            except Exception:
                scores.append(-np.inf)
        
        return np.mean(scores) if scores else -np.inf

    def _proxy_screen_candidates(self, batch, X, y):
        """Pre-filter candidates using FeatureBoost proxy scoring.
        
        Returns the top proxy_top_pct fraction of non-pipeline candidates,
        plus all pipeline-required candidates (which skip proxy).
        """
        if not self.use_proxy_evaluation or not self._check_lgb_available():
            return batch
        
        # Separate pipeline-required (skip proxy) from scorable candidates
        pipeline_candidates = [i for i in batch if i.require_pipeline]
        scorable_candidates = [i for i in batch if not i.require_pipeline]
        
        if len(scorable_candidates) <= 5:
            return batch  # Not enough to filter
        
        try:
            # Get CV splitter
            cv = self._get_cv_splitter()
            
            # Train base model and get OOF predictions (once per generation)
            if not hasattr(self, '_current_oof_preds') or self._oof_preds_stale:
                self._current_oof_preds = self._train_base_model_and_get_residuals(X, y, cv)
                self._oof_preds_stale = False
            
            # Score each scorable candidate
            fb_scores = {}
            for interaction in scorable_candidates:
                try:
                    # Generate candidate values
                    parent_names = [interaction.feature_1.name]
                    if interaction.feature_2 is not None:
                        parent_names.append(interaction.feature_2.name)
                    
                    if not all(p in X.columns for p in parent_names):
                        continue
                    
                    name, vals = self._feature_cache.get_or_compute(
                        parent_names, interaction.op,
                        lambda inter=interaction: (inter.name, inter.generate(X))
                    )
                    
                    score = self._featureboost_score(
                        vals, y, self._current_oof_preds, cv
                    )
                    fb_scores[id(interaction)] = (interaction, score)
                except Exception:
                    pass  # Skip failed candidates
            
            if not fb_scores:
                return batch
            
            # Keep top proxy_top_pct
            n_keep = max(3, int(len(fb_scores) * self.proxy_top_pct))
            sorted_candidates = sorted(fb_scores.values(), key=lambda x: x[1], reverse=True)
            top_candidates = [interaction for interaction, _ in sorted_candidates[:n_keep]]
            
            return top_candidates + pipeline_candidates
            
        except Exception as e:
            self._log(f"  Proxy screening failed ({e}), falling back to full evaluation")
            return batch

    def _get_cv_splitter(self):
        """Get the CV splitter object from self.cv (handles int and splitter)."""
        if isinstance(self.cv, int):
            y = getattr(self, "_current_y", None)
            if y is None:
                return self.cv
            return make_cv_splitter(
                self.cv,
                y,
                shuffle=True,
                random_state=self.random_state,
                groups=self._groups_active,
            )
        return self.cv

    def _compute_baseline_drift(self, X: pd.DataFrame) -> dict:
        """Adversarial drift score per ORIGINAL column vs the held-out test features.

        Returns {} when adversarial validation is disabled or no test set is set.
        Cheap: one adversarial-validation fit on the raw (original) columns shared
        with ``self.X_test``. Engineered features inherit these scores via parents.
        """
        if not self.use_adversarial_validation or self.X_test is None:
            return {}
        try:
            from tabularaml.inspect.adversarial import AdversarialValidator
            shared = [c for c in X.columns if c in set(self.X_test.columns)]
            if not shared:
                return {}
            av = AdversarialValidator(cv=min(5, self.cv if isinstance(self.cv, int) else 5),
                                      random_state=self.random_state,
                                      use_gpu=self.device == "cuda", n_jobs=self.n_jobs)
            av.fit(X[shared], self.X_test[shared])
            self._adv_auc = av.auc_
            scores = av.feature_drift_scores()
            self._log(f"  Adversarial validation: train/test AUC={av.auc_:.4f} "
                      f"({'shift detected' if av.auc_ > 0.6 else 'distributions match'})")
            return scores
        except Exception as e:
            self._log(f"  Adversarial drift computation failed: {e}")
            return {}

    def _candidate_drift(self, inter) -> float:
        """Drift proxy for a candidate = mean drift of its parent features."""
        if not self._adv_drift_scores:
            return 0.0
        parents = [inter.feature_1.name]
        if getattr(inter, "feature_2", None) is not None:
            parents.append(inter.feature_2.name)
        vals = [self._adv_drift_scores.get(p, 0.0) for p in parents]
        return float(np.mean(vals)) if vals else 0.0

    def _adv_final_drift_drop(self, X: pd.DataFrame, y: pd.Series) -> list:
        """Drop generated features whose true (engineered) train/test drift is high.

        Transforms the held-out test set through the fitted pipeline so engineered
        columns exist on both sides, then runs adversarial validation and drops
        generated columns with drift score above ``adv_drift_max``. Original
        features are never dropped. Caps removal at 50% of generated features.
        """
        if not self.use_adversarial_validation or self.X_test is None:
            return []
        generated = [c for c in X.columns if c not in self.initial_features]
        if not generated:
            return []
        try:
            from tabularaml.inspect.adversarial import AdversarialValidator
            # Fit a throwaway copy on the original columns so the live search
            # pipeline/state is never mutated, then engineer the test matrix.
            tmp = deepcopy(self)
            tmp.X_test = None  # avoid recursion / extra work in the copy
            tmp.fit(X[self.initial_features], y)
            X_test_t = tmp.transform(self.X_test)
            shared_gen = [c for c in generated if c in set(X_test_t.columns)]
            if not shared_gen:
                return []
            av = AdversarialValidator(cv=min(5, self.cv if isinstance(self.cv, int) else 5),
                                      random_state=self.random_state,
                                      use_gpu=self.device == "cuda", n_jobs=self.n_jobs)
            av.fit(X[shared_gen], X_test_t[shared_gen])
            scores = av.feature_drift_scores()
            flagged = [(c, s) for c, s in scores.items() if s > self.adv_drift_max]
            flagged.sort(key=lambda kv: kv[1], reverse=True)
            cap = max(0, int(0.5 * len(generated)))
            to_drop = [c for c, _ in flagged[:cap]]
            if to_drop:
                self._log(f"  Adversarial drift drop (AUC={av.auc_:.4f}): removing "
                          f"{len(to_drop)} high-drift generated features (>{self.adv_drift_max})")
            return to_drop
        except Exception as e:
            self._log(f"  Adversarial final drift drop failed: {e}")
            return []

    def _final_regularized_selection(self, X, y):
        """After search, use L1 regularization + tree importance to jointly select the best feature subset.
        
        Only applies when ≥10 generated features exist. Original features are always kept.
        Returns a list of features to drop (generated features that didn't survive selection).
        """
        generated_features = [col for col in X.columns if col not in self.initial_features]
        
        if len(generated_features) < 10:
            return []  # Not enough generated features to warrant selection
        
        self._log(f"Regularized post-selection: evaluating {len(generated_features)} generated features...")
        
        try:
            from sklearn.preprocessing import StandardScaler
            
            # Prepare data — only numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 3:
                return []
            
            X_numeric = X[numeric_cols].copy()
            X_numeric = X_numeric.fillna(X_numeric.median())
            X_numeric = X_numeric.replace([np.inf, -np.inf], 0)
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_numeric)
            
            # Phase 1: L1 regularization
            l1_selected = set()
            try:
                if self.task == "regression":
                    from sklearn.linear_model import LassoCV
                    model = LassoCV(cv=5, alphas=np.logspace(-4, 1, 50), max_iter=10000, n_jobs=self.n_jobs)
                    model.fit(X_scaled, y)
                    coef_mask = np.abs(model.coef_) > 1e-6
                else:
                    from sklearn.linear_model import LogisticRegressionCV
                    model = LogisticRegressionCV(cv=5, penalty='l1', solver='saga',
                                                 max_iter=5000, n_jobs=self.n_jobs, Cs=50)
                    model.fit(X_scaled, y)
                    # For multi-class, take absolute max across classes
                    if model.coef_.ndim > 1:
                        coef_mask = np.abs(model.coef_).max(axis=0) > 1e-6
                    else:
                        coef_mask = np.abs(model.coef_.ravel()) > 1e-6
                
                l1_selected = set(np.array(numeric_cols)[coef_mask].tolist())
                self._log(f"  L1 selected {len(l1_selected)} features")
            except Exception as e:
                self._log(f"  L1 selection failed: {e}")
                l1_selected = set(numeric_cols)  # Fallback: keep all
            
            # Phase 2: Tree-based importance
            tree_selected = set()
            try:
                if self.device == "cuda":
                    from xgboost import XGBRegressor, XGBClassifier
                    if self.task == "regression":
                        tree_model = XGBRegressor(n_estimators=300, max_depth=6, verbosity=0, n_jobs=self.n_jobs, device="cuda", enable_categorical=True)
                    else:
                        tree_model = XGBClassifier(n_estimators=300, max_depth=6, verbosity=0, n_jobs=self.n_jobs, device="cuda", enable_categorical=True)
                else:
                    from lightgbm import LGBMRegressor, LGBMClassifier
                    if self.task == "regression":
                        tree_model = LGBMRegressor(n_estimators=300, max_depth=6, n_jobs=self.n_jobs, verbose=-1)
                    else:
                        tree_model = LGBMClassifier(n_estimators=300, max_depth=6, n_jobs=self.n_jobs, verbose=-1)
                tree_model.fit(X_numeric, y)
                importances = pd.Series(tree_model.feature_importances_, index=numeric_cols)
                # Keep top-K where K = number of L1-selected features (or at least initial features count)
                n_keep = max(len(l1_selected), len(self.initial_features))
                tree_selected = set(importances.nlargest(n_keep).index.tolist())
                self._log(f"  Tree importance selected top {len(tree_selected)} features")
            except Exception as e:
                self._log(f"  Tree importance failed: {e}")
                tree_selected = set(numeric_cols)
            
            # Final set: union of L1 and tree selected
            selected = l1_selected | tree_selected
            
            # Original features are ALWAYS kept
            selected.update(self.initial_features)
            
            # Also keep non-numeric generated features (categorical encodings, etc.)
            non_numeric_generated = [col for col in generated_features if col not in numeric_cols]
            selected.update(non_numeric_generated)
            
            # Features to drop
            features_to_drop = [col for col in generated_features 
                               if col in numeric_cols and col not in selected]
            
            if features_to_drop:
                self._log(f"  Regularized selection: dropping {len(features_to_drop)} weak generated features")
                self._log(f"  Dropped: {features_to_drop}")
            else:
                self._log(f"  Regularized selection: all generated features retained")
            
            return features_to_drop
            
        except Exception as e:
            self._log(f"  Regularized post-selection failed: {e}")
            return []

    def _softmax_temp_sampling(self, pool, weights, n=1, tau=0.5) -> list:
        """Sample items using softmax temperature sampling."""
        if n >= len(pool):
            return pool
        weights = np.array(weights)
        w = weights / tau
        w -= np.max(w)
        probs = np.exp(w) / np.sum(np.exp(w))
        return random.choices(pool, k=n, weights=probs)
    
    def _analyze_feature_interactions(self, X: pd.DataFrame, y: pd.Series, max_pairs: int = 200) -> Dict[tuple, float]:
        """Use SHAP interaction values to identify feature pairs with strong interactions."""
        importance_analyzer = FeatureImportanceAnalyzer(
            task_type=self.task, use_gpu=self.device == "cuda", verbose=0, n_jobs=self.n_jobs, preferred_gbm="xgboost" if self.device == "cuda" else "lightgbm")
        return importance_analyzer.get_feature_interactions(X, y, max_pairs=max_pairs)
    
    def _get_feature_family(self, feature_name: str) -> str:
        """Get the root/family name of a feature by extracting the original column name."""
        # Split on common separators and take the first part as the family
        for sep in self.ops["num"]["binary"] + self.ops["cat"]["binary"]:
            if sep in feature_name:
                return feature_name.split(sep)[0]
        return feature_name

    def _sample_parents_with_memory(self, generation: List[Feature], n=20, tau=0.5) -> tuple[list[Feature], list[tuple[Feature, Feature]]]:
        """Enhanced parent sampling using adaptive controller's memory."""
        generation = [f for f in generation if not f.require_pipeline]
        if not generation:
            return [], []
        
        # Get parent quality scores
        parent_scores = []
        for feat in generation:
            base_score = feat.weight
            parent_quality = self.adaptive_controller.get_parent_quality_score(feat.name)
            usage_penalty = 1.0 / (1.0 + self._parent_usage.get(feat.name, 0) * 0.1)
            
            # During stagnation, heavily weight parent quality and usage
            if self.adaptive_controller.state.stagnation_level.value >= StagnationLevel.SEVERE.value:
                total_score = 0.2 * base_score + 0.5 * parent_quality + 0.3 * usage_penalty
            else:
                total_score = 0.5 * base_score + 0.3 * parent_quality + 0.2 * usage_penalty
            
            parent_scores.append((feat, total_score))
        
        # Sort by score
        parent_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Sample with diversity enforcement
        unary_features = []
        selected_families = set()
        
        # First, select some high-quality parents
        for feat, score in parent_scores[:n//2]:
            family = self._get_feature_family(feat.name)
            unary_features.append(feat)
            selected_families.add(family)
            if len(unary_features) >= n//2:
                break
        
        # Then, add diverse parents
        for feat, score in parent_scores[n//2:]:
            family = self._get_feature_family(feat.name)
            if family not in selected_families or len(selected_families) < 3:
                unary_features.append(feat)
                selected_families.add(family)
            if len(unary_features) >= n:
                break
        
        # Create feature pairs with preference for cross-family and high-interaction pairs
        feature_pairs = []
        
        # Use SHAP interactions if available
        if hasattr(self, 'feature_interactions') and self.feature_interactions:
            name_to_feature = {f.name: f for f in generation}
            interaction_pairs = []
            
            for (f1_name, f2_name), interaction_strength in self.feature_interactions.items():
                if f1_name in name_to_feature and f2_name in name_to_feature:
                    f1, f2 = name_to_feature[f1_name], name_to_feature[f2_name]
                    parent1_quality = self.adaptive_controller.get_parent_quality_score(f1_name)
                    parent2_quality = self.adaptive_controller.get_parent_quality_score(f2_name)
                    
                    pair_score = interaction_strength * (parent1_quality + parent2_quality) / 2
                    interaction_pairs.append(((f1, f2), pair_score))
            
            # Sort by score and take top pairs
            interaction_pairs.sort(key=lambda x: x[1], reverse=True)
            for (f1, f2), _ in interaction_pairs[:n//2]:
                feature_pairs.append((f1, f2))
        
        # Add random diverse pairs
        families = list(set(self._get_feature_family(f.name) for f in generation))
        while len(feature_pairs) < n:
            if len(families) >= 2 and random.random() < 0.7:
                # Cross-family pair
                f1_family, f2_family = random.sample(families, 2)
                f1_candidates = [f for f in generation if self._get_feature_family(f.name) == f1_family]
                f2_candidates = [f for f in generation if self._get_feature_family(f.name) == f2_family]
                if f1_candidates and f2_candidates:
                    feature_pairs.append((random.choice(f1_candidates), random.choice(f2_candidates)))
            else:
                # Random pair
                if len(generation) >= 2:
                    feature_pairs.append(tuple(random.sample(generation, 2)))
        
        # Update usage tracking
        for f in unary_features:
            self._parent_usage[f.name] = self._parent_usage.get(f.name, 0) + 1
        for f1, f2 in feature_pairs:
            self._parent_usage[f1.name] = self._parent_usage.get(f1.name, 0) + 1
            self._parent_usage[f2.name] = self._parent_usage.get(f2.name, 0) + 1
            
        return unary_features, feature_pairs[:n]

    def _sample_children_with_creativity(self, candidates_pool: List[Interaction], n=200, 
                                       tau=0.7, force_creative=False) -> List[Interaction]:
        """Enhanced child sampling with creativity injection."""
        if not candidates_pool:
            return []
        
        if n >= len(candidates_pool):
            return candidates_pool
        
        # During severe stagnation, inject creative operations
        if force_creative or self.adaptive_controller.state.stagnation_level.value >= StagnationLevel.SEVERE.value:
            # Group by operation type
            op_groups = defaultdict(list)
            for interaction in candidates_pool:
                op_groups[interaction.op].append(interaction)
            
            # Get creative operations
            creative_ops = []
            for dtype in ["num", "cat"]:
                for op_type in ["unary", "binary"]:
                    creative_ops.extend(self.adaptive_controller.get_creative_operations(dtype, op_type, 3))
            
            # Prioritize interactions with creative operations
            creative_candidates = []
            normal_candidates = []
            
            for interaction in candidates_pool:
                if interaction.op in creative_ops:
                    creative_candidates.append(interaction)
                else:
                    normal_candidates.append(interaction)
            
            # Take more creative candidates during stagnation
            creative_ratio = min(0.7, 0.3 + 0.1 * self.adaptive_controller.state.stagnation_level.value)
            n_creative = int(n * creative_ratio)
            n_normal = n - n_creative
            
            result = []
            if creative_candidates:
                weights = [i.weight for i in creative_candidates]
                result.extend(self._softmax_temp_sampling(creative_candidates, weights, n_creative, tau * 1.5))
            
            if normal_candidates and len(result) < n:
                weights = [i.weight for i in normal_candidates]
                result.extend(self._softmax_temp_sampling(normal_candidates, weights, n - len(result), tau))
            
            return result
        else:
            # Normal sampling
            weights = [i.weight for i in candidates_pool]
            return self._softmax_temp_sampling(candidates_pool, weights, n, tau)

    def _creative_hopeful_monster(self, X: pd.DataFrame, y: pd.Series, generation: list, 
                                n_features: int = 10, callback: Optional[Callable] = None) -> tuple[list, pd.DataFrame, PipelineWrapper]:
        """Completely revamped hopeful monster strategy with true creativity."""
        valid_generation = [feat for feat in generation if feat.name in X.columns]
        
        if len(valid_generation) < 2:
            return [], X, self.pipeline
        
        # Apply budget constraint
        remaining_budget = self.max_gen_new_feats - self.state['counters']['total_new_features'] if self.max_gen_new_feats != float('inf') else float('inf')
        max_features_to_find = min(n_features, remaining_budget) if remaining_budget > 0 else 0
        
        if max_features_to_find <= 0:
            self._log(f"  Creative HM: No remaining feature budget")
            return [], X, self.pipeline
        
        if callback:
            callback(0, 0)
        
        candidates_pool = []
        
        # Strategy 1: Use completely random combinations
        random_features = random.sample(valid_generation, min(len(valid_generation), self.n_parents))
        
        # Strategy 2: Use creative operations that haven't been tried much
        creative_ops = {
            "num": {
                "unary": self.adaptive_controller.get_creative_operations("num", "unary", 5),
                "binary": self.adaptive_controller.get_creative_operations("num", "binary", 5)
            },
            "cat": {
                "unary": self.adaptive_controller.get_creative_operations("cat", "unary", 5),
                "binary": self.adaptive_controller.get_creative_operations("cat", "binary", 5)
            }
        }
        
        # Generate candidates with creative operations
        for feat in random_features[:self.n_parents//2]:
            dtype = feat.dtype
            # Use creative unary operations
            for op in creative_ops[dtype]["unary"]:
                if op in self.ops[dtype]["unary"]:
                    candidates_pool.append(Interaction(feat, op))
        
        # Generate random pairs with creative binary operations
        for _ in range(self.n_parents//2):
            f1, f2 = random.sample(random_features, 2)
            dtype = "num" if f1.dtype == f2.dtype == "num" else "cat"
            for op in creative_ops[dtype]["binary"]:
                if op in self.ops[dtype]["binary"]:
                    candidates_pool.append(Interaction(f1, op, f2))
        
        # Strategy 3: Multi-step transformations (transform already transformed features)
        transformed_features = [f for f in valid_generation if f.depth > 0]
        if transformed_features:
            for feat in random.sample(transformed_features, min(len(transformed_features), 5)):
                # Apply another transformation
                dtype = feat.dtype
                for op in random.sample(self.ops[dtype]["unary"], min(3, len(self.ops[dtype]["unary"]))):
                    candidates_pool.append(Interaction(feat, op))
        
        # Strategy 4: Use features that have never been used as parents
        unused_parents = [f for f in valid_generation 
                         if self.adaptive_controller.feature_as_parent_attempts.get(f.name, 0) == 0]
        if unused_parents:
            for feat in random.sample(unused_parents, min(len(unused_parents), 10)):
                dtype = feat.dtype
                # Try multiple operations
                for op in random.sample(self.ops[dtype]["unary"], min(2, len(self.ops[dtype]["unary"]))):
                    candidates_pool.append(Interaction(feat, op))
        
        # Remove duplicates and blacklisted
        seen = set()
        unique_candidates = []
        for interaction in candidates_pool:
            if interaction.name not in seen and interaction.name not in getattr(self, 'blacklisted_features', set()):
                seen.add(interaction.name)
                unique_candidates.append(interaction)
        
        # Sample with heavy randomization
        batch = self._sample_children_with_creativity(unique_candidates, self.n_children * 2, tau=2.0, force_creative=True)
        
        self._log(f"  Creative HM: evaluating {len(batch)} highly creative candidates...")
        elites, X_new, pipeline_new = self._select_elites(batch, max_features_to_find, X, y, callback=callback)
        
        self._log(f"  Creative HM: found {len(elites)} features from {len(batch)} candidates")
        return elites, X_new, pipeline_new

    def _prepare_pipeline(self, interactions: List[Interaction]) -> PipelineWrapper:
        """Prepare PipelineWrapper for encoding operations."""
        target_enc_cols = [i.feature_1.name for i in interactions if i.op == "target"]
        count_enc_cols = [i.feature_1.name for i in interactions if i.op == "count"]
        freq_enc_cols = [i.feature_1.name for i in interactions if i.op == "freq"]
        
        # Collect GroupBy encoders for agg interactions
        groupby_encoders = []
        for i in interactions:
            if i.is_agg and i.feature_2 is not None:
                agg_func = i.op.replace("groupby_", "")
                groupby_encoders.append(
                    GroupByEncoder(cat_col=i.feature_1.name, num_col=i.feature_2.name,
                                  agg_func=agg_func, output_col=i.name)
                )
        
        pipeline = PipelineWrapper(imputer=None, scaler=None,
                                   encoder=self._make_cat_encoder(target_enc_cols, count_enc_cols, freq_enc_cols))
        pipeline.groupby_encoders = groupby_encoders
        
        # Collect Temporal encoders for temporal interactions
        temporal_encoders = []
        for i in interactions:
            if getattr(i, 'is_temporal', False):
                temporal_encoders.append(
                    TemporalEncoder(col=i.feature_1.name, id_col=self.id_col,
                                   time_col=self.time_col, op_name=i.op, output_col=i.name)
                )
        pipeline.temporal_encoders = temporal_encoders
        return pipeline

    def _extend_pipeline(self, pipeline: PipelineWrapper, new_pipeline: PipelineWrapper) -> PipelineWrapper:
        """Extend pipeline with new_pipeline for categorical encoding."""
        # Merge existing GroupBy encoders
        existing_gb = getattr(pipeline, 'groupby_encoders', [])
        new_gb = getattr(new_pipeline, 'groupby_encoders', [])
        # Deduplicate by output_col name
        seen_gb = {gb.output_col for gb in existing_gb}
        merged_gb = list(existing_gb)
        for gb in new_gb:
            if gb.output_col not in seen_gb:
                merged_gb.append(gb)
                seen_gb.add(gb.output_col)
        
        result = PipelineWrapper(imputer=None, scaler=None,
            encoder=self._make_cat_encoder(
                target_enc_cols=list(set(pipeline.encoder.target_enc_cols + new_pipeline.encoder.target_enc_cols)),
                count_enc_cols=list(set(pipeline.encoder.count_enc_cols + new_pipeline.encoder.count_enc_cols)),
                freq_enc_cols=list(set(pipeline.encoder.freq_enc_cols + new_pipeline.encoder.freq_enc_cols))))
        result.groupby_encoders = merged_gb
        
        # Merge temporal encoders
        existing_te = getattr(pipeline, 'temporal_encoders', [])
        new_te = getattr(new_pipeline, 'temporal_encoders', [])
        seen_te = {te.output_col for te in existing_te}
        merged_te = list(existing_te)
        for te in new_te:
            if te.output_col not in seen_te:
                merged_te.append(te)
                seen_te.add(te.output_col)
        result.temporal_encoders = merged_te
        return result
        
    def _apply_interactions(self, X: pd.DataFrame, interactions: List[Interaction]) -> tuple[pd.DataFrame, PipelineWrapper]:
        """Apply non-pipeline feature interactions to X."""
        new_features = {}
        for interaction in interactions:
            if not interaction.require_pipeline:
                required_features = [interaction.feature_1.name]
                if interaction.feature_2 is not None:
                    required_features.append(interaction.feature_2.name)
                if all(feat in X.columns for feat in required_features):
                    try:
                        parent_names = [interaction.feature_1.name]
                        if interaction.feature_2 is not None:
                            parent_names.append(interaction.feature_2.name)
                        name, val = self._feature_cache.get_or_compute(
                            parent_names, interaction.op,
                            lambda inter=interaction: (inter.name, inter.generate(X))
                        )
                        if name not in X.columns and name not in new_features:  # Avoid duplicates
                            new_features[name] = val
                    except Exception as e:
                        self._log(f"Warning: Failed to generate {interaction.name}: {str(e)}")
        
        if new_features:
            # Check for duplicate columns before concatenating
            X = self._ensure_no_duplicates(X, "before adding features in _apply_interactions")
            
            X_copy = pd.concat([X.copy(), pd.DataFrame(new_features, index=X.index)], axis=1)
            
            # Verify no duplicates after concatenation
            X_copy = self._ensure_no_duplicates(X_copy, "after adding features in _apply_interactions")
        else:
            X_copy = X.copy()
            
        # Replace infs generated by non-pipeline features
        X_copy = X_copy.replace([np.inf, -np.inf], np.nan)
        return X_copy, self._prepare_pipeline(interactions)

    def _is_redundant(self, cand_name: str, X_with_cand: pd.DataFrame,
                      X_accepted: pd.DataFrame, strong: bool = False) -> bool:
        """True if a candidate is near-duplicate of an already-accepted engineered feature.

        Compares absolute Pearson correlation against accepted *generated* columns
        only. Features whose standalone gain is materially higher (``strong``) are
        kept regardless, so genuinely better replacements are not blocked.
        """
        if strong or cand_name not in X_with_cand.columns:
            return False
        cand = pd.to_numeric(X_with_cand[cand_name], errors="coerce")
        if not np.isfinite(cand.to_numpy(dtype=float)).any() or cand.nunique(dropna=True) <= 1:
            return False
        accepted_generated = [c for c in X_accepted.columns
                              if c not in self.initial_features and c != cand_name
                              and pd.api.types.is_numeric_dtype(X_accepted[c])]
        if not accepted_generated:
            return False
        cand_vals = cand.to_numpy(dtype=float)
        for col in accepted_generated:
            other = pd.to_numeric(X_accepted[col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(cand_vals) & np.isfinite(other)
            if mask.sum() < 10:
                continue
            a, b = cand_vals[mask], other[mask]
            if a.std() == 0 or b.std() == 0:
                continue
            corr = abs(np.corrcoef(a, b)[0, 1])
            if np.isfinite(corr) and corr >= self.redundancy_corr_threshold:
                return True
        return False

    def _seed_template_candidates(self, X: pd.DataFrame, generation: list) -> list:
        """Deterministic 2nd-order + groupby-cross template pool to guarantee coverage.

        Seeds the first generation with the highest-value region of the search
        space (OpenFE-style): all arithmetic crosses among top-importance numeric
        features, every categorical x numeric groupby aggregation, and count/freq
        of top categoricals. Genetic search then explores beyond these.
        """
        if not self.seed_templates:
            return []
        num_cols, cat_cols = self._get_num_cat_cols(X)
        num_set, cat_set = set(num_cols), set(cat_cols)
        feats = [f for f in generation if f.name in X.columns]
        num_feats = sorted([f for f in feats if f.dtype == "num" and f.name in num_set],
                           key=lambda f: f.weight, reverse=True)[:self.seed_top_k]
        cat_feats = sorted([f for f in feats if f.dtype == "cat" and f.name in cat_set],
                           key=lambda f: f.weight, reverse=True)[:self.seed_top_k]

        cands = []
        # Numeric x numeric arithmetic crosses (classic 2nd-order pool)
        bin_ops = [op for op in ("add", "sub", "mul", "div", "absdiff")
                   if op in self.ops.get("num", {}).get("binary", [])]
        for i in range(len(num_feats)):
            for j in range(i + 1, len(num_feats)):
                for op in bin_ops:
                    cands.append(Interaction(num_feats[i], op, num_feats[j]))
        # Categorical x numeric groupby aggregations
        if "agg" in self.ops:
            for cf in cat_feats:
                for nf in num_feats:
                    for op in self.ops["agg"]["binary"]:
                        cands.append(Interaction(cf, op, nf))
        # Count / frequency encodings of top categoricals
        for cf in cat_feats:
            for op in ("count", "freq"):
                if op in self.ops.get("cat", {}).get("unary", []):
                    cands.append(Interaction(cf, op))

        # Drop templates whose feature already exists; cap total (preserve priority order)
        existing = set(X.columns)
        cands = [c for c in cands if c.name not in existing]
        if len(cands) > self.seed_max_candidates:
            cands = cands[:self.seed_max_candidates]
        self._log(f"Seeded {len(cands)} template candidates from "
                  f"{len(num_feats)} numeric + {len(cat_feats)} categorical top features")
        return cands

    def _select_elites_batch(self, batch: list[Interaction], n: int, X: pd.DataFrame, y: pd.Series,
                             callback: Optional[Callable] = None) -> tuple[list[Interaction], pd.DataFrame, PipelineWrapper]:
        """Batched joint selection: evaluate non-pipeline candidates in groups.

        Fits one model per batch (instead of one per candidate), so complementary
        "suppressor" features that only help together can be admitted as a set.
        Pipeline-required candidates (encodings/groupby/temporal) are deferred to
        the sequential selector, which fits their leakage-safe encoders per fold.
        """
        self._in_batch_select = True
        X = self._ensure_no_duplicates(X, "in _select_elites_batch")
        valid = [i for i in batch if all(feat in X.columns for feat in
                 ([i.feature_1.name] + ([i.feature_2.name] if i.feature_2 else [])))
                 and i.name not in getattr(self, 'blacklisted_features', set())]
        nonpipe = [i for i in valid if not i.require_pipeline]
        pipe = [i for i in valid if i.require_pipeline]

        ranked = self.adaptive_controller.rank_candidates_with_memory(nonpipe, X, y)
        _, best_val = self._eval_baseline(X, y, self.pipeline)
        selected, X_base = [], X.copy()
        bs = max(2, int(self.batch_size))
        min_gain = self.adaptive_controller.get_adaptive_min_gain()
        evals = 0

        for start in range(0, len(ranked), bs):
            if len(selected) >= n:
                break
            if hasattr(self, 'stop_requested') and self.stop_requested:
                break
            chunk = ranked[start:start + bs]
            X_try = X_base.copy()
            added = []
            for inter in chunk:
                if len(selected) + len(added) >= n or inter.name in X_try.columns:
                    continue
                try:
                    val = inter.generate(X_try)
                except Exception:
                    continue
                X_try[inter.name] = np.asarray(val)
                added.append(inter)
            if not added:
                continue
            X_try = X_try.replace([np.inf, -np.inf], np.nan)
            evals += 1
            try:
                _, new_val = self._eval_baseline(X_try, y, self.pipeline)
            except Exception:
                continue
            delta = (new_val - best_val) if self.scorer.greater_is_better else (best_val - new_val)
            gain = delta / (abs(best_val) + 1e-8)
            if callback and callback(start + len(chunk), len(selected)):
                break
            if gain >= min_gain:
                # Keep the improving batch, dropping redundant near-duplicates.
                for inter in added:
                    if self.redundancy_prune and self._is_redundant(inter.name, X_try, X_base):
                        X_try = X_try.drop(columns=[inter.name])
                        continue
                    selected.append(inter)
                    self.adaptive_controller.update_operation_stats(inter, success=True, gain=gain)
                X_base, best_val = X_try, new_val

        # Defer pipeline-required candidates to the sequential selector. The
        # re-entrancy guard stays set so the nested call runs the greedy path
        # (the dispatcher resets it in a finally once the whole batch completes).
        if pipe and len(selected) < n:
            pipe_sel, X_base, pipe_pipeline = self._select_elites(
                pipe, n - len(selected), X_base, y, callback)
            selected.extend(pipe_sel)
            if callback:
                callback(len(ranked) + len(pipe), len(selected), force_complete=True)
            return selected, X_base, self._extend_pipeline(self.pipeline, self._prepare_pipeline(selected))

        if callback:
            callback(len(ranked), len(selected), force_complete=True)
        return selected, X_base, self._extend_pipeline(self.pipeline, self._prepare_pipeline(selected))

    def _select_elites(self, batch: list[Interaction], n: int, X: pd.DataFrame, y: pd.Series,
                      callback: Optional[Callable] = None) -> tuple[list[Interaction], pd.DataFrame, PipelineWrapper]:
        """Greedy forward-selection with adaptive thresholds."""
        # Joint/batch selection path (opt-in, or auto during severe stagnation).
        # The re-entrancy guard prevents infinite recursion when the batch selector
        # defers its pipeline-required candidates back to this greedy selector.
        if (getattr(self, 'batch_evaluation', False) and batch
                and not getattr(self, '_in_batch_select', False)):
            try:
                return self._select_elites_batch(batch, n, X, y, callback)
            finally:
                self._in_batch_select = False
        if not batch:
            if callback: callback(0, 0, force_complete=True)
            return [], X, self.pipeline

        # Ensure X has no duplicate columns
        X = self._ensure_no_duplicates(X, "in _select_elites")

        # Filter valid interactions
        valid_batch = [i for i in batch if all(feat in X.columns for feat in 
                      ([i.feature_1.name] + ([i.feature_2.name] if i.feature_2 else [])))
                      and not (hasattr(self, 'blacklisted_features') and i.name in getattr(self, 'blacklisted_features', set()))]
        
        if not valid_batch:
            if callback: callback(0, 0, force_complete=True)
            return [], X, self.pipeline

        # Prepare features and ranking
        try:
            X_copy, pipe_batch = self._apply_interactions(X, valid_batch)
        except Exception as e:
            self._log(f"Error in _apply_interactions: {str(e)}")
            if callback: callback(0, 0, force_complete=True)
            return [], X, self.pipeline
            
        pipe_ext = self._extend_pipeline(self.pipeline, pipe_batch)
        
        # Use memory-aware ranking
        ranked = self.adaptive_controller.rank_candidates_with_memory(valid_batch, X, y)

        # Selection loop with adaptive threshold
        _, best_val = self._eval_baseline(X, y, self.pipeline)
        selected, X_base = [], X.copy()
        evals = consec_no_gain = 0
        
        # Adjust early stopping based on stagnation
        # if self.adaptive_controller.state.stagnation_level.value >= StagnationLevel.SEVERE.value:
        #     # Be more patient during severe stagnation
        #     early_thr = len(ranked)  # Evaluate all
        # else:
        #     early_thr = (int(len(ranked) * self.early_stopping_child_eval) 
        #                 if isinstance(self.early_stopping_child_eval, float) 
        #                 else self.early_stopping_child_eval 
        #                 if isinstance(self.early_stopping_child_eval, int) 
        #                 else len(ranked))
        
        # Respect user's early stopping parameter
        early_thr = (int(len(ranked) * self.early_stopping_child_eval) 
                    if isinstance(self.early_stopping_child_eval, float) 
                    else self.early_stopping_child_eval 
                    if isinstance(self.early_stopping_child_eval, int) 
                    else len(ranked))
        
        min_evals = max(5, int(0.05 * len(ranked)))

        for inter in ranked:
            if hasattr(self, 'stop_requested') and self.stop_requested:
                break
                
            evals += 1
            if callback and callback(evals, len(selected)):
                break
                
            if len(selected) >= n or not all(feat in X_base.columns for feat in ([inter.feature_1.name] + ([inter.feature_2.name] if inter.feature_2 else []))):
                continue

            # Evaluate interaction
            X_try = X_base.copy()
            if not inter.require_pipeline and inter.name in X_copy.columns:
                X_try[inter.name] = X_copy[inter.name].values
            
            # Check for duplicates before evaluation
            if X_try.columns.duplicated().any():
                self._log(f"Warning: Duplicate columns in X_try for {inter.name}, skipping")
                continue
                
            pipe_iter = self._extend_pipeline(self.pipeline, self._prepare_pipeline([inter] + selected))
            
            try:
                _, new_val = self._eval_baseline(X_try, y, pipe_iter)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._log(f"Error evaluating {inter.name}: {str(e)}")
                continue
            
            delta = (new_val - best_val) if self.scorer.greater_is_better else (best_val - new_val)
            gain = delta / (abs(best_val) + 1e-8)

            # Adversarial drift penalty: discount the gain of candidates built from
            # train/test-shifting parents so they must clear a higher acceptance bar.
            if self.use_adversarial_validation and self._adv_drift_scores:
                drift = self._candidate_drift(inter)
                if drift > 0:
                    gain -= self.adv_drift_weight * drift * abs(self.adaptive_controller.get_adaptive_min_gain())

            # Use adaptive threshold
            success = gain >= self.adaptive_controller.get_adaptive_min_gain()

            # Redundancy guard: reject near-duplicate features that barely beat the bar.
            if (success and self.redundancy_prune and not inter.require_pipeline
                    and inter.name in X_copy.columns):
                if self._is_redundant(inter.name, X_copy, X_base,
                                      strong=gain >= 2 * self.adaptive_controller.get_adaptive_min_gain()):
                    success = False

            self.adaptive_controller.update_operation_stats(inter, success=success, gain=gain)

            if success:
                selected.append(inter)
                X_base, best_val, consec_no_gain = X_try, new_val, 0
                # Engineered feature inherits its parents' drift for downstream candidates.
                if self.use_adversarial_validation and self._adv_drift_scores:
                    self._adv_drift_scores[inter.name] = self._candidate_drift(inter)
            else:
                consec_no_gain += 1

            # Always apply early stopping based on configured threshold
            if evals >= min_evals and consec_no_gain >= early_thr:
                break

        if callback: callback(len(ranked), len(selected), force_complete=True)
        return selected, X_base, self._extend_pipeline(self.pipeline, self._prepare_pipeline(selected))

    def _get_feature_dependencies(self, generation: list) -> dict:
        """Build a compact dependency graph for features."""
        deps = {}
        for feat in generation:
            if hasattr(feat, 'generating_interaction') and feat.generating_interaction:
                i = feat.generating_interaction
                if i.type == "unary":
                    deps.setdefault(i.feature_1.name, []).append(feat.name)
                elif i.type == "binary":
                    deps.setdefault(i.feature_1.name, []).append(feat.name)
                    deps.setdefault(i.feature_2.name, []).append(feat.name)
        return deps

    def _intelligent_pruning(self, X: pd.DataFrame, y: pd.Series, generation: list, 
                           prune_pct: float = 0.2) -> tuple[pd.DataFrame, PipelineWrapper, list]:
        """Intelligent pruning that considers feature history and dependencies."""
        if X.shape[1] <= len(self.initial_features) + 2:
            return X, self.pipeline, generation
            
        weights = self._get_top_k_features(X, y, k=-1, pipeline=self.pipeline)
        new_features = [col for col in X.columns if col not in self.initial_features]
        features_to_remove = max(1, int(len(new_features) * prune_pct))
        
        if len(new_features) < features_to_remove:
            return X, self.pipeline, generation
        
        # Get dependencies
        deps = self._get_feature_dependencies(generation)
        
        # Get features that should be protected
        protected_features = set()
        
        # Protect features that others depend on
        for parent_feat, dependent_list in deps.items():
            protected_features.add(parent_feat)
        
        # Protect features that have been successful parents
        for feat_name in new_features:
            if len(self.adaptive_controller.successful_children.get(feat_name, [])) > 0:
                protected_features.add(feat_name)
        
        # Sort by importance and filter out protected
        new_feat_weights = weights[weights.index.isin(new_features)].sort_values('weighted_importance')
        candidates = [f for f in new_feat_weights.index if f not in protected_features]
        
        # Also consider how many times a feature has been pruned before
        previously_pruned = getattr(self, 'previously_pruned_features', set())
        
        # Prioritize features that have been pruned before for removal
        priority_remove = [f for f in candidates if f in previously_pruned]
        other_candidates = [f for f in candidates if f not in previously_pruned]
        
        to_remove = priority_remove[:features_to_remove]
        if len(to_remove) < features_to_remove:
            to_remove.extend(other_candidates[:features_to_remove - len(to_remove)])
        
        if not to_remove:
            self._log("  No features can be pruned due to dependencies")
            return X, self.pipeline, generation
        
        # Update tracking
        self.blacklisted_features = getattr(self, 'blacklisted_features', set())
        self.previously_pruned_features = getattr(self, 'previously_pruned_features', set())
        
        for feat_name in to_remove:
            self.previously_pruned_features.add(feat_name)
            # Blacklist if pruned multiple times
            if feat_name in previously_pruned:
                self.blacklisted_features.add(feat_name)
        
        self.pruned_features = getattr(self, 'pruned_features', set())
        self.pruned_features.update(to_remove)
        
        X_pruned = X.drop(columns=to_remove)
        self._log(f"  Intelligently pruned features: {to_remove}")
        
        # Update pipeline
        updated_pipeline = deepcopy(self.pipeline)
        for feature_name in to_remove:
            if feature_name in updated_pipeline.encoder.target_enc_cols:
                updated_pipeline.encoder.target_enc_cols.remove(feature_name)
            if feature_name in updated_pipeline.encoder.count_enc_cols:
                updated_pipeline.encoder.count_enc_cols.remove(feature_name)
            if feature_name in updated_pipeline.encoder.freq_enc_cols:
                updated_pipeline.encoder.freq_enc_cols.remove(feature_name)
        
        return X_pruned, updated_pipeline, [feat for feat in generation if feat.name not in to_remove]

    def _partial_restart(self, X: pd.DataFrame, y: pd.Series, generation: list, 
                        keep_top_n: int = 10) -> tuple[pd.DataFrame, list]:
        """Perform a partial restart keeping only the best features."""
        self._log(f"  Performing partial restart (restart #{self.adaptive_controller.state.total_restarts + 1})")
        
        # Get the best features to keep from generation
        best_features = self.adaptive_controller.get_restart_features(
            generation, keep_top_n, current_columns=list(X.columns)
        )
        best_feature_names = {f.name for f in best_features}
        
        # Filter to only features that actually exist in X
        best_feature_names = {name for name in best_feature_names if name in X.columns}
        
        # Keep initial features and best generated features (avoiding duplicates)
        columns_to_keep = list(self.initial_features)
        for col in X.columns:
            if col in best_feature_names and col not in columns_to_keep:
                columns_to_keep.append(col)
        
        # Ensure no duplicate columns
        columns_to_keep = list(dict.fromkeys(columns_to_keep))  # Remove duplicates while preserving order
        
        # Check for and handle duplicate columns in X
        X = self._ensure_no_duplicates(X, "before restart")
        
        X_restart = X[columns_to_keep].copy()
        
        # Verify no duplicates in result
        X_restart = self._ensure_no_duplicates(X_restart, "after restart")
        
        # Update generation to match actual columns
        new_generation = [f for f in generation if f.name in X_restart.columns]
        
        # Update adaptive controller state
        self.adaptive_controller.state.last_restart_gen = self.state['counters']['current_gen']
        self.adaptive_controller.state.total_restarts += 1
        self.adaptive_controller.state.stagnation_level = StagnationLevel.MILD  # Reset to mild
        
        # Clear some tracking but keep learned patterns
        self.blacklisted_features = set()
        self.previously_pruned_features = set()
        self._parent_usage = {}
        self._feature_cache.clear()
        
        self._log(f"  Restart complete: kept {len(X_restart.columns)} features (was {X.shape[1]})")
        
        return X_restart, new_generation

    def _sync_state_components(self, X: pd.DataFrame, pipeline, generation: list,
                                preserve_pruned: bool = False):
        """Ensure all state components are consistent.

        Args:
            preserve_pruned: If True, keep pruned_features as-is (used when reverting to best)
        """
        self.X = X.copy()
        self.pipeline = pipeline
        self.generation = list(generation)  # Shallow copy is fine here
        self.interactions = [feat.generating_interaction for feat in self.generation
                           if hasattr(feat, 'generating_interaction') and feat.generating_interaction]
        if hasattr(self, 'pruned_features') and not preserve_pruned:
            # Only filter pruned features during normal search, not when reverting
            self.pruned_features = {feat for feat in self.pruned_features if feat not in X.columns}
    
    def _save_current_as_best(self):
        """Save current state as the best state and auto-save if path is provided."""
        if hasattr(self, 'state') and 'best' in self.state:
            self.state['best'].update(
                X=self.X.copy(),
                pipeline=deepcopy(self.pipeline),  # Deep copy to prevent mutation
                generation=deepcopy(self.generation),  # Deep copy to preserve interaction refs
                pruned_features=getattr(self, 'pruned_features', set()).copy(),
                interactions=deepcopy(getattr(self, 'interactions', []))  # Save interactions too
            )
            if hasattr(self, 'save_path') and self.save_path:
                self.save(self.save_path)
    
    def _revert_to_best(self):
        """Revert to best saved state."""
        if hasattr(self, 'state') and 'best' in self.state and self.state['best']['X'] is not None:
            self.X = self.state['best']['X'].copy()
            self.pipeline = deepcopy(self.state['best']['pipeline'])  # Deep copy to prevent mutation
            self.generation = deepcopy(self.state['best']['generation'])
            self.pruned_features = self.state['best']['pruned_features'].copy()
            # Restore interactions from saved state if available (backwards compat)
            if 'interactions' in self.state['best'] and self.state['best']['interactions']:
                self.interactions = deepcopy(self.state['best']['interactions'])
            else:
                # Fallback: rebuild from generation
                self.interactions = [feat.generating_interaction for feat in self.generation
                                   if hasattr(feat, 'generating_interaction') and feat.generating_interaction]
            return True
        return False

    def _get_search_parameters(self, progress: float, generation_num: int) -> tuple[float, float, float, float]:
        """Get search parameters with more aggressive exploration during stagnation."""
        # Base progressive annealing
        base_tau = max(0.1, 0.1 + 0.9 * (1 + np.cos(progress * np.pi)) / 2)
        base_beta = 0.8 * (1 / (1 + np.exp(10 * progress - 5)))
        base_gamma = 0.2 * np.exp(-3 * progress)
        base_lambda = 0.1 + 0.4 * (1 / (1 + np.exp(-10 * (progress - 0.5))))
        
        # More aggressive adjustments for stagnation
        intensity = self.adaptive_controller.state.exploration_intensity
        return (
            base_tau * (1 + 3 * intensity),  # Much higher temperature for more randomness
            base_beta * (1 + 2 * intensity),  # More exploration
            base_gamma * (1 + 4 * intensity),  # Much more diversity
            base_lambda * (1 - 0.7 * intensity)  # Less exploitation
        )

    def _drop_id_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that appear to be IDs to not be considered for feature generation."""
        cols_to_drop = []
        for col in X.columns:
            if hasattr(self, 'time_col') and col == self.time_col:
                continue
            if hasattr(self, 'id_col') and col == self.id_col:
                continue
            
            col_str = str(col).lower()
            is_id_name = col_str in ["id", "index"] or col_str.endswith("_id")
            
            # If it's explicitly named like an ID, or acts like a perfect ID (all uniquely categorical)
            if is_id_name or (X[col].nunique() == len(X) and not pd.api.types.is_float_dtype(X[col])):
                cols_to_drop.append(col)
        
        if cols_to_drop:
            self._log(f"Dropping ID columns from generation: {cols_to_drop}")
            if not hasattr(self, 'dropped_id_cols'):
                self.dropped_id_cols = set()
            self.dropped_id_cols.update(cols_to_drop)
            return X.drop(columns=cols_to_drop)
        return X

    def search(self, X: pd.DataFrame, y: pd.Series, X_test: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, PipelineWrapper, list[Feature], list[Interaction]]:
        """Enhanced genetic algorithm with better stagnation handling.

        If ``X_test`` (unlabeled test features) is provided and
        ``use_adversarial_validation`` is enabled, engineered features that drift
        between train and test distributions are penalized during selection and
        pruned at the end. Only test *features* are used -- never a target.
        """
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        start_time = time.time()
        X = self._drop_id_columns(X)
        if X_test is not None:
            self.X_test = self._drop_id_columns(X_test.copy())
        self._adv_drift_scores = {}
        self._set_defaults(X, y)
        self.cv = normalize_rotatable_splitter(self.cv)
        self.initial_features = list(X.columns)
        num_cols, cat_cols = self._get_num_cat_cols(X)
        self.max_gen_new_feats = (int(self.max_new_feats * len(self.initial_features)) if isinstance(self.max_new_feats, float)
                                 else self.max_new_feats if isinstance(self.max_new_feats, int) else float('inf'))
        if self.max_gen_new_feats == float('inf') and hasattr(self, 'max_gen_new_feats_pct'):
            self.max_gen_new_feats = int(self.max_gen_new_feats_pct * len(self.initial_features))

        # Label encode target for GBMs
        if self.task != "regression":
            unique_vals = np.unique(y)
            if not np.array_equal(unique_vals, np.arange(len(unique_vals))):
                y_encoded, _ = y.factorize(sort=True)
                y = pd.Series(y_encoded, index=y.index, name=y.name)

        # Instance sampling for large datasets
        X_full, y_full = None, None
        sample_size = getattr(self, 'search_sample_size', None)
        if sample_size and len(X) > sample_size:
            X_full, y_full = X, y
            X, y, groups_sub = self._create_search_subsample(X, y, sample_size, self.groups)
            self._groups_active = groups_sub
            if hasattr(self.cv, '_groups'):
                self.cv._groups = groups_sub
            self._log(f"Instance sampling: {len(X_full)} -> {len(X)} rows for search (search_sample_size={sample_size})")
        else:
            self._groups_active = self.groups
            if hasattr(self.cv, '_groups'):
                self.cv._groups = self.groups

        # Initialize
        self.pruned_features = set()
        self._parent_usage = {}
        self._seeds_injected = False
        self._feature_cache.clear()
        self._oof_preds_stale = True  # Proxy evaluation: force recompute on first generation
        self._current_y = y  # Reference for proxy eval objective detection

        # Meta-validation split (CV bias fix)
        X_meta, y_meta, groups_meta = None, None, None
        if self.meta_validation_frac > 0 and len(X) > 2000:
            try:
                split_meta = getattr(self.cv, "split_meta", None)
                if callable(split_meta):
                    search_idx, meta_idx = split_meta(
                        X,
                        y=y,
                        groups=self._groups_active,
                        frac=self.meta_validation_frac,
                        random_state=self.random_state,
                    )
                elif self._groups_active is not None:
                    from sklearn.model_selection import GroupShuffleSplit
                    gss = GroupShuffleSplit(n_splits=1, test_size=self.meta_validation_frac, random_state=self.random_state)
                    search_idx, meta_idx = next(gss.split(X, y, groups=self._groups_active))
                else:
                    from sklearn.model_selection import train_test_split
                    stratify = y if self.task != "regression" else None
                    search_idx, meta_idx = train_test_split(
                        np.arange(len(X)), test_size=self.meta_validation_frac,
                        stratify=stratify, random_state=self.random_state
                    )
                X_meta, y_meta = X.iloc[meta_idx].copy(), y.iloc[meta_idx].copy()
                if self._groups_active is not None:
                    groups_arr = np.asarray(self._groups_active)
                    groups_meta = groups_arr[meta_idx]
                X, y = X.iloc[search_idx].copy(), y.iloc[search_idx].copy()
                if self._groups_active is not None:
                    self._groups_active = np.asarray(self._groups_active)[search_idx]
                    if hasattr(self.cv, '_groups'):
                        self.cv._groups = self._groups_active
                self._log(f"Meta-validation split: {len(X)} search + {len(X_meta)} meta-validation rows")
            except Exception as e:
                self._log(f"Meta-validation split failed: {e}")
                X_meta, y_meta, groups_meta = None, None, None  # Fallback: no meta split

        self._log(f"Starting {self.task} on {self.device} - {X.shape[0]} samples, {X.shape[1]} features")
        self._log(f"Params: gen={self.n_generations}, parents={self.n_parents}, children={self.n_children}, limit={self.max_gen_new_feats}, time_budget={self.time_budget}s.")
        self.adaptive_controller.initialize_operations(self.ops)
        self.adaptive_controller.reset_for_new_run()
        self.state['best']['train_score'], self.state['best']['val_score'] = self._eval_baseline(X, y, self.pipeline)
        gen0_log = f"Gen 0: Train {self.scorer.name}={self.state['best']['train_score']:.5f}, Val {self.scorer.name}={self.state['best']['val_score']:.5f}"
        if self.logging_scorers:
            logging_scores = self._eval_logging_scorers(X, y, self.pipeline)
            gen0_log += f" | {self._format_logging_scores(logging_scores)}"
        self._log(gen0_log)
        self.state['best']['X'], self.state['best']['pipeline'] = X.copy(), deepcopy(self.pipeline)
        self.state['best']['pruned_features'] = getattr(self, 'pruned_features', set()).copy()
        
        # Baseline adversarial drift on original columns (engineered feats inherit via parents)
        if self.use_adversarial_validation and self.X_test is not None:
            self._adv_drift_scores = self._compute_baseline_drift(X)

        # Initialize interactions and generation
        self.feature_interactions = self._analyze_feature_interactions(X, y, max_pairs=10000)
        top_feats_df = self._get_top_k_features(X, y, k=2*self.n_parents, pipeline=self.pipeline)
        generation = [Feature(name=feat, dtype="num" if feat in num_cols else "cat", 
                             weight=top_feats_df.loc[feat, "weighted_importance"]) for feat in top_feats_df.index]
        self.state['best']['generation'] = generation.copy()

        # Main loop
        stagnation_counter = 0
        hopeful_monster_consecutive_fails = 0
        cv_rotation_splits = self.cv if isinstance(self.cv, int) else None
        
        with tqdm(total=self.n_generations, desc="Generations") as pbar:
            for N in range(self.n_generations):
                self.state['counters']['current_gen'] = N
                
                # Check for stop request
                if hasattr(self, 'stop_requested') and self.stop_requested:
                    self._log(f"🛑 Generation stopped by user request at generation {N}")
                    break
                
                if self.time_budget and (time.time() - start_time) > self.time_budget:
                    self._log(f"Time budget exceeded. Stopping.")
                    break
                
                progress = N / self.n_generations
                tau, beta, gamma, lambda_ = self._get_search_parameters(progress, N)
                self.adaptive_controller.assess_stagnation(
                    self.state['counters']['no_feature_gens_count'],
                    self.state['counters']['consecutive_no_improvement_iters']
                )
                
                # CV fold rotation (CV bias fix)
                if self.rotate_cv_folds and N > 0 and N % self.fold_rotation_period == 0:
                    cv_rotated = False
                    if cv_rotation_splits is not None:
                        if self._groups_active is not None:
                            self.cv = RotatedGroupKFold(cv_rotation_splits, rotation=self.random_state + N)
                        else:
                            self.cv = make_cv_splitter(
                                cv_rotation_splits,
                                y,
                                shuffle=True,
                                random_state=self.random_state + N,
                                groups=self._groups_active,
                            )
                        cv_rotated = True
                    else:
                        rotate_cv = getattr(self.cv, "rotated", None)
                        if callable(rotate_cv):
                            self.cv = rotate_cv(N)
                            cv_rotated = True

                    if not cv_rotated:
                        self._log(f"  CV fold rotation skipped for {type(self.cv).__name__}")
                    else:
                        self._oof_preds_stale = True
                        self._log(f"  CV fold rotation: rotated to {type(self.cv).__name__}")
                
                # Check for restart conditions
                if self.adaptive_controller.should_restart(N):
                    X, generation = self._partial_restart(X, y, generation, keep_top_n=15)
                    self.state['counters']['total_new_features'] = X.shape[1] - len(self.initial_features)
                    stagnation_counter = 0
                    hopeful_monster_consecutive_fails = 0
                    continue
                
                # Periodic check for duplicate columns
                X = self._ensure_no_duplicates(X, f"at Gen {N+1}")
                if X.shape[1] < len([f for f in generation if f.name in X.columns]):
                    # Update generation to match if columns were removed
                    generation = [f for f in generation if f.name in X.columns]
                    self.state['counters']['total_new_features'] = X.shape[1] - len(self.initial_features)
                
                # Intelligent pruning after extended stagnation
                if stagnation_counter >= 5:
                    prune_pct = 0.3 if self.adaptive_controller.state.stagnation_level == StagnationLevel.CRITICAL else 0.2
                    X, self.pipeline, generation = self._intelligent_pruning(X, y, generation, prune_pct=prune_pct)
                    self.state['counters']['total_new_features'] = X.shape[1] - len(self.initial_features)
                    stagnation_counter = 0
                    self._log(f"  Applied intelligent pruning after stagnation")
                
                # Creative hopeful monster during severe/critical stagnation
                adaptive_status = self.adaptive_controller.get_status_summary()
                hopeful_monster_success = False
                
                if (adaptive_status['stagnation_level'] in ['SEVERE', 'CRITICAL'] and 
                    (random.random() < 0.5 or hopeful_monster_consecutive_fails >= 3)):
                    
                    self._log(f"  Attempting creative hopeful monster...")
                    
                    def monster_callback(ec, sc, force_complete=False):
                        return self.time_budget and (time.time() - start_time) > self.time_budget
                    
                    # Use enhanced creative hopeful monster
                    monster_elites, X_monster, pipe_monster = self._creative_hopeful_monster(
                        X, y, generation, n_features=10, callback=monster_callback
                    )
                    
                    if monster_elites:
                        _, monster_score = self._eval_baseline(X_monster, y, pipe_monster)
                        best_score = self.state['best']['val_score']
                        is_better = (monster_score > best_score) == self.scorer.greater_is_better
                        
                        if is_better:
                            X, self.pipeline, elites = X_monster, pipe_monster, monster_elites
                            hopeful_monster_success = True
                            hopeful_monster_consecutive_fails = 0
                            self._log(f"  Creative HM SUCCESS! Score: {best_score:.5f} → {monster_score:.5f}")
                            self.adaptive_controller.update_strategy_stats("hopeful_monster", True)
                            
                            # Update generation
                            new_generation = generation.copy()
                            for interaction in elites:
                                feat = interaction.get_new_feature_instance()
                                feat.set_generating_interaction(interaction)
                                new_generation.append(feat)
                            generation = new_generation
                            
                            # Update counters and state
                            self.state['counters']['total_new_features'] = X.shape[1] - len(self.initial_features)
                            self.state['counters']['no_feature_gens_count'] = 0
                            new_train_score, new_val_score = self._eval_baseline(X, y, self.pipeline)
                            delta = new_val_score - self.state['best']['val_score'] if self.scorer.greater_is_better else self.state['best']['val_score'] - new_val_score
                            self.state['best'].update(gen_num=N+1, val_score=new_val_score)
                            self.state['counters']['consecutive_no_improvement_iters'] = 0
                            stagnation_counter = 0
                            
                            self._sync_state_components(X, self.pipeline, generation)
                            self._save_current_as_best()
                            self.feature_interactions = self._analyze_feature_interactions(X, y, max_pairs=10000)

                            for elite in elites:
                                self.adaptive_controller.update_operation_stats(elite, success=True, gain=delta/(abs(best_score) + 1e-8))
                        else:
                            self._log(f"  Creative HM: no improvement")
                            hopeful_monster_consecutive_fails += 1
                            self.adaptive_controller.update_strategy_stats("hopeful_monster", False)
                    else:
                        self._log(f"  Creative HM: no features found")
                        hopeful_monster_consecutive_fails += 1
                        self.adaptive_controller.update_strategy_stats("hopeful_monster", False)
                
                # Normal generation if hopeful monster wasn't successful
                if not hopeful_monster_success:
                    # Enhanced parent sampling
                    unary, binary = self._sample_parents_with_memory(generation, n=self.n_parents, tau=tau)
                    valid_unary = [feat for feat in unary if feat.name in X.columns]
                    valid_binary = [(f1, f2) for f1, f2 in binary if f1.name in X.columns and f2.name in X.columns]

                    candidates_pool = []
                    for feat in valid_unary:
                        self.state['seen_feats'].add(feat)
                        candidates_pool.extend([Interaction(feat, op) for op in self.ops[feat.dtype]["unary"]])
                    for feat1, feat2 in valid_binary:
                        self.state['seen_feats'].update({feat1, feat2})
                        op_list = self.ops["num" if feat1.dtype == feat2.dtype == "num" else "cat"]["binary"]
                        candidates_pool.extend([Interaction(feat1, op, feat2) for op in op_list])
                    
                    # Generate GroupBy aggregation candidates (cat × num pairs)
                    if "agg" in self.ops:
                        cat_parents = [f for f in valid_unary if f.dtype == "cat"]
                        num_parents = [f for f in valid_unary if f.dtype == "num"]
                        # Also get num features from binary pairs
                        for f1, f2 in valid_binary:
                            if f1.dtype == "num" and f1 not in num_parents:
                                num_parents.append(f1)
                            if f2.dtype == "num" and f2 not in num_parents:
                                num_parents.append(f2)
                        
                        if cat_parents and num_parents:
                            # Sample a subset to avoid explosion
                            n_agg_pairs = min(len(cat_parents) * len(num_parents), self.n_parents)
                            for _ in range(n_agg_pairs):
                                cat_feat = random.choice(cat_parents)
                                num_feat = random.choice(num_parents)
                                for agg_op in self.ops["agg"]["binary"]:
                                    candidates_pool.append(Interaction(cat_feat, agg_op, num_feat))

                    # Generate Temporal candidates (when time_col is specified)
                    if self.time_col and self.id_col and "temporal" in self.ops:
                        num_parents_temporal = [f for f in valid_unary if f.dtype == "num" 
                                                and f.name != self.time_col and f.name != self.id_col]
                        if num_parents_temporal:
                            # Sample a reasonable number of temporal candidates
                            n_temporal = min(len(num_parents_temporal), self.n_parents // 2)
                            temporal_feats = random.sample(num_parents_temporal, n_temporal)
                            for feat in temporal_feats:
                                for temp_op in self.ops["temporal"]["unary"]:
                                    candidates_pool.append(Interaction(feat, temp_op))

                    # Enhanced child sampling
                    batch = self._sample_children_with_creativity(candidates_pool, self.n_children, tau=tau)

                    # Seed deterministic template candidates once (first normal generation)
                    if self.seed_templates and not getattr(self, '_seeds_injected', False):
                        seeds = self._seed_template_candidates(X, generation)
                        if seeds:
                            seen_names = {i.name for i in batch}
                            batch = [s for s in seeds if s.name not in seen_names] + batch
                        self._seeds_injected = True

                    # Phase 1: Proxy screening (fast FeatureBoost pre-filter)
                    n_before_proxy = len(batch)
                    batch = self._proxy_screen_candidates(batch, X, y)
                    proxy_info = f" [{len(batch)}/{n_before_proxy} after proxy]" if self.use_proxy_evaluation and len(batch) < n_before_proxy else ""
                    pbar.set_description(f"Gen {N+1}: Testing {len(batch)} candidates{proxy_info}")
                
                    remaining_budget = self.max_gen_new_feats - self.state['counters']['total_new_features'] if self.max_gen_new_feats != float('inf') else float('inf')
                    features_per_gen = max(min(20, remaining_budget), 1) if remaining_budget > 0 else 1
                    
                    if remaining_budget <= 0:
                        self._log(f"Gen {N+1}: No budget remaining. Skipping.")
                        continue
                    
                    with tqdm(total=len(batch), desc="Evaluating features", leave=False) as inner_pbar:
                        def update_callback(ec, sc, force_complete=False):
                            inner_pbar.update(max(0, ec - inner_pbar.n if not force_complete else len(batch) - inner_pbar.n))
                            inner_pbar.set_description(f"Evaluating features - Selected: {sc}")
                            return self.time_budget and (time.time() - start_time) > self.time_budget

                        elites, X, self.pipeline = self._select_elites(batch, features_per_gen, X, y, update_callback)
                        self._oof_preds_stale = True  # Mark OOF preds as stale after features change

                    if elites:
                        self.adaptive_controller.update_strategy_stats("normal", True)
                    else:
                        self.adaptive_controller.update_strategy_stats("normal", False)
                
                # Handle generation update (same logic as before)
                if hopeful_monster_success:
                    features_added = len(elites)
                    new_feature_names = [elite.name for elite in elites]
                else:
                    new_feature_names = [elite.name for elite in elites]
                    new_generation = generation.copy()
                    for interaction in elites:
                        feat = interaction.get_new_feature_instance()
                        feat.set_generating_interaction(interaction)
                        new_generation.append(feat)
                    
                    # Update weights if changes made
                    if new_feature_names or elites:
                        weights = self._get_top_k_features(X, y, k=-1, pipeline=self.pipeline)
                        for feat in new_generation:
                            if feat.name in weights.index:
                                feat.update_weight(weights.loc[feat.name, "weighted_importance"])
                            elif hasattr(feat, 'weight') and feat.weight > 0:
                                feat.update_weight(feat.weight * 0.95)
                    
                    generation = new_generation
                    features_added = len(elites)
                    
                    # Safety check for duplicates after adding features
                    X = self._ensure_no_duplicates(X, f"after adding {features_added} features in Gen {N+1}")
                    
                    self.state['counters']['total_new_features'] = X.shape[1] - len(self.initial_features)
                    self.state['counters']['no_feature_gens_count'] = 0 if features_added > 0 else self.state['counters']['no_feature_gens_count'] + 1
                    
                    new_train_score, new_val_score = self._eval_baseline(X, y, self.pipeline)
                    delta = new_val_score - self.state['best']['val_score'] if self.scorer.greater_is_better else self.state['best']['val_score'] - new_val_score
                    
                    # Revert if no improvement
                    if delta <= 0 and features_added > 0:
                        self._log(f"  Gen {N+1} added {features_added} features but no improvement. Reverting to best gen.")
                        if self._revert_to_best():
                            X, self.pipeline, generation = self.X, self.pipeline, self.generation
                            new_val_score, delta = self.state['best']['val_score'], 0
                            self.state['counters']['total_new_features'] = X.shape[1] - len(self.initial_features)
                            elites = []
                
                # Update best state
                if not hopeful_monster_success:
                    if delta > 0:
                        self.state['best'].update(gen_num=N+1, val_score=new_val_score)
                        self.state['counters']['consecutive_no_improvement_iters'] = 0
                        stagnation_counter = 0
                        self.feature_interactions = self._analyze_feature_interactions(X, y, max_pairs=10000)

                        self._sync_state_components(X, self.pipeline, generation)
                        self._save_current_as_best()
                    else:
                        self.state['counters']['consecutive_no_improvement_iters'] += 1
                        stagnation_counter += 1
                
                # Enhanced logging
                if delta <= 0:
                    improvement = "No improvement."
                elif delta < 0.5e-5:
                    improvement = "Score improved by <0.00001."
                else:
                    improvement = f"Score improved by {delta:.5f}."
                adaptive_status = self.adaptive_controller.get_status_summary()
                
                n_groupby = len(getattr(self.pipeline, "groupby_encoders", []))
                n_temporal = len(getattr(self.pipeline, "temporal_encoders", []))
                n_encoded = (self.pipeline.encoder.n_new_feats if hasattr(self.pipeline, 'encoder') else 0) + n_groupby + n_temporal

                gen_log = f"Gen {N+1}: Added {features_added} features, {X.shape[1] + n_encoded} total ({self.state['counters']['total_new_features'] + n_encoded} new)."
                gen_log += f" Train {self.scorer.name}={new_train_score:.5f}, Val {self.scorer.name}={new_val_score:.5f}. {improvement}"
                gen_log += f" Status: {adaptive_status['stagnation_level']}, Strategy success: {adaptive_status['strategy_success']}"

                self._log(gen_log)

                # Log additional scorers if configured
                if self.logging_scorers:
                    logging_scores = self._eval_logging_scorers(X, y, self.pipeline)
                    if logging_scores:
                        self._log(f"  Logging scorers: {self._format_logging_scores(logging_scores)}")
                
                # Log new features
                if features_added > 0 and elites:
                    encoder_feats = set(
                        self.pipeline.encoder.target_enc_cols +
                        self.pipeline.encoder.count_enc_cols +
                        self.pipeline.encoder.freq_enc_cols
                    )
                    new_simple = [elite.name for elite in elites if not elite.require_pipeline]
                    new_pipeline = [elite.name for elite in elites if elite.require_pipeline and elite.name not in encoder_feats]
                    if new_simple: self._log(f"  Simple: {new_simple}")
                    if new_pipeline: self._log(f"  Pipeline: {new_pipeline}")

                    if self.pipeline.encoder.target_enc_cols: self._log(f"  Target encoded: {self.pipeline.encoder.target_enc_cols}")
                    if self.pipeline.encoder.count_enc_cols: self._log(f"  Count encoded: {self.pipeline.encoder.count_enc_cols}")
                    if self.pipeline.encoder.freq_enc_cols: self._log(f"  Freq encoded: {self.pipeline.encoder.freq_enc_cols}")
                
                pbar.set_postfix({f"{self.scorer.name}": f"{new_val_score:.5f}", "features": X.shape[1] + n_encoded,
                                 "new": self.state['counters']['total_new_features'] + n_encoded, "best_gen": self.state['best']['gen_num']})
                pbar.update(1)

                # Save after each trial if enabled
                if self.save_each_trial and self.save_path:
                    self._sync_state_components(X, self.pipeline, generation)
                    self.save(self.save_path)

                # Check termination conditions
                if self.max_gen_new_feats != float('inf') and self.state['counters']['total_new_features'] >= self.max_gen_new_feats:
                    self._log(f"Reached max new features ({self.state['counters']['total_new_features']}/{self.max_gen_new_feats}). Stopping.")
                    break
                
                if self.state['counters']['consecutive_no_improvement_iters'] >= self.early_stopping_iter:
                    self._log(f"Early stopping after {self.state['counters']['consecutive_no_improvement_iters']} generations without improvement.")
                    break
        
        elapsed_time = time.time() - start_time

        # Replay on full data if instance sampling was used
        if X_full is not None:
            # Revert to best state from search (sample-sized)
            if self.state['best']['X'] is not None and not X.equals(self.state['best']['X']):
                self._revert_to_best()
                generation = self.generation

            # Replay discovered features on full dataset
            self._log("Replaying discovered features on full dataset...")
            X = X_full.copy()
            for interaction in getattr(self, 'interactions', []):
                if interaction.name not in X.columns and not interaction.require_pipeline:
                    try:
                        val = interaction.generate(X)
                        X[interaction.name] = val
                    except Exception as e:
                        self._log(f"Replay warning: {interaction.name}: {e}")

            # Drop pruned features
            if hasattr(self, 'pruned_features') and self.pruned_features:
                X = X.drop(columns=[c for c in self.pruned_features if c in X.columns], errors='ignore')

            y = y_full
            self._groups_active = self.groups  # restore full groups after subsampled search
            if hasattr(self.cv, '_groups'):
                self.cv._groups = self.groups
            self._sync_state_components(X, self.pipeline, generation)
            self._save_current_as_best()

            # Re-evaluate on full data for accurate metrics
            train_score, val_score = self._eval_baseline(X, y, self.pipeline)
            self.state['best']['train_score'], self.state['best']['val_score'] = train_score, val_score
            self._log(f"Full-data validation: {self.scorer.name}={val_score:.5f}")
        else:
            # No sampling — standard revert-to-best logic
            if self.state['best']['gen_num'] < self.n_generations and not X.equals(self.state['best']['X']):
                self._log(f"Reverting to best generation ({self.state['best']['gen_num']}).")
                if self._revert_to_best():
                    X, self.pipeline, generation = self.X, self.pipeline, self.generation
            else:
                self._sync_state_components(X, self.pipeline, generation)

        # Meta-validation evaluation (CV bias diagnostic)
        if X_meta is not None and y_meta is not None:
            try:
                # Replay features on meta split
                X_meta_transformed = X_meta.copy()
                for interaction in getattr(self, 'interactions', []):
                    if interaction.name not in X_meta_transformed.columns and not interaction.require_pipeline:
                        try:
                            val = interaction.generate(X_meta_transformed)
                            X_meta_transformed[interaction.name] = val
                        except Exception:
                            pass
                
                # Drop pruned features from meta
                if hasattr(self, 'pruned_features') and self.pruned_features:
                    X_meta_transformed = X_meta_transformed.drop(
                        columns=[c for c in self.pruned_features if c in X_meta_transformed.columns], errors='ignore')
                
                meta_train, meta_val = self._eval_baseline(X_meta_transformed, y_meta, self.pipeline, groups=groups_meta)
                search_val = self.state['best']['val_score']
                
                if self.scorer.greater_is_better:
                    gap = search_val - meta_val
                else:
                    gap = meta_val - search_val
                
                self._log(f"Meta-validation: search_val={search_val:.5f}, meta_val={meta_val:.5f}, gap={gap:.5f}")
                if abs(gap) > 0.02:  # Significant gap suggests overfitting to search folds
                    self._log(f"  Warning: Notable gap between search and meta-validation scores - possible selection overfitting")
            except Exception as e:
                self._log(f"Meta-validation evaluation failed: {e}")

        # Adversarial drift drop: remove generated features that don't transfer to test.
        # Mirrors the regularized-selection path: pruned_features drives removal at
        # transform() time, so interactions/encoders stay internally consistent.
        if self.use_adversarial_validation and self.X_test is not None and getattr(self, 'interactions', None):
            drift_drop = self._adv_final_drift_drop(X, y)
            if drift_drop:
                X = X.drop(columns=[c for c in drift_drop if c in X.columns], errors='ignore')
                if not hasattr(self, 'pruned_features'):
                    self.pruned_features = set()
                self.pruned_features.update(drift_drop)
                self._sync_state_components(X, self.pipeline, generation, preserve_pruned=True)

        # Regularized post-selection (Enhancement 5)
        if self.final_selection and hasattr(self, 'interactions') and self.interactions:
            features_to_drop = self._final_regularized_selection(X, y)
            if features_to_drop:
                X = X.drop(columns=[c for c in features_to_drop if c in X.columns], errors='ignore')
                if not hasattr(self, 'pruned_features'):
                    self.pruned_features = set()
                self.pruned_features.update(features_to_drop)
                self._sync_state_components(X, self.pipeline, generation, preserve_pruned=True)
                # Re-evaluate after pruning
                train_score, val_score = self._eval_baseline(X, y, self.pipeline)
                self.state['best']['val_score'] = val_score
                self.state['best']['train_score'] = train_score
                self._log(f"Post-selection validation: {self.scorer.name}={val_score:.5f}")

        # Calculate and store metrics
        n_init_feats = len(self.initial_features)
        n_groupby = len(getattr(self.pipeline, "groupby_encoders", []))
        n_temporal = len(getattr(self.pipeline, "temporal_encoders", []))
        n_added_feats = len(X.columns) - n_init_feats + self.pipeline.encoder.n_new_feats + n_groupby + n_temporal

        # Use a clean pipeline for baseline evaluation to get true initial performance
        baseline_pipeline = PipelineWrapper(imputer=None, scaler=None, encoder=self._make_cat_encoder())
        self.initial_train_metric, self.initial_val_metric = self._eval_baseline(X[self.initial_features], y, baseline_pipeline)
        self.final_metric = self.state['best']['val_score']
        self.gain = self.final_metric - self.initial_val_metric if self.scorer.greater_is_better else self.initial_val_metric - self.final_metric
        self.pct_gain = self.gain / (abs(self.initial_val_metric) + 1e-8)
        
        self.n_samples, self.n_init_feats, self.n_added_feats = len(X), n_init_feats, n_added_feats
        self.n_final_feats, self.elapsed_time = n_init_feats + n_added_feats, elapsed_time
        
        # Log summary
        self._log(f"\nComplete: {elapsed_time:.2f}s, Best gen: {self.state['best']['gen_num']}, "
                 f"Best {self.scorer.name}: {self.state['best']['val_score']:.5f}, "
                 f"Features: {n_added_feats}/{n_init_feats + n_added_feats}")
        
        # Log strategy performance
        strategy_summary = self.adaptive_controller.get_status_summary()
        self._log(f"Strategy performance: {strategy_summary['strategy_success']}")
        self._log(f"Total restarts: {strategy_summary['total_restarts']}")
        
        # Log new features by type
        encoder_feats_final = set(
            self.pipeline.encoder.target_enc_cols +
            self.pipeline.encoder.count_enc_cols +
            self.pipeline.encoder.freq_enc_cols
        )
        all_generated = set(X.columns) - set(self.initial_features)
        
        gb_names = [gb.output_col for gb in getattr(self.pipeline, "groupby_encoders", [])]
        te_names = [te.output_col for te in getattr(self.pipeline, "temporal_encoders", [])]
        
        new_features = {
            "generated": all_generated - encoder_feats_final,
            "target encoded": self.pipeline.encoder.target_enc_cols,
            "count encoded": self.pipeline.encoder.count_enc_cols,
            "freq encoded": self.pipeline.encoder.freq_enc_cols,
            "groupby": gb_names,
            "temporal": te_names
        }
        for feat_type, features in new_features.items():
            if features: self._log(f"New {feat_type}: {features}")

        # Log final logging scorer results
        if self.logging_scorers:
            self._log("Final logging scorer results:")
            logging_scores = self._eval_logging_scorers(X, y, self.pipeline)
            if logging_scores:
                self._log(f"  {self._format_logging_scores(logging_scores)}")

        # Reset for further calls
        if self.infer_task: self.baseline_model = self.task = self.scorer = None

        # Final sync - use current pipeline (don't create new one which would lose encoder settings)
        # Pipeline will be converted to sklearn Pipeline in fit() when needed
        self._sync_state_components(X, self.pipeline, generation, preserve_pruned=True)
        return self.X, self.pipeline, self.generation, self.interactions

    def _set_defaults(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Set default values for task, model, and scorer."""
        # Task/model/scorer
        self.task = self.task or ("regression" if type_of_target(y) == "continuous" else "classification")
        is_reg = self.task == "regression"
        if self.baseline_model is None:
            if self.device == "cuda":
                from xgboost import XGBRegressor, XGBClassifier
                self.baseline_model = (XGBRegressor if is_reg else XGBClassifier)(
                    device=self.device, n_jobs=self.n_jobs, enable_categorical=True, verbosity=0,
                    random_state=self.random_state)
            else:
                from lightgbm import LGBMRegressor, LGBMClassifier
                self.baseline_model = (LGBMRegressor if is_reg else LGBMClassifier)(
                    n_jobs=self.n_jobs, verbose=-1, random_state=self.random_state)
        if self.scorer is None:
            self.scorer = (PREDEFINED_REG_SCORERS["rmse"] if is_reg else 
                          PREDEFINED_CLS_SCORERS["binary_crossentropy"] 
                          if len(np.unique(y)) == 2 else 
                          PREDEFINED_CLS_SCORERS["categorical_crossentropy"])

        # Pipeline & adaptive controller
        self.pipeline = PipelineWrapper(imputer=None, scaler=None, encoder=self._make_cat_encoder())
        self.adaptive_controller.reset_for_new_run()
        self.adaptive_controller.initialize_operations(self.ops)

        # Search state
        self.state = {
            "best": dict(gen_num=0, val_score=0, train_score=0, X=None,
                        generation=None, pipeline=None, pruned_features=set(),
                        interactions=None),  # Store interactions for proper restoration
            "counters": dict(total_new_features=0, no_feature_gens_count=0,
                           consecutive_no_improvement_iters=0, current_gen=0),
            "seen_feats": set(),
        }

        # Align X and y index if needed
        if not X.index.equals(y.index):
            if len(X) == len(y):
                # If lengths match, we just reset both safely
                X.reset_index(drop=True, inplace=True)
                if hasattr(y, "reset_index"):
                    y.reset_index(drop=True, inplace=True)
            else:
                self._log(f"Warning: X length ({len(X)}) != y length ({len(y)}).")

        # Keep these attributes available even when no generated feature
        # is accepted before replay/transform paths run.
        self.interactions = []
        self.generation = []

        # Reset metrics
        self.initial_metric = self.final_metric = self.gain = self.pct_gain = None
        self.n_samples = self.n_init_feats = self.n_added_feats = self.n_final_feats = self.elapsed_time = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureGenerator':
        """Fit pipeline on input data with generated features."""
        if not getattr(self, 'interactions', None):
            self._log("Warning: No interactions. No features generated.")
            return self
            
        if not getattr(self, 'pipeline', None):
            self._log("Warning: No pipeline. Creating default.")
            self.pipeline = PipelineWrapper(imputer=None, scaler=None, encoder=self._make_cat_encoder()).get_pipeline(X, y)

        # Label encode target (same as search) — category_encoders internally
        # converts non-numeric y to numpy via LabelEncoder without wrapping back in Series
        if y is not None and getattr(self, 'task', None) != "regression":
            unique_vals = np.unique(y)
            if not np.array_equal(unique_vals, np.arange(len(unique_vals))):
                y_encoded, _ = y.factorize(sort=True)
                y = pd.Series(y_encoded, index=y.index, name=y.name)      
        X_transformed = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Generate non-pipeline features
        for interaction in self.interactions:
            if interaction.name not in X_transformed.columns and not interaction.require_pipeline:
                try:
                    val = interaction.generate(X_transformed)
                    if val is not None:
                        X_transformed[interaction.name] = val
                except Exception as e:
                    self._log(f"Error generating {interaction.name}: {str(e)}")

        # Replace infs generated by non-pipeline features
        X_transformed = X_transformed.replace([np.inf, -np.inf], np.nan)

        # Fit pipeline - use X_transformed to build pipeline with correct columns
        if isinstance(self.pipeline, PipelineWrapper):
            self.pipeline = self.pipeline.get_pipeline(X_transformed, y)
        self.pipeline.fit(X_transformed, y)
        return self

    def transform(self, X: pd.DataFrame):
        """Transform data by applying interactions and pipeline."""
        if not getattr(self, 'interactions', None):
            self._log("Warning: No interactions. Returning unchanged.")
            return X
            
        X_transformed = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
            
        # Generate features
        for interaction in self.interactions:
            if interaction.name not in X_transformed.columns and not interaction.require_pipeline:
                try:
                    val = interaction.generate(X_transformed)
                    if val is not None:
                        X_transformed[interaction.name] = val
                except Exception as e:
                    self._log(f"Error generating {interaction.name}: {str(e)}")

        # Replace infs generated by non-pipeline features
        X_transformed = X_transformed.replace([np.inf, -np.inf], np.nan)
        
        # Apply pipeline
        pipeline = getattr(self, 'pipeline', None)
        if pipeline is not None:
            if isinstance(pipeline, PipelineWrapper):
                pipeline = pipeline.get_pipeline(X_transformed)
                self.pipeline = pipeline
            try:
                X_transformed = _restore_missing_pipeline_columns(
                    X_transformed,
                    pipeline,
                    log_fn=self._log,
                )
                X_transformed = pipeline.transform(X_transformed)
            except Exception as e:
                self._log(f"Error applying pipeline: {str(e)}")
        
        # Remove pruned features
        if hasattr(self, 'pruned_features') and self.pruned_features:
            columns_to_drop = [col for col in self.pruned_features if col in X_transformed.columns]
            if columns_to_drop:
                X_transformed = X_transformed.drop(columns=columns_to_drop)
                self._log(f"Removed {len(columns_to_drop)} pruned features")
        
        return X_transformed

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def _build_save_metadata(self, serializer: str) -> Dict[str, Optional[str]]:
        """Capture the runtime that produced the save file for troubleshooting."""
        return {
            "python_version": sys.version.split()[0],
            "numpy_version": getattr(np, "__version__", None),
            "pandas_version": getattr(pd, "__version__", None),
            "serializer": serializer,
            "pickle_protocol": str(_FEATURE_GENERATOR_PICKLE_PROTOCOL),
        }

    def _get_serializable_state(self) -> Dict[str, Any]:
        """Return instance state without forcing class-level code serialization."""
        return dict(self.__dict__)

    @classmethod
    def _from_serialized_state(cls, state: Dict[str, Any]) -> "FeatureGenerator":
        """Rebuild an instance from a serialized state dict."""
        instance = cls.__new__(cls)
        instance.__dict__.update(state)
        instance._ensure_backwards_compat()
        return instance

    @classmethod
    def _warn_for_runtime_mismatch(cls, metadata: Optional[Dict[str, str]]) -> None:
        """Warn when loading across different runtimes that may affect pickle compatibility."""
        if not metadata:
            return

        current_versions = {
            "python_version": sys.version.split()[0],
            "numpy_version": getattr(np, "__version__", None),
            "pandas_version": getattr(pd, "__version__", None),
        }

        mismatches = []
        for key, current_value in current_versions.items():
            saved_value = metadata.get(key)
            if saved_value and current_value and saved_value != current_value:
                mismatches.append(f"{key} saved={saved_value}, current={current_value}")

        if mismatches:
            warnings.warn(
                "Loading a FeatureGenerator saved in a different runtime: "
                + "; ".join(mismatches),
                RuntimeWarning,
            )
    
    def save(self, filepath):
        """Save current state using a versioned, more portable state envelope."""
        import os
        
        # Ensure current state is consistent before saving
        if hasattr(self, 'X') and hasattr(self, 'pipeline') and hasattr(self, 'generation'):
            self._sync_state_components(self.X, self.pipeline, self.generation)
        
        save_dir = os.path.dirname(filepath)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        removed_X = getattr(self, "X", None) if hasattr(self, "X") else None
        removed_y = getattr(self, "y", None) if hasattr(self, "y") else None
        removed_best_X = None
        removed_best_y = None
        had_X = hasattr(self, "X")
        had_y = hasattr(self, "y")
        had_best_X = hasattr(self, "state") and "best" in self.state and "X" in self.state["best"]
        had_best_y = hasattr(self, "state") and "best" in self.state and "y" in self.state["best"]

        if had_X:
            del self.X
        if had_y:
            del self.y
        if had_best_X:
            removed_best_X = self.state["best"]["X"]
            del self.state["best"]["X"]
        if had_best_y:
            removed_best_y = self.state["best"]["y"]
            del self.state["best"]["y"]

        try:
            state = self._get_serializable_state()
            serializer = "pickle"

            try:
                state_bytes = pickle.dumps(state, protocol=_FEATURE_GENERATOR_PICKLE_PROTOCOL)
            except Exception:
                # Fall back to cloudpickle when users pass custom objects that stdlib pickle cannot handle.
                try:
                    import cloudpickle
                except ImportError:
                    raise ImportError("cloudpickle required for this generator state. Install with: pip install cloudpickle")
                serializer = "cloudpickle"
                state_bytes = cloudpickle.dumps(state, protocol=_FEATURE_GENERATOR_PICKLE_PROTOCOL)
        finally:
            if had_X:
                self.X = removed_X
            if had_y:
                self.y = removed_y
            if had_best_X:
                self.state["best"]["X"] = removed_best_X
            if had_best_y:
                self.state["best"]["y"] = removed_best_y

        payload = {
            "format": _FEATURE_GENERATOR_SAVE_FORMAT,
            "format_version": _FEATURE_GENERATOR_SAVE_VERSION,
            "class_name": self.__class__.__name__,
            "metadata": self._build_save_metadata(serializer),
            "state": state_bytes,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(payload, f, protocol=_FEATURE_GENERATOR_PICKLE_PROTOCOL)
        self._log(f"State saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load state from file, supporting both new portable saves and legacy pickles."""
        import os

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found.")

        try:
            with open(filepath, 'rb') as f:
                loaded_obj = _compatible_pickle_load(f)

            if isinstance(loaded_obj, dict) and loaded_obj.get("format") == _FEATURE_GENERATOR_SAVE_FORMAT:
                metadata = loaded_obj.get("metadata", {})
                state_payload = loaded_obj.get("state")

                if not isinstance(state_payload, (bytes, bytearray)):
                    raise ValueError("Invalid serialized state payload.")

                state = _compatible_pickle_loads(state_payload)
                instance = cls._from_serialized_state(state)
                cls._warn_for_runtime_mismatch(metadata)
                return instance

            if isinstance(loaded_obj, cls):
                loaded_obj._ensure_backwards_compat()
                return loaded_obj

            if isinstance(loaded_obj, dict):
                return cls._from_serialized_state(loaded_obj)

            raise TypeError(f"Unsupported save format: {type(loaded_obj).__name__}")
        except Exception as e:
            raise ValueError(f"Failed to load: {str(e)}")

    def _ensure_backwards_compat(self):
        """Ensure backwards compatibility with older saved pkl files."""
        # Ensure pruned_features exists
        if not hasattr(self, 'pruned_features'):
            self.pruned_features = set()

        # Ensure save_each_trial exists
        if not hasattr(self, 'save_each_trial'):
            self.save_each_trial = False

        # Ensure logging_scorers exists
        if not hasattr(self, 'logging_scorers'):
            self.logging_scorers = []

        # Ensure interactions exist and are populated
        if not hasattr(self, 'interactions') or not self.interactions:
            if hasattr(self, 'generation') and self.generation:
                self.interactions = [feat.generating_interaction for feat in self.generation
                                   if hasattr(feat, 'generating_interaction') and feat.generating_interaction]
            else:
                self.interactions = []

        # Ensure search_sample_size exists
        if not hasattr(self, 'search_sample_size'):
            self.search_sample_size = None

        # Enhancement 1: Feature cache
        if not hasattr(self, '_feature_cache'):
            self._feature_cache = FeatureCache(max_size_mb=2000)

        # Enhancement 2: Proxy evaluation
        if not hasattr(self, 'use_proxy_evaluation'):
            self.use_proxy_evaluation = True
        if not hasattr(self, 'proxy_top_pct'):
            self.proxy_top_pct = 0.15
        if not hasattr(self, '_lgb_available'):
            self._lgb_available = None

        # Enhancement 4: CV bias fix
        if not hasattr(self, 'meta_validation_frac'):
            self.meta_validation_frac = 0.15
        if not hasattr(self, 'rotate_cv_folds'):
            self.rotate_cv_folds = True
        if not hasattr(self, 'fold_rotation_period'):
            self.fold_rotation_period = 5

        # Enhancement 5: Regularized post-selection
        if not hasattr(self, 'final_selection'):
            self.final_selection = True

        # Enhancement 6: Temporal operators
        if not hasattr(self, 'time_col'):
            self.time_col = None
        if not hasattr(self, 'id_col'):
            self.id_col = None

        # Ensure state dict has interactions in best (for future reverts)
        if hasattr(self, 'state') and 'best' in self.state:
            if 'X' not in self.state['best']:
                self.state['best']['X'] = None
            if 'interactions' not in self.state['best']:
                self.state['best']['interactions'] = deepcopy(self.interactions)
            if 'pruned_features' not in self.state['best']:
                self.state['best']['pruned_features'] = set()

    def generate(self, X: pd.DataFrame, y: pd.Series):
        """Main entry point for feature generation."""
        return self.search(X, y)
