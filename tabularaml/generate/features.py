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

from tabularaml.eval.cv import (cross_val_score, cross_val_fold_scores, make_cv_splitter,
                                sanitize_model_features, FoldScores)
from tabularaml.eval.scorers import PREDEFINED_REG_SCORERS, PREDEFINED_CLS_SCORERS, PREDEFINED_SCORERS, Scorer
from tabularaml.eval.splitters import RotatedGroupKFold, normalize_rotatable_splitter
from tabularaml.generate.ops import OPS, ALL_OPS_LAMBDAS, AGG_OPS, TEMPORAL_OPS, GLOBAL_OPS, build_temporal_ops
from tabularaml.generate.expanders import BaselineFeatureExpander
from tabularaml.inspect.importance import FeatureImportanceAnalyzer
from tabularaml.preprocessing.encoders import (CategoricalEncoder, GroupByEncoder, TemporalEncoder,
                                               GlobalTransformEncoder)
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
        self.is_global = op in GLOBAL_OPS

        if self.is_temporal:
            # Temporal: feature_1 is the numeric column, feature_2 unused (unary-style)
            self.type = "unary"
            self.dtype = "num"
            self.depth = feature_1.depth + 1
            self.weight = feature_1.weight
            self.require_pipeline = True  # Must go through pipeline
            self.name = f"{op}_{feature_1.name}"
        elif self.is_global:
            # Global transform: rank/bin/winsor maps fitted on the train fold
            self.type = "unary"
            self.dtype = "num"
            self.depth = feature_1.depth + 1
            self.weight = feature_1.weight
            self.require_pipeline = True  # Fit-state must come from train folds only
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


@dataclass
class FoldEvalState:
    """Cached per-fold CV scores of the current baseline feature set.

    Valid only while the CV splitter state (cv_epoch), row count and column set
    are unchanged; any mismatch forces a recompute so paired per-fold deltas are
    never taken against a stale baseline vector.
    """
    fold_scores: Optional[np.ndarray] = None
    cv_epoch: int = -1
    n_rows: int = -1
    cols_hash: int = 0
    per_era: Optional[dict] = None

    @staticmethod
    def hash_cols(X: pd.DataFrame) -> int:
        return hash(tuple(sorted(map(str, X.columns))))

    def matches(self, cv_epoch: int, X: pd.DataFrame) -> bool:
        return (self.fold_scores is not None
                and self.cv_epoch == cv_epoch
                and self.n_rows == len(X)
                and self.cols_hash == self.hash_cols(X))


# Initial priority overrides for specific operations (see initialize_operations)
_OP_PRIORS = {"concat": 0.8}


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
                        # Start with higher scores for rarely used operations;
                        # explicit priors boost ops that unlock follow-up moves
                        # (concat keys enable multi-key group-bys next round)
                        initial_score = _OP_PRIORS.get(op, 0.7 if self.op_usage[op] < 5 else 0.5)
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
        elif getattr(interaction, 'is_global', False):
            dtype, op_type = "global", "unary"
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
            
            # Complexity penalty: linear term acts as a tiebreaker at every depth
            # (prefer simpler, more robust features at equal promise), quadratic
            # term kicks in past depth 3
            complexity_penalty = 0.1 * interaction.depth + (
                (interaction.depth / 5.0) ** 2 if interaction.depth > 3 else 0)
            
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
                 proxy_mode: Literal["batched", "featureboost", "none"] = "batched",
                 proxy_ram_budget_mb: int = 512,
                 proxy_halving: bool = False,
                 proxy_top_pct: float = 0.15,
                 meta_validation_frac: float = 0.15,
                 rotate_cv_folds: bool = True,
                 fold_rotation_period: int = 5,
                 final_selection: bool = True,
                 time_col: Optional[str] = None,
                 id_col: Optional[str] = None,
                 temporal_windows: Optional[list] = None,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 cv_n_jobs: Union[int, str] = "auto",
                 acceptance: Literal["statistical", "mean"] = "statistical",
                 acceptance_folds_frac: float = 0.7,
                 confirmation_seeds: int = 1,
                 null_importance_selection: bool = True,
                 null_importance_n_perm: int = 4,
                 null_importance_pct: float = 75.0,
                 expand_datetime: bool = True,
                 expand_row_stats: bool = True,
                 era_col: Optional[str] = None,
                 era_acceptance_frac: float = 0.55,
                 adversarial_auc_warn: float = 0.75,
                 adversarial_drop: bool = False):

        # Capture provided parameters
        provided_params = locals().copy()
        provided_params.pop('self')
        
        self.mode = mode
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.cv_n_jobs = cv_n_jobs
        self._cv_n_jobs_resolved = 1

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

        # Competition-grade knobs that presets may override: must be assigned
        # BEFORE the mode override so presets can change them
        self.acceptance = acceptance
        self.acceptance_folds_frac = acceptance_folds_frac
        self.confirmation_seeds = confirmation_seeds
        self.null_importance_selection = null_importance_selection
        self.null_importance_n_perm = null_importance_n_perm
        self.null_importance_pct = null_importance_pct
        self.expand_datetime = expand_datetime
        self.expand_row_stats = expand_row_stats
        self.use_proxy_evaluation = use_proxy_evaluation
        self.proxy_mode = proxy_mode if use_proxy_evaluation else "none"
        self.proxy_ram_budget_mb = proxy_ram_budget_mb
        self.proxy_halving = proxy_halving
        self.proxy_top_pct = proxy_top_pct

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
        self.pipeline = PipelineWrapper(imputer=None, scaler=None, encoder=CategoricalEncoder())
        
        # Feature value cache
        self._feature_cache = FeatureCache(max_size_mb=cache_size_mb)
        
        # Proxy evaluation settings (preset-overridable values assigned above)
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

        # Paired-fold evaluation state (statistical acceptance; knobs assigned above)
        self._cv_epoch = 0
        self._best_fold_state = FoldEvalState()

        # Base-table expansion runtime state
        self.base_expander = None
        self._priority_candidates = []

        # Era mode (CrunchDAO/Numerai-style grouped time-series data)
        self.era_col = era_col
        self.era_acceptance_frac = era_acceptance_frac

        # Adversarial validation (active only when X_test is passed to search)
        self.adversarial_auc_warn = adversarial_auc_warn
        self.adversarial_drop = adversarial_drop
        self.adversarial_report = None

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
            if groups_arr is not None and getattr(self, 'era_col', None):
                # Era mode: sample WHOLE eras (never split one) until the row
                # budget is met, preserving within-era structure for per-era
                # metrics and grouped CV.
                rng = np.random.RandomState(self.random_state)
                eras = rng.permutation(pd.unique(groups_arr))
                take, count = [], 0
                for e in eras:
                    idx_e = np.where(groups_arr == e)[0]
                    take.append(idx_e)
                    count += len(idx_e)
                    if count >= sample_size:
                        break
                indices = np.sort(np.concatenate(take))
                n_splits_hint = getattr(self.cv, 'n_splits', self.cv if isinstance(self.cv, int) else 5)
                if len(take) < 2 * n_splits_hint:
                    self._log(f"Warning: era subsample keeps only {len(take)} eras for "
                              f"{n_splits_hint}-fold grouped CV — consider a larger search_sample_size")
            elif groups_arr is not None:
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

    def _eval_baseline(self, X: pd.DataFrame, y: pd.Series, pipeline=None, groups=None,
                       update_fold_cache: bool = False) -> tuple[float, float]:
        """Evaluate baseline model performance."""
        pipeline = pipeline.get_pipeline(X, y) if pipeline is not None else pipeline
        eval_groups = self._groups_active if groups is None else groups
        cv_dict = cross_val_score(self.baseline_model, X, y, self.scorer, cv=self.cv,
                                 return_dict=True, pipeline=pipeline, model_fit_kwargs=self.model_fit_kwargs,
                                 groups=eval_groups, n_jobs_folds=self._cv_n_jobs_resolved)
        if update_fold_cache:
            self._store_baseline_fold_scores(
                X, FoldScores(mean_val=cv_dict["mean_val_score"],
                              fold_scores=np.asarray(cv_dict["fold_val_scores"], dtype=float),
                              per_group=cv_dict.get("val_group_scores")))
        return cv_dict["mean_train_score"], cv_dict["mean_val_score"]

    def _eval_cv_light(self, X: pd.DataFrame, y: pd.Series, pipeline=None, groups=None) -> FoldScores:
        """Light CV evaluation for the candidate hot loop: per-fold val scores only
        (no train-side predictions, no fold-model retention)."""
        pipeline = pipeline.get_pipeline(X, y) if pipeline is not None else pipeline
        eval_groups = self._groups_active if groups is None else groups
        return cross_val_fold_scores(self.baseline_model, X, y, self.scorer, cv=self.cv,
                                     pipeline=pipeline, model_fit_kwargs=self.model_fit_kwargs,
                                     groups=eval_groups, n_jobs_folds=self._cv_n_jobs_resolved)

    def _store_baseline_fold_scores(self, X: pd.DataFrame, res: FoldScores) -> None:
        """Cache the baseline per-fold vector for paired candidate comparisons."""
        self._best_fold_state = FoldEvalState(
            fold_scores=np.asarray(res.fold_scores, dtype=float),
            cv_epoch=self._cv_epoch,
            n_rows=len(X),
            cols_hash=FoldEvalState.hash_cols(X),
            per_era=res.per_group)

    def _get_baseline_fold_scores(self, X: pd.DataFrame, y: pd.Series,
                                  pipeline=None) -> FoldScores:
        """Baseline per-fold scores for X, from cache when still valid.

        The cache key (cv_epoch, n_rows, cols_hash) guarantees paired deltas are
        only ever computed against a vector produced under the same splitter
        state and feature set.
        """
        if self._best_fold_state.matches(self._cv_epoch, X):
            return FoldScores(mean_val=float(np.mean(self._best_fold_state.fold_scores)),
                              fold_scores=self._best_fold_state.fold_scores,
                              per_group=self._best_fold_state.per_era)
        res = self._eval_cv_light(X, y, pipeline if pipeline is not None else self.pipeline)
        self._store_baseline_fold_scores(X, res)
        return res

    def _bump_cv_epoch(self, reason: str = "") -> None:
        """Invalidate cached fold vectors after any change to splitter state or data."""
        self._cv_epoch += 1

    def _acceptance_gate(self, gain: float, fold_deltas: Optional[np.ndarray],
                         era_deltas: Optional[np.ndarray] = None) -> bool:
        """Decide candidate acceptance.

        "statistical" (default): the mean relative gain must clear the adaptive
        threshold AND the paired per-fold deltas must be positive in at least
        k_req of K folds (sign-test gate), so a candidate cannot be accepted on
        the strength of a single lucky fold. k_req relaxes by one step (floored
        at a simple majority) when stagnation reaches SEVERE. Falls back to the
        mean-only rule when paired vectors are unavailable (mismatched splitter
        state) or K < 3 (no statistical power).

        In era mode, era_deltas (paired per-era deltas) additionally require the
        candidate to help in at least era_acceptance_frac of shared eras — a
        feature that wins on a few eras but loses broadly is rejected.
        """
        if gain < self.adaptive_controller.get_adaptive_min_gain():
            return False
        if getattr(self, "acceptance", "statistical") != "statistical":
            return True
        if era_deltas is not None and len(era_deltas) >= 4:
            frac_pos = float(np.mean(era_deltas > 0))
            if frac_pos < getattr(self, "era_acceptance_frac", 0.55):
                return False
        if fold_deltas is None or len(fold_deltas) < 3:
            return True
        K = len(fold_deltas)
        frac = getattr(self, "acceptance_folds_frac", 0.7)
        k_req = min(K, max(2, int(np.ceil(frac * K))))
        if self.adaptive_controller.state.stagnation_level.value >= StagnationLevel.SEVERE.value:
            k_req = max(K // 2 + 1, k_req - 1)
        return int(np.sum(fold_deltas > 0)) >= k_req

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

                score_kwargs = {}
                if getattr(self.scorer, 'needs_groups', False) and self._groups_active is not None:
                    score_kwargs['groups'] = np.asarray(self._groups_active)[val_idx]
                base_score = self.scorer.score(y.iloc[val_idx], base_preds, **score_kwargs)
                new_score = self.scorer.score(y.iloc[val_idx], new_preds, **score_kwargs)
                
                if self.scorer.greater_is_better:
                    scores.append(new_score - base_score)
                else:
                    scores.append(base_score - new_score)  # Lower is better, so improvement = base - new
            except Exception:
                scores.append(-np.inf)
        
        return np.mean(scores) if scores else -np.inf

    def _proxy_screen_candidates(self, batch, X, y):
        """Pre-filter candidates with the configured proxy strategy.

        proxy_mode="batched" (default): one LightGBM over ALL candidate columns
        at once with init_score = base-model OOF margins, ranked by gain
        importance. Falls back to the per-candidate FeatureBoost screen on any
        failure. proxy_mode="featureboost": the original per-candidate screen.
        proxy_mode="none": pass everything through.

        Returns kept scorable candidates plus pipeline-required candidates,
        the latter capped so they cannot flood elite selection (they carry no
        proxy score).
        """
        proxy_mode = getattr(self, 'proxy_mode', 'batched' if self.use_proxy_evaluation else 'none')
        if proxy_mode == "none" or not self.use_proxy_evaluation or not self._check_lgb_available():
            return batch

        pipeline_candidates = [i for i in batch if i.require_pipeline]
        scorable_candidates = [i for i in batch if not i.require_pipeline]

        if len(scorable_candidates) <= 5:
            return batch  # Not enough to filter

        try:
            cv = self._get_cv_splitter()

            # Train base model and get OOF predictions (once per generation)
            if not hasattr(self, '_current_oof_preds') or self._oof_preds_stale:
                self._current_oof_preds = self._train_base_model_and_get_residuals(X, y, cv)
                self._oof_preds_stale = False

            top_candidates = None
            n_final = max(3, int(len(scorable_candidates) * self.proxy_top_pct))
            if proxy_mode == "batched":
                # Two-stage screen: one joint residual-boosting model coarsely
                # filters the batch (cheap), then per-candidate FeatureBoost
                # ranks the survivors (gain importance in a joint model splits
                # credit among correlated candidates, so the final ranking
                # must measure individual marginal value).
                try:
                    coarse = self._batched_proxy_rank(scorable_candidates, X, y, cv)
                    if coarse is not None:
                        if len(coarse) > n_final:
                            top_candidates = self._featureboost_screen(
                                coarse, X, y, cv, n_keep=n_final)
                        else:
                            top_candidates = coarse
                except (Exception, MemoryError) as e:
                    self._log(f"  Batched proxy failed ({e}), falling back to FeatureBoost")
            if top_candidates is None:
                top_candidates = self._featureboost_screen(scorable_candidates, X, y, cv,
                                                           n_keep=n_final)
            if top_candidates is None:
                return batch

            # Cap pipeline-required candidates (they skipped proxy and carry no
            # score): keep the best-ranked ones up to half the scored survivors.
            if pipeline_candidates:
                cap = max(10, len(top_candidates) // 2)
                if len(pipeline_candidates) > cap:
                    ranked_pipe = self.adaptive_controller.rank_candidates_with_memory(
                        pipeline_candidates, X, y)
                    self._log(f"  Pipeline candidates capped: {cap}/{len(pipeline_candidates)} kept")
                    pipeline_candidates = ranked_pipe[:cap]

            return top_candidates + pipeline_candidates

        except Exception as e:
            self._log(f"  Proxy screening failed ({e}), falling back to full evaluation")
            return batch

    def _featureboost_screen(self, scorable_candidates, X, y, cv, n_keep=None):
        """Per-candidate FeatureBoost screen (refinement and fallback path)."""
        fb_scores = {}
        for interaction in scorable_candidates:
            try:
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
            return None

        if n_keep is None:
            n_keep = max(3, int(len(fb_scores) * self.proxy_top_pct))
        sorted_candidates = sorted(fb_scores.values(), key=lambda x: x[1], reverse=True)
        return [interaction for interaction, _ in sorted_candidates[:n_keep]]

    def _materialize_candidate_matrix(self, scorable_candidates, X):
        """Build a float32/category frame of candidate columns via the feature cache.

        Returns (C, kept_interactions); candidates that fail to generate, are
        mostly non-finite, or duplicate an earlier name are dropped.
        """
        cols, kept = {}, []
        for interaction in scorable_candidates:
            parent_names = [interaction.feature_1.name]
            if interaction.feature_2 is not None:
                parent_names.append(interaction.feature_2.name)
            if not all(p in X.columns for p in parent_names):
                continue
            if interaction.name in cols:
                continue
            try:
                _, vals = self._feature_cache.get_or_compute(
                    parent_names, interaction.op,
                    lambda inter=interaction: (inter.name, inter.generate(X))
                )
                vals = pd.Series(np.asarray(vals).ravel() if not isinstance(vals, pd.Series) else vals.values,
                                 index=X.index)
                if vals.dtype == object or isinstance(vals.dtype, pd.CategoricalDtype):
                    vals = pd.Series(pd.Categorical(vals.astype(str)), index=X.index)
                else:
                    vals = pd.to_numeric(vals, errors="coerce").astype(np.float32)
                    finite_frac = np.isfinite(vals.values).mean()
                    if finite_frac < 0.5:
                        continue
                    vals = vals.replace([np.inf, -np.inf], np.nan)
                cols[interaction.name] = vals
                kept.append(interaction)
            except Exception:
                continue
        if not kept:
            return None, []
        return pd.DataFrame(cols, index=X.index), kept

    def _batched_proxy_rank(self, scorable_candidates, X, y, cv):
        """Rank all candidates with ONE residual-boosting LightGBM per fold.

        Trains on the full candidate matrix with init_score = base-model OOF
        margins, so gain importance measures each candidate's contribution to
        explaining what the current feature set cannot. Orders of magnitude
        fewer model fits than per-candidate FeatureBoost.
        """
        import lightgbm as lgb

        C, kept = self._materialize_candidate_matrix(scorable_candidates, X)
        if C is None or len(kept) < 5:
            return None

        oof = self._current_oof_preds
        groups = self._groups_active

        # RAM guard: row-subsample to fit the configured budget
        ram_budget = getattr(self, 'proxy_ram_budget_mb', 512)
        est_mb = len(C) * C.shape[1] * 4 / 2**20
        row_idx = None
        if est_mb > ram_budget:
            n_rows = max(5000, int(len(C) * ram_budget / est_mb))
            if n_rows < len(C):
                rng = np.random.RandomState(self.random_state)
                row_idx = np.sort(rng.choice(len(C), n_rows, replace=False))

        def _gain_for(C_sub, y_sub, oof_sub, groups_sub, max_folds=2):
            objective = self._get_lgb_objective()
            params = {"objective": objective, "num_leaves": 31, "verbosity": -1,
                      "n_jobs": self.n_jobs, "learning_rate": 0.1,
                      "feature_fraction": 0.8, "random_state": self.random_state}
            if objective == "multiclass":
                params["num_class"] = len(np.unique(y))
            gain = np.zeros(C_sub.shape[1])
            cv_local = self._get_cv_splitter()
            for fold_i, (tr, va) in enumerate(cv_local.split(C_sub, y_sub, groups=groups_sub)):
                if fold_i >= max_folds:
                    break
                dtrain = lgb.Dataset(C_sub.iloc[tr], y_sub.iloc[tr], init_score=oof_sub[tr])
                dval = lgb.Dataset(C_sub.iloc[va], y_sub.iloc[va], init_score=oof_sub[va],
                                   reference=dtrain)
                booster = lgb.train(params, dtrain, num_boost_round=300,
                                    valid_sets=[dval],
                                    callbacks=[lgb.early_stopping(30, verbose=False),
                                               lgb.log_evaluation(period=0)])
                gain += booster.feature_importance("gain")
            return gain

        def _subset(idx):
            C_s = C.iloc[idx]
            y_s = y.iloc[idx]
            oof_s = oof[idx]
            g_s = np.asarray(groups)[idx] if groups is not None else None
            return C_s, y_s, oof_s, g_s

        if getattr(self, 'proxy_halving', False) and len(C) > 8000:
            # Stage A: cheap screen on a small sample keeps the top half
            rng = np.random.RandomState(self.random_state)
            idx_a = np.sort(rng.choice(len(C), 8000, replace=False))
            gain_a = _gain_for(*_subset(idx_a), max_folds=1)
            order_a = np.argsort(-gain_a)
            survivors = order_a[:max(5, len(kept) // 2)]
            C = C.iloc[:, survivors]
            kept = [kept[i] for i in survivors]

        if row_idx is not None:
            gain = _gain_for(*_subset(row_idx))
        else:
            g_all = np.asarray(groups) if groups is not None else None
            gain = _gain_for(C, y, oof, g_all)

        # Coarse keep: ~3x the final quota; the FeatureBoost refinement stage
        # makes the final per-candidate call among these survivors.
        n_final = max(3, int(len(kept) * self.proxy_top_pct))
        n_coarse = min(len(kept), max(15, 3 * n_final))
        order = np.argsort(-gain)
        top = [kept[i] for i in order[:n_coarse] if gain[i] > 0]
        if not top:  # all-zero gain: nothing distinguishable, keep best-ranked few
            top = [kept[i] for i in order[:3]]
        return top

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

    def _make_selection_tree_model(self, random_state=None):
        """Tree model used by post-search selection (L1 companion and null importance)."""
        rs = self.random_state if random_state is None else random_state
        if self.device == "cuda":
            from xgboost import XGBRegressor, XGBClassifier
            cls = XGBRegressor if self.task == "regression" else XGBClassifier
            return cls(n_estimators=300, max_depth=6, verbosity=0, n_jobs=self.n_jobs,
                       device="cuda", enable_categorical=True, random_state=rs)
        from lightgbm import LGBMRegressor, LGBMClassifier
        cls = LGBMRegressor if self.task == "regression" else LGBMClassifier
        return cls(n_estimators=300, max_depth=6, n_jobs=self.n_jobs, verbose=-1,
                   random_state=rs)

    def _null_importance_selection(self, X, y, era_groups=None):
        """Drop generated features whose importance does not beat a target-permutation null.

        Olivier-style null importances: fit the selection tree on the real
        target for actual gains, then n_perm times on a permuted target; a
        generated feature survives only if its actual importance exceeds the
        configured percentile of its own null distribution. Original (and
        expander) features are never dropped here. When era_groups is given,
        the target is permuted within eras so era-level structure is preserved
        in the null.
        """
        if not getattr(self, 'null_importance_selection', True):
            return []
        generated = [c for c in X.columns if c not in self.initial_features]
        if len(generated) < 10:
            return []
        n_perm = int(getattr(self, 'null_importance_n_perm', 4))
        pct = float(getattr(self, 'null_importance_pct', 75.0))
        self._log(f"Null-importance selection: {len(generated)} generated features, {n_perm} permutations...")
        if era_groups is not None and len(era_groups) != len(X):
            era_groups = None  # row mismatch (e.g. after replay) — plain permutation
        try:
            X_fit = sanitize_model_features(X)
            y_fit = y
            groups_fit = np.asarray(era_groups) if era_groups is not None else None
            if len(X_fit) > 200_000:
                rng = np.random.RandomState(self.random_state)
                idx = np.sort(rng.choice(len(X_fit), 100_000, replace=False))
                X_fit, y_fit = X_fit.iloc[idx], y.iloc[idx]
                if groups_fit is not None:
                    groups_fit = groups_fit[idx]

            def _importances(target, seed):
                model = self._make_selection_tree_model(random_state=seed)
                model.fit(X_fit, target)
                names = getattr(model, 'feature_names_in_', X_fit.columns)
                return dict(zip(names, model.feature_importances_))

            actual = _importances(y_fit, self.random_state)

            nulls = {f: [] for f in generated}
            y_arr = np.asarray(y_fit)
            for s in range(n_perm):
                rng = np.random.RandomState(self.random_state + 1000 + s)
                if groups_fit is not None:
                    y_perm = y_arr.copy()
                    for g in np.unique(groups_fit):
                        mask = groups_fit == g
                        y_perm[mask] = rng.permutation(y_perm[mask])
                else:
                    y_perm = rng.permutation(y_arr)
                null_imp = _importances(pd.Series(y_perm, index=X_fit.index), self.random_state + 1000 + s)
                for f in generated:
                    nulls[f].append(null_imp.get(f, 0.0))

            drop = []
            for f in generated:
                act = actual.get(f, 0.0)
                if act <= 0 or act <= np.percentile(nulls[f], pct):
                    drop.append(f)
            if drop:
                self._log(f"  Null importance: dropping {len(drop)} features not beating the null")
                self._log(f"  Dropped: {drop}")
            else:
                self._log("  Null importance: all generated features beat the null")
            return drop
        except Exception as e:
            self._log(f"  Null-importance selection failed: {e}")
            return []

    def _adversarial_validation_report(self, X_final: pd.DataFrame, X_test: pd.DataFrame):
        """Train-vs-test discriminability of the final feature set.

        Applies the base expander and non-pipeline interactions to X_test,
        fits a LightGBM to distinguish train rows from test rows, and reports
        3-fold AUC plus the features driving the shift. High AUC means the
        feature set encodes train-specific structure that will not transfer.
        Drops shift-driving generated features only when adversarial_drop=True.
        """
        from lightgbm import LGBMClassifier
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score

        X_t = X_test.copy()
        if getattr(self, 'base_expander', None) is not None:
            X_t = self.base_expander.transform(X_t)
        for interaction in getattr(self, 'interactions', []):
            if interaction.name not in X_t.columns and not interaction.require_pipeline:
                try:
                    X_t[interaction.name] = interaction.generate(X_t)
                except Exception:
                    pass

        common = [c for c in X_final.columns if c in X_t.columns]
        if len(common) < 2:
            self._log("Adversarial validation skipped: no common columns")
            return None

        cap = 50_000
        rng = np.random.RandomState(self.random_state)
        A = X_final[common]
        B = X_t[common]
        if len(A) > cap:
            A = A.iloc[np.sort(rng.choice(len(A), cap, replace=False))]
        if len(B) > cap:
            B = B.iloc[np.sort(rng.choice(len(B), cap, replace=False))]

        XX = sanitize_model_features(pd.concat([A, B], axis=0, ignore_index=True))
        yy = np.r_[np.zeros(len(A)), np.ones(len(B))]

        clf = LGBMClassifier(n_estimators=200, num_leaves=31, n_jobs=self.n_jobs,
                             verbose=-1, random_state=self.random_state,
                             importance_type="gain")
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        aucs = []
        for tr, va in skf.split(XX, yy):
            m = deepcopy(clf)
            m.fit(XX.iloc[tr], yy[tr])
            aucs.append(roc_auc_score(yy[va], m.predict_proba(XX.iloc[va])[:, 1]))
        auc = float(np.mean(aucs))

        clf.fit(XX, yy)
        imp = pd.Series(clf.feature_importances_, index=common).astype(float)
        share = imp / max(imp.sum(), 1e-9)
        top = share.sort_values(ascending=False).head(10)
        generated = set(X_final.columns) - set(self.initial_features)
        self.adversarial_report = {
            "auc": auc,
            "top_shift_features": [(name, float(s), name in generated) for name, s in top.items()],
        }

        warn_thr = getattr(self, 'adversarial_auc_warn', 0.75)
        self._log(f"Adversarial validation: train-vs-test AUC={auc:.3f}"
                  + (f" (> {warn_thr}: feature set encodes train-specific structure!)" if auc > warn_thr else ""))
        if auc > warn_thr:
            for name, s, is_gen in self.adversarial_report["top_shift_features"][:5]:
                self._log(f"  shift driver: {name} ({s:.1%}{', generated' if is_gen else ''})")

        drop = []
        if getattr(self, 'adversarial_drop', False) and auc > warn_thr:
            drop = [name for name, s, is_gen in self.adversarial_report["top_shift_features"]
                    if is_gen and s > 0.10]
            if drop:
                self._log(f"  Adversarial drop: removing {drop}")
        return drop

    def _era_feature_corr_report(self):
        """Future work: per-era correlation of generated features with the span
        of existing features, to flag candidates for neutralization
        (CrunchDAO/Numerai feature-exposure analysis). Currently a no-op."""
        return None

    def _make_alternate_splitter(self, s: int, y):
        """Fresh CV splitter with an alternate seed for confirmation runs.

        Deterministic per s, so calling twice yields identical folds and the
        new-vs-best fold vectors pair correctly. Returns None when the user's
        splitter cannot be reseeded safely.
        """
        hint = getattr(self, '_cv_int_hint', None)
        seed = self.random_state + 10_000 + 7919 * s
        if hint:
            if self._groups_active is not None:
                return RotatedGroupKFold(hint, rotation=seed)
            return make_cv_splitter(hint, y, shuffle=True, random_state=seed,
                                    groups=self._groups_active)
        rotated = getattr(self.cv, 'rotated', None)
        if callable(rotated):
            return rotated(1_000_000 + s)
        return None

    def _confirm_generation(self, X_new: pd.DataFrame, y: pd.Series, pipe_new) -> bool:
        """Re-test an improving feature set against the previous best under
        alternate CV seeds before committing it.

        Both states are evaluated under the SAME alternate splitter (paired
        fold deltas); the improvement is confirmed iff the pooled mean delta
        over all alternate folds is positive. Costs 2*confirmation_seeds light
        CVs, paid only on improving generations.
        """
        seeds = int(getattr(self, 'confirmation_seeds', 0) or 0)
        if seeds <= 0:
            return True
        best_X = self.state['best'].get('X')
        best_pipe = self.state['best'].get('pipeline')
        if best_X is None or len(best_X) != len(X_new):
            return True
        sign = 1.0 if self.scorer.greater_is_better else -1.0
        deltas = []
        for s in range(seeds):
            alt_a = self._make_alternate_splitter(s, y)
            alt_b = self._make_alternate_splitter(s, y)  # fresh twin, identical folds
            if alt_a is None:
                self._log("  Confirmation skipped: splitter cannot be reseeded")
                return True
            try:
                res_new = cross_val_fold_scores(
                    self.baseline_model, X_new, y, self.scorer, cv=alt_a,
                    pipeline=pipe_new.get_pipeline(X_new, y) if pipe_new is not None else None,
                    model_fit_kwargs=self.model_fit_kwargs, groups=self._groups_active,
                    n_jobs_folds=self._cv_n_jobs_resolved)
                res_best = cross_val_fold_scores(
                    self.baseline_model, best_X, y, self.scorer, cv=alt_b,
                    pipeline=best_pipe.get_pipeline(best_X, y) if best_pipe is not None else None,
                    model_fit_kwargs=self.model_fit_kwargs, groups=self._groups_active,
                    n_jobs_folds=self._cv_n_jobs_resolved)
                if len(res_new.fold_scores) == len(res_best.fold_scores):
                    deltas.extend((sign * (res_new.fold_scores - res_best.fold_scores)).tolist())
            except Exception as e:
                self._log(f"  Confirmation eval failed ({e}); accepting unconfirmed")
                return True
        if not deltas:
            return True
        return float(np.mean(deltas)) > 0

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
                tree_model = self._make_selection_tree_model()
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
                                   encoder=CategoricalEncoder(target_enc_cols, count_enc_cols, freq_enc_cols))
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

        # Collect Global transform encoders (rank/bin/winsor fitted on train fold)
        global_encoders = []
        for i in interactions:
            if getattr(i, 'is_global', False):
                global_encoders.append(
                    GlobalTransformEncoder(col=i.feature_1.name, kind=i.op, output_col=i.name)
                )
        pipeline.global_encoders = global_encoders
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
            encoder=CategoricalEncoder(
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

        # Merge global transform encoders
        existing_g = getattr(pipeline, 'global_encoders', [])
        new_g = getattr(new_pipeline, 'global_encoders', [])
        seen_g = {g.output_col for g in existing_g}
        merged_g = list(existing_g)
        for g in new_g:
            if g.output_col not in seen_g:
                merged_g.append(g)
                seen_g.add(g.output_col)
        result.global_encoders = merged_g
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

    def _select_elites(self, batch: list[Interaction], n: int, X: pd.DataFrame, y: pd.Series,
                      callback: Optional[Callable] = None,
                      early_thr_override: Optional[int] = None) -> tuple[list[Interaction], pd.DataFrame, PipelineWrapper]:
        """Greedy forward-selection with adaptive thresholds."""
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

        # Selection loop with adaptive threshold and paired per-fold acceptance.
        # Baseline fold vector comes from the cache when the splitter state and
        # feature set are unchanged (saves one full CV per generation).
        base_res = self._get_baseline_fold_scores(X, y)
        best_val, best_folds = base_res.mean_val, base_res.fold_scores
        best_eras = base_res.per_group
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
        
        # Respect user's early stopping parameter (overridable by the
        # time-budget-aware sizing policy)
        if early_thr_override is not None:
            early_thr = early_thr_override
        else:
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

            # Compose candidate frame without copying X_base: pandas CoW makes
            # this lazy, and cross_val_score sanitizes (copies) internally.
            if not inter.require_pipeline and inter.name in X_copy.columns:
                X_try = pd.concat([X_base, X_copy[[inter.name]]], axis=1)
            else:
                X_try = X_base


            # Check for duplicates before evaluation
            if X_try.columns.duplicated().any():
                self._log(f"Warning: Duplicate columns in X_try for {inter.name}, skipping")
                continue
                
            pipe_iter = self._extend_pipeline(self.pipeline, self._prepare_pipeline([inter] + selected))

            try:
                res = self._eval_cv_light(X_try, y, pipe_iter)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._log(f"Error evaluating {inter.name}: {str(e)}")
                continue

            sign = 1.0 if self.scorer.greater_is_better else -1.0
            new_val = res.mean_val
            delta = sign * (new_val - best_val)
            gain = delta / (abs(best_val) + 1e-8)
            fold_deltas = None
            if best_folds is not None and len(res.fold_scores) == len(best_folds):
                fold_deltas = sign * (res.fold_scores - best_folds)
            era_deltas = None
            if best_eras and res.per_group:
                shared = [e for e in res.per_group if e in best_eras]
                if len(shared) >= 4:
                    era_deltas = sign * np.array([res.per_group[e] - best_eras[e] for e in shared])

            success = self._acceptance_gate(gain, fold_deltas, era_deltas)

            self.adaptive_controller.update_operation_stats(inter, success=success, gain=gain)

            if success:
                selected.append(inter)
                # Consolidate once per accepted feature (rare), not per candidate
                X_base, best_val, best_folds, consec_no_gain = X_try.copy(), new_val, res.fold_scores, 0
                best_eras = res.per_group
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
        self._bump_cv_epoch("partial restart")
        
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
            fold_state = getattr(self, '_best_fold_state', None)
            self.state['best'].update(
                X=self.X.copy(),
                pipeline=deepcopy(self.pipeline),  # Deep copy to prevent mutation
                generation=deepcopy(self.generation),  # Deep copy to preserve interaction refs
                pruned_features=getattr(self, 'pruned_features', set()).copy(),
                interactions=deepcopy(getattr(self, 'interactions', [])),  # Save interactions too
                val_fold_scores=(fold_state.fold_scores.copy()
                                 if fold_state is not None and fold_state.fold_scores is not None else None),
                fold_cv_epoch=(fold_state.cv_epoch if fold_state is not None else -1),
                val_fold_per_era=(fold_state.per_era if fold_state is not None else None),
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
            # Restore the paired-fold baseline vector of the best state; a stale
            # epoch (e.g. rotation since the save) fails matches() and triggers
            # a lazy recompute rather than a corrupted pairing.
            saved_scores = self.state['best'].get('val_fold_scores')
            if saved_scores is not None:
                self._best_fold_state = FoldEvalState(
                    fold_scores=np.asarray(saved_scores, dtype=float),
                    cv_epoch=self.state['best'].get('fold_cv_epoch', -1),
                    n_rows=len(self.X),
                    cols_hash=FoldEvalState.hash_cols(self.X),
                    per_era=self.state['best'].get('val_fold_per_era'))
            else:
                self._best_fold_state = FoldEvalState()
            return True
        return False

    def _budget_scaled_sizes(self, start_time: float, N: int) -> tuple[int, Optional[int]]:
        """Shrink per-generation work when wall-clock burn outpaces generation progress.

        Returns (n_children_eff, early_thr_override). scale < 1 when the
        remaining-time fraction lags the remaining-generation fraction; floors
        keep every generation meaningful (>= 20 children, >= 8 evals).
        """
        if not self.time_budget:
            return self.n_children, None
        elapsed = time.time() - start_time
        remaining_frac = max(0.0, 1.0 - elapsed / self.time_budget)
        gen_frac = max(1e-6, 1.0 - (N + 1) / max(1, self.n_generations))
        scale = float(np.clip(remaining_frac / gen_frac, 0.3, 1.0))
        if scale >= 1.0:
            return self.n_children, None
        n_children_eff = max(20, int(self.n_children * scale))
        if isinstance(self.early_stopping_child_eval, float):
            proxy_active = getattr(self, 'proxy_mode', 'none') != 'none' and self.use_proxy_evaluation
            approx_ranked = max(1, int(n_children_eff * (self.proxy_top_pct if proxy_active else 1.0)))
            base_thr = int(self.early_stopping_child_eval * approx_ranked)
        elif isinstance(self.early_stopping_child_eval, int):
            base_thr = self.early_stopping_child_eval
        else:
            return n_children_eff, None
        return n_children_eff, max(8, int(base_thr * scale))

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

            # Datetime columns are never IDs (the base expander decomposes them)
            if pd.api.types.is_datetime64_any_dtype(X[col]) and not is_id_name:
                continue

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

    def search(self, X: pd.DataFrame, y: pd.Series,
               X_test: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, PipelineWrapper, list[Feature], list[Interaction]]:
        """Enhanced genetic algorithm with better stagnation handling.

        X_test (optional, unlabeled): enables post-search adversarial
        validation — a train-vs-test report flagging generated features that
        encode train-specific structure.
        """
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        start_time = time.time()

        # Era mode: the era column becomes the CV grouping (never a feature),
        # and an int cv is upgraded to era-grouped folds.
        if getattr(self, 'era_col', None) and self.era_col in X.columns:
            era_series = X[self.era_col]
            if self.groups is None:
                self.groups = np.asarray(era_series)
            X = X.drop(columns=[self.era_col])
            if isinstance(self.cv, int):
                self.cv = RotatedGroupKFold(self.cv, rotation=self.random_state)
            self._log(f"Era mode: {pd.Series(self.groups).nunique()} eras drive grouped CV "
                      f"and era-stability acceptance")

        # Base-table expansion: datetime decomposition + row stats become part
        # of the base table (parent-eligible, protected from pruning). Runs
        # BEFORE ID dropping so unique-valued datetime columns are decomposed
        # rather than mistaken for IDs.
        self.base_expander = None
        if getattr(self, 'expand_datetime', True) or getattr(self, 'expand_row_stats', True):
            try:
                expander = BaselineFeatureExpander(
                    datetime_features=getattr(self, 'expand_datetime', True),
                    row_stats=getattr(self, 'expand_row_stats', True),
                    exclude_cols=tuple(c for c in (self.time_col, self.id_col) if c))
                X_expanded = expander.fit(X).transform(X)
                if expander.added_cols_:
                    self.base_expander = expander
                    X = X_expanded
                    self._log(f"Base expansion: {expander.summary()}")
            except Exception as e:
                self._log(f"Base expansion failed: {e}")

        X = self._drop_id_columns(X)
        self._set_defaults(X, y)
        self._cv_int_hint = self.cv if isinstance(self.cv, int) else None
        self.cv = normalize_rotatable_splitter(self.cv)
        self.initial_features = list(X.columns)
        self._priority_candidates = []
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
        self._bump_cv_epoch("search init")  # final data shape known only now (subsample/meta split)
        self.state['best']['train_score'], self.state['best']['val_score'] = self._eval_baseline(
            X, y, self.pipeline, update_fold_cache=True)
        gen0_log = f"Gen 0: Train {self.scorer.name}={self.state['best']['train_score']:.5f}, Val {self.scorer.name}={self.state['best']['val_score']:.5f}"
        if self.logging_scorers:
            logging_scores = self._eval_logging_scorers(X, y, self.pipeline)
            gen0_log += f" | {self._format_logging_scores(logging_scores)}"
        self._log(gen0_log)
        self.state['best']['X'], self.state['best']['pipeline'] = X.copy(), deepcopy(self.pipeline)
        self.state['best']['pruned_features'] = getattr(self, 'pruned_features', set()).copy()
        
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
                        self._bump_cv_epoch("fold rotation")
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

                        if is_better and not self._confirm_generation(X_monster, y, pipe_monster):
                            self._log("  Creative HM improvement not confirmed on alternate folds; discarded.")
                            is_better = False

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
                            new_train_score, new_val_score = self._eval_baseline(
                                X, y, self.pipeline, update_fold_cache=True)
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

                    # Generate Global transform candidates (rank/bin/winsor via pipeline);
                    # shallow parents only, and never re-rank a rank
                    if "global" in self.ops:
                        global_op_names = self.ops["global"]["unary"]
                        global_parents = [f for f in valid_unary if f.dtype == "num"
                                          and (f.depth or 0) < 2
                                          and not any(f.name.startswith(g + "_") for g in global_op_names)]
                        for feat in global_parents:
                            for gop in global_op_names:
                                candidates_pool.append(Interaction(feat, gop))

                    # Enhanced child sampling (budget-aware sizing)
                    n_children_eff, early_thr_eff = self._budget_scaled_sizes(start_time, N)
                    if n_children_eff < self.n_children:
                        self._log(f"  Budget-aware sizing: children {self.n_children} -> {n_children_eff}")
                    batch = self._sample_children_with_creativity(candidates_pool, n_children_eff, tau=tau)

                    # Guaranteed evaluation of queued follow-up candidates
                    # (e.g. group-bys over a just-accepted concat key)
                    if getattr(self, '_priority_candidates', None):
                        batch_names = {b.name for b in batch}
                        queued, seen_q = [], set()
                        for c in self._priority_candidates:
                            if (c.name in batch_names or c.name in seen_q or c.name in X.columns
                                    or c.feature_1.name not in X.columns
                                    or (c.feature_2 is not None and c.feature_2.name not in X.columns)):
                                continue
                            queued.append(c)
                            seen_q.add(c.name)
                        queued = queued[:max(1, n_children_eff // 4)]
                        if queued:
                            self._log(f"  Priority queue: injecting {len(queued)} follow-up candidates")
                            batch = queued + batch
                        self._priority_candidates = []

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

                        elites, X, self.pipeline = self._select_elites(
                            batch, features_per_gen, X, y, update_callback,
                            early_thr_override=early_thr_eff)
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

                        # Accepted concat keys unlock multi-key group-bys: queue
                        # them for guaranteed evaluation next generation
                        if interaction.op == "concat" and "agg" in self.ops:
                            num_parents_q = sorted(
                                [f for f in generation if f.dtype == "num" and f.name in X.columns
                                 and not f.require_pipeline],
                                key=lambda f: f.weight, reverse=True)[:5]
                            for np_feat in num_parents_q:
                                for agg_op in self.ops["agg"]["binary"]:
                                    self._priority_candidates.append(Interaction(feat, agg_op, np_feat))

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
                    
                    new_train_score, new_val_score = self._eval_baseline(
                        X, y, self.pipeline, update_fold_cache=True)
                    delta = new_val_score - self.state['best']['val_score'] if self.scorer.greater_is_better else self.state['best']['val_score'] - new_val_score
                    
                    # Revert if no improvement
                    if delta <= 0 and features_added > 0:
                        self._log(f"  Gen {N+1} added {features_added} features but no improvement. Reverting to best gen.")
                        if self._revert_to_best():
                            X, self.pipeline, generation = self.X, self.pipeline, self.generation
                            new_val_score, delta = self.state['best']['val_score'], 0
                            self.state['counters']['total_new_features'] = X.shape[1] - len(self.initial_features)
                            elites = []
                
                # Multi-seed confirmation before committing an improvement
                if not hopeful_monster_success and delta > 0 and features_added > 0:
                    if not self._confirm_generation(X, y, self.pipeline):
                        self._log(f"  Gen {N+1} improvement not confirmed on alternate folds; reverting.")
                        if self._revert_to_best():
                            X, self.pipeline, generation = self.X, self.pipeline, self.generation
                            new_val_score, delta = self.state['best']['val_score'], 0
                            self.state['counters']['total_new_features'] = X.shape[1] - len(self.initial_features)
                            elites = []
                            features_added = 0

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
                n_global = len(getattr(self.pipeline, "global_encoders", []))
                n_encoded = (self.pipeline.encoder.n_new_feats if hasattr(self.pipeline, 'encoder') else 0) + n_groupby + n_temporal + n_global

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
            if getattr(self, 'base_expander', None) is not None:
                X = self.base_expander.transform(X)
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
            self._bump_cv_epoch("full-data replay")
            train_score, val_score = self._eval_baseline(X, y, self.pipeline, update_fold_cache=True)
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
                if getattr(self, 'base_expander', None) is not None:
                    X_meta_transformed = self.base_expander.transform(X_meta_transformed)
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

        # Regularized post-selection (Enhancement 5) + null-importance gate
        if self.final_selection and hasattr(self, 'interactions') and self.interactions:
            features_to_drop = set(self._final_regularized_selection(X, y))
            X_survivors = (X.drop(columns=[c for c in features_to_drop if c in X.columns], errors='ignore')
                           if features_to_drop else X)
            null_era_groups = self._groups_active if getattr(self, 'era_col', None) else None
            features_to_drop.update(self._null_importance_selection(X_survivors, y,
                                                                    era_groups=null_era_groups))
            features_to_drop = sorted(features_to_drop)
            if features_to_drop:
                X = X.drop(columns=[c for c in features_to_drop if c in X.columns], errors='ignore')
                if not hasattr(self, 'pruned_features'):
                    self.pruned_features = set()
                self.pruned_features.update(features_to_drop)
                self._sync_state_components(X, self.pipeline, generation, preserve_pruned=True)
                # Re-evaluate after pruning
                train_score, val_score = self._eval_baseline(X, y, self.pipeline, update_fold_cache=True)
                self.state['best']['val_score'] = val_score
                self.state['best']['train_score'] = train_score
                self._log(f"Post-selection validation: {self.scorer.name}={val_score:.5f}")

        # Adversarial validation against unlabeled test features (optional)
        if X_test is not None:
            try:
                adv_drop = self._adversarial_validation_report(X, X_test)
                if adv_drop:
                    X = X.drop(columns=[c for c in adv_drop if c in X.columns], errors='ignore')
                    self.pruned_features.update(adv_drop)
                    self._sync_state_components(X, self.pipeline, generation, preserve_pruned=True)
                    train_score, val_score = self._eval_baseline(X, y, self.pipeline, update_fold_cache=True)
                    self.state['best']['val_score'], self.state['best']['train_score'] = val_score, train_score
                    self._log(f"Post-adversarial validation: {self.scorer.name}={val_score:.5f}")
            except Exception as e:
                self._log(f"Adversarial validation failed: {e}")

        # Calculate and store metrics
        n_init_feats = len(self.initial_features)
        n_groupby = len(getattr(self.pipeline, "groupby_encoders", []))
        n_temporal = len(getattr(self.pipeline, "temporal_encoders", []))
        n_global = len(getattr(self.pipeline, "global_encoders", []))
        n_added_feats = len(X.columns) - n_init_feats + self.pipeline.encoder.n_new_feats + n_groupby + n_temporal + n_global

        # Use a clean pipeline for baseline evaluation to get true initial performance
        baseline_pipeline = PipelineWrapper(imputer=None, scaler=None, encoder=CategoricalEncoder())
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
        g_names = [g.output_col for g in getattr(self.pipeline, "global_encoders", [])]

        new_features = {
            "generated": all_generated - encoder_feats_final,
            "target encoded": self.pipeline.encoder.target_enc_cols,
            "count encoded": self.pipeline.encoder.count_enc_cols,
            "freq encoded": self.pipeline.encoder.freq_enc_cols,
            "groupby": gb_names,
            "temporal": te_names,
            "global": g_names
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

        # Parallel fold fitting: conservative default to avoid oversubscription
        # (model n_jobs is clamped per fold inside cross_val_score).
        cv_n_jobs = getattr(self, "cv_n_jobs", "auto")
        if cv_n_jobs == "auto":
            self._cv_n_jobs_resolved = 1 if self.device == "cuda" else 2
        else:
            self._cv_n_jobs_resolved = max(1, int(cv_n_jobs))

        # Pipeline & adaptive controller
        self.pipeline = PipelineWrapper(imputer=None, scaler=None, encoder=CategoricalEncoder())
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
            self.pipeline = PipelineWrapper(imputer=None, scaler=None, encoder=CategoricalEncoder()).get_pipeline(X, y)

        # Label encode target (same as search) — category_encoders internally
        # converts non-numeric y to numpy via LabelEncoder without wrapping back in Series
        if y is not None and getattr(self, 'task', None) != "regression":
            unique_vals = np.unique(y)
            if not np.array_equal(unique_vals, np.arange(len(unique_vals))):
                y_encoded, _ = y.factorize(sort=True)
                y = pd.Series(y_encoded, index=y.index, name=y.name)
        X_transformed = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        # Re-apply base expansion (datetime parts, row stats)
        if getattr(self, 'base_expander', None) is not None:
            X_transformed = self.base_expander.transform(X_transformed)

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
        X_transformed = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()

        # Re-apply base expansion (datetime parts, row stats) — also when no
        # interactions were accepted, so the output schema matches search()
        if getattr(self, 'base_expander', None) is not None:
            X_transformed = self.base_expander.transform(X_transformed)

        if not getattr(self, 'interactions', None):
            self._log("Warning: No interactions. Returning base-expanded data.")
            return X_transformed

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
        if not hasattr(self, 'proxy_mode'):
            self.proxy_mode = "batched" if self.use_proxy_evaluation else "none"
        if not hasattr(self, 'proxy_ram_budget_mb'):
            self.proxy_ram_budget_mb = 512
        if not hasattr(self, 'proxy_halving'):
            self.proxy_halving = False

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

        # Competition-grade additions (paired-fold acceptance, parallel CV)
        if not hasattr(self, 'acceptance'):
            self.acceptance = "statistical"
        if not hasattr(self, 'acceptance_folds_frac'):
            self.acceptance_folds_frac = 0.7
        if not hasattr(self, 'confirmation_seeds'):
            self.confirmation_seeds = 1
        if not hasattr(self, 'null_importance_selection'):
            self.null_importance_selection = True
        if not hasattr(self, 'null_importance_n_perm'):
            self.null_importance_n_perm = 4
        if not hasattr(self, 'null_importance_pct'):
            self.null_importance_pct = 75.0
        if not hasattr(self, '_cv_int_hint'):
            self._cv_int_hint = None
        if not hasattr(self, 'expand_datetime'):
            self.expand_datetime = True
        if not hasattr(self, 'expand_row_stats'):
            self.expand_row_stats = True
        if not hasattr(self, 'base_expander'):
            self.base_expander = None
        if not hasattr(self, '_priority_candidates'):
            self._priority_candidates = []
        if not hasattr(self, 'era_col'):
            self.era_col = None
        if not hasattr(self, 'era_acceptance_frac'):
            self.era_acceptance_frac = 0.55
        if not hasattr(self, 'adversarial_auc_warn'):
            self.adversarial_auc_warn = 0.75
        if not hasattr(self, 'adversarial_drop'):
            self.adversarial_drop = False
        if not hasattr(self, 'adversarial_report'):
            self.adversarial_report = None
        if not hasattr(self, 'cv_n_jobs'):
            self.cv_n_jobs = "auto"
        if not hasattr(self, '_cv_n_jobs_resolved'):
            self._cv_n_jobs_resolved = 1
        if not hasattr(self, '_cv_epoch'):
            self._cv_epoch = 0
        if not hasattr(self, '_best_fold_state'):
            self._best_fold_state = FoldEvalState()

        # Ensure state dict has interactions in best (for future reverts)
        if hasattr(self, 'state') and 'best' in self.state:
            if 'X' not in self.state['best']:
                self.state['best']['X'] = None
            if 'interactions' not in self.state['best']:
                self.state['best']['interactions'] = deepcopy(self.interactions)
            if 'pruned_features' not in self.state['best']:
                self.state['best']['pruned_features'] = set()
            if 'val_fold_scores' not in self.state['best']:
                self.state['best']['val_fold_scores'] = None
            if 'fold_cv_epoch' not in self.state['best']:
                self.state['best']['fold_cv_epoch'] = -1

    def generate(self, X: pd.DataFrame, y: pd.Series, X_test: Optional[pd.DataFrame] = None):
        """Main entry point for feature generation."""
        return self.search(X, y, X_test=X_test)
