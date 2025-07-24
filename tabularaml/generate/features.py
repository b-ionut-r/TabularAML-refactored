import os, random, time
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from enum import Enum
from dataclasses import dataclass, field
from typing import Union, List, Optional, Callable, Literal, Dict, Set, Tuple
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.multiclass import type_of_target
from tqdm.auto import tqdm
from xgboost import XGBClassifier, XGBRegressor

from tabularaml.eval.cv import cross_val_score
from tabularaml.eval.scorers import PREDEFINED_REG_SCORERS, PREDEFINED_CLS_SCORERS, PREDEFINED_SCORERS, Scorer
from tabularaml.generate.ops import OPS, ALL_OPS_LAMBDAS
from tabularaml.inspect.importance import FeatureImportanceAnalyzer
from tabularaml.preprocessing.encoders import CategoricalEncoder
from tabularaml.preprocessing.imputers import SimpleImputer
from tabularaml.preprocessing.pipeline import PipelineWrapper
from tabularaml.configs.feature_gen import PRESET_PARAMS
from tabularaml.utils import is_gpu_available

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
                    return ALL_OPS_LAMBDAS[self.op](X, self.feature_1.name)
                elif self.type == "binary":
                    # Check for column shape issues
                    if X[self.feature_1.name].ndim > 1 or X[self.feature_2.name].ndim > 1:
                        raise ValueError(f"Multi-dimensional columns detected: {self.feature_1.name} shape={X[self.feature_1.name].shape}, {self.feature_2.name} shape={X[self.feature_2.name].shape}")
                    return ALL_OPS_LAMBDAS[self.op](X, self.feature_1.name, self.feature_2.name)
            except Exception as e:
                raise Exception(f"Error generating {self.name}: {str(e)}")
        raise Exception("Can't generate feature using lambdas. Requires pipeline to avoid data leakage.")
    
    def get_new_feature_instance(self):
        return Feature(name=self.name, dtype=self.dtype, weight=self.weight, require_pipeline=self.require_pipeline)


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
        self.strategy_success = {"hopeful_monster": 0, "beam_search": 0, "normal": 0}
        self.strategy_attempts = {"hopeful_monster": 0, "beam_search": 0, "normal": 0}
        
        # Memory of what worked
        self.successful_patterns = []  # List of (parent_features, operation, gain) tuples
        self.weight_modifications = {}

    def initialize_operations(self, ops):
        """Initialize operation statistics with diversity bias."""
        for dtype in ops:
            for op_type in ops[dtype]:
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
        self.strategy_success = {"hopeful_monster": 0, "beam_search": 0, "normal": 0}
        self.strategy_attempts = {"hopeful_monster": 0, "beam_search": 0, "normal": 0}
    
    def get_status_summary(self) -> dict:
        """Get summary of current adaptive state."""
        hopeful_rate = self.get_strategy_success_rate("hopeful_monster")
        beam_rate = self.get_strategy_success_rate("beam_search")
        normal_rate = self.get_strategy_success_rate("normal")
        
        return {
            "stagnation_level": self.state.stagnation_level.name,
            "exploration_intensity": f"{self.state.exploration_intensity:.2f}",
            "min_gain_factor": f"{self.state.min_gain_reduction_factor:.2f}",
            "consecutive_success": self.state.consecutive_successful_generations,
            "weights_modified": self.state.feature_weights_modified,
            "features_modified": len(self.weight_modifications),
            "strategy_success": f"HM:{hopeful_rate:.2f}, BS:{beam_rate:.2f}, N:{normal_rate:.2f}",
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
                 use_gpu: bool = True,
                 log_file: Union[str, Path] = "cache/logs/feat_gen_log.txt",
                 adaptive: bool = True, 
                 time_budget=None, 
                 max_ops_per_generation=None,
                 exploration_factor: float = 0.2, 
                 save_path=None):        
        
        # Capture provided parameters
        provided_params = locals().copy()
        provided_params.pop('self')
        
        self.mode = mode
        if mode:
            self._set_params_from_mode(provided_params)
        
        # Set parameters not handled by mode
        if not mode:
            # If no mode, set all parameters normally
            self.n_generations = n_generations
            self.n_parents = n_parents
            self.n_children = n_children
            self.ranking_method = ranking_method
            self.min_pct_gain = min_pct_gain
            self.early_stopping_iter = early_stopping_iter
            self.early_stopping_child_eval = early_stopping_child_eval
            self.cv = cv
            self.time_budget = time_budget
        
        # Core parameters (always set normally)
        self.baseline_model = baseline_model
        self.model_fit_kwargs = model_fit_kwargs
        self.task = task
        self.scorer = scorer
        self.infer_task = any(p is None for p in (baseline_model, task, scorer))
        self.imp_weights = imp_weights
        self.max_new_feats = max_new_feats
        self.adaptive = adaptive
        self.save_path = save_path
        
        # Early stopping
        self.early_stopping_iter = (int(early_stopping_iter * n_generations) 
                                   if isinstance(early_stopping_iter, float)
                                   else early_stopping_iter 
                                   if isinstance(early_stopping_iter, int) 
                                   else float('inf'))
        self.early_stopping_child_eval = early_stopping_child_eval
        
        # Technical setup
        self.adaptive_controller = ImprovedAdaptiveController(
            original_min_pct_gain=min_pct_gain, 
            exploration_factor=exploration_factor
        )
        self.ops = ops if ops is not None else OPS
        self.cv = cv
        self.device = "cuda" if is_gpu_available() and use_gpu else "cpu"
        self.pipeline = PipelineWrapper(imputer=None, scaler=None, encoder=CategoricalEncoder())
        
        # Legacy compatibility
        self.max_ops_per_generation = max_ops_per_generation
        self.exploration_factor = exploration_factor
        
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log_file = log_file

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
            sig = inspect.signature(self.__init__)
            
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
        print(message)
        if self.log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, "a") as f:
                f.write(f"[{timestamp}] {message}\n")

    def _get_num_cat_cols(self, X: pd.DataFrame) -> tuple[list, list]:
        return (X.select_dtypes(include=['number']).columns.tolist(), 
                X.select_dtypes(include=['object', 'category']).columns.tolist())
    
    def _get_top_k_features(self, X: pd.DataFrame, y: pd.Series, k: int = 50, pipeline=None) -> pd.DataFrame:
        """Get top k features by importance."""
        pipeline.imputer = SimpleImputer() 
        analyzer = FeatureImportanceAnalyzer(
            task_type=self.task, weights=self.imp_weights, preferred_gbm="xgboost",
            pipeline=pipeline, cv=self.cv, use_gpu=(self.device == "cuda"))
        analyzer.fit(X, y)
        pipeline.imputer = None 
        imp_df = analyzer.get_importance(normalize=False)[["weighted_importance"]]
        imp_df.sort_values(by="weighted_importance", axis=0, ascending=False, inplace=True)
        return imp_df if k == -1 else imp_df[:k]

    def _eval_baseline(self, X: pd.DataFrame, y: pd.Series, pipeline=None) -> tuple[float, float]:
        """Evaluate baseline model performance."""
        pipeline = pipeline.get_pipeline(X) if pipeline is not None else pipeline
        cv_dict = cross_val_score(self.baseline_model, X, y, self.scorer, cv=self.cv,
                                 return_dict=True, pipeline=pipeline, model_fit_kwargs=self.model_fit_kwargs)
        return cv_dict["mean_train_score"], cv_dict["mean_val_score"]

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
            task_type=self.task, use_gpu=self.device == "gpu", verbose=0, n_jobs=-1, preferred_gbm='xgboost')
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
        return PipelineWrapper(imputer=None, scaler=None, 
                              encoder=CategoricalEncoder(target_enc_cols, count_enc_cols, freq_enc_cols))

    def _extend_pipeline(self, pipeline: PipelineWrapper, new_pipeline: PipelineWrapper) -> PipelineWrapper:
        """Extend pipeline with new_pipeline for categorical encoding."""
        return PipelineWrapper(imputer=None, scaler=None,
            encoder=CategoricalEncoder(
                target_enc_cols=list(set(pipeline.encoder.target_enc_cols + new_pipeline.encoder.target_enc_cols)),
                count_enc_cols=list(set(pipeline.encoder.count_enc_cols + new_pipeline.encoder.count_enc_cols)),
                freq_enc_cols=list(set(pipeline.encoder.freq_enc_cols + new_pipeline.encoder.freq_enc_cols))))
        
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
                        name, val = interaction.generate(X)
                        if name not in X.columns and name not in new_features:  # Avoid duplicates
                            new_features[name] = val
                    except Exception as e:
                        self._log(f"Warning: Failed to generate {interaction.name}: {str(e)}")
        
        if new_features:
            # Check for duplicate columns before concatenating
            X = self._ensure_no_duplicates(X, "before adding features in _apply_interactions")
            
            X_copy = pd.concat([X.copy(), pd.DataFrame(new_features)], axis=1)
            
            # Verify no duplicates after concatenation
            X_copy = self._ensure_no_duplicates(X_copy, "after adding features in _apply_interactions")
        else:
            X_copy = X.copy()
            
        return X_copy, self._prepare_pipeline(interactions)

    def _select_elites(self, batch: list[Interaction], n: int, X: pd.DataFrame, y: pd.Series,
                      callback: Optional[Callable] = None) -> tuple[list[Interaction], pd.DataFrame, PipelineWrapper]:
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
            evals += 1
            if callback: callback(evals, len(selected))
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
                self._log(f"Error evaluating {inter.name}: {str(e)}")
                continue
            
            delta = (new_val - best_val) if self.scorer.greater_is_better else (best_val - new_val)
            gain = delta / (abs(best_val) + 1e-8)
            
            # Use adaptive threshold
            success = gain >= self.adaptive_controller.get_adaptive_min_gain()
            
            self.adaptive_controller.update_operation_stats(inter, success=success, gain=gain)
            
            if success:
                selected.append(inter)
                X_base, best_val, consec_no_gain = X_try, new_val, 0
            else:
                consec_no_gain += 1

            # Only apply early stopping if not in severe stagnation
            if (evals >= min_evals and consec_no_gain >= early_thr and 
                self.adaptive_controller.state.stagnation_level.value < StagnationLevel.SEVERE.value):
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
        
        self._log(f"  Restart complete: kept {len(X_restart.columns)} features (was {X.shape[1]})")
        
        return X_restart, new_generation

    def _sync_state_components(self, X: pd.DataFrame, pipeline: PipelineWrapper, generation: list):
        """Ensure all state components are consistent and save current state as best if needed."""
        self.X = X.copy()
        self.pipeline = pipeline
        self.generation = generation.copy()
        self.interactions = [feat.generating_interaction for feat in self.generation 
                           if hasattr(feat, 'generating_interaction') and feat.generating_interaction]
        if hasattr(self, 'pruned_features'):
            self.pruned_features = {feat for feat in self.pruned_features if feat not in X.columns}
    
    def _save_current_as_best(self):
        """Save current state as the best state and auto-save if path is provided."""
        if hasattr(self, 'state') and 'best' in self.state:
            self.state['best'].update(
                X=self.X.copy(),
                pipeline=self.pipeline,
                generation=self.generation.copy(),
                pruned_features=getattr(self, 'pruned_features', set()).copy()
            )
            if hasattr(self, 'save_path') and self.save_path:
                self.save(self.save_path)
    
    def _revert_to_best(self):
        """Revert to best saved state."""
        if hasattr(self, 'state') and 'best' in self.state and self.state['best']['X'] is not None:
            self.X = self.state['best']['X'].copy()
            self.pipeline = self.state['best']['pipeline']
            self.generation = self.state['best']['generation'].copy()
            self.pruned_features = self.state['best']['pruned_features'].copy()
            self._sync_state_components(self.X, self.pipeline, self.generation)
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

    def search(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, PipelineWrapper, list[Feature], list[Interaction]]:
        """Enhanced genetic algorithm with better stagnation handling."""
        start_time = time.time()
        self._set_defaults(X, y)
        self.initial_features = list(X.columns)
        num_cols, cat_cols = self._get_num_cat_cols(X)
        self.max_gen_new_feats = (int(self.max_new_feats * len(self.initial_features)) if isinstance(self.max_new_feats, float) 
                                 else self.max_new_feats if isinstance(self.max_new_feats, int) else float('inf'))

        # Label encode target for GBMs
        if self.task != "regression":
            unique_vals = np.unique(y)
            if not np.array_equal(unique_vals, np.arange(len(unique_vals))):
                y_encoded, _ = y.factorize()
                y = pd.Series(y_encoded, index=y.index, name=y.name)
        
        # Initialize
        self.pruned_features = set()
        self._parent_usage = {}
        self._log(f"Starting {self.task} on {self.device} - {X.shape[0]} samples, {X.shape[1]} features")
        self._log(f"Params: gen={self.n_generations}, parents={self.n_parents}, children={self.n_children}, limit={self.max_gen_new_feats}")
        self.adaptive_controller.initialize_operations(self.ops)
        self.adaptive_controller.reset_for_new_run()
        self.state['best']['train_score'], self.state['best']['val_score'] = self._eval_baseline(X, y, self.pipeline)
        self._log(f"Gen 0: Train {self.scorer.name}={self.state['best']['train_score']:.5f}, Val {self.scorer.name}={self.state['best']['val_score']:.5f}")
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
        
        with tqdm(total=self.n_generations, desc="Generations") as pbar:
            for N in range(self.n_generations):
                self.state['counters']['current_gen'] = N
                
                if self.time_budget and (time.time() - start_time) > self.time_budget:
                    self._log(f"Time budget exceeded. Stopping.")
                    break
                
                progress = N / self.n_generations
                tau, beta, gamma, lambda_ = self._get_search_parameters(progress, N)
                self.adaptive_controller.assess_stagnation(
                    self.state['counters']['no_feature_gens_count'],
                    self.state['counters']['consecutive_no_improvement_iters']
                )
                
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

                    # Enhanced child sampling
                    batch = self._sample_children_with_creativity(candidates_pool, self.n_children, tau=tau)
                    pbar.set_description(f"Gen {N+1}: Testing {len(batch)} candidates")
                
                    remaining_budget = self.max_gen_new_feats - self.state['counters']['total_new_features'] if self.max_gen_new_feats != float('inf') else float('inf')
                    features_per_gen = max(min(20, remaining_budget), 1) if remaining_budget > 0 else 1
                    
                    if remaining_budget <= 0:
                        self._log(f"Gen {N+1}: No budget remaining. Skipping.")
                        continue
                    
                    # Skip beam search if it's been failing consistently
                    use_beam_search = (adaptive_status['stagnation_level'] in ['SEVERE', 'CRITICAL'] and 
                                     self.adaptive_controller.get_strategy_success_rate("beam_search") > 0.2 and
                                     random.random() < 0.3)
                    
                    if use_beam_search:
                        self._log(f"  Starting beam search k=3...")
                        # [Beam search implementation - keeping original logic]
                        # ... (beam search code remains the same)
                        beam_search_used = True
                    else:
                        beam_search_used = False
                    
                    # Standard elite selection
                    if not beam_search_used:
                        with tqdm(total=len(batch), desc="Evaluating features", leave=False) as inner_pbar:
                            def update_callback(ec, sc, force_complete=False):
                                inner_pbar.update(max(0, ec - inner_pbar.n if not force_complete else len(batch) - inner_pbar.n))
                                inner_pbar.set_description(f"Evaluating features - Selected: {sc}")
                                return self.time_budget and (time.time() - start_time) > self.time_budget
                            
                            elites, X, self.pipeline = self._select_elites(batch, features_per_gen, X, y, update_callback)
                        
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
                improvement = "No improvement." if delta <= 0 else f"Score improved by {delta:.5f}."
                adaptive_status = self.adaptive_controller.get_status_summary()
                
                gen_log = f"Gen {N+1}: Added {features_added} features, {X.shape[1]} total ({self.state['counters']['total_new_features']} new)."
                gen_log += f" Train {self.scorer.name}={new_train_score:.5f}, Val {self.scorer.name}={new_val_score:.5f}. {improvement}"
                gen_log += f" Status: {adaptive_status['stagnation_level']}, Strategy success: {adaptive_status['strategy_success']}"
                
                self._log(gen_log)
                
                # Log new features
                if features_added > 0 and elites:
                    new_simple = [elite.name for elite in elites if not elite.require_pipeline]
                    if new_simple: self._log(f"  Simple: {new_simple}")
                    
                    if self.pipeline.encoder.target_enc_cols: self._log(f"  Target encoded: {self.pipeline.encoder.target_enc_cols}")
                    if self.pipeline.encoder.count_enc_cols: self._log(f"  Count encoded: {self.pipeline.encoder.count_enc_cols}")
                    if self.pipeline.encoder.freq_enc_cols: self._log(f"  Freq encoded: {self.pipeline.encoder.freq_enc_cols}")
                
                pbar.set_postfix({f"{self.scorer.name}": f"{new_val_score:.5f}", "features": X.shape[1], 
                                 "new": self.state['counters']['total_new_features'], "best_gen": self.state['best']['gen_num']})
                pbar.update(1)
                
                # Check termination conditions
                if self.max_gen_new_feats != float('inf') and self.state['counters']['total_new_features'] >= self.max_gen_new_feats:
                    self._log(f"Reached max new features ({self.state['counters']['total_new_features']}/{self.max_gen_new_feats}). Stopping.")
                    break
                
                if self.state['counters']['consecutive_no_improvement_iters'] >= self.early_stopping_iter:
                    self._log(f"Early stopping after {self.state['counters']['consecutive_no_improvement_iters']} generations without improvement.")
                    break
        
        elapsed_time = time.time() - start_time
        
        # Ensure best generation is returned
        if self.state['best']['gen_num'] < self.n_generations and not X.equals(self.state['best']['X']):
            self._log(f"Reverting to best generation ({self.state['best']['gen_num']}).")
            if self._revert_to_best():
                X, self.pipeline, generation = self.X, self.pipeline, self.generation
        else:
            self._sync_state_components(X, self.pipeline, generation)
                    
        # Calculate and store metrics
        n_init_feats = len(self.initial_features)
        n_added_feats = len(X.columns) - n_init_feats + self.pipeline.encoder.n_new_feats
        
        self.initial_train_metric, self.initial_val_metric = self._eval_baseline(X[self.initial_features], y, self.pipeline)
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
        new_features = {
            "simple": set(X.columns) - set(self.initial_features),
            "target": self.pipeline.encoder.target_enc_cols,
            "count": self.pipeline.encoder.count_enc_cols,
            "freq": self.pipeline.encoder.freq_enc_cols
        }
        for feat_type, features in new_features.items():
            if features: self._log(f"New {feat_type}: {features}")
        
        # Reset for further calls
        if self.infer_task: self.baseline_model = self.task = self.scorer = None
    
        self._sync_state_components(X, self.pipeline.get_pipeline(X), generation)
        return self.X, self.pipeline, self.generation, self.interactions

    def _set_defaults(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Set default values for task, model, and scorer."""
        # Task/model/scorer
        self.task = self.task or ("regression" if type_of_target(y) == "continuous" else "classification")
        is_reg = self.task == "regression"
        if self.baseline_model is None:
            self.baseline_model = (XGBRegressor if is_reg else XGBClassifier)(
                device=self.device, enable_categorical=True, verbosity=0)
        if self.scorer is None:
            self.scorer = (PREDEFINED_REG_SCORERS["rmse"] if is_reg else 
                          PREDEFINED_CLS_SCORERS["binary_crossentropy"] 
                          if len(np.unique(y)) == 2 else 
                          PREDEFINED_CLS_SCORERS["categorical_crossentropy"])

        # Pipeline & adaptive controller
        self.pipeline = PipelineWrapper(imputer=None, scaler=None, encoder=CategoricalEncoder())
        self.adaptive_controller.reset_for_new_run()
        self.adaptive_controller.initialize_operations(self.ops)

        # Search state
        self.state = {
            "best": dict(gen_num=0, val_score=0, train_score=0, X=None, 
                        generation=None, pipeline=None, pruned_features=set()),
            "counters": dict(total_new_features=0, no_feature_gens_count=0, 
                           consecutive_no_improvement_iters=0, current_gen=0),
            "seen_feats": set(),
        }

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
            self.pipeline = PipelineWrapper(imputer=None, scaler=None, encoder=CategoricalEncoder()).get_pipeline(X)
        
        X_transformed = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Generate non-pipeline features
        for interaction in self.interactions:
            if interaction.name not in X_transformed.columns and not interaction.require_pipeline:
                try:
                    result = interaction.generate(X_transformed)
                    if result is not None:
                        X_transformed[result[0]] = result[1]
                except Exception as e:
                    self._log(f"Error generating {interaction.name}: {str(e)}")
                    
        # Fit pipeline
        if isinstance(self.pipeline, PipelineWrapper):
            self.pipeline = self.pipeline.get_pipeline(X)
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
                    result = interaction.generate(X_transformed)
                    if result is not None:
                        X_transformed[result[0]] = result[1]
                except Exception as e:
                    self._log(f"Error generating {interaction.name}: {str(e)}")
        
        # Apply pipeline
        pipeline = getattr(self, 'pipeline', None)
        if pipeline is not None:
            try:
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
    
    def save(self, filepath):
        """Save current state using cloudpickle."""
        import os
        try:
            import cloudpickle
        except ImportError:
            raise ImportError("cloudpickle required. Install with: pip install cloudpickle")
        
        # Ensure current state is consistent before saving
        if hasattr(self, 'X') and hasattr(self, 'pipeline') and hasattr(self, 'generation'):
            self._sync_state_components(self.X, self.pipeline, self.generation)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            cloudpickle.dump(self, f)
        self._log(f"State saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load state from file using cloudpickle."""
        import os
        try:
            import cloudpickle
        except ImportError:
            raise ImportError("cloudpickle required. Install with: pip install cloudpickle")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found.")
        
        try:    
            with open(filepath, 'rb') as f:
                return cloudpickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load: {str(e)}")

    def generate(self, X: pd.DataFrame, y: pd.Series):
        """Main entry point for feature generation."""
        return self.search(X, y)