############################################################################################################################################################################################################################################

import inspect
import os
import random
import time
from pathlib import Path
from collections import Counter
from datetime import datetime
from itertools import combinations
from enum import Enum
from dataclasses import dataclass
from typing import Union, List, Optional, Callable, Literal, Dict

import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import BaseCrossValidator
from tqdm.auto import tqdm
from xgboost import XGBClassifier, XGBRegressor


from tabularaml.eval.cv import cross_val_score
from tabularaml.eval.scorers import (PREDEFINED_REG_SCORERS, PREDEFINED_CLS_SCORERS, 
                                     PREDEFINED_SCORERS, Scorer)
from tabularaml.generate.ops import OPS, CAT_OPS_LAMBDAS, NUM_OPS_LAMBDAS, ALL_OPS_LAMBDAS
from tabularaml.inspect.importance import FeatureImportanceAnalyzer
from tabularaml.preprocessing.encoders import CategoricalEncoder
from tabularaml.preprocessing.imputers import SimpleImputer
from tabularaml.preprocessing.pipeline import PipelineWrapper
from tabularaml.configs.feature_gen import PRESET_PARAMS
from tabularaml.utils import is_gpu_available

############################################################################################################################################################################################################################################

class Feature:
    """
    Feature class for representing data columns with their properties and metadata.
    Attributes:
        name (str): Feature name or identifier.
        dtype (Literal["num", "cat"]): Data type of the feature - 'num' for numeric, 'cat' for categorical.
        weight (float): Importance weight assigned to this feature.
        depth (int): Feature complexity or transformation depth (number of operations applied).
        require_pipeline (bool): Indicates if feature requires specific preprocessing pipeline.
        generating_interaction: Reference to the interaction that generated this feature, if any.
    Methods:
        get_feature_depth(): Calculates feature complexity by counting operations in its name.
        get_col_from_df(X): Extracts feature values from a dataframe.
        update_weight(new_weight): Updates the feature's importance weight.
        set_generating_interaction(interaction): Sets reference to generating interaction.
    """

    def __init__(self, 
                 name: str, 
                 dtype: Literal["num", "cat"], 
                 weight: float, 
                 depth: Optional[int] = None, 
                 require_pipeline: Optional[bool] = False):
        
        self.name = name
        self.dtype = dtype
        self.weight = weight
        self.depth = depth if depth is not None else self.get_feature_depth()
        self.require_pipeline = require_pipeline
        self.generating_interaction = None
        
    def get_feature_depth(self):
        n = 0
        for ending in OPS["num"]["unary"] + OPS["cat"]["unary"]:
            n += self.name.count(f"_{ending}")
        for middle in OPS["num"]["binary"] + OPS["cat"]["binary"]:
            n += self.name.count(f"_{middle}_")
        return n

    def get_col_from_df(self, X: pd.DataFrame): 
        return X[self.name].values
    
    def update_weight(self, new_weight: float): 
        self.weight = new_weight

    def set_generating_interaction(self, interaction: 'Interaction'): 
        self.generating_interaction = interaction


############################################################################################################################################################################################################################################

class Interaction:

    """
    Represents feature interactions for engineering new features.
    Creates features by applying unary operations to single features or binary operations between two features.

    Parameters:
        feature_1: First input feature.
        op: Operation to apply (e.g.,  "+", "-", "*", "/", "concat", "count", "target" etc.)/
        feature_2: Second feature for binary operations (default: None).

    Attributes:
        type: "unary" or "binary" interaction.
        dtype: Data type of resulting feature.
        depth: Complexity level in feature engineering tree.
        weight: Importance score.
        require_pipeline: Whether pipeline is needed to prevent data leakage (for "count", "target", "freq").
        name: Generated feature name.
    """

    def __init__(self, 
                 feature_1: Feature, 
                 op: str, 
                 feature_2: Optional[Feature] = None):
        
        self.feature_1 = feature_1
        self.op = op
        self.feature_2 = feature_2
        self.type = "unary" if self.feature_2 is None else "binary"
        self.dtype = (self.feature_1.dtype if self.feature_2 is None else 
                    "num" if self.feature_1.dtype == self.feature_2.dtype == "num" else "cat")
        self.depth = (feature_1.depth + 1 if self.feature_2 is None else 
                    max(self.feature_1.depth, self.feature_2.depth) + 1)
        self.weight = feature_1.weight if self.feature_2 is None else (feature_1.weight + feature_2.weight) / 2
        self.require_pipeline = self.feature_2 is None and self.op in ["target", "count", "freq"]
        self.name = f"{self.feature_1.name}_{op}" if self.type == "unary" else f"{self.feature_1.name}_{op}_{self.feature_2.name}"
         
    def generate(self, X, y = None):
        if not self.require_pipeline:
            if self.type == "unary":
                return ALL_OPS_LAMBDAS[self.op](X, self.feature_1.name)
            elif self.type == "binary":
                return ALL_OPS_LAMBDAS[self.op](X, self.feature_1.name, self.feature_2.name)
        raise Exception("Can't generate feature using lambdas. Requires pipeline to avoid data leakage.")
    
    def get_new_feature_instance(self):
        return Feature(name=self.name, dtype=self.dtype, weight=self.weight, require_pipeline=self.require_pipeline)


############################################################################################################################################################################################################################################

class FeatureGenerator:
    """
    A genetic algorithm-based feature generator for tabular data.
    This class implements an evolutionary approach to automatically create and select new features
    that improve model performance. It iteratively constructs features through operations like
    arithmetic combinations, aggregations, and categorical encoding, evaluating them by their
    impact on model performance.
    Parameters
    ----------
    baseline_model : model object, default = None
        The model used to evaluate features. If None, XGBoost will be used based on task type.
    model_fit_kwargs : dict, default = {}
        Additional parameters passed to the model's fit method.
    task : str, default = None
        'classification' or 'regression'. If None, will be inferred from target data.
    scorer : Scorer, default = None
        Metric used to evaluate features. If None, will use appropriate default for task.
    mode : str, default = None
        Preset parameter configuration (e.g., 'lite', 'medium', 'best', 'extreme').
        See API documentation for further details.
    n_generations : int, default = 15
        Maximum number of generations to evolve.
    n_parents : int, default = 40
        Number of parent features selected in each generation, of each
        interaction type (n_parents features for unary and n_parents pairs for binary).
    n_children : int, default = 200
        Number of candidate features to evaluate in each generation.
    ranking_method : str, default = "multi_criteria"
        Method used to rank candidate features before greedy selection.
        Options: "multi_criteria" (recommended), "shap", or "none".
        - multi_criteria: Uses a composite score based on information gain,
          correlation, novelty, and historical performance.
        - shap: Uses SHAP values to rank features (computationally expensive).
        - none: Uses the original order of candidates without ranking.
    min_pct_gain : float, default = 0.001
        Minimum percentage improvement required to keep a feature.
    imp_weights : dict, default = None
        Weights for feature importance metrics.
    max_gen_new_feats_pct : float or int, default = 2.0
        Maximum number of new features to create, as percentage or absolute value.
    early_stopping_iter : float or int or None, default = 0.4
        Number of generations without improvement before early stopping.
    early_stopping_child_eval : float or int or None, default = 0.3
        Controls early stopping during candidate evaluation.
    ops : dict, default=None
        Operations to use for feature generation.
    cv : int, default = 5
        Number of cross-validation folds / cross validator instance.
    use_gpu : bool, default = True
        Whether to use GPU if available.
    log_to_file : str, default = "cache/logs/feat_gen_log.txt"
        File path for logging.
    adaptive : bool, default = True
        Whether to adapt parameters during feature generation.
    time_budget : int, default = None
        Maximum time (seconds) to run feature generation.
    save_path : str or Path, default = None
        Path where to save the best generation when found. When provided, automatically 
        saves the feature generator state when a new best generation is discovered.
    Methods
    -------
    generate(X, y)
        Generate new features based on provided data.
        Returns augmented dataframe, pipeline, and feature representation.
    Notes
    -----
    The feature generation process combines operations on parent features to create
    candidate features, then evaluates their impact using cross-validation.
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
                 min_pct_gain: float = 0.001, # 0.1%
                 imp_weights: Optional[Dict[Literal["tree", "correlation", "permutation", "shap"], float]] = None,
                 max_new_feats = None,
                 early_stopping_iter: Union[float, int, bool] = 0.4, 
                 early_stopping_child_eval: Union[float, int, bool] = 0.3,
                 ops: Optional[dict] = None,
                 cv: Union[int, BaseCrossValidator] = 5,
                 use_gpu: bool = True,
                 log_file: Union[str, Path] = "cache/logs/feat_gen_log.txt",
                 adaptive: bool = True, 
                 time_budget: Optional[int] = None,
                 max_ops_per_generation: Optional[int] = None,
                 exploration_factor: float = 0.2,
                 save_path: Optional[Union[str, Path]] = None):        
        self.mode = mode
        if mode:
            self._set_params_from_mode()
        
        self.baseline_model = baseline_model
        self.model_fit_kwargs = model_fit_kwargs
        self.task = task
        self.scorer = scorer
        self.infer_task = any(param is None for param in (baseline_model, task, scorer))
        self.n_generations = n_generations
        self.n_parents = n_parents
        self.n_children = n_children
        self.ranking_method = ranking_method    
        self.min_pct_gain = min_pct_gain
        self.imp_weights = imp_weights
        self.max_new_feats = max_new_feats
        self.early_stopping_iter = (
            int(early_stopping_iter * n_generations)
            if isinstance(early_stopping_iter, float)
            else early_stopping_iter
            if isinstance(early_stopping_iter, int)
            else float('inf')
        )
        self.early_stopping_child_eval = early_stopping_child_eval
        self.adaptive = adaptive
        self.time_budget = time_budget 
        self.save_path = save_path
        
        # Initialize unified adaptive controller
        self.adaptive_controller = AdaptiveController(
            original_min_pct_gain=min_pct_gain,
            exploration_factor=exploration_factor
        )

        self.ops = ops if ops is not None else OPS
        self.cv = cv
        self.device = "cuda" if is_gpu_available() and use_gpu else "cpu"
        self.pipeline = PipelineWrapper(
            imputer = None,
            scaler = None,
            encoder = CategoricalEncoder()
        )
        
        # Legacy parameters for backward compatibility
        self.max_ops_per_generation = max_ops_per_generation
        self.exploration_factor = exploration_factor
         
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log_file = log_file




    def _set_params_from_mode(self):
        """
        Sets instance generation parameters to the ones assigned to the 
        'mode' string key in the PRESET_PARAMS dictionary.
        Used in __init__ constructor. If at instantiation user overwrites any
        of the parameters, that takes precedence.
        """
        mode_dict = PRESET_PARAMS.get(self.mode, None)
        if mode_dict:
            for param, value in mode_dict.items():
                setattr(self, param, value)
        else:
            raise Exception(f"{self.mode.upper()} mode is undefinded. Use one of the following: \
                            'low', 'medium', 'best', 'extreme'")
        

    
    def _log(self, message):
        """Log message to both terminal and file if specified."""
        print(message)
        if self.log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, "a") as f:
                f.write(f"[{timestamp}] {message}\n")
    


    def _get_num_cat_cols(self, X: pd.DataFrame) -> tuple[list, list]:
        num_cols = X.select_dtypes(include=['number']).columns.to_list()
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        return num_cols, cat_cols
    
    def _get_top_k_features(self,
                            X: pd.DataFrame, 
                            y: pd.Series, 
                            k: int = 50, 
                            pipeline: Optional[PipelineWrapper] = None) -> pd.DataFrame:
        """
        Retrieves top k features according to FeatureImportanceAnalyzer.
        Return type: pd.DataFrame, with index features names and a column
        for their assigned weighted feature importance.
        """
        # FIA needs imputing
        pipeline.imputer = SimpleImputer() 
        
        # Create and fit the analyzer
        analyzer = FeatureImportanceAnalyzer(
            task_type=self.task,
            weights=self.imp_weights,
            preferred_gbm="xgboost",
            pipeline=pipeline,
            cv=self.cv,
            use_gpu=(self.device == "cuda")
        )
        analyzer.fit(X, y)
        
        # After analysis, drop imputing from pipeline
        pipeline.imputer = None 
        
        # Get unnormalized feature importances for more accurate feature weighting in sampling
        imp_df = analyzer.get_importance(normalize=False)[["weighted_importance"]]
        imp_df.sort_values(by="weighted_importance", axis=0, ascending=False, inplace=True)
        
        # Return all features or top k
        return imp_df if k == -1 else imp_df[:k]  # index: feats, col: weights of importance

    

    def _eval_baseline(self, 
                       X: pd.DataFrame, 
                       y: pd.Series,
                       pipeline: Optional[PipelineWrapper] = None) -> tuple[float, float]:
        """
        Evaluates given (X, y) state with baseline model.
        Returns mean_train_score, mean_val_score.
        """
        pipeline = pipeline.get_pipeline(X) if pipeline is not None else pipeline
        cv_dict =  cross_val_score(self.baseline_model, X, y, self.scorer, cv = self.cv,
                                   return_dict = True, pipeline = pipeline, 
                                   model_fit_kwargs = self.model_fit_kwargs)
        return cv_dict["mean_train_score"], cv_dict["mean_val_score"]
            



    def _softmax_temp_sampling(self, pool, weights, n=1, tau=0.5) -> list:
        """
        General sampling method.
        Samples items from a pool using softmax temperature sampling.
            
        Args:
            pool: Collection of items to sample from.
            weights: Corresponding weights for each item in pool.
            n: Number of items to sample. Returns entire pool if n > len(pool).
            tau: Temperature parameter, controls randomness (lower = more deterministic).
            
        Returns:
            List of n sampled items from pool.
        """
        if n > len(pool):
            return pool
        weights = np.array(weights)
        w = weights / tau
        w -= np.max(w)
        probs = np.exp(w) / np.sum(np.exp(w))
        return random.choices(pool, k = n, weights = probs)
    


        
    def _analyze_feature_interactions(self, X: pd.DataFrame, y: pd.Series, max_pairs: int = 200) -> Dict[tuple, float]:
        """
        Use SHAP interaction values to identify feature pairs with strong interactions.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series): Target variable
            max_pairs (int): Maximum number of feature pairs to consider
            
        Returns:
            dict: Dictionary with feature pairs as keys and interaction strength as values
        """        
        # Initialize the importance analyzer with appropriate parameters
        importance_analyzer = FeatureImportanceAnalyzer(
            task_type=self.task, 
            use_gpu=self.device == "gpu",
            verbose=0,
            n_jobs=-1,
            preferred_gbm='xgboost'
        )
        
        # Calculate interaction values
        interaction_values = importance_analyzer.get_feature_interactions(X, y, max_pairs=max_pairs)
        return interaction_values
    



    def _sample_parents(self, generation: List[Feature], n=20, tau=0.5) -> tuple[list[Feature], list[tuple[Feature, Feature]]]:
        """Sample n features for mutation and n feature pairs for crossover using weighted probability."""
        # Filter, initialize and return empty lists if no valid features
        generation = [f for f in generation if not f.require_pipeline]
        if not generation:
            return [], []
        
        # Initialize usage tracking and get state
        self._parent_usage = getattr(self, '_parent_usage', {})
        is_stagnating = hasattr(self, 'adaptive_controller') and \
                       self.adaptive_controller.state.stagnation_level in ['MODERATE', 'SEVERE']
        
        # Analyze feature families and detect imbalance
        family_counts = {}
        for f in generation:
            family = self._get_feature_family(f.name)
            family_counts[family] = family_counts.get(family, 0) + 1
        
        avg_family_size = len(generation) / max(1, len(family_counts))
        overrepresented = {f for f, c in family_counts.items() if c > avg_family_size * 1.5}
        family_to_features = {family: [] for family in family_counts}
        for f in generation:
            family_to_features[self._get_feature_family(f.name)].append(f)
        
        # Calculate weights with penalties for overused features and overrepresented families
        adjusted_weights = []
        for f in generation:
            usage = self._parent_usage.get(f.name, 0)
            family = self._get_feature_family(f.name)
            penalty = (0.1 if usage > 5 else 0.5 if usage > 3 else 1.0) * (0.5 if family in overrepresented else 1.0)
            adjusted_weights.append((f, f.weight * penalty))
        
        # Split features and calculate diversity parameters
        sorted_features = sorted(adjusted_weights, key=lambda x: x[1])
        split_idx = len(sorted_features) // 2
        bottom_half = [f for f, _ in sorted_features[:split_idx]]
        top_half = [f for f, _ in sorted_features[split_idx:]]
        diversity_pct = max(0.4 if is_stagnating else 0.25, 0.5 if overrepresented else 0)
        
        # PART 1: Select unary features with family diversity
        unary_features, selected_families = [], set()
        
        # Sample from bottom half (boost small families) and top half (boost unused families)
        if bottom_half:
            # Bottom half with small family boost
            weights = [f.weight * (1.0 + 1.0/max(1, family_counts[self._get_feature_family(f.name)])) for f in bottom_half]
            selected = self._softmax_temp_sampling(bottom_half, weights, int(n * diversity_pct), tau * 1.5)
            selected_families.update(self._get_feature_family(f.name) for f in selected)
            unary_features.extend(selected)
        
        # Fill remainder from top half (boost unused families)
        if len(unary_features) < n and top_half:
            weights = [f.weight * (2.0 if self._get_feature_family(f.name) not in selected_families else 0.5) for f in top_half]
            unary_features.extend(self._softmax_temp_sampling(top_half, weights, min(n - len(unary_features), len(top_half)), tau))
        
        # PART 2: Create binary feature pairs with cross-family preference
        feature_pairs = []
        families = list(family_counts.keys())
        
        # Create cross-family pairs
        if len(families) >= 2:
            cross_family_count = int(n * (0.7 if is_stagnating else 0.5))
            while len(feature_pairs) < cross_family_count and len(feature_pairs) < n:
                f1_family, f2_family = random.sample(families, 2)
                if family_to_features[f1_family] and family_to_features[f2_family]:
                    feature_pairs.append((random.choice(family_to_features[f1_family]), 
                                         random.choice(family_to_features[f2_family])))
        
        # Use SHAP interaction data if available
        remaining = n - len(feature_pairs)
        if remaining > 0 and hasattr(self, 'feature_interactions') and self.feature_interactions:
            name_to_feature = {f.name: f for f in generation}
            valid_pairs = []
            
            for (f1_name, f2_name), interaction in self.feature_interactions.items():
                if f1_name in name_to_feature and f2_name in name_to_feature:
                    f1, f2 = name_to_feature[f1_name], name_to_feature[f2_name]
                    # Apply usage and family penalties
                    usage_penalty = 0.1 if (self._parent_usage.get(f1_name, 0) > 3 and 
                                          self._parent_usage.get(f2_name, 0) > 3) else 1.0
                    family_penalty = 0.3 if self._get_feature_family(f1_name) == self._get_feature_family(f2_name) else 1.0
                    weight = (f1.weight * f2.weight * interaction * usage_penalty * family_penalty) ** 0.5
                    valid_pairs.append(((f1, f2), weight))
            
            if valid_pairs:
                pairs, weights = zip(*valid_pairs)
                feature_pairs.extend(self._softmax_temp_sampling(list(pairs), list(weights), 
                                                               min(int(remaining * 0.7), len(pairs)), tau))
        
        # Fill remaining with diverse random pairs
        while len(feature_pairs) < n:
            # Try to find cross-family pair with limited attempts
            for _ in range(3):
                f1, f2 = random.sample(generation, 2)
                if self._get_feature_family(f1.name) != self._get_feature_family(f2.name):
                    feature_pairs.append((f1, f2))
                    break
            else:
                # Fallback to any pair if needed
                feature_pairs.append(tuple(random.sample(generation, 2)))
            
            if len(feature_pairs) >= n:
                break
        
        # Update usage tracking
        for f in unary_features:
            self._parent_usage[f.name] = self._parent_usage.get(f.name, 0) + 1
        for f1, f2 in feature_pairs:
            self._parent_usage[f1.name] = self._parent_usage.get(f1.name, 0) + 1
            self._parent_usage[f2.name] = self._parent_usage.get(f2.name, 0) + 1
            
        return unary_features, feature_pairs

   



    def _sample_children(self,
                         candidates_pool: List[Interaction], n = 200, 
                         beta = 0.2, gamma = 0.1, lambda_ = 0.2, delta = 0.3, tau = 0.7) -> list[Interaction]:
        """
        Sample candidate interactions using the adaptive controller's prioritization.
        """
        if not candidates_pool:
            return []

        # Use adaptive controller to prioritize operations
        prioritized_candidates = []
        
        # Group candidates by operation type for prioritization
        op_groups = {}
        for interaction in candidates_pool:
            dtype = interaction.dtype
            op_type = interaction.type
            key = (dtype, op_type)
            op_groups.setdefault(key, []).append(interaction)

        # Apply operation prioritization within each group
        for (dtype, op_type), group in op_groups.items():
            prioritized_ops = self.adaptive_controller.get_prioritized_operations(dtype, op_type)
            if prioritized_ops:
                # Sort group by operation priority, then by feature weight
                group_sorted = sorted(group, 
                    key=lambda i: (
                        prioritized_ops.index(i.op) if i.op in prioritized_ops else len(prioritized_ops),
                        -i.weight  # Higher weight features preferred
                    ))
                prioritized_candidates.extend(group_sorted)
            else:
                # Fallback: sort by feature weight only
                group_sorted = sorted(group, key=lambda i: -i.weight)
                prioritized_candidates.extend(group_sorted)

        # Fallback if prioritization fails
        if not prioritized_candidates:
            prioritized_candidates = candidates_pool

        # Use top candidates first, then add diversity via sampling
        if n >= len(prioritized_candidates):
            return prioritized_candidates
            
        # Take top portion deterministically, sample rest for diversity
        top_portion = int(n * 0.6)  # 60% top candidates
        sample_portion = n - top_portion
        
        result = prioritized_candidates[:top_portion]
        if sample_portion > 0:
            # Sample remaining with weights from the rest
            remaining = prioritized_candidates[top_portion:]
            weights = [interaction.weight for interaction in remaining]
            sampled = self._softmax_temp_sampling(remaining, weights, sample_portion, tau)
            result.extend(sampled)
            
        return result
    


    def _prepare_pipeline(self, interactions: List[Interaction]) -> PipelineWrapper:
        """
        From a given list of interactions, prepare, if needed
        the PipelineWrapper needed for encoding operations.
        """
        target_enc_cols, count_enc_cols, freq_enc_cols = [], [], []
        for interaction in interactions:
            op = interaction.op
            feat = interaction.feature_1.name
            if op == "target":
                target_enc_cols.append(feat)
            elif op == "count":
                count_enc_cols.append(feat)
            elif op == "freq":
                freq_enc_cols.append(feat)
        if target_enc_cols or count_enc_cols or freq_enc_cols:
            return PipelineWrapper(imputer = None,
                                   scaler = None,
                                   encoder = CategoricalEncoder(target_enc_cols,
                                                                count_enc_cols,
                                                                freq_enc_cols))
        return PipelineWrapper(imputer = None,
                               scaler = None,
                               encoder = CategoricalEncoder())
    



    def _extend_pipeline(self, pipeline: PipelineWrapper, new_pipeline: PipelineWrapper) -> PipelineWrapper:
        """
        Extends pipeline with new_pipeline. Used for categorical encoding features, where
        PipelineWrapper's encoder is  as needed.
        """
        return PipelineWrapper(
            imputer = None,
            scaler = None,
            encoder = CategoricalEncoder(
                target_enc_cols = list(set(pipeline.encoder.target_enc_cols 
                + new_pipeline.encoder.target_enc_cols)),
                count_enc_cols = list(set(pipeline.encoder.count_enc_cols 
                + new_pipeline.encoder.count_enc_cols)),
                freq_enc_cols = list(set(pipeline.encoder.freq_enc_cols 
                + new_pipeline.encoder.freq_enc_cols)),
            )
        )
        
    


    def _apply_interactions(self, X: pd.DataFrame, 
                            interactions: List[Interaction]) -> tuple[pd.DataFrame, PipelineWrapper]:
        """
        Applies non-pipeline feature interactions to X and returns the updated DataFrame 
        along with a PipelineWrapper for deferred transformations.
        """
        # Generate all features first
        new_features = {}
        for interaction in interactions:
            if not interaction.require_pipeline:
                # Check if required features exist in DataFrame
                required_features = [interaction.feature_1.name]
                if interaction.feature_2 is not None:
                    required_features.append(interaction.feature_2.name)
                
                # Skip interaction if any required feature is missing
                if not all(feat in X.columns for feat in required_features):
                    continue
                    
                name, val = interaction.generate(X)
                if name not in X.columns:
                    new_features[name] = val
        # Join all new features at once
        if new_features:
            X_copy = pd.concat([X.copy(), pd.DataFrame(new_features)], axis=1)
        else:
            X_copy = X.copy()
        pipeline = self._prepare_pipeline(interactions)
        return X_copy, pipeline
    
    


    def _multi_criteria_ranking(self, batch: list[Interaction], X: pd.DataFrame, y: pd.Series) -> list[Interaction]:
        """
        Ranks candidate features using the adaptive controller's unified ranking system.
        """
        return self.adaptive_controller.rank_candidates_with_adaptive_criteria(batch, X, y)

    def _select_elites(self, batch: list[Interaction], n: int,
                   X: pd.DataFrame, y: pd.Series,
                   callback: Optional[Callable] = None
                   ) -> tuple[list[Interaction], pd.DataFrame, PipelineWrapper]:
        """ Greedy forward-selection of ≤ n interactions (multi-criteria ranking + early-stop). """

        # ── early exit ──────────────────────────────────────────────────────────
        if not batch:
            if callback: callback(0, 0, force_complete=True)
            return [], X, self.pipeline

        # ── filter out interactions with missing features or blacklisted ──────────────
        valid_batch = []
        for interaction in batch:
            required_features = [interaction.feature_1.name]
            if interaction.feature_2 is not None:
                required_features.append(interaction.feature_2.name)
            
            # Check if this would create a blacklisted feature
            skip_interaction = False
            if hasattr(self, 'blacklisted_features') and interaction.name in self.blacklisted_features:
                skip_interaction = True
                
            # Only include interaction if all required features exist and not blacklisted
            if all(feat in X.columns for feat in required_features) and not skip_interaction:
                valid_batch.append(interaction)
        
        # Update batch with valid interactions only
        batch = valid_batch
        
        if not batch:
            if callback: callback(0, 0, force_complete=True)
            return [], X, self.pipeline

        # ── prepare batch features & pipeline ──────────────────────────────────
        X_copy, pipe_batch = self._apply_interactions(X, batch)
        pipe_ext = self._extend_pipeline(self.pipeline, pipe_batch)

        # ── rank candidates using multi-criteria approach ─────────────────────────
        if hasattr(self, 'ranking_method') and self.ranking_method == 'shap':
            # Original SHAP-based ranking if explicitly requested
            fi = FeatureImportanceAnalyzer(weights={"shap": 1.0},
                                        task_type=self.task, cv=self.cv,
                                        pipeline=pipe_ext)
            fi.fit(X_copy, y)
            # Get unnormalized SHAP values for accurate feature importance ranking
            shap = fi.get_importance(normalize=False)
            shap = shap[shap.index.isin([i.name for i in batch])]

            ranked = sorted(batch,
                            key=lambda i: shap.loc[i.name, "weighted_importance"]
                            if i.name in shap.index else -float("inf"),
                            reverse=True)
        elif hasattr(self, 'ranking_method') and self.ranking_method == 'none':
            # Use original order if ranking is explicitly disabled
            ranked = batch
        else:
            # Use multi-criteria ranking by default (recommended approach)
            ranked = self._multi_criteria_ranking(batch, X, y)

        # ── baseline ───────────────────────────────────────────────────────────
        _base_train, best_val = self._eval_baseline(X, y, self.pipeline)
        selected: list[Interaction] = []
        X_base = X.copy()
        evals = consec_no_gain = 0
        min_evals = max(5, int(0.05 * len(ranked)))
        early_thr = (int(len(ranked) * self.early_stopping_child_eval)
                    if isinstance(self.early_stopping_child_eval, float)
                    else self.early_stopping_child_eval
                    if isinstance(self.early_stopping_child_eval, int)
                    else len(ranked))

        # ── main loop ──────────────────────────────────────────────────────────
        for inter in ranked:
            evals += 1
            if callback:
                callback(evals, len(selected))  # progress tick first

            if len(selected) >= n:  # quota hit
                if callback:
                    callback(len(ranked), len(selected), force_complete=True)
                break

            # Check if required features exist in DataFrame
            required_features = [inter.feature_1.name]
            if inter.feature_2 is not None:
                required_features.append(inter.feature_2.name)
            
            # Skip interaction if any required feature is missing
            if not all(feat in X_base.columns for feat in required_features):
                continue

            # trial dataframe
            if inter.require_pipeline or inter.name not in X_copy.columns:
                X_try = X_base
            else:
                X_try = X_base.copy()
                X_try[inter.name] = X_copy[inter.name].values

            # trial pipeline & score
            pipe_iter = self._extend_pipeline(self.pipeline, self._prepare_pipeline([inter] + selected))
            _new_train, new_val = self._eval_baseline(X_try, y, pipe_iter)

            delta = (new_val - best_val) if self.scorer.greater_is_better else (best_val - new_val)
            gain = delta / (abs(best_val) + 1e-8)
            success = gain >= self.adaptive_controller.get_adaptive_min_gain()
            
            # Update adaptive controller with result
            self.adaptive_controller.update_operation_stats(inter, success=success, gain=gain)

            if success:  # keep
                selected.append(inter)
                X_base, best_val, consec_no_gain = X_try, new_val, 0
            else:
                consec_no_gain += 1

            if evals >= min_evals and consec_no_gain >= early_thr:  # early stop
                if callback:
                    callback(len(ranked), len(selected), force_complete=True)
                break

        if callback and evals < len(ranked):
            callback(len(ranked), len(selected), force_complete=True)

        # ── final pipeline ─────────────────────────────────────────────────────
        final_pipe = self._extend_pipeline(self.pipeline,
                                        self._prepare_pipeline(selected))
        return selected, X_base, final_pipe



        

    def _get_search_parameters(self, progress: float, generation_num: int) -> tuple[float, float, float, float]:
        """
        Get unified search parameters using the adaptive controller.
        """
        # Update adaptive controller's stagnation assessment
        self.adaptive_controller.assess_stagnation(
            self.state['counters']['no_feature_gens_count'],
            self.state['counters']['consecutive_no_improvement_iters']
        )
        
        # Get coordinated parameters
        return self.adaptive_controller.get_search_parameters(progress)
    


    def _apply_adaptive_mechanisms(self, generation: list[Feature]) -> dict:
        """
        Apply all adaptive mechanisms through the unified controller.
        Returns a dictionary with logging information instead of logging directly.
        """
        if not self.adaptive:
            return {}
            
        # Let the adaptive controller handle feature weight adjustments
        modifications = self.adaptive_controller.adapt_feature_weights(generation)
        
        if modifications > 0:
            status = self.adaptive_controller.get_status_summary()
            return {
                'modifications': modifications,
                'stagnation_level': status['stagnation_level'],
                'consecutive_success': status['consecutive_success'],
                'exploration_intensity': status['exploration_intensity'],
                'min_gain_factor': status['min_gain_factor']
            }
        return {}


    ###
    def _get_feature_family(self, feature_name: str) -> str:
        """Get the root/family name of a feature by extracting the original column name."""
        # Split on common separators and take the first part as the family
        for sep in self.ops["num"]["binary"] + self.ops["cat"]["binary"]:
            if sep in feature_name:
                return feature_name.split(sep)[0]
        return feature_name
    
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
    
    def _prune_weak_features(self, X: pd.DataFrame, y: pd.Series, generation: list, n_remove: int = 2, prune_pct: float = 0.1) -> tuple[pd.DataFrame, PipelineWrapper, list]:
        """
        Remove weakest features based on permutation importance.
        
        Parameters:
        -----------
        X : pd.DataFrame
            The feature dataframe
        y : pd.Series
            Target values
        generation : list
            List of Feature objects
        n_remove : int, default=2
            Fixed number of features to remove (used if prune_pct is None)
        prune_pct : float, default=0.1
            Percentage of features to remove (has priority over n_remove)
            
        Returns:
        --------
        tuple
            (pruned dataframe, pipeline, updated generation list)
        """
        if X.shape[1] <= len(self.initial_features) + 2:  # Don't prune if too few features
            return X, self.pipeline, generation
            
        # Get feature importances
        weights = self._get_top_k_features(X, y, k=-1, pipeline=self.pipeline)
        
        # Find weakest non-original features
        new_features = [col for col in X.columns if col not in self.initial_features]
        
        # Calculate number of features to remove based on percentage
        if prune_pct is not None:
            features_to_remove = max(1, int(len(new_features) * prune_pct))
        else:
            features_to_remove = n_remove
            
        if len(new_features) < features_to_remove:
            return X, self.pipeline, generation
        
        # Get dependencies to prevent removal of features other features rely on
        deps = self._get_feature_dependencies(generation)
        
        # Flatten dependencies to get all features that are depended upon
        protected_features = set()
        for parent_feat, dependent_list in deps.items():
            protected_features.add(parent_feat)  # Protect the parent feature
        
        # Sort new features by importance and filter out those that are protected
        new_feat_weights = weights[weights.index.isin(new_features)].sort_values('weighted_importance')
        candidates = [f for f in new_feat_weights.index if f not in protected_features]
        
        # Get the features to remove (weakest first, respecting features_to_remove limit)
        to_remove = candidates[:min(features_to_remove, len(candidates))]
        
        # Log dependency protection if needed
        if len(to_remove) < features_to_remove and len(protected_features) > 0:
            self._log(f"  Only pruning {len(to_remove)} features (not {features_to_remove}) due to {len(protected_features)} protected dependencies")
            
        # If no features can be pruned, return original
        if not to_remove:
            self._log("  No features can be pruned due to dependencies")
            return X, self.pipeline, generation
                
        # Track blacklisted features to prevent re-adding them later
        if not hasattr(self, 'blacklisted_features'):
            self.blacklisted_features = set()
        
        # For features pruned multiple times, add to blacklist
        for feat_name in to_remove:
            if feat_name in getattr(self, 'previously_pruned_features', set()):
                self.blacklisted_features.add(feat_name)
                
        # Remember currently pruned features for next time
        if not hasattr(self, 'previously_pruned_features'):
            self.previously_pruned_features = set(to_remove)
        else:
            self.previously_pruned_features.update(to_remove)
            
        # Store pruned features separately (including original features) for later use in transform
        if not hasattr(self, 'pruned_features'):
            self.pruned_features = set(to_remove)
        else:
            self.pruned_features.update(to_remove)
            
        X_pruned = X.drop(columns=to_remove)
        self._log(f"  Pruned weak features: {to_remove}")
        
        # Update pipeline to remove references to pruned features
        updated_pipeline = deepcopy(self.pipeline)
        for feature_name in to_remove:
            # Remove from encoder columns if they exist
            if feature_name in updated_pipeline.encoder.target_enc_cols:
                updated_pipeline.encoder.target_enc_cols.remove(feature_name)
            if feature_name in updated_pipeline.encoder.count_enc_cols:
                updated_pipeline.encoder.count_enc_cols.remove(feature_name)
            if feature_name in updated_pipeline.encoder.freq_enc_cols:
                updated_pipeline.encoder.freq_enc_cols.remove(feature_name)
        
        # Notify adaptive controller about pruned features (mark as unsuccessful)
        for feat in generation:
            if feat.name in to_remove and hasattr(feat, 'generating_interaction') and feat.generating_interaction:
                self.adaptive_controller.update_operation_stats(feat.generating_interaction, success=False, gain=-0.001)
        
        # Update generation list to remove pruned features
        updated_generation = [feat for feat in generation if feat.name not in to_remove]
        
        return X_pruned, updated_pipeline, updated_generation
    
    def _hopeful_monster_generation(self, X: pd.DataFrame, y: pd.Series, generation: list, 
                               n_features: int = 10, callback: Optional[Callable] = None) -> tuple[list, pd.DataFrame, PipelineWrapper]:
        """High-risk, high-reward generation using only under-utilized parents."""
        # Filter generation to only include features that exist in current DataFrame
        valid_generation = [feat for feat in generation if feat.name in X.columns]
        
        # Apply max features constraint
        remaining_budget = self.max_gen_new_feats - self.state['counters']['total_new_features'] if self.max_gen_new_feats != float('inf') else float('inf')
        max_features_to_find = min(n_features, remaining_budget) if remaining_budget > 0 else 0
        
        if max_features_to_find <= 0:
            self._log(f"  Hopeful monster: No remaining feature budget ({remaining_budget})")
            return [], X, self.pipeline
        
        # Initialize callback if provided
        if callback:
            callback(0, 0)
        
        # Group features by their root/family name to encourage diversity
        feature_families = {}
        for feat in valid_generation:
            family = self._get_feature_family(feat.name)
            if family not in feature_families:
                feature_families[family] = []
            feature_families[family].append(feat)
        
        # Enforce diversity by limiting overrepresented feature families
        MAX_FEATURES_PER_FAMILY = max(2, len(valid_generation) // (len(feature_families) or 1))
        diverse_features = []
        for family, feats in feature_families.items():
            # Take at most MAX_FEATURES_PER_FAMILY from each family, prioritizing by weight
            sorted_family = sorted(feats, key=lambda f: f.weight, reverse=True)
            diverse_features.extend(sorted_family[:MAX_FEATURES_PER_FAMILY])
            
        # Sort by weight for the bottom 30% selection
        sorted_features = sorted(diverse_features, key=lambda f: f.weight)
        bottom_30_pct = sorted_features[:max(1, len(sorted_features) // 3)]
        
        if len(bottom_30_pct) < 2:
            return [], X, self.pipeline
            
        # Generate candidates from low-weight parents only
        candidates_pool = []
        for feat in bottom_30_pct[:self.n_parents // 2]:
            candidates_pool.extend([Interaction(feat, op) for op in self.ops[feat.dtype]["unary"]])
        
        # Force diverse pairings across different feature families when possible
        family_samples = list(feature_families.keys())
        for _ in range(min(self.n_parents // 2, len(bottom_30_pct) // 2)):
            # Try to sample from different families
            if len(family_samples) >= 2:
                family1, family2 = random.sample(family_samples, 2)
                f1 = random.choice(feature_families[family1])
                f2 = random.choice(feature_families[family2])
            else:
                f1, f2 = random.sample(bottom_30_pct, 2)
                
            op_list = self.ops["num" if f1.dtype == f2.dtype == "num" else "cat"]["binary"]
            candidates_pool.extend([Interaction(f1, op, f2) for op in op_list])
        
        # Filter out blacklisted feature patterns
        if hasattr(self, 'blacklisted_features') and self.blacklisted_features:
            filtered_pool = []
            for interaction in candidates_pool:
                # Skip if this would generate a blacklisted feature
                if interaction.name in self.blacklisted_features:
                    continue
                filtered_pool.append(interaction)
            candidates_pool = filtered_pool
        
        # Sample and evaluate with higher quota
        batch = self._sample_children(candidates_pool, self.n_children, 0.3, 0.4, 0.3, 1.0)
        
        # Forward the provided callback or use a placeholder
        if not callback:
            # Create a default silent callback if none provided
            def monster_callback(evaluated_count, selected_count, force_complete=False):
                pass  # Silent progress for hopeful monster
            actual_callback = monster_callback
        else:
            actual_callback = callback
            
        # Log the start of elite selection
        self._log(f"  Hopeful monster: evaluating {len(batch)} candidates...")
        elites, X_new, pipeline_new = self._select_elites(batch, max_features_to_find, X, y, callback=actual_callback)
        
        # Don't log success/failure here - let the caller determine that based on final score comparison
        # We only log the basic statistics about the evaluation
        self._log(f"  Hopeful monster: found {len(elites)} features from {len(batch)} candidates")
        return elites, X_new, pipeline_new

    def _beam_search_elites(self, batch: list, k: int, X: pd.DataFrame, y: pd.Series, 
                         callback: Optional[Callable] = None) -> tuple[list, pd.DataFrame, PipelineWrapper]:
        """Small-scale beam search keeping top-k feature sets."""
        if not batch or k <= 1:
            self._log(f"  Beam search skipped (batch size: {len(batch)}, k: {k}). Using standard elite selection instead.")
            return self._select_elites(batch, min(k, len(batch)), X, y, callback=callback)
            
        # Start with single best feature
        self._log(f"  Beam search: ranking {len(batch)} candidates...")
        ranked = self._multi_criteria_ranking(batch, X, y)
        best_sets = []  # List of (features_list, X_df, pipeline, score)
        
        # Apply early stopping and max features constraints
        remaining_budget = self.max_gen_new_feats - self.state['counters']['total_new_features'] if self.max_gen_new_feats != float('inf') else float('inf')
        max_features_to_find = min(k, remaining_budget) if remaining_budget > 0 else 0
        
        # Early stopping threshold for beam search
        early_thr = (int(len(ranked) * self.early_stopping_child_eval)
                    if isinstance(self.early_stopping_child_eval, float)
                    else self.early_stopping_child_eval
                    if isinstance(self.early_stopping_child_eval, int)
                    else len(ranked))
        
        if max_features_to_find <= 0:
            self._log(f"  Beam search: No remaining feature budget ({remaining_budget})")
            return [], X, self.pipeline
        
        # Report initial progress
        if callback:
            callback(0, 0)
        
        # Initialize with top individual features
        evaluations_done = 0
        consec_no_gain = 0
        min_evals = max(3, k)  # Minimum evaluations for beam search
        
        for i in range(min(k, len(ranked))):
            feat = ranked[i]
            X_try, pipe_try = self._apply_interactions(X, [feat])
            pipe_ext = self._extend_pipeline(self.pipeline, pipe_try)
            _, score = self._eval_baseline(X_try, y, pipe_ext)
            best_sets.append(([feat], X_try, pipe_ext, score))
            
            # Update progress
            evaluations_done += 1
            if callback:
                callback(evaluations_done, len(best_sets))
        
        # Expand each set by trying to add one more feature
        self._log(f"  Beam search: initialized with {len(best_sets)} seed features, expanding combinations...")
        final_features = []
        best_score = float('-inf') if self.scorer.greater_is_better else float('inf')
        best_X, best_pipeline = X, self.pipeline
        features_found = 0
        
        for feature_set, X_set, pipe_set, _ in best_sets:
            if features_found >= max_features_to_find:
                break
                
            for candidate in ranked:
                if candidate in feature_set:
                    continue
                
                # Check early stopping conditions
                if evaluations_done >= early_thr and evaluations_done >= min_evals:
                    self._log(f"  Beam search early stopping: {evaluations_done} evaluations reached threshold")
                    break
                
                # Check if time budget is exceeded via callback
                if callback and callback(evaluations_done, features_found, False):
                    self._log(f"  Beam search interrupted: time budget exceeded after {evaluations_done} evaluations")
                    break
                    
                # Try adding this candidate to the current set
                test_set = feature_set + [candidate]
                X_test, pipe_test = self._apply_interactions(X, test_set)
                pipe_ext = self._extend_pipeline(self.pipeline, pipe_test)
                _, test_score = self._eval_baseline(X_test, y, pipe_ext)
                
                # Update progress tracking
                evaluations_done += 1
                
                # Keep if better than current best
                is_better = (test_score > best_score) if self.scorer.greater_is_better else (test_score < best_score)
                if is_better:
                    best_score = test_score
                    final_features = test_set
                    best_X, best_pipeline = X_test, pipe_ext
                    features_found = len(final_features)
                    consec_no_gain = 0
                    
                    # Save local best state but don't update global best state yet
                    # That will happen in the main search loop if the beam search actually improves the global best
                    # We only update the instance attributes here to track the current feature set
                    self.X = best_X.copy()
                    self.pipeline = best_pipeline
                    
                    # Check against global best, but don't update the global best state yet
                    # We'll let the main search loop handle that after comparing scores
                    orig_score = self.state['best']['val_score']
                    is_better_global = (best_score > orig_score) if self.scorer.greater_is_better else (best_score < orig_score)
                    if is_better_global and self.save_path:
                        # Before saving, ensure we have rebuilt interactions from these features
                        self.interactions = []
                        for interaction in test_set:
                            self.interactions.append(interaction)
                        self.save(self.save_path)
                    
                    # Report feature selection through callback
                    if callback:
                        callback(evaluations_done, features_found)
                else:
                    consec_no_gain += 1
                
                # Break if we've found enough features
                if features_found >= max_features_to_find:
                    self._log(f"  Beam search: reached max features limit ({max_features_to_find})")
                    break
        
        orig_score = self.state['best']['val_score']
        score_diff = abs(best_score - orig_score)
        
        if len(final_features) > 0:
            self._log(f"  Beam search complete: found {len(final_features)} features with score: {best_score:.5f} after {evaluations_done} evaluations")
            is_better = (best_score > orig_score) == self.scorer.greater_is_better
            is_better_text = "better" if is_better else "worse"
            improvement_direction = "higher" if self.scorer.greater_is_better else "lower"
            self._log(f"  Score is {score_diff:.5f} points {is_better_text} than original: {orig_score:.5f} ({improvement_direction} is better for {self.scorer.name})")
        
        # Final progress report
        if callback:
            callback(evaluations_done, len(final_features), True)
            
        return final_features, best_X, best_pipeline

    ###

    def search(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, PipelineWrapper, list[Feature], list[Interaction]]:
        """
        Searches new features using a genetic algorithm approach.
        This method implements an evolutionary algorithm that iteratively creates, 
        evaluates, and selects features to improve model performance. It uses 
        unary and binary operations to create candidate features, evaluates their 
        impact on model performance, and selectively adds the most beneficial ones.
        Parameters
        ----------
        X : pandas.DataFrame
            Input features dataset
        y : pandas.Series
            Target variable
        Returns
        -------
        pandas.DataFrame
            Enhanced dataset with generated features, where possible without pipeline
        Pipeline
            Pipeline containing encoding transformations
        list
            Final generation of Feature objects. Includes all features (original + all generated features)
        list
            Includes all used Interaction objects to generate new features
        Notes
        -----        - Uses adaptive exploration strategies if enabled
        - Implements early stopping if no improvement is observed
        - Respects time budget if specified
        - Tracks performance metrics throughout generations
        - Maintains global counters for operation frequency to encourage diversity
        - Can revert to best generation if later generations don't improve
        - Uses a progress bar (tqdm)
        """

        # Initialize start time for time budget tracking, set defaults, infer dtypes
        start_time = time.time()
        self._set_defaults(X, y)
        self.initial_features = list(X.columns)
        num_cols, cat_cols = self._get_num_cat_cols(X)
        self.max_gen_new_feats = int(self.max_new_feats * len(self.initial_features)) if isinstance(self.max_new_feats, float) \
                                 else self.max_new_feats if isinstance(self.max_new_feats, int) \
                                 else float('inf') # depends on number of features in X, can't be handled in the constructor

        # Ensure the target is label encoded as expected by GBMs (if needed)
        if self.task != "regression":
            unique_vals = np.unique(y)
            if not np.array_equal(unique_vals, np.arange(len(unique_vals))):
                self._log("Factorizing labels...")
                y, _ = y.factorize()
                self._log("Done.")
            else:
                self._log("Label factorization not needed.")
        
        # Initial logging
        self._log(f"Starting feature generation - Task: {self.task}, Device: {self.device}\n")
        self._log(f"Params: generations = {self.n_generations}, parents = {self.n_parents}, children = {self.n_children}, min_gain = {self.min_pct_gain}")
        if self.time_budget:
            self._log(f"Time budget: {self.time_budget} seconds")
        self._log(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, " + 
                  f"Limit: {self.max_gen_new_feats if self.max_gen_new_feats != float('inf') else 'Unlimited'} new features max")
        
        # Initialize adaptive controller
        self.pruned_features = set()  # Reset pruned features for new run
        self.adaptive_controller.initialize_operations(self.ops)
        self.adaptive_controller.reset_for_new_run()
        self.state['best']['train_score'], self.state['best']['val_score'] = self._eval_baseline(X, y, self.pipeline)
        self._log(f"Gen 0: Training with original features. Mean Train {self.scorer.name} = {self.state['best']['train_score']:.5f}. " +
                  f"Mean Val {self.scorer.name} = {self.state['best']['val_score']:.5f}")
        self.state['best']['X'], self.state['best']['pipeline'] = X.copy(), deepcopy(self.pipeline)
        self.state['best']['pruned_features'] = getattr(self, 'pruned_features', set()).copy()
        
        # Analyze feature interactions using SHAP
        self.feature_interactions = self._analyze_feature_interactions(X, y, max_pairs=10000)
        
        # Generation 0: top (2 * n_parents original) features
        top_feats_df = self._get_top_k_features(X, y, k = 2 * self.n_parents, pipeline = self.pipeline)
        generation = [Feature(name = feat, dtype = "num" if feat in num_cols else "cat", 
                              weight = top_feats_df.loc[feat, "weighted_importance"], 
                            #   depth=0
                              ) for feat in top_feats_df.index]
        self.state['best']['generation'] = generation.copy()
        


        # Main genetic algorithm loop
        stagnation_counter = 0  # Track consecutive generations without improvement
        with tqdm(total=self.n_generations, desc = "Generations") as pbar:
            for N in range(self.n_generations):

                # Check if time budget is exceeded
                if self.time_budget and (time.time() - start_time) > self.time_budget:
                    self._log(f"Time budget of {self.time_budget} seconds exceeded. Stopping optimization.")
                    break 
                
                # Get adaptive search parameters
                progress = N / self.n_generations
                tau, beta, gamma, lambda_ = self._get_search_parameters(progress, N)
                
                # Apply adaptive mechanisms if needed (collect info for later logging)
                adaptive_adjustments = self._apply_adaptive_mechanisms(generation)
                
                # Archive & Prune: Remove weak features after 5 stagnant generations
                if stagnation_counter >= 5:
                    stagnation_before_prune = stagnation_counter
                    # Prune 10% of features instead of fixed number
                    X, self.pipeline, generation = self._prune_weak_features(X, y, generation, prune_pct=0.1)
                    # Update feature counter after pruning
                    self.state['counters']['total_new_features'] = X.shape[1] - len(self.initial_features)
                    stagnation_counter = 0  # Reset counter after pruning
                    self._log(f"  Applied archive & prune after {stagnation_before_prune} stagnant generations")
                
                # Hopeful Monster: High-risk strategy during severe stagnation
                adaptive_status = self.adaptive_controller.get_status_summary()
                hopeful_monster_success = False
                if adaptive_status['stagnation_level'] == 'SEVERE' and random.random() < 0.3:  # 30% chance
                    self._log(f"  Attempting hopeful monster generation...")
                    
                    # Create a callback for hopeful monster progress
                    def monster_update_callback(evaluated_count, selected_count, force_complete=False):
                        # Check time budget during feature evaluation
                        if self.time_budget and (time.time() - start_time) > self.time_budget:
                            return True  # Signal to stop evaluation
                        return False
                    
                    monster_elites, X_monster, pipe_monster = self._hopeful_monster_generation(X, y, generation, n_features=5, callback=monster_update_callback)
                    if monster_elites:
                        # Evaluate if monster generation is better than the best state
                        _, monster_score = self._eval_baseline(X_monster, y, pipe_monster)
                        best_score = self.state['best']['val_score']
                        is_better = (monster_score > best_score) if self.scorer.greater_is_better else (monster_score < best_score)
                        if is_better:
                            X, self.pipeline = X_monster, pipe_monster
                            elites = monster_elites
                            hopeful_monster_success = True
                            new_feature_names = [elite.name for elite in elites]
                            self._log(f"  Hopeful monster: successful! Improved score from {best_score:.5f} to {monster_score:.5f}, selected {len(elites)} features")
                            self._log(f"  Hopeful monster features: {new_feature_names}")
                            
                            # ... continue with the normal flow after elite selection
                            new_generation = generation.copy()
                            for interaction in elites:
                                feat = interaction.get_new_feature_instance()
                                feat.set_generating_interaction(interaction)
                                new_generation.append(feat)
                            generation = new_generation
                            # Always sync the counter with actual feature count
                            self.state['counters']['total_new_features'] = X.shape[1] - len(self.initial_features)
                            self.state['counters']['no_feature_gens_count'] = 0
                            # Evaluate and update best state
                            new_train_score, new_val_score = self._eval_baseline(X, y, self.pipeline)
                            delta = new_val_score - self.state['best']['val_score'] if self.scorer.greater_is_better else self.state['best']['val_score'] - new_val_score
                            
                            # Update best state since monster was better
                            self.state['best']['gen_num'], self.state['best']['val_score'], self.state['best']['X'] = N + 1, new_val_score, X.copy()
                            self.state['best']['generation'], self.state['best']['pipeline'] = generation.copy(), self.pipeline
                            self.state['counters']['consecutive_no_improvement_iters'] = 0
                            stagnation_counter = 0
                            
                            # Update instance attributes to match best state
                            self.X, self.pipeline, self.generation = X.copy(), self.pipeline, generation.copy()
                            # Collect interactions from generation
                            self.interactions = []
                            for feat in self.generation:
                                if hasattr(feat, 'generating_interaction') and feat.generating_interaction:
                                    self.interactions.append(feat.generating_interaction)
                            
                            # Save best state if save_path is provided
                            if self.save_path:
                                self.save(self.save_path)
                            
                            # Update feature interactions after hopeful monster success
                            self.feature_interactions = self._analyze_feature_interactions(X, y, max_pairs=10000)
                            
                            # Notify adaptive controller of hopeful monster success
                            for elite in elites:
                                self.adaptive_controller.update_operation_stats(elite, success=True, gain=delta/(abs(best_score) + 1e-8))
                            
                            # We'll log the standard generation info after the hopeful monster section
                        else:
                            self._log(f"  Hopeful monster: features selected individually but didn't improve global best score")
                            self._log(f"  (Selected {len(monster_elites)} features, best: {best_score:.5f}, monster: {monster_score:.5f})")
                    else:
                        self._log(f"  Hopeful monster: no features found during evaluation")
                
                # Skip normal generation process if hopeful monster was successful
                if not hopeful_monster_success:
                    # Parents sampling selection
                    unary, binary = self._sample_parents(generation, n = self.n_parents, tau = tau)

                    # Filter parents to only include features that exist in current DataFrame
                    valid_unary = [feat for feat in unary if feat.name in X.columns]
                    valid_binary = [(feat1, feat2) for feat1, feat2 in binary 
                                   if feat1.name in X.columns and feat2.name in X.columns]

                    # Create candidate interactions pool
                    candidates_pool = []
                    for feat in valid_unary:
                        self.state['seen_feats'].add(feat)
                        candidates_pool.extend([Interaction(feat, op) for op in self.ops[feat.dtype]["unary"]])
                    for feat1, feat2 in valid_binary:
                        self.state['seen_feats'].update({feat1, feat2})
                        op_list = self.ops["num" if feat1.dtype == feat2.dtype == "num" else "cat"]["binary"]
                        candidates_pool.extend([Interaction(feat1, op, feat2) for op in op_list])

                    # Sample children and run selection
                    batch = self._sample_children(candidates_pool, self.n_children, beta, gamma, lambda_, tau)
                    pbar.set_description(f"Gen {N+1}: Testing batch of {len(batch)} candidates children")
                
                    # Calculate remaining feature budget for this generation
                    remaining_budget = self.max_gen_new_feats - self.state['counters']['total_new_features'] if self.max_gen_new_feats != float('inf') else float('inf')
                    
                    # Set a reasonable max features per generation limit - don't restrict to just 1 feature
                    # This should allow multiple useful features to be found in each generation
                    features_per_gen = max(min(20, remaining_budget), 1) if remaining_budget > 0 else 1
                    
                    # Skip expensive evaluation if no budget left
                    if remaining_budget <= 0:
                        self._log(f"Gen {N+1}: No remaining feature budget. Skipping generation.")
                        continue
                    
                    # Select elites with progress tracking
                    beam_search_used = False
                    if adaptive_status['stagnation_level'] == 'SEVERE' and random.random() < 0.2:  # 20% chance
                        self._log(f"  Starting beam search with k=3 for elite selection in SEVERE stagnation...")
                        
                        # Calculate actual beam search evaluations for progress bar
                        k_beam = 3
                        max_beam_evals = k_beam + k_beam * (len(batch) - 1)
                        
                        with tqdm(total = max_beam_evals, desc = "Beam search evaluations", leave = False) as inner_pbar:
                            def beam_callback(evaluated_count, selected_count, force_complete=False):
                                inner_pbar.update(max(0, evaluated_count - inner_pbar.n if not force_complete else max_beam_evals - inner_pbar.n))
                                inner_pbar.set_description(f"Beam search: {evaluated_count}/{max_beam_evals}, Selected: {selected_count}")
                                # Check time budget during feature evaluation
                                if self.time_budget and (time.time() - start_time) > self.time_budget:
                                    return True  # Signal to stop evaluation
                            
                            elites, X, self.pipeline = self._beam_search_elites(batch, k=k_beam, X=X, y=y, callback=beam_callback)
                        
                        beam_search_used = True
                        success = len(elites) > 0
                        if success:
                            elite_names = [elite.name for elite in elites]
                            self._log(f"  Beam search successful: found {len(elites)} features")
                            self._log(f"  Beam search features: {elite_names}")
                            
                            # Verify if the beam search actually improved the score compared to the best score
                            _, beam_score = self._eval_baseline(X, y, self.pipeline)
                            best_score = self.state['best']['val_score']
                            is_better = (beam_score > best_score) if self.scorer.greater_is_better else (beam_score < best_score)
                            
                            if not is_better:
                                # Immediately revert to best generation if beam search didn't improve
                                improvement_direction = "higher" if self.scorer.greater_is_better else "lower"
                                comparison_result = "worse than" if self.scorer.greater_is_better else "better than"
                                self._log(f"  Gen {N+1} beam search added {len(elites)} features but didn't improve global best.")
                                self._log(f"  Reverting to best gen ({self.state['best']['gen_num']}) - current: {beam_score:.5f} is {comparison_result} best: {best_score:.5f} ({improvement_direction} is better)")
                                X, self.pipeline = self.state['best']['X'].copy(), self.state['best']['pipeline']
                                elites = []
                        else:
                            self._log(f"  Beam search failed: found {len(elites)} features")
                    
                    if not beam_search_used:
                        with tqdm(total = len(batch), desc = "Evaluating children features", leave = False) as inner_pbar:
                            def update_callback(evaluated_count, selected_count, force_complete=False):
                                inner_pbar.update(max(0, evaluated_count - inner_pbar.n if not force_complete else len(batch) - inner_pbar.n))
                                inner_pbar.set_description(f"Evaluated: {evaluated_count}/{len(batch)}, Selected: {selected_count}")
                                # Check time budget during feature evaluation
                                if self.time_budget and (time.time() - start_time) > self.time_budget:
                                    return True  # Signal to stop evaluation
                            
                            elites, X, self.pipeline = self._select_elites(
                                batch, 
                                features_per_gen,  # Use the per-generation limit, not the total remaining budget 
                                X, y, update_callback
                            )
                else:
                    # Hopeful monster was successful, elites were already set above
                    pass
                    
                # Check again if time budget is exceeded after feature evaluation
                if self.time_budget and (time.time() - start_time) > self.time_budget:
                    self._log(f"Time budget of {self.time_budget} seconds exceeded during feature evaluation. Stopping optimization.")
                    break
                
                # Handle feature generation and evaluation
                if hopeful_monster_success:
                    # For hopeful monster, we already have the elites, new generation, and scores calculated above
                    features_added = len(elites)
                    new_feature_names = [elite.name for elite in elites]
                    # new_train_score and new_val_score were already calculated above
                    # delta was already calculated above
                    # No need to recreate generation or recalculate scores
                else:
                    # Update weights and generation - Always recalculate weights when elites exist
                    new_feature_names = [elite.name for elite in elites]
                    
                    # Create new generation with elite features first
                    new_generation = generation.copy()
                    for interaction in elites:
                        feat = interaction.get_new_feature_instance()
                        feat.set_generating_interaction(interaction)
                        new_generation.append(feat)
                    
                    # Recalculate weights for all features if we have any changes
                    if new_feature_names or len(elites) > 0:
                        weights = self._get_top_k_features(X, y, k=-1, pipeline=self.pipeline)
                        # Update weights for all features in the new generation
                        for feat in new_generation:
                            if feat.name in weights.index:
                                feat.update_weight(weights.loc[feat.name, "weighted_importance"])
                            elif hasattr(feat, 'weight') and feat.weight > 0:
                                # Apply small decay for features not found in importance ranking
                                feat.update_weight(feat.weight * 0.95)
                    
                    generation = new_generation
                    
                    # Update feature counts and tracking variables
                    features_added = len(elites)
                    # Always sync the counter with actual feature count to avoid drift
                    self.state['counters']['total_new_features'] = X.shape[1] - len(self.initial_features)
                    self.state['counters']['no_feature_gens_count'] = 0 if features_added > 0 else self.state['counters']['no_feature_gens_count'] + 1
                    
                    # Track operation diversity for this generation
                    if elites:
                        ops_used = [elite.op for elite in elites]
                        # self._log(f"  Operations used: {Counter(ops_used)}")
                    
                    # Evaluate performance
                    new_train_score, new_val_score = self._eval_baseline(X, y, self.pipeline)
                    delta = new_val_score - self.state['best']['val_score'] if self.scorer.greater_is_better else self.state['best']['val_score'] - new_val_score
                    
                    # Revert if no improvement but features were added
                    if delta <= 0 and features_added > 0:
                        improvement_direction = "higher" if self.scorer.greater_is_better else "lower"
                        comparison_result = "worse than" if self.scorer.greater_is_better else "better than"
                        self._log(f"  Gen {N+1} added {features_added} features but didn't improve. Reverting to best gen ({self.state['best']['gen_num']}).")
                        self._log(f"  Current score: {new_val_score:.5f} is {comparison_result} best: {self.state['best']['val_score']:.5f} ({improvement_direction} is better for {self.scorer.name})")
                        X, self.pipeline, generation = self.state['best']['X'].copy(), self.state['best']['pipeline'], self.state['best']['generation'].copy()
                        self.pruned_features = self.state['best']['pruned_features'].copy()
                        new_val_score, delta = self.state['best']['val_score'], 0
                        # Recalculate total new features based on current X after revert
                        self.state['counters']['total_new_features'] = X.shape[1] - len(self.initial_features)
                        elites = []
                
                # Update best state tracking (only if not already done for hopeful monster)
                if not hopeful_monster_success:
                    if delta > 0:
                        self.state['best']['gen_num'], self.state['best']['val_score'], self.state['best']['X'] = N + 1, new_val_score, X.copy()
                        self.state['best']['generation'], self.state['best']['pipeline'] = generation.copy(), self.pipeline
                        self.state['best']['pruned_features'] = getattr(self, 'pruned_features', set()).copy()
                        self.state['counters']['consecutive_no_improvement_iters'] = 0
                        stagnation_counter = 0  # Reset stagnation counter on improvement
                        # Update feature interactions for the next generation when we have improvements
                        self.feature_interactions = self._analyze_feature_interactions(X, y, max_pairs=10000)
                        
                        # Update instance attributes to match best state
                        self.X, self.pipeline, self.generation = X.copy(), self.pipeline, generation.copy()
                        # Collect interactions from generation
                        self.interactions = []
                        for feat in self.generation:
                            if hasattr(feat, 'generating_interaction') and feat.generating_interaction:
                                self.interactions.append(feat.generating_interaction)
                        
                        # Save best state if save_path is provided
                        if self.save_path:
                            self.save(self.save_path)
                    else:
                        self.state['counters']['consecutive_no_improvement_iters'] += 1
                        stagnation_counter += 1  # Increment stagnation counter
                
                # Log progress and update display
                improvement = f"No improvement yet. Best generation: {self.state['best']['gen_num']}." if delta <= 0 else f"Score improved by {delta:.5f}. Best generation: {N+1}."
                
                # Get adaptive status for logging
                adaptive_status = self.adaptive_controller.get_status_summary()
                adaptive_info = f"\n  Adaptive status: {adaptive_status['stagnation_level']}, intensity: {adaptive_status['exploration_intensity']}, min_gain: {self.adaptive_controller.get_adaptive_min_gain():.5f}" if self.adaptive else ""
                
                # Start the generation log entry
                gen_log = f"Gen {N+1}:\n  Added {features_added} features, now using {X.shape[1]} features ({self.state['counters']['total_new_features']} new).\n"
                gen_log += f"  Mean Train {self.scorer.name} = {new_train_score:.5f}.  Mean Val {self.scorer.name} = {new_val_score:.5f}.\n"
                gen_log += f"  {improvement}{adaptive_info}"
                
                # Add annealing parameters if not hopeful monster
                if not hopeful_monster_success:
                    gen_log += f"\n  Annealing: τ={tau:.2f}, β={beta:.2f}, γ={gamma:.2f}, λ={lambda_:.2f}."
                
                # Add adaptive adjustments info if any were made
                if adaptive_adjustments and 'modifications' in adaptive_adjustments:
                    gen_log += f"\n  Adaptive adjustments: {adaptive_adjustments['modifications']} features modified"
                    gen_log += f"\n    Stagnation level: {adaptive_adjustments['stagnation_level']} (consecutive success: {adaptive_adjustments['consecutive_success']})"
                    gen_log += f"\n    Exploration intensity: {adaptive_adjustments['exploration_intensity']}"
                    gen_log += f"\n    Min gain factor: {adaptive_adjustments['min_gain_factor']}"
                
                self._log(gen_log)
                
                # Display newly added features if any were added in this generation
                if features_added > 0:
                    # Find new simple features (non-pipeline features)
                    new_simple = set(X.columns) - set(self.initial_features)
                    if features_added > 0 and elites:
                        new_simple_from_this_gen = set([elite.name for elite in elites if not elite.require_pipeline])
                        if new_simple_from_this_gen:
                            self._log(f"  New simple features: {new_simple_from_this_gen}.")
                    
                    # Get encoded features from the pipeline
                    new_target = self.pipeline.encoder.target_enc_cols
                    new_count = self.pipeline.encoder.count_enc_cols
                    new_freq = self.pipeline.encoder.freq_enc_cols
                    
                    # Print the encoded features only if they exist
                    if new_target:
                        self._log(f"  New target encoded features: {new_target}.")
                    if new_count:
                        self._log(f"  New count encoded features: {new_count}.")
                    if new_freq:
                        self._log(f"  New freq encoded features: {new_freq}.")
                
                pbar.set_postfix({f"{self.scorer.name}": f"{new_val_score:.5f}", "features": X.shape[1], 
                                 "new_features": self.state['counters']['total_new_features'], "best_gen": self.state['best']['gen_num']})
                pbar.update(1)
                
                # Check termination conditions
                if self.max_gen_new_feats != float('inf') and self.state['counters']['total_new_features'] >= self.max_gen_new_feats:
                    self._log(f"Reached maximum allowed new features ({self.state['counters']['total_new_features']}/{self.max_gen_new_feats}). Stopping optimization.")
                    break
                
                if self.state['counters']['consecutive_no_improvement_iters'] >= self.early_stopping_iter:
                    self._log(f"Early stopping after {self.state['counters']['consecutive_no_improvement_iters']} generations without improvement.")
                    break
        
        # Calculate execution time
        elapsed_time = time.time() - start_time
        
        # Ensure we return the best generation found
        if self.state['best']['gen_num'] < self.n_generations and not X.equals(self.state['best']['X']):
            self._log(f"Reverting to best generation ({self.state['best']['gen_num']}).")
            X, self.pipeline, generation = self.state['best']['X'], self.state['best']['pipeline'], self.state['best']['generation']
            self.pruned_features = self.state['best']['pruned_features'].copy()
                    
        # Calculate metrics and store as instance attributes
        n_init_feats = len(self.initial_features)
        n_added_feats = len(X.columns) - n_init_feats + self.pipeline.encoder.n_new_feats
        
        # Evaluate and store performance metrics
        self.initial_train_metric, self.initial_val_metric = self._eval_baseline(X[self.initial_features], y, self.pipeline)
        self.final_metric = self.state['best']['val_score']
        self.gain = self.final_metric - self.initial_val_metric if self.scorer.greater_is_better else self.initial_val_metric - self.final_metric
        self.pct_gain = self.gain / (abs(self.initial_val_metric) + 1e-8)
        
        # Store additional statistics
        self.n_samples, self.n_init_feats, self.n_added_feats = len(X), n_init_feats, n_added_feats
        self.n_final_feats, self.elapsed_time = n_init_feats + n_added_feats, elapsed_time
        
        # Log summary statistics
        self._log(
            f"\nFeature generation complete: {elapsed_time:.2f}s, Best gen: {self.state['best']['gen_num']}, "
            f"Best {self.scorer.name}: {self.state['best']['val_score']:.5f}, "
            f"Features added/total: {n_added_feats}/{n_init_feats + n_added_feats}"
        )
        
        # Log new features by type (more compact)
        new_features = {
            "simple": set(X.columns) - set(self.initial_features),
            "target": self.pipeline.encoder.target_enc_cols,
            "count": self.pipeline.encoder.count_enc_cols,
            "freq": self.pipeline.encoder.freq_enc_cols
        }
        
        for feat_type, features in new_features.items():
            self._log(f"New {feat_type} features: {features}")
        
        # Reset for further calls if needed
        if self.infer_task:
            self.baseline_model = self.task = self.scorer = None
    
        self.X, self.pipeline, self.generation = X, self.pipeline.get_pipeline(X), generation
        self.interactions = []
        for feat in self.generation:
            if hasattr(feat, 'generating_interaction') and feat.generating_interaction:
                self.interactions.append(feat.generating_interaction)
        
        return self.X, self.pipeline, self.generation, self.interactions
        





    def _set_defaults(self, X: pd.DataFrame, y: pd.Series) -> None:

        #  task / model / scorer
        self.task = self.task or ("regression" if type_of_target(y) == "continuous" else "classification")
        is_reg = self.task == "regression"
        if self.baseline_model is None:
            self.baseline_model = (XGBRegressor if is_reg else XGBClassifier)(
                device=self.device, enable_categorical=True, verbosity=0
            )
        if self.scorer is None:
            self.scorer = (
                PREDEFINED_REG_SCORERS["rmse"] if is_reg
                else PREDEFINED_CLS_SCORERS["binary_crossentropy"]
                if len(np.unique(y)) == 2
                else PREDEFINED_CLS_SCORERS["categorical_crossentropy"]
            )

        #  pipeline & hyper-params
        self.pipeline = PipelineWrapper(imputer=None, scaler=None, encoder=CategoricalEncoder())
        
        # Initialize adaptive controller for this run
        self.adaptive_controller.reset_for_new_run()
        self.adaptive_controller.initialize_operations(self.ops)

        #  search-state containers (simplified - adaptive state moved to controller)
        self.state = {
            "best": dict(gen_num=0, val_score=0, train_score=0, X=None, generation=None, pipeline=None, pruned_features=set()),
            "counters": dict(
                total_new_features=0,
                no_feature_gens_count=0,
                consecutive_no_improvement_iters=0
            ),
            "seen_feats": set(),
        }

        #  metric reset
        self.initial_metric = self.final_metric = self.gain = self.pct_gain = None
        self.n_samples = self.n_init_feats = self.n_added_feats = self.n_final_feats = self.elapsed_time = None








    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureGenerator':
        """
        Fit the feature generator's pipeline on the input data.
        This method uses self.interactions to create features and then fits 
        the pipeline on these features. This is needed because some of the categorical
        features handled by the pipeline might have as parents features generated without it.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : array-like, optional
            Target values for supervised transformations
            
        Returns
        -------
        self : FeatureGenerator
            Returns self for method chaining
        """
        if not getattr(self, 'interactions', None):
            self._log("Warning: No interactions found in self.interactions. No features will be generated.")
            return self
            
        # Create default pipeline if needed
        if not getattr(self, 'pipeline', None):
            self._log("Warning: No pipeline found. Creating a default pipeline.")
            self.pipeline = PipelineWrapper(
                imputer=None, scaler=None, encoder=CategoricalEncoder()
            ).get_pipeline(X)
        
        # Convert X to DataFrame if needed and apply transformations
        X_transformed = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Generate features using non-pipeline requiring interactions
        for interaction in self.interactions:
            if interaction.name not in X_transformed.columns and not interaction.require_pipeline:
                try:
                    result = interaction.generate(X_transformed)
                    if result is not None:
                        X_transformed[result[0]] = result[1]
                except Exception as e:
                    self._log(f"Error generating feature {interaction.name}: {str(e)}")
                    
        # Fit the pipeline with the generated features
        if isinstance(self.pipeline, PipelineWrapper):
            self.pipeline = self.pipeline.get_pipeline(X)
        self.pipeline.fit(X_transformed, y)
        return self
    

    def transform(self, X: pd.DataFrame):
        """
        Transform input data by applying all interactions in self.interactions
        and then transforming with the fitted pipeline.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features to transform
            
        Returns
        -------
        pd.DataFrame
            Transformed features with all generated features added and pruned features removed
        """
        if not getattr(self, 'interactions', None):
            self._log("Warning: No interactions found in self.interactions. Returning input features unchanged.")
            return X
            
        # Convert X to DataFrame if needed
        X_transformed = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
            
        # Create features using interactions
        for interaction in self.interactions:
            if interaction.name not in X_transformed.columns and not interaction.require_pipeline:
                try:
                    result = interaction.generate(X_transformed)
                    if result is not None:
                        X_transformed[result[0]] = result[1]
                except Exception as e:
                    self._log(f"Error generating feature {interaction.name}: {str(e)}")
        
        # Apply pipeline transformations
        pipeline = getattr(self, 'pipeline', None)
        if pipeline is not None:
            try:
                X_transformed = pipeline.transform(X_transformed)
            except Exception as e:
                self._log(f"Error applying pipeline transformations: {str(e)}")
        
        # Remove pruned features (including pruned original features)
        if hasattr(self, 'pruned_features') and self.pruned_features:
            # Only drop columns that exist in the dataframe
            columns_to_drop = [col for col in self.pruned_features if col in X_transformed.columns]
            if columns_to_drop:
                X_transformed = X_transformed.drop(columns=columns_to_drop)
                self._log(f"Removed {len(columns_to_drop)} pruned features from transformed data")
        
        return X_transformed
        


    def fit_transform(self, X, y=None):

        """
        Fit the feature generator on the input data and then transform it.
        This is equivalent to calling fit(X, y) followed by transform(X), but is more efficient.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : array-like, optional
            Target values for supervised transformations
            
        Returns
        -------
        pd.DataFrame
            Transformed features with all generated features added        
            """
        return self.fit(X, y).transform(X)
    
    def save(self, filepath):
        """
        Save the feature generator state to a file using cloudpickle for complete serialization.
        
        Recursively serializes the entire class instance with all attributes, including complex
        objects like Feature instances, Interactions, lambda functions, and nested data structures.
        
        Parameters:
        -----------
        filepath : str
            Path where the state will be saved
        """
        import os
        
        try:
            import cloudpickle
        except ImportError:
            raise ImportError("cloudpickle is required for serialization. Install with: pip install cloudpickle")
        
        # Ensure all critical attributes are set before saving
        # If we have state['best'] with X and generation but no self.X/self.generation
        if not hasattr(self, 'X') and hasattr(self, 'state') and 'best' in self.state and self.state['best']['X'] is not None:
            self.X = self.state['best']['X']
            self.pipeline = self.state['best']['pipeline']
            self.generation = self.state['best']['generation']
            
        # Make sure interactions are extracted from generation
        if hasattr(self, 'generation'):
            # Always rebuild interactions from generation to ensure completeness
            self.interactions = []
            # Get features currently in X, which include pruned features that were later regenerated
            current_features = set(self.X.columns) if hasattr(self, 'X') else set()
            for feat in self.generation:
                if hasattr(feat, 'generating_interaction') and feat.generating_interaction:
                    # Include the feature if it's in the current X dataframe
                    if feat.name in current_features:
                        self.interactions.append(feat.generating_interaction)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            cloudpickle.dump(self, f)
        self._log(f"Feature generator state saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load a feature generator state from a file using cloudpickle.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved state file
            
        Returns:
        --------
        FeatureGenerator
            Loaded feature generator instance with all attributes restored
        """
        import os
        
        try:
            import cloudpickle
        except ImportError:
            raise ImportError("cloudpickle is required for deserialization. Install with: pip install cloudpickle")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file {filepath} does not exist. Please check the path.")
        
        try:    
            with open(filepath, 'rb') as f:
                return cloudpickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load file: {str(e)}")

############################################################################################################################################################################################################################################

class StagnationLevel(Enum):
    NONE = 0
    MILD = 1  
    MODERATE = 2
    SEVERE = 3

@dataclass
class AdaptiveState:
    """Container for adaptive system state"""
    stagnation_level: StagnationLevel = StagnationLevel.NONE
    generations_without_features: int = 0
    generations_without_improvement: int = 0
    consecutive_successful_generations: int = 0  # Track consecutive successful generations
    exploration_intensity: float = 0.0  # 0.0 = exploitation, 1.0 = pure exploration
    min_gain_reduction_factor: float = 1.0
    feature_weights_modified: bool = False

class AdaptiveController:
    """
   
    Unified controller for all adaptive mechanisms in feature generation.
    
    This class coordinates all adaptive behaviors to prevent conflicts and ensure
    mechanisms work together harmoniously. It follows a hierarchical approach where
    stagnation assessment drives all other adaptive behaviors.
    """
    
    def __init__(self, original_min_pct_gain: float = 0.005, 
                 exploration_factor: float = 0.2):
        self.original_min_pct_gain = original_min_pct_gain
        self.exploration_factor = exploration_factor
        
        # Adaptive state
        self.state = AdaptiveState()
        
        # Operation tracking for intelligent candidate prioritization
        self.op_stats = {
            "num": {"unary": {}, "binary": {}},
            "cat": {"unary": {}, "binary": {}}
        }
        self.op_usage = Counter()
        self.op_success = Counter()
        
        # Feature usage tracking for diversity
        self.feature_usage = Counter()
        self.failed_interactions = Counter()
        
        # Weight modification tracking
        self.weight_modifications = {}



    def initialize_operations(self, ops):
        """Initialize operation statistics."""
        for dtype in ops:
            for op_type in ops[dtype]:
                for op in ops[dtype][op_type]:
                    if op not in self.op_stats[dtype][op_type]:
                        self.op_stats[dtype][op_type][op] = {
                            "success_rate": 0.5,
                            "avg_gain": 0.0,
                            "priority_score": 0.5
                        }


    def assess_stagnation(self, no_features_count: int, no_improvement_count: int) -> None:
        """
        Centralized stagnation assessment that drives all other adaptive behaviors.
        Requires multiple consecutive successful generations to gradually decrease stagnation level.
        """
        self.state.generations_without_features = no_features_count
        self.state.generations_without_improvement = no_improvement_count
        
        # Track consecutive successful generations (a generation is successful if it has both features and improvement)
        if no_features_count == 0 and no_improvement_count == 0:
            self.state.consecutive_successful_generations += 1
        else:
            # Reset the counter if there's any stagnation
            self.state.consecutive_successful_generations = 0
        
        # First, determine the stagnation level based on current conditions
        current_level = None
        if no_features_count >= 4 or no_improvement_count >= 6:
            current_level = StagnationLevel.SEVERE
        elif no_features_count >= 2 or no_improvement_count >= 4:
            current_level = StagnationLevel.MODERATE
        elif no_features_count >= 1 or no_improvement_count >= 2:
            current_level = StagnationLevel.MILD
        else:
            current_level = StagnationLevel.NONE
        
        # If current conditions indicate stagnation, immediately increase to that level
        if current_level.value > self.state.stagnation_level.value:
            self.state.stagnation_level = current_level
        # Otherwise, only SEVERE stagnation can be escaped with consecutive successes
        elif current_level.value < self.state.stagnation_level.value:
            # For SEVERE stagnation only, remove it entirely after 2 consecutive successful generations
            if self.state.stagnation_level == StagnationLevel.SEVERE:
                # Completely remove stagnation after 2 consecutive successes when in SEVERE state
                if self.state.consecutive_successful_generations >= 2:
                    self.state.stagnation_level = StagnationLevel.NONE
            # No automatic decrease behavior for other stagnation levels
        
        # Set exploration intensity based on stagnation level
        if self.state.stagnation_level == StagnationLevel.SEVERE:
            self.state.exploration_intensity = 1.0
        elif self.state.stagnation_level == StagnationLevel.MODERATE:
            self.state.exploration_intensity = 0.6
        elif self.state.stagnation_level == StagnationLevel.MILD:
            self.state.exploration_intensity = 0.3
        else:
            self.state.exploration_intensity = 0.0
    
    def get_adaptive_min_gain(self) -> float:
        """Get adaptively adjusted minimum gain threshold."""
        if self.state.stagnation_level == StagnationLevel.SEVERE:
            self.state.min_gain_reduction_factor = 0.25  # Very lenient
        elif self.state.stagnation_level == StagnationLevel.MODERATE:
            self.state.min_gain_reduction_factor = 0.5   # Moderately lenient
        elif self.state.stagnation_level == StagnationLevel.MILD:
            self.state.min_gain_reduction_factor = 0.75  # Slightly lenient
        else:
            self.state.min_gain_reduction_factor = 1.0   # Original threshold
            
        return self.original_min_pct_gain * self.state.min_gain_reduction_factor
    
    def get_search_parameters(self, progress: float) -> tuple[float, float, float, float]:
        """
        Get unified search parameters (tau, beta, gamma, lambda) that coordinate
        both progressive annealing and adaptive exploration.
        """
        # Base progressive annealing schedule
        base_tau = max(0.1, 0.1 + 0.9 * (1 + np.cos(progress * np.pi)) / 2)
        base_beta = 0.8 * (1 / (1 + np.exp(10 * progress - 5)))
        base_gamma = 0.2 * np.exp(-3 * progress)
        base_lambda = 0.1 + 0.4 * (1 / (1 + np.exp(-10 * (progress - 0.5))))
        
        # Apply adaptive adjustments based on stagnation
        intensity = self.state.exploration_intensity
        
        # Temperature (exploration)
        tau = base_tau * (1 + 2 * intensity)
        
        # Operation novelty weight  
        beta = base_beta * (1 + intensity)
        
        # Feature novelty weight
        gamma = base_gamma * (1 + 3 * intensity)
        
        # Complexity penalty (reduce during exploration)
        lambda_ = base_lambda * (1 - 0.5 * intensity)
        
        return tau, beta, gamma, lambda_
    
    def update_operation_stats(self, interaction: 'Interaction', success: bool, gain: float = 0.0):
        """Update operation performance statistics."""
        op = interaction.op
        dtype = interaction.dtype
        op_type = interaction.type
        
        # Update counters
        self.op_usage[op] += 1
        if success:
            self.op_success[op] += 1
        else:
            # Track failed interactions
            interaction_key = f"{interaction.feature_1.name}|{op}|{interaction.feature_2.name if interaction.feature_2 else ''}"
            self.failed_interactions[interaction_key] += 1
        
        # Initialize if not exists
        if op not in self.op_stats[dtype][op_type]:
            self.op_stats[dtype][op_type][op] = {
                "success_rate": 0.5, "avg_gain": 0.0, "priority_score": 0.5
            }
        
        # Update with exponential decay (favor recent performance)
        stats = self.op_stats[dtype][op_type][op]
        decay = 0.9
        
        if self.op_usage[op] > 0:
            stats["success_rate"] = (stats["success_rate"] * decay + 
                                   (1.0 if success else 0.0) * (1 - decay))
        
        if success:
            stats["avg_gain"] = (stats["avg_gain"] * decay + gain * (1 - decay))
        
        # Combined priority score
        stats["priority_score"] = (0.7 * stats["success_rate"] + 
                                 0.3 * min(1.0, stats["avg_gain"] * 10))
    
    def get_prioritized_operations(self, dtype: str, op_type: str) -> list[str]:
        """Get operations prioritized by performance and exploration needs."""
        if dtype not in self.op_stats or op_type not in self.op_stats[dtype]:
            return []
        
        ops_with_scores = []
        for op, stats in self.op_stats[dtype][op_type].items():
            # Base exploitation score
            exploitation_score = stats["priority_score"]
            
            # Add exploration bonus - favor underused operations during exploration
            usage_factor = 1.0 / (1.0 + self.op_usage.get(op, 0) * 0.1)  # Penalize overused ops
            exploration_bonus = usage_factor * self.state.exploration_intensity * 0.5  # Bounded bonus
            
            final_score = exploitation_score + exploration_bonus
            ops_with_scores.append((op, final_score))
        
        ops_with_scores.sort(key=lambda x: x[1], reverse=True)
        return [op for op, score in ops_with_scores]
    
    def adapt_feature_weights(self, generation: list['Feature']) -> int:
        """
        Unified feature weight adaptation that coordinates decay and boosting.
        Returns number of features modified.
        """
        if self.state.stagnation_level == StagnationLevel.NONE:
            # Recovery phase - gradually restore slashed weights when stagnation is completely gone
            if self.weight_modifications:
                modifications = 0
                for feature in generation:
                    if feature.name in self.weight_modifications and "severe_slash" in str(self.weight_modifications[feature.name].get("modifications", [])):
                        # Recover weights gradually (double them, up to the original)
                        original_weight = self.weight_modifications[feature.name]["original"]
                        recovery_weight = min(original_weight, feature.weight * 2.0)
                        if recovery_weight > feature.weight:
                            feature.weight = recovery_weight
                            modifications += 1
                            if feature.weight >= original_weight * 0.9:  # Close enough to original
                                del self.weight_modifications[feature.name]
                
                if modifications > 0:
                    return modifications
            return 0
        
        # Track feature usage
        for feat in generation:
            if not feat.require_pipeline:
                self.feature_usage[feat.name] += 1
        
        modifications = 0
        
        # Get usage statistics for diversity assessment
        usage_values = list(self.feature_usage.values()) if self.feature_usage else [1]
        mean_usage = np.mean(usage_values)
        max_usage = max(usage_values)
        
        # Drastic weight reduction for top percentage of features during SEVERE stagnation
        if self.state.stagnation_level == StagnationLevel.SEVERE:
            # Find top 3 highest weighted features and slash their weights
            sorted_features = [f for f in generation if not f.require_pipeline]
            sorted_features = sorted(sorted_features, key=lambda f: f.weight, reverse=True)

            # Calculate how many features to prune (top 3 features)
            top_n = min(3, len(sorted_features))

            for i, feature in enumerate(sorted_features[:top_n]):
                feature.weight *= 0.1  # Slash to 10% of original
                modifications += 1
                self.weight_modifications[feature.name] = {
                    "original": feature.weight * 10,  # Store original before slash
                    "new": feature.weight,
                    "modifications": ["severe_slash_0.1"],
                    "stagnation_level": self.state.stagnation_level.name
                }
        
        for feature in generation:
            if feature.require_pipeline:
                continue
                
            original_weight = feature.weight
            new_weight = original_weight
            modifications_applied = []
            
            # Skip if already processed in severe stagnation
            if (self.state.stagnation_level == StagnationLevel.SEVERE and 
                feature.name in self.weight_modifications and
                "severe_slash" in str(self.weight_modifications[feature.name].get("modifications", []))):
                continue
            
            # Diversity boosting for underutilized features
            if self.state.stagnation_level.value >= StagnationLevel.MODERATE.value:
                usage = self.feature_usage.get(feature.name, 0)
                if usage < mean_usage and max_usage > 0:
                    boost_factor = 1.0 + 0.5 * (1.0 - usage / max_usage)
                    new_weight *= boost_factor
                    modifications_applied.append(f"boost_{boost_factor:.2f}")
            
            # Stagnation decay for exploration
            if self.state.stagnation_level.value >= StagnationLevel.MODERATE.value:
                # Apply probabilistic decay, but skip recently boosted features
                if "boost" not in "".join(modifications_applied):  # This check is good!
                    decay_prob = 0.3 + 0.2 * self.state.stagnation_level.value / 3.0
                    if random.random() < decay_prob:
                        decay_factor = 0.7 + 0.2 * (1 - self.state.exploration_intensity)
                        new_weight *= decay_factor
                        modifications_applied.append(f"decay_{decay_factor:.2f}")
                # Add: Also prevent boosting of recently decayed features
                elif len(self.weight_modifications.get(feature.name, {}).get("modifications", [])) > 0:
                    # Skip decay if feature was recently modified
                    pass
            
            # Apply weight change and track
            if new_weight != original_weight:
                feature.weight = new_weight
                modifications += 1
                self.weight_modifications[feature.name] = {
                    "original": original_weight,
                    "new": new_weight,
                    "modifications": modifications_applied,
                    "stagnation_level": self.state.stagnation_level.name
                }
        
        self.state.feature_weights_modified = modifications > 0
        return modifications
    
    def rank_candidates_with_adaptive_criteria(self, batch: list['Interaction'], 
                                             X: pd.DataFrame, y: pd.Series) -> list['Interaction']:
        """
        Multi-criteria ranking with adaptive weighting based on current exploration needs.
        """
        if not batch:
            return []
        
        # Calculate base scores
        weights = np.array([interaction.weight for interaction in batch])
        depths = np.array([interaction.depth for interaction in batch])
        
        # Preserve relative weight importance by using a less aggressive normalization
        # that maintains the proportional differences between weights
        if len(weights) > 1 and np.sum(weights) > 0:
            # Softmax-based normalization preserves relative differences better
            # than min-max scaling while still ensuring values are bounded
            exp_weights = np.exp(weights - np.max(weights))  # Subtract max for numerical stability
            norm_weights = exp_weights / np.sum(exp_weights)
        else:
            norm_weights = np.ones_like(weights) * 0.5
        
        # Complexity penalty (exponential to heavily penalize deep features)
        max_depth = np.max(depths) if len(depths) > 0 else 1
        if max_depth > 0:
            # Exponential penalty: heavily punish features that are just transformations of existing complex ones
            complexity_penalty = (depths / max_depth) ** 2
        else:
            complexity_penalty = np.zeros_like(depths)
        
        # Operation success scores
        op_scores = np.array([self._get_op_priority_score(interaction.op) for interaction in batch])
        
        # Novelty scores (favor rare operations during exploration)
        op_counts = Counter([interaction.op for interaction in batch])
        total_ops = sum(op_counts.values())
        novelty_scores = np.array([1 - (op_counts[interaction.op] / total_ops) for interaction in batch])
        
        # Failed interaction penalties
        failed_penalties = np.array([self._get_failure_penalty(interaction) for interaction in batch])
        
        # Adaptive weighting based on exploration intensity
        intensity = self.state.exploration_intensity
        
        # During high stagnation, emphasize novelty and reduce complexity penalties
        # Give feature importance weights a higher priority to preserve their influence
        weight_importance = 0.6 + 0.1 * (1 - intensity)      # Higher weight to preserve importance
        operation_weight = 0.15 + 0.05 * (1 - intensity)     # Reduce during exploration  
        novelty_weight = 0.05 + 0.2 * intensity              # Increase during exploration
        complexity_weight = 0.15 + 0.05 * (1 - intensity)    # Reduce during exploration

        failure_weight = 0.05 + 0.05 * (1 - intensity)       # Reduce during exploration
        
        # Ensure weights sum to 1.0 to prevent scaling issues
        total = weight_importance + operation_weight + novelty_weight + complexity_weight + failure_weight
        weight_importance /= total
        operation_weight /= total
        novelty_weight /= total
        complexity_weight /= total
        failure_weight /= total
        
        # Composite scoring - prioritize feature importance weights
        composite_scores = (
            weight_importance * norm_weights +
            operation_weight * op_scores +
            novelty_weight * novelty_scores -
            complexity_weight * complexity_penalty -
            failure_weight * failed_penalties
        )
        
        # Sort by composite score
        ranking = np.argsort(-composite_scores)
        return [batch[i] for i in ranking]
    
    def _get_op_priority_score(self, op: str) -> float:
        """Get priority score for an operation."""
        for dtype in self.op_stats:
            for op_type in self.op_stats[dtype]:
                if op in self.op_stats[dtype][op_type]:
                    return self.op_stats[dtype][op_type][op]["priority_score"]
        return 0.5
    
    def _get_failure_penalty(self, interaction: 'Interaction') -> float:
        """Get penalty for previously failed similar interactions."""
        interaction_key = f"{interaction.feature_1.name}|{interaction.op}|{interaction.feature_2.name if interaction.feature_2 else ''}"
        failures = self.failed_interactions.get(interaction_key, 0)
        return min(0.8, 0.1 * failures)
    
    def reset_for_new_run(self):
        """Reset state for a new feature generation run."""
        self.state = AdaptiveState()
        self.weight_modifications.clear()
        # Keep operation stats for learning across runs
    
    def get_status_summary(self) -> dict:
        """Get summary of current adaptive state for logging."""
        return {
            "stagnation_level": self.state.stagnation_level.name,
            "exploration_intensity": f"{self.state.exploration_intensity:.2f}",
            "min_gain_factor": f"{self.state.min_gain_reduction_factor:.2f}",
            "consecutive_success": self.state.consecutive_successful_generations,
            "weights_modified": self.state.feature_weights_modified,
            "features_modified": len(self.weight_modifications)
        }
