import category_encoders as ce
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted, check_array, _check_feature_names_in

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    A custom transformer that applies various category encodings and ensures proper handling of column names.
    """
    def __init__(self, target_enc_cols=None, count_enc_cols=None, freq_enc_cols=None,
                 return_original=True, handle_unknown='value', handle_missing='value'):

        # Validate and initialize columns
        self.target_enc_cols = self._validate_columns(target_enc_cols)
        self.count_enc_cols = self._validate_columns(count_enc_cols)
        self.freq_enc_cols = self._validate_columns(freq_enc_cols)
        
        self.return_original = return_original
        self.handle_unknown = handle_unknown
        self.handle_missing = handle_missing

        self._all_configured_cols = sorted(list(set(
            self.target_enc_cols + self.count_enc_cols + self.freq_enc_cols
        )))

        # Initialize encoders with validated columns
        self.n_new_feats = 0
        self.target_encoder = None
        self.count_encoder = self._init_encoder(ce.CountEncoder, self.count_enc_cols, normalize=False)
        self.freq_encoder = self._init_encoder(ce.CountEncoder, self.freq_enc_cols, normalize=True)

        self._target_encoding_mode = 'standard'
        self._target_encoded_output_cols = []
        self._planned_target_output_cols = []
        self._planned_multiclass = False

        self.feature_names_in_ = None
        self.n_features_in_ = None
        self._encoder_input_features = None
        self._feature_names_out = None
      

    def _validate_columns(self, columns):
        """Ensure columns are a list of strings."""
        if columns is None:
            return []
        if not isinstance(columns, list) or not all(isinstance(c, str) for c in columns):
            raise ValueError("Columns must be a list of strings.")
        return columns

    def _init_encoder(self, encoder_class, cols, **kwargs):
        """Initialize an encoder if columns are specified."""
        if cols:
            self.n_new_feats += len(cols)
            return encoder_class(cols=cols, handle_unknown=self.handle_unknown, 
                               handle_missing=self.handle_missing, **kwargs)
        return None

    def _count_target_classes(self, y):
        """Return number of classes for 1D targets, otherwise None."""
        if y is None:
            return None
        y_arr = np.asarray(y)
        if y_arr.ndim != 1:
            return None
        return int(pd.Series(y_arr).dropna().nunique())

    def _is_multiclass_target(self, y):
        """Detect if y is multiclass for target-encoding mode selection."""
        if y is None:
            return bool(self._planned_multiclass)
        try:
            tgt_type = type_of_target(y)
        except Exception:
            n_classes = self._count_target_classes(y)
            return bool(n_classes is not None and n_classes > 2)

        if tgt_type in ("multiclass", "multiclass-multioutput"):
            return True
        return False

    def set_target_info(self, y=None):
        """Pre-compute expected target output names for pipeline planning."""
        if not self.target_enc_cols:
            self._planned_target_output_cols = []
            self._planned_multiclass = False
            self.n_new_feats = len(self.count_enc_cols) + len(self.freq_enc_cols)
            return self

        is_multiclass = self._is_multiclass_target(y)
        self._planned_multiclass = is_multiclass
        n_classes = self._count_target_classes(y)

        if is_multiclass:
            y_arr = np.asarray(y) if y is not None else np.array([])
            labels = pd.Series(y_arr).dropna().unique().tolist() if y_arr.ndim == 1 and y_arr.size else []

            if not labels:
                if n_classes is None:
                    labels = [0, 1]
                else:
                    labels = list(range(n_classes))

            if n_classes is None or n_classes <= 2:
                self._planned_target_output_cols = [f"{col}_target" for col in self.target_enc_cols]
            else:
                self._planned_target_output_cols = [
                    f"{col}_target_{class_label}"
                    for class_label in labels
                    for col in self.target_enc_cols
                ]
        else:
            self._planned_target_output_cols = [f"{col}_target" for col in self.target_enc_cols]

        n_target_per_col = 1 if not is_multiclass or n_classes is None else max(1, n_classes - 1)
        self.n_new_feats = (
            n_target_per_col * len(self.target_enc_cols)
            + len(self.count_enc_cols)
            + len(self.freq_enc_cols)
        )

        return self

    def _init_target_encoder(self, y):
        """Initialize target encoder, using polynomial wrapper for multiclass."""
        if not self.target_enc_cols:
            return None

        base_encoder = ce.TargetEncoder(
            cols=self.target_enc_cols,
            handle_unknown=self.handle_unknown,
            handle_missing=self.handle_missing,
        )

        if self._is_multiclass_target(y):
            from category_encoders.wrapper import PolynomialWrapper
            self._target_encoding_mode = 'multiclass'
            return PolynomialWrapper(base_encoder)

        self._target_encoding_mode = 'standard'
        return base_encoder

    @staticmethod
    def _ensure_dataframe(values, index=None):
        """Normalize encoder outputs to DataFrame for consistent column handling."""
        if isinstance(values, pd.DataFrame):
            return values
        if isinstance(values, pd.Series):
            return values.to_frame()
        return pd.DataFrame(values, index=index)

    def _rename_target_output_columns(self, transformed, current_cols):
        """Rename target-encoder outputs to stable framework-friendly names."""
        transformed_df = self._ensure_dataframe(transformed)
        if transformed_df.empty:
            return transformed_df

        if self._target_encoding_mode == 'multiclass':
            sorted_cols = sorted(current_cols, key=len, reverse=True)
            rename_map = {}
            for out_col in transformed_df.columns:
                out_col_str = str(out_col)
                mapped_name = None
                for source_col in sorted_cols:
                    if out_col_str == source_col:
                        mapped_name = f"{source_col}_target"
                        break
                    prefix = f"{source_col}_"
                    if out_col_str.startswith(prefix):
                        suffix = out_col_str[len(source_col):]
                        mapped_name = f"{source_col}_target{suffix}"
                        break
                if mapped_name is None:
                    mapped_name = f"target_{out_col_str}"
                rename_map[out_col] = mapped_name
            return transformed_df.rename(columns=rename_map)

        rename_map = {}
        for source_col in current_cols:
            if source_col in transformed_df.columns:
                rename_map[source_col] = f"{source_col}_target"
        return transformed_df.rename(columns=rename_map)

    def _capture_target_output_columns(self, encoder, X_subset, current_cols):
        """Capture fitted target-encoder output columns for downstream consistency."""
        transformed = encoder.transform(X_subset)
        renamed = self._rename_target_output_columns(transformed, current_cols)
        self._target_encoded_output_cols = list(renamed.columns)

    def get_reserved_output_columns(self):
        """Return output columns created by this encoder for collision avoidance."""
        output_cols = []

        if self._target_encoded_output_cols:
            output_cols.extend(self._target_encoded_output_cols)
        elif self._planned_target_output_cols:
            output_cols.extend(self._planned_target_output_cols)
        else:
            output_cols.extend(f"{col}_target" for col in self.target_enc_cols)

        output_cols.extend(f"{col}_count" for col in self.count_enc_cols)
        output_cols.extend(f"{col}_freq" for col in self.freq_enc_cols)
        return output_cols

    def fit(self, X, y=None):
        # Convert to DataFrame and validate
        X_df = self._check_and_convert_X(X)
        self.feature_names_in_ = np.array(X_df.columns, dtype=object)
        self.n_features_in_ = X_df.shape[1]

        # Determine valid columns present in the data
        self._encoder_input_features = [col for col in self._all_configured_cols if col in X_df.columns]

        # Build target encoder after we know the target type.
        self.set_target_info(y)
        self.target_encoder = self._init_target_encoder(y)
        
        # Fit each encoder on the valid columns
        self._fit_encoder(self.target_encoder, X_df, y, self.target_enc_cols, is_target=True)
        self._fit_encoder(self.count_encoder, X_df, None, self.count_enc_cols)
        self._fit_encoder(self.freq_encoder, X_df, None, self.freq_enc_cols)

        self._feature_names_out = self._generate_output_feature_names(self.feature_names_in_)
        self.n_new_feats = len(self._generate_output_feature_names(self.feature_names_in_, only_new=True))
        return self

    def _check_and_convert_X(self, X):
        """Convert X to DataFrame and ensure proper column names."""
        if not isinstance(X, pd.DataFrame):
            try:
                n_features = X.shape[1] if hasattr(X, 'shape') else len(X[0])
                columns = [f"col_{i}" for i in range(n_features)]
                return pd.DataFrame(X, columns=columns)
            except Exception as e:
                raise ValueError(f"Failed to convert input to DataFrame: {e}")
        return X

    def _fit_encoder(self, encoder, X_df, y, cols, is_target=False):
        """Fit an encoder on the relevant columns if present."""
        if encoder is None:
            return
        current_cols = [col for col in cols if col in X_df.columns]
        if not current_cols:
            return

        # Create a subset with only the required columns
        # This ensures we pass only the exact columns needed to the encoder
        X_subset = X_df[current_cols].copy()

        # For single column case, ensure we don't have DataFrame indexing issues
        # Category encoders has issues when you pass a DataFrame slice and then tries to do X[col]
        if len(current_cols) == 1:
            # Create a new DataFrame with explicit column name to avoid any indexing issues
            # Must preserve the original index to avoid mismatch with y
            col_name = current_cols[0]
            new_df = pd.DataFrame({col_name: X_subset[col_name]}, index=X_subset.index)
            X_subset = new_df

        encoder.fit(X_subset, y)
        if is_target:
            self._capture_target_output_columns(encoder, X_subset, current_cols)

    def transform(self, X):
        check_is_fitted(self)
        X_df = self._check_and_convert_X(X)
        cols_to_process = [col for col in self._encoder_input_features if col in X_df.columns]

        X_encoded = pd.DataFrame(index=X_df.index)
        X_encoded = self._transform_target_encoder(self.target_encoder, X_df, cols_to_process, X_encoded)
        X_encoded = self._transform_encoder(self.count_encoder, X_df, cols_to_process, X_encoded, '_count')
        X_encoded = self._transform_encoder(self.freq_encoder, X_df, cols_to_process, X_encoded, '_freq')

        if self.return_original:
            X_final = pd.concat([X_df, X_encoded], axis=1)
        else:
            X_final = X_encoded

        expected_cols = self._generate_output_feature_names(X_df.columns, only_new=not self.return_original)
        return X_final.reindex(columns=expected_cols, copy=False)

    def _transform_target_encoder(self, encoder, X_df, cols_to_process, X_encoded):
        """Apply target encoder transform and append renamed output columns."""
        if encoder is None:
            return X_encoded

        current_cols = [col for col in cols_to_process if col in self.target_enc_cols]
        if not current_cols:
            return X_encoded

        transformed = encoder.transform(X_df[current_cols])
        transformed_df = self._rename_target_output_columns(transformed, current_cols)

        target_cols = self._target_encoded_output_cols or list(transformed_df.columns)
        for col in target_cols:
            if col in transformed_df.columns:
                X_encoded[col] = transformed_df[col].values

        return X_encoded

    def _transform_encoder(self, encoder, X_df, cols_to_process, X_encoded, suffix):
        """Apply encoder transform and add features."""
        if encoder is None:
            return X_encoded
        current_cols = [col for col in cols_to_process if col in encoder.cols]
        if not current_cols:
            return X_encoded
        transformed = encoder.transform(X_df[current_cols])
        for col in current_cols:
            X_encoded[f"{col}{suffix}"] = transformed[col].values
        return X_encoded

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        input_features = _check_feature_names_in(self, input_features)
        return self._generate_output_feature_names(input_features, only_new=not self.return_original)

    def _generate_output_feature_names(self, input_features, only_new=False):
        input_list = input_features.tolist() if isinstance(input_features, np.ndarray) else list(input_features)
        new_features = []

        if self._target_encoded_output_cols:
            new_features.extend(self._target_encoded_output_cols)
        else:
            for col in input_list:
                if col in self.target_enc_cols:
                    new_features.append(f"{col}_target")

        for col in input_list:
            if col in self.count_enc_cols:
                new_features.append(f"{col}_count")
            if col in self.freq_enc_cols:
                new_features.append(f"{col}_freq")
        return np.array(input_list + new_features if not only_new else new_features, dtype=object)

    def _more_tags(self):
        return {
            'allow_nan': True,
            'requires_y': bool(self.target_enc_cols),
            "_xfail_checks": {
                "check_dtype_object": "Handles object types internally."
            }
        }


class GroupByEncoder(BaseEstimator, TransformerMixin):
    """Fit-transform group-by statistics within CV folds to prevent leakage.
    
    Computes aggregation statistics (mean, std, etc.) of a numeric column
    grouped by a categorical column. Fitting learns the mapping from the
    training fold; transform applies it to any fold with unseen-category fallback.
    """
    def __init__(self, cat_col, num_col, agg_func, output_col=None):
        self.cat_col = cat_col
        self.num_col = num_col
        self.agg_func = agg_func
        self.output_col = output_col or f"groupby_{agg_func}({cat_col}, {num_col})"
        self.mapping_ = None
        self.global_fallback_ = None
        self.rank_values_by_group_ = None
        self.global_rank_values_ = None

    @staticmethod
    def _empirical_percentile(sorted_values, value):
        if sorted_values is None or len(sorted_values) == 0:
            return np.nan
        if pd.isna(value):
            return np.nan
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return np.nan
        pos = np.searchsorted(sorted_values, numeric_value, side="right")
        return pos / float(len(sorted_values))

    @staticmethod
    def _map_numeric_with_fallback(cat_series, mapping, fallback):
        """Map category keys to numeric stats and safely fill missing values.

        Pandas can keep mapped output as categorical dtype when input is
        categorical. Filling missing values with a float fallback then raises:
        "Cannot setitem on a Categorical with a new category".
        """
        mapped = cat_series.map(mapping)
        mapped = pd.to_numeric(pd.Series(mapped, index=cat_series.index), errors="coerce")
        return mapped.fillna(fallback)

    def fit(self, X, y=None):
        if self.cat_col not in X.columns or self.num_col not in X.columns:
            self.mapping_ = pd.Series(dtype=float)
            self.global_fallback_ = 0.0
            return self
        
        if self.agg_func == "zscore":
            # For zscore, store both mean and std
            self.group_mean_ = X.groupby(self.cat_col)[self.num_col].mean()
            self.group_std_ = X.groupby(self.cat_col)[self.num_col].std().fillna(1e-8)
            self.global_mean_ = X[self.num_col].mean()
            self.global_std_ = max(X[self.num_col].std(), 1e-8)
        elif self.agg_func == "rank":
            # For rank, store sorted train-only distributions and score by
            # empirical percentile at transform time to avoid transductive leakage.
            self.rank_values_by_group_ = {}
            grouped = X.groupby(self.cat_col)[self.num_col]
            for group_value, series in grouped:
                values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
                self.rank_values_by_group_[group_value] = np.sort(values)

            all_values = pd.to_numeric(X[self.num_col], errors="coerce").dropna().to_numpy(dtype=float)
            self.global_rank_values_ = np.sort(all_values)
            self.global_fallback_ = 0.5
        else:
            self.mapping_ = X.groupby(self.cat_col)[self.num_col].agg(self.agg_func)
            self.global_fallback_ = self.mapping_.mean() if self.agg_func != "count" else 0.0
        return self

    def transform(self, X):
        if self.cat_col not in X.columns or self.num_col not in X.columns:
            return pd.DataFrame({self.output_col: np.zeros(len(X))}, index=X.index)
        
        if self.agg_func == "zscore":
            group_means = self._map_numeric_with_fallback(X[self.cat_col], self.group_mean_, self.global_mean_)
            group_stds = self._map_numeric_with_fallback(X[self.cat_col], self.group_std_, self.global_std_)
            result = (X[self.num_col] - group_means) / (group_stds + 1e-8)
        elif self.agg_func == "rank":
            # Train-distribution percentile rank to keep transform batch-independent.
            group_values = self.rank_values_by_group_ if self.rank_values_by_group_ is not None else {}
            global_values = self.global_rank_values_
            numeric = pd.to_numeric(X[self.num_col], errors="coerce")

            out = np.empty(len(X), dtype=float)
            cats = X[self.cat_col].values
            vals = numeric.values

            for idx, (cat_value, num_value) in enumerate(zip(cats, vals)):
                rank_values = group_values.get(cat_value)
                if rank_values is None:
                    rank_values = global_values
                percentile = self._empirical_percentile(rank_values, num_value)
                if pd.isna(percentile):
                    percentile = self.global_fallback_
                out[idx] = percentile

            result = pd.Series(out, index=X.index)
        else:
            result = self._map_numeric_with_fallback(X[self.cat_col], self.mapping_, self.global_fallback_)
        
        return pd.DataFrame({self.output_col: result.values}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array([self.output_col])


class TemporalEncoder(BaseEstimator, TransformerMixin):
    """Fit-transform temporal/lag features within CV folds to prevent leakage.
    
    Computes time-series features (lags, rolling stats, momentum, pct_change)
    grouped by entity ID and sorted by time column. Only uses backward-looking
    operations to prevent future data leakage.
    
    The op_name encodes both the operation type and window size, e.g.:
      - 'lag_3'          → shift by 3
      - 'rolling_mean_7' → rolling mean with window 7
      - 'momentum_12'    → value - lag_12
    """
    
    # Regex patterns: op_type → (regex, has_window)
    _OP_PATTERNS = [
        ("rolling_mean_", "rolling_mean"),
        ("rolling_std_",  "rolling_std"),
        ("pct_change_",   "pct_change"),
        ("momentum_",     "momentum"),
        ("lag_",          "lag"),
    ]
    
    def __init__(self, col, id_col, time_col, op_name, output_col=None,
                 strict_no_leakage=True):
        self.col = col
        self.id_col = id_col
        self.time_col = time_col
        self.op_name = op_name
        self.output_col = output_col or f"{op_name}({col})"
        self.strict_no_leakage = bool(strict_no_leakage)
        self.global_fallback_ = None
        self.id_history_ = {}
        self.global_history_ = np.array([], dtype=float)
        
        # Parse op_type and window from op_name
        self.op_type, self.window = self._parse_op_name(op_name)

    @staticmethod
    def _parse_op_name(op_name):
        """Extract (op_type, window) from names like 'rolling_mean_7' or 'lag_3'."""
        for prefix, op_type in TemporalEncoder._OP_PATTERNS:
            if op_name.startswith(prefix):
                try:
                    window = int(op_name[len(prefix):])
                    return op_type, window
                except ValueError:
                    pass
        return op_name, 1  # Fallback

    @staticmethod
    def _history_lag(history, window):
        if history is None or len(history) < window or window < 1:
            return np.nan
        return float(history[-window])

    @staticmethod
    def _history_tail(history, window):
        if history is None or len(history) == 0:
            return np.array([], dtype=float)
        width = int(min(window, len(history)))
        return history[-width:]

    def _get_history_for_id(self, entity_id):
        history = self.id_history_.get(entity_id)
        if history is None or len(history) == 0:
            history = self.global_history_
        return history

    def fit(self, X, y=None):
        if self.col not in X.columns or self.id_col not in X.columns or self.time_col not in X.columns:
            self.global_fallback_ = 0.0
            return self
            
        w = self.window
        X_sorted = X.sort_values([self.id_col, self.time_col]).copy()
        X_sorted[self.col] = pd.to_numeric(X_sorted[self.col], errors="coerce")

        # Persist train-only per-entity history for strict no-leakage transforms.
        self.id_history_ = {}
        for entity_id, series in X_sorted.groupby(self.id_col)[self.col]:
            values = series.dropna().to_numpy(dtype=float)
            self.id_history_[entity_id] = values

        all_values = X_sorted[self.col].dropna().to_numpy(dtype=float)
        self.global_history_ = all_values

        grouped = X_sorted.groupby(self.id_col)[self.col]
        
        if self.op_type == "lag":
            result = grouped.shift(w)
        elif self.op_type == "rolling_mean":
            result = grouped.transform(lambda x: x.rolling(w, min_periods=1).mean())
        elif self.op_type == "rolling_std":
            result = grouped.transform(lambda x: x.rolling(w, min_periods=1).std())
        elif self.op_type == "momentum":
            result = X_sorted[self.col] - grouped.shift(w)
        elif self.op_type == "pct_change":
            result = grouped.pct_change(w)
        else:
            result = pd.Series(np.zeros(len(X)), index=X_sorted.index)
            
        self.global_fallback_ = result.median()
        if pd.isna(self.global_fallback_) or np.isinf(self.global_fallback_):
            self.global_fallback_ = 0.0
            
        return self

    def transform(self, X):
        if self.col not in X.columns or self.id_col not in X.columns or self.time_col not in X.columns:
            return pd.DataFrame({self.output_col: np.zeros(len(X))}, index=X.index)
        
        if self.strict_no_leakage:
            w = self.window
            numeric = pd.to_numeric(X[self.col], errors="coerce")
            out = np.empty(len(X), dtype=float)

            entity_values = X[self.id_col].values
            current_values = numeric.values

            for idx, (entity_id, current_value) in enumerate(zip(entity_values, current_values)):
                history = self._get_history_for_id(entity_id)

                if self.op_type == "lag":
                    value = self._history_lag(history, w)
                elif self.op_type == "rolling_mean":
                    tail = self._history_tail(history, w)
                    value = float(np.mean(tail)) if len(tail) else np.nan
                elif self.op_type == "rolling_std":
                    tail = self._history_tail(history, w)
                    value = float(np.std(tail, ddof=1)) if len(tail) > 1 else np.nan
                elif self.op_type == "momentum":
                    lag_value = self._history_lag(history, w)
                    if np.isfinite(current_value) and np.isfinite(lag_value):
                        value = float(current_value - lag_value)
                    else:
                        value = np.nan
                elif self.op_type == "pct_change":
                    lag_value = self._history_lag(history, w)
                    if np.isfinite(current_value) and np.isfinite(lag_value) and abs(lag_value) > 1e-12:
                        value = float((current_value / lag_value) - 1.0)
                    else:
                        value = np.nan
                else:
                    value = 0.0

                if not np.isfinite(value):
                    value = self.global_fallback_
                out[idx] = value

            result = pd.Series(out, index=X.index)
            return pd.DataFrame({self.output_col: result.values}, index=X.index)

        w = self.window
        
        # Sort by time within groups
        X_sorted = X.sort_values([self.id_col, self.time_col]).copy()
        X_sorted[self.col] = pd.to_numeric(X_sorted[self.col], errors="coerce")
        grouped = X_sorted.groupby(self.id_col)[self.col]
        
        if self.op_type == "lag":
            result = grouped.shift(w)
        elif self.op_type == "rolling_mean":
            result = grouped.transform(lambda x: x.rolling(w, min_periods=1).mean())
        elif self.op_type == "rolling_std":
            result = grouped.transform(lambda x: x.rolling(w, min_periods=1).std())
        elif self.op_type == "momentum":
            result = X_sorted[self.col] - grouped.shift(w)
        elif self.op_type == "pct_change":
            result = grouped.pct_change(w)
        else:
            result = pd.Series(np.zeros(len(X)), index=X_sorted.index)
        
        # Fill NaN from insufficient history
        result = result.fillna(self.global_fallback_)
        # Replace inf values
        result = result.replace([np.inf, -np.inf], self.global_fallback_)
        
        # Re-index to match original X order
        result = result.reindex(X.index)
        
        return pd.DataFrame({self.output_col: result.values}, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array([self.output_col])
