import category_encoders as ce
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
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
        self.target_encoder = self._init_encoder(ce.TargetEncoder, self.target_enc_cols)
        self.count_encoder = self._init_encoder(ce.CountEncoder, self.count_enc_cols, normalize=False)
        self.freq_encoder = self._init_encoder(ce.CountEncoder, self.freq_enc_cols, normalize=True)

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

    def fit(self, X, y=None):
        # Convert to DataFrame and validate
        X_df = self._check_and_convert_X(X)
        self.feature_names_in_ = np.array(X_df.columns, dtype=object)
        self.n_features_in_ = X_df.shape[1]

        # Determine valid columns present in the data
        self._encoder_input_features = [col for col in self._all_configured_cols if col in X_df.columns]
        
        # Fit each encoder on the valid columns
        self._fit_encoder(self.target_encoder, X_df, y, self.target_enc_cols)
        self._fit_encoder(self.count_encoder, X_df, None, self.count_enc_cols)
        self._fit_encoder(self.freq_encoder, X_df, None, self.freq_enc_cols)

        self._feature_names_out = self._generate_output_feature_names(self.feature_names_in_)
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

    def _fit_encoder(self, encoder, X_df, y, cols):
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

    def transform(self, X):
        check_is_fitted(self)
        X_df = self._check_and_convert_X(X)
        cols_to_process = [col for col in self._encoder_input_features if col in X_df.columns]

        X_encoded = pd.DataFrame(index=X_df.index)
        X_encoded = self._transform_encoder(self.target_encoder, X_df, cols_to_process, X_encoded, '_target')
        X_encoded = self._transform_encoder(self.count_encoder, X_df, cols_to_process, X_encoded, '_count')
        X_encoded = self._transform_encoder(self.freq_encoder, X_df, cols_to_process, X_encoded, '_freq')

        if self.return_original:
            X_final = pd.concat([X_df, X_encoded], axis=1)
        else:
            X_final = X_encoded

        expected_cols = self._generate_output_feature_names(X_df.columns, only_new=not self.return_original)
        return X_final.reindex(columns=expected_cols, copy=False)

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
        new_features = []
        for col in input_features:
            if col in self.target_enc_cols:
                new_features.append(f"{col}_target")
            if col in self.count_enc_cols:
                new_features.append(f"{col}_count")
            if col in self.freq_enc_cols:
                new_features.append(f"{col}_freq")
        return np.array(input_features.tolist() + new_features if not only_new else new_features, dtype=object)

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
            # For rank, store the full distribution per group
            self.mapping_ = X.groupby(self.cat_col)[self.num_col].agg("mean")  # Fallback
            self.global_fallback_ = 0.5  # Median percentile rank
        else:
            self.mapping_ = X.groupby(self.cat_col)[self.num_col].agg(self.agg_func)
            self.global_fallback_ = self.mapping_.mean() if self.agg_func != "count" else 0.0
        return self

    def transform(self, X):
        if self.cat_col not in X.columns or self.num_col not in X.columns:
            return pd.DataFrame({self.output_col: np.zeros(len(X))}, index=X.index)
        
        if self.agg_func == "zscore":
            group_means = X[self.cat_col].map(self.group_mean_).fillna(self.global_mean_)
            group_stds = X[self.cat_col].map(self.group_std_).fillna(self.global_std_)
            result = (X[self.num_col] - group_means) / (group_stds + 1e-8)
        elif self.agg_func == "rank":
            # Rank within group using transform
            result = X.groupby(self.cat_col)[self.num_col].rank(pct=True)
            result = result.fillna(self.global_fallback_)
        else:
            result = X[self.cat_col].map(self.mapping_)
            result = result.fillna(self.global_fallback_)
        
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
    
    def __init__(self, col, id_col, time_col, op_name, output_col=None):
        self.col = col
        self.id_col = id_col
        self.time_col = time_col
        self.op_name = op_name
        self.output_col = output_col or f"{op_name}({col})"
        self.global_fallback_ = None
        
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

    def fit(self, X, y=None):
        if self.col not in X.columns or self.id_col not in X.columns or self.time_col not in X.columns:
            self.global_fallback_ = 0.0
            return self
            
        w = self.window
        X_sorted = X.sort_values([self.id_col, self.time_col])
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
        
        w = self.window
        
        # Sort by time within groups
        X_sorted = X.sort_values([self.id_col, self.time_col])
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
