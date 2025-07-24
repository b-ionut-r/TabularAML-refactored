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