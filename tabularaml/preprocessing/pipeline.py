import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder, 
                                   MinMaxScaler,  StandardScaler,
                                   FunctionTransformer)
                                   
from tabularaml.preprocessing.imputers import SimpleImputer, AdvancedImputer


class PipelineWrapper:
    """
    Data preprocessing pipeline wrapper.
    Used for feature meta-data detection (dtypes and necessary conversions), 
    imputing missing values, encoding categorical features etc.
    """

    def __init__(self,
                 imputer = AdvancedImputer(strategy='auto'),
                 scaler = MinMaxScaler(),
                 encoder = OneHotEncoder(sparse_output = False,
                                         handle_unknown = "ignore"),
                
        ):

        """
        Parameters:
        -----------
        * imputer: Missing values imputer. Default is AdvancedImputer with auto strategy
        (intelligently selects appropriate imputation method for each column).
        * scaler: Numerical features scaler. Default is MinMaxScaler.
        * encoder: Categorical features encoder. Default is OneHotEncoder.
        * cols_to_scale: List of numerical columns to scale. If None, scale
        
        All transformers must be SciKit-Learn API-compatible. If None is passed
        instead, preprocessing step will be skipped.
        """
        self.imputer = imputer
        self.scaler = scaler
        self.encoder = encoder



    def get_pipeline(self, X, y=None):

        self._get_dtypes(X)

        # Fallbacks for None
        scaler = self.scaler if self.scaler is not None else "passthrough"
        encoder = self.encoder if self.encoder is not None else "passthrough"
        imputer = self.imputer if self.imputer is not None else FunctionTransformer()

        # ColumnTransformer
        ct = ColumnTransformer(
            transformers = [
                ("scaling", scaler, self.numerical_columns),
                ("encoding", encoder, self.categorical_columns),
            ],
            remainder = "passthrough",
            verbose_feature_names_out = False
        )
        ct.set_output(transform="pandas")

        # Final pipeline
        pipeline = Pipeline([
            ("imputing", imputer),
            ("scaling_encoding", ct)
        ])

        return pipeline
            


    def _get_dtypes(self, X: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Internal method. Used to infer features' datatypes and make certain conversion(s)
        (like numbers/percentages from string to float). 
        Drops ID column(s) and non-categorical text features based on the self.drop_non_categorical_text flag.

        Parameters:
            X (pd.DataFrame): Input pandas DataFrame.

        Returns:
            (pd.DataFrame, dict): Modified dataframe and dictionary mapping column names to data types.
        """
        # Create an explicit copy of the dataframe to avoid SettingWithCopyWarning
        X = X.copy()
        
        dtypes_dict = {}
        dropped_columns = []
        
        # Process each column to determine data type and perform conversions
        for col in X.columns:
            # Handle object or categorical types
            if X[col].dtype == np.dtype('O') or isinstance(X[col].dtype, pd.CategoricalDtype):
                # Determine if column is likely categorical based on unique value ratio
                is_likely_categorical = X[col].nunique() / len(X[col]) < 0.1
                
                if is_likely_categorical:
                    X.loc[:, col] = self._convert_to_category(X[col])
                    dtypes_dict[col] = "cat"
                else:
                    # Try numeric conversion
                    numeric_result = self._try_numeric_conversion(X[col])
                    if numeric_result is not None:
                        X.loc[:, col] = numeric_result
                        dtypes_dict[col] = "float" if X[col].dtype == np.float64 else "int"
                    else:
                        # Handle non-categorical text
                        if hasattr(self, 'drop_non_categorical_text') and self.drop_non_categorical_text:
                            dropped_columns.append(col)
                        else:
                            X.loc[:, col] = self._convert_to_category(X[col])
                            dtypes_dict[col] = "cat"
            
            # Handle numeric types directly
            elif np.issubdtype(X[col].dtype, np.integer):
                dtypes_dict[col] = "int"
            elif np.issubdtype(X[col].dtype, np.floating):
                dtypes_dict[col] = "float"
            elif isinstance(X[col].dtype, pd.CategoricalDtype):
                dtypes_dict[col] = "cat"
        
        # Add ID columns to dropped list
        for col in list(X.columns):
            if col.lower() == "id" and col not in dropped_columns:
                dropped_columns.append(col)
        
        # Drop columns and update the dtypes dictionary
        if dropped_columns:
            X = X.drop(columns=dropped_columns)
            for col in dropped_columns:
                if col in dtypes_dict:
                    del dtypes_dict[col]
        
        # Store dropped, numerical, categorical columns for reference
        self.dropped_columns = dropped_columns
        self.numerical_columns = [col for col, dtype in dtypes_dict.items() 
                                  if dtype in ["int", "float"]]
        self.categorical_columns = [col for col, dtype in dtypes_dict.items()
                                    if dtype == "cat"]
        
        self.dtypes_dict = dtypes_dict
    
    def _convert_to_category(self, series):
        """Helper method to convert a series to a categorical type with missing value handling."""
        if not isinstance(series.dtype, pd.CategoricalDtype):
            return series.fillna('missing').astype('category')
        else:
            if 'missing' not in series.cat.categories:
                series = series.cat.add_categories('missing')
            return series.fillna('missing')
    
    def _try_numeric_conversion(self, series):
        """Helper method to try different numeric conversions on a series."""
        conversions = [
            lambda x: pd.to_numeric(x),  # Direct numeric conversion
            lambda x: pd.to_numeric(x.str.replace(',', '')),  # Comma as thousands separator
            lambda x: pd.to_numeric(x.str.strip().str.rstrip('%')) / 100,  # Percentage
        ]
        
        for conversion in conversions:
            try:
                return conversion(series)
            except (ValueError, TypeError):
                continue
        
        return None
    



MODELS_PIPELINES_WRAPPERS = {
    "XGB": PipelineWrapper(imputer = None,
                           scaler = None,
                           encoder = None),
    "LGB": PipelineWrapper(imputer = None,
                           scaler = None,
                           encoder = None),
    "CAT": PipelineWrapper(imputer = None,
                           scaler = None,
                           encoder = None)
}




