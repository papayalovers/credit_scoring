from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder, FunctionTransformer,
    RobustScaler
)
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import logging 

logger = logging.getLogger(__name__)


class OrdinalMapper(BaseEstimator, TransformerMixin):
    """
    Custom transformer for ordinal encoding using predefined mapping.

    Parameters
    ----------
    mapping : dict
        Dictionary where keys are column names and values are mapping dicts.
        Example:
        {
            "loan_grade": {"A": 1, "B": 2, "C": 3}
        }

    Raises
    ------
    ValueError
        If mapping is empty or if unseen categories are found during transform.
    """

    def __init__(self, mapping: dict):
        self.mapping = mapping
    
    def fit(self, X, y=None):
        """Validate mapping before transformation."""
        if not self.mapping:
            raise ValueError("Ordinal mapping is empty! Please check your config.")
        
        self.is_fitted_ = True
        return self 
    
    def transform(self, X):
        """Apply ordinal mapping to specified columns."""
        X = X.copy()

        for col, map_dict in self.mapping.items():
            if col not in X.columns:
                raise ValueError(f"Column '{col}' not found in input data during Ordinal Mapping")

            X[col] = X[col].map(map_dict)

            # Check for unseen categories
            if X[col].isna().any():
                raise ValueError(
                    f"Column '{col}' contains unseen categories not in mapping"
                )

        return X

    def set_output(self, transform=None):
        """Compatibility method for sklearn set_output API."""
        return self


class Preprocessing:
    """
    Preprocessing pipeline builder for machine learning workflow.

    This class constructs a ColumnTransformer that applies:
    - Numerical pipeline: Imputation + (optional) Log Transform + Robust Scaling
    - Nominal categorical pipeline: OneHotEncoding
    - Ordinal categorical pipeline: Custom Ordinal Mapping

    Parameters
    ----------
    config : dict
        Configuration dictionary containing feature definitions:

    is_log : bool, optional (default=False)
        Whether to apply log1p transformation before scaling on numerical features.

    Raises
    ------
    ValueError
        If no features are defined in config.
    """

    def __init__(self, config: dict, is_log: bool = False):
        self.config = config
        self.is_log = is_log

        self.num_cols = config.get('features', {}).get('numerical', [])
        self.cat_nominal_cols = config.get('features', {}).get('cat_nominal', []) 
        self.cat_ordinal_map = config.get('features', {}).get('cat_ordinal', {})

        # Validation
        if not self.num_cols and not self.cat_nominal_cols and not self.cat_ordinal_map:
            raise ValueError("No features defined in config. Please check config['features']")

        logger.info(f"Numerical cols: {self.num_cols}")
        logger.info(f"Nominal cols: {self.cat_nominal_cols}")
        logger.info(f"Ordinal cols: {list(self.cat_ordinal_map.keys())}")

    def num_pipeline(self):
        """
        Build numerical preprocessing pipeline.

        Steps:
        - Impute missing values using median
        - Optional log transformation
        - Apply RobustScaler

        Returns
        -------
        Pipeline
        """
        steps = [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ]

        if self.is_log:
            steps.insert(1, ("log", FunctionTransformer(np.log1p)))
        
        return Pipeline(steps)
    
    def cat_nom_pipeline(self):
        """
        Build nominal categorical pipeline using OneHotEncoder.

        Returns
        -------
        Pipeline
        """
        steps = [
            ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
        ]
        return Pipeline(steps)
    
    def cat_ord_pipeline(self):
        """
        Build ordinal categorical pipeline using custom OrdinalMapper.

        Returns
        -------
        Pipeline
        """
        steps = [
            ('mapper', OrdinalMapper(self.cat_ordinal_map))
        ]
        return Pipeline(steps)

    def build_pipeline(self):
        """
        Construct full preprocessing pipeline using ColumnTransformer.

        Returns
        -------
        ColumnTransformer
            Combined preprocessing pipeline for all feature types.
        """
        transformers = []

        # Numerical
        if self.num_cols:
            transformers.append(
                ("num", self.num_pipeline(), self.num_cols)
            )
        else:
            logger.warning("No numerical columns provided!")

        # Nominal
        if self.cat_nominal_cols:
            transformers.append(
                ("cat_nom", self.cat_nom_pipeline(), self.cat_nominal_cols)
            )
        else:
            logger.warning("No nominal categorical columns provided!")

        # Ordinal
        if self.cat_ordinal_map:
            transformers.append(
                ("cat_ord", self.cat_ord_pipeline(), list(self.cat_ordinal_map.keys()))
            )
        else:
            logger.warning("No ordinal mapping provided!")

        if not transformers:
            raise ValueError("No transformers available to build pipeline!")

        if self.is_log:
            logger.info("Pipeline flow: OHE -> Ordinal -> Log -> RobustScaler")
        else:
            logger.info("Pipeline flow: OHE -> Ordinal -> RobustScaler")

        return ColumnTransformer(transformers=transformers)