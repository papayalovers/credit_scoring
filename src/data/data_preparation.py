import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataPreparation:
    def __init__(self, config: dict) -> None:
        self.config = config.copy()
        self.schema = config.get('schema', {})
        self.path_raw_data = config.get('path', {}).get('raw_data', '')

    def load_raw_data(self) -> pd.DataFrame:
        """Load data from the specified path."""
        df = pd.read_csv(self.path_raw_data)
        logger.info(f"Data loaded from {self.path_raw_data} with shape {df.shape}")
        return df
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """ 
        Validate data against the schema.

        Parameters:
            data (pd.DataFrame): The data to validate.

        Returns:
            bool: True if data is valid, False otherwise.
        """
        TYPE_MAPPING = {
            'integer': ['int64', 'int32'],
            'decimal': ['float64', 'float32'],
            'string': ['object', 'string'],
        }

        # 
        is_valid = True

        for column, column_schema in self.schema.items():
            # Check data type
            data_type = data[column].dtype
            expected_type = column_schema.get('type', None)
            if expected_type in TYPE_MAPPING:
                if data_type not in TYPE_MAPPING[expected_type]:
                    logger.warning(f"Column '{column}' has type '{data_type}', expected '{expected_type}'")
                    is_valid = False
            else:
                logger.warning(f"Column '{column}' has unexpected type '{data_type}'")
                is_valid = False

            # Check minimum value
            if 'minimum' in column_schema:
                min_value = column_schema['minimum']
                if not data[column].dropna().ge(min_value).all():
                    logger.warning(f"Column '{column}' has values below minimum of {min_value}")
                    is_valid = False
            
            # Check maximum value
            if 'maximum' in column_schema:
                max_value = column_schema['maximum']
                if not data[column].dropna().le(max_value).all():
                    logger.warning(f"Column '{column}' has values above maximum of {max_value}")
                    is_valid = False
            
            # Check enum values
            if 'enum' in column_schema:
                enum_values = set(column_schema['enum'])
                data_values = set(data[column].dropna().unique())
                invalid_values = data_values - enum_values
                if invalid_values:
                    logger.warning(f"Column '{column}' has invalid enum values: {invalid_values}")
                    is_valid = False
        
        return is_valid

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the data by minimum and maximum values based on the schema after check by notebooks.

        Parameters:
            data (pd.DataFrame): The data to clean.

        Returns:
            pd.DataFrame: The cleaned data.
        """
        data = data.copy()
        # Store the index to remove
        idx_to_drop = set()

        for column, rules in self.schema.items():
            # Clean data based on minimum and maximum values
            min_value = rules.get('minimumm', None)
            max_value = rules.get('maximum', None)

            if min_value is not None and max_value is not None:
                invalid = data[(data[column]<min_value) | (data[column]>max_value)]
                invalid_count = len(invalid)
                if invalid_count > 0:
                    idx_invalid = set(invalid.index)
                    idx_to_drop.update(idx_invalid)
                    logger.info(f"Column '{column}' has {invalid_count} invalid rows. Indices marked for removal.")            

            if min_value is not None and max_value is None:
                invalid = data[data[column]<min_value]
                invalid_count = len(invalid)
                if invalid_count > 0:
                    idx_invalid = set(invalid.index)
                    idx_to_drop.update(idx_invalid)
                    logger.info(f"Column '{column}' has {invalid_count} invalid rows. Indices marked for removal.")            
            
            if max_value is not None and min_value is None:
                invalid = data[data[column]>max_value]
                invalid_count = len(invalid)
                if invalid_count > 0:
                    idx_invalid = set(invalid.index)
                    idx_to_drop.update(idx_invalid)
                    logger.info(f"Column '{column}' has {invalid_count} invalid rows. Indices marked for removal.")            
        
        data = data.drop(list(idx_to_drop), axis=0)
        return data
    
    def data_defense(self, data: pd.DataFrame) -> None:
        """Validate input data based on features configuration.
        
        Parameters:
            data (pd.DataFrame): The input data to validate.
        Returns:
            None
        """
        data = data.copy()
        cat_cols = self.config.get('features', {}).get('categorical', [])
        num_cols = self.config.get('features', {}).get('numerical', [])

        # Type Check
        for column in cat_cols:
            if column in data.columns:
                if not pd.api.types.is_object_dtype(data[column]):
                    logger.error(f"Column '{column}' is expected to be categorical but has type '{data[column].dtype}'")
                    raise ValueError(f"Column '{column}' is expected to be categorical but has type '{data[column].dtype}'")    
                
        for column in num_cols:
            if column in data.columns:
                if not pd.api.types.is_numeric_dtype(data[column]):
                    logger.error(f"Column '{column}' is expected to be numerical but has type '{data[column].dtype}'")
                    raise ValueError(f"Column '{column}' is expected to be numerical but has type '{data[column].dtype}'")
                
        # Check Ranges
        for col, rules in self.schema.items():
            if 'minimum' in rules and 'maximum' in rules:
                min_value = rules['minimum']
                max_value = rules['maximum']
                if not data[col].dropna().between(min_value, max_value).all():
                    logger.error(f"Column '{col}' has values outside the range [{min_value}, {max_value}]")
                    raise ValueError(f"Column '{col}' has values outside the range [{min_value}, {max_value}]")
            if 'minimum' in rules and 'maximum' not in rules:
                min_value = rules['minimum']
                max_value = data[col].max()
                if not data[col].dropna().between(min_value, max_value).all():
                    logger.error(f"Column '{col}' has values outside the minimum value [{min_value}]")
                    raise ValueError(f"Column '{col}' has values outside the minimum value [{min_value}]")
            if 'maximum' in rules and 'minimum' not in rules:
                min_value = data[col].min()
                max_value = rules['maximum']
                if not data[col].dropna().between(min_value, max_value).all():
                    logger.error(f"Column '{col}' has values outside the maximum value [{max_value}]")
                    raise ValueError(f"Column '{col}' has values outside the maximum value [{max_value}]")

    def split_input_output(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Split the data into input features and target variable.

        Parameters:
            data (pd.DataFrame): The data to split.

        Returns:
            tuple[pd.DataFrame, pd.Series]: The input features and target variable.
        """
        target_cols = self.config.get('target', {})

        if not target_cols:
            logger.error("No target column specified in the configuration.")
            raise ValueError("No target column specified in the configuration.")
        
        # Splitting input and output
        X = data.drop(columns=target_cols)
        y = data[target_cols[0]]
        return X, y
    
    def split_x_y(
            self, X: pd.DataFrame, 
            y: pd.Series, 
            test_size: float = 0.2, 
            random_state: int = 42
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split the data into training and testing sets.

        Parameters:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target variable.
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): Controls the shuffling applied to the data before applying the split.
        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: The training and testing datasets.
        """
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test