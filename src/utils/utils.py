import yaml
import joblib
import logging
from pathlib import Path

# Logger name
logger = logging.getLogger(__name__)
# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class ConfigManager:
    def __init__(self, config_path: str) -> None:
        self.config_path = BASE_DIR / config_path
        self.config = None

    def load_config(self) -> dict:
        """ 
        Load configuration parameters from a YAML file.

        Returns:
            dict: Configuration parameters.
        """
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Convert relative paths in config to absolute paths based on BASE_DIR
        for key, path in self.config.get("path", {}).items():
            self.config["path"][key] = str(BASE_DIR / path)

        return self.config

    def update_config(self, key: str, value: any) -> None:
        """
        Update a configuration parameter and save it back to the YAML file.
        Args:
            key (str): The configuration key to update (can be nested using dot notation).
            value (any): The new value for the configuration key.
        Returns:
            None
        """
        if self.config is None:
            raise RuntimeError("Config not loaded. Please load the config before updating.")

        # split if the key is nested
        keys = key.split(".")
        cfg = self.config

        for k in keys[:-1]:
            # auto create dict if key doesn't exist
            if k not in cfg or not isinstance(cfg[k], dict):
                cfg[k] = {}  
            cfg = cfg[k]

        cfg[keys[-1]] = value

        with open(self.config_path, 'w') as file:
            yaml.safe_dump(self.config, file)

        logger.info(f"Configuration updated: {key} = {value}")

    def serialized_data(self, object_to_serialize: any, path: str) -> None:
        """
        Serialize a Python object to a file using YAML.

        Args:
            object_to_serialize (any): The Python object to serialize.
            path (str): The file path where the serialized data will be saved.

        Returns:
            None
        """
        logger.info(f"Serializing data to {path}...")
        joblib.dump(object_to_serialize, path)
    
    def deserialize_data(self, path: str) -> any:
        """
        Deserialize a Python object from a YAML file.

        Args:
            path (str): The file path from which to deserialize the data. 
        Returns:
            any: The deserialized Python object.
        """
        logger.info(f"Deserializing data from {path}...")
        file_ = joblib.load(path)
        return file_