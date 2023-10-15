import yaml
import logging

class ConfigManager:
    """
    A class to manage the configuration settings from a YAML file.
    """
    
    def __init__(self, config_path="config.yaml"):
        """
        Initializes the ConfigManager with the specified configuration file path.
        
        Args:
            config_path (str): The path to the configuration file. Defaults to "config.yaml".
        """
        self.config_path = config_path
        self.config_data = self.load_config()
        self.logger = logging.getLogger(__name__)

    def load_config(self) -> dict:
        """
        Loads the configuration data from the YAML file.
        
        Returns:
            dict: The configuration data as a dictionary.
        """
        try:
            with open(self.config_path, 'r') as stream:
                return yaml.safe_load(stream)
        except FileNotFoundError:
            self.logger.error(f"Error: The configuration file '{self.config_path}' was not found.")
            return {}
        except yaml.YAMLError as exc:
            self.logger.error(f"Error: Could not parse the configuration file: {exc}")
            return {}

    def get_config(self, section: str = None, key: str = None) -> dict:
        """
        Gets a configuration setting from the specified section and key.
        
        Args:
            section (str): The section in the configuration data. Defaults to None.
            key (str): The key in the section. Defaults to None.
        
        Returns:
            dict: The configuration setting.
        """
        try:
            if section is None:
                return self.config_data
            elif key:
                return self.config_data.get(section, {}).get(key)
            else:
                return self.config_data.get(section)
        except AttributeError:
            self.logger.error("Error: Configuration data is not loaded correctly.")
            return None

    def get_constant(self, key: str) -> any:
        """Gets a constant value from the configuration data."""
        return self.get_config('constants', key)

    def get_model_settings(self, key: str = None) -> any:
        """
        Gets an EDA setting from the configuration data.

        Args:
            key (str): The key for the specific EDA setting. If not provided, returns the entire EDA settings.

        Returns:
            any: The EDA setting value.
        """
        model_settings = self.get_config('model_settings')
        if key:
            return model_settings.get(key)
        return model_settings

    def get_json_dir(self) -> str:
        """Gets the JSON directory path from the configuration data."""
        return self.get_config('paths', 'json_dir')

    def odds_dir(self) -> str:
        """Gets the JSON directory path from the configuration data."""
        return self.get_config('paths', 'odds_dir')

    def get_data_dir(self) -> str:
        """Gets the data directory path from the configuration data."""
        return self.get_config('paths', 'data_dir')

    def get_model_dir(self) -> str:
        """Gets the model directory path from the configuration data."""
        return self.get_config('paths', 'model_dir')
    
    def get_static_dir(self) -> str:
        """Gets the static directory path from the configuration data."""
        return self.get_config('paths', 'static_dir')

    def get_db_path(self) -> str:
        """Gets the database path from the configuration data."""
        return self.get_config('database', 'db_path')

    def get_mongo_uri(self) -> str:
        """Gets the MongoDB URI from the configuration data."""
        return self.get_config('database', 'mongo_uri')

    def get_database_name(self) -> str:
        """Gets the MongoDB database name from the configuration data."""
        return self.get_config('database', 'database_name')
