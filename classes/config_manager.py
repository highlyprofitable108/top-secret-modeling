import yaml


class ConfigManager:
    """A class to manage the configuration settings from a YAML file."""
    
    def __init__(self, config_path="config.yaml"):
        """Initializes the ConfigManager with the specified configuration file path."""
        self.config_path = config_path
        self.config_data = self.load_config()

    def load_config(self):
        """Loads the configuration data from the YAML file."""
        try:
            with open(self.config_path, 'r') as stream:
                return yaml.safe_load(stream)
        except FileNotFoundError:
            print(f"Error: The configuration file '{self.config_path}' was not found.")
            return {}
        except yaml.YAMLError as exc:
            print(f"Error: Could not parse the configuration file: {exc}")
            return {}

    def get_config(self, section=None, key=None):
        """Gets a configuration setting from the specified section and key."""
        try:
            if section is None:
                return self.config_data
            elif key:
                return self.config_data.get(section, {}).get(key)
            else:
                return self.config_data.get(section)
        except AttributeError:
            print("Error: Configuration data is not loaded correctly.")
            return None

    def get_json_dir(self):
        """Gets the JSON directory path from the configuration data."""
        return self.get_config('paths', 'json_dir')

    def get_data_dir(self):
        """Gets the data directory path from the configuration data."""
        return self.get_config('paths', 'data_dir')

    def get_model_dir(self):
        """Gets the model directory path from the configuration data."""
        return self.get_config('paths', 'model_dir')

    def get_db_path(self):
        """Gets the database path from the configuration data."""
        return self.get_config('database', 'db_path')

    def get_mongo_uri(self):
        """Gets the MongoDB URI from the configuration data."""
        return self.get_config('database', 'mongo_uri')

    def get_database_name(self):
        """Gets the MongoDB database name from the configuration data."""
        return self.get_config('database', 'database_name')
