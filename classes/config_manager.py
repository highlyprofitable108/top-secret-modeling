import yaml


class ConfigManager:
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.config_data = self.load_config()

    def load_config(self):
        with open(self.config_path, 'r') as stream:
            return yaml.safe_load(stream)

    def get_config(self, section, key=None):
        if key:
            return self.config_data.get(section, {}).get(key)
        return self.config_data.get(section)

    def get_json_dir(self):
        return self.config_data['paths']['json_dir']
    
    def get_data_dir(self):
        return self.config_data['paths']['data_dir']

    def get_db_path(self):
        return self.config_data['database']['db_path']