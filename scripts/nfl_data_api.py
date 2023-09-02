from classes.config_manager import ConfigManager
from classes.api_handler import APIHandler

config_manager = ConfigManager()
api_config = config_manager.get_config("nfl_api")
api_handler = APIHandler(api_config["base_url"], api_config["api_key"])

# Fetch game schedule
game_schedule = api_handler.fetch_game_schedule(
    2018, "REG", api_config["endpoint_schedule"]
)
