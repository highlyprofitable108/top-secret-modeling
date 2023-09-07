from classes.config_manager import ConfigManager
from classes.api_handler import APIHandler


def fetch_nfl_data(year, season_type):
    try:
        config_manager = ConfigManager()
        api_config = config_manager.get_config("nfl_api")
        api_handler = APIHandler(api_config["base_url"], api_config["api_key"])

        # Fetch game schedule
        game_schedule = api_handler.fetch_game_schedule(
            year, season_type, api_config["endpoint_schedule"]
        )
        return game_schedule
    except Exception as e:
        print(f"Error fetching NFL data: {e}")
        return None


# Example usage:
# fetch_nfl_data(2018, "REG")
