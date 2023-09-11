import logging
from classes.config_manager import ConfigManager
from classes.api_handler import APIHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_nfl_data(year: int, season_type: str) -> dict:
    """
    Fetches NFL data for a given year and season type.

    Parameters:
    year (int): The year for which to fetch the data.
    season_type (str): The type of the season (e.g., "REG" for regular season).

    Returns:
    dict: The game schedule data.
    """
    try:
        # Validate parameters
        if not isinstance(year, int) or year < 2000:
            logging.error("Invalid year parameter")
            return None
        if season_type not in ["REG", "POST"]:
            logging.error("Invalid season_type parameter")
            return None

        config_manager = ConfigManager()
        api_config = config_manager.get_config("nfl_api")

        # Validate configuration
        if not api_config or "base_url" not in api_config or "api_key" not in api_config or "endpoint_schedule" not in api_config:
            logging.error("Invalid API configuration")
            return None

        api_handler = APIHandler(api_config["base_url"], api_config["api_key"])

        # Fetch game schedule
        game_schedule = api_handler.fetch_game_schedule(year, season_type, api_config["endpoint_schedule"])

        # Validate response
        if not game_schedule:
            logging.error("Failed to fetch game schedule")
            return None

        logging.info("Successfully fetched NFL data")
        return game_schedule
    except Exception as e:
        logging.error(f"Error fetching NFL data: {e}")
        return None

if __name__ == "__main__":
    # Example usage:
    fetch_nfl_data(2020, "REG")
