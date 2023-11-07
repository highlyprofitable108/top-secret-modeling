from classes.api_handler import APIHandler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def validate_parameters(year: int, season_type: str) -> bool:
    """Validate the input parameters."""
    if not isinstance(year, int) or year < 2000:
        logging.error("Invalid year parameter")
        return False
    if season_type not in ["REG", "PST"]:
        logging.error("Invalid season_type parameter")
        return False
    return True


def validate_configuration(api_config: dict) -> bool:
    """Validate the API configuration."""
    if not api_config or "base_url" not in api_config or "api_key" not in api_config or "endpoint_schedule" not in api_config:
        logging.error("Invalid API configuration")
        return False
    return True


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
        api_handler = APIHandler()

        if not validate_parameters(year, season_type):
            return None

        # Fetch game schedule
        game_schedule = api_handler.fetch_game_schedule(year, season_type)

        logging.info("Successfully fetched NFL data")
        return game_schedule
    except Exception as e:
        logging.error(f"Error fetching NFL data: {e}")
        return None


if __name__ == "__main__":
    # Example usage:
    fetch_nfl_data(2023, "REG")
