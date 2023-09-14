import requests
import logging


class APIHandler:
    """
    A class to handle API interactions.

    Attributes:
        base_url (str): The base URL for the API.
        api_key (str): The API key for authentication.
        headers (dict): The headers for the API requests.
    """

    def __init__(self, base_url, api_key):
        """
        Initializes the APIHandler with the base URL and API key.

        Args:
            base_url (str): The base URL for the API.
            api_key (str): The API key for authentication.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        self.logger = logging.getLogger(__name__)

    def fetch_game_schedule(self, year, season, endpoint):
        """
        Fetches the game schedule for a given year and season.

        Args:
            year (int): The year for which to fetch the schedule.
            season (str): The season for which to fetch the schedule.
            endpoint (str): The API endpoint for fetching the schedule.

        Returns:
            dict: The JSON response from the API.
        """
        url = f"{self.base_url}{endpoint.format(year=year, season=season)}?api_key={self.api_key}"
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            return response.json()
        else:
            self.logger.error(f"Failed to fetch game schedule: {response.status_code} - {response.text}")
            return None

    def fetch_game_statistics(self, game_id, endpoint):
        """
        Fetches the game statistics for a given game ID.

        Args:
            game_id (int): The ID of the game for which to fetch statistics.
            endpoint (str): The API endpoint for fetching the statistics.

        Returns:
            dict: The JSON response from the API.
        """
        url = f"{self.base_url}{endpoint.format(game_id=game_id)}?api_key={self.api_key}"
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            return response.json()
        else:
            self.logger.error(f"Failed to fetch game statistics: {response.status_code} - {response.text}")
            return None
