import requests
import logging
import time
import json
from classes.config_manager import ConfigManager

HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}


class APIHandler:
    """
    A class to handle API interactions.

    Attributes:
        base_url (str): The base URL for the API.
        api_key (str): The API key for authentication.
        headers (dict): The headers for the API requests.
    """

    def __init__(self):
        """
        Initializes the APIHandler with the base URL and API key.

        Args:
            base_url (str): The base URL for the API.
            api_key (str): The API key for authentication.
        """
        self.config_manager = ConfigManager()
        self.base_url = self.config_manager.get_config('nfl_api', 'base_url')
        self.api_key = self.config_manager.get_config('nfl_api', 'api_key')
        self.json_dir = self.config_manager.get_config('paths', 'json_dir')
        self.headers = HEADERS
        self.logger = logging.getLogger(__name__)

    def fetch_game_schedule(self, year, season):
        """
        Fetches the game schedule for a given year and season.

        Args:
            year (int): The year for which to fetch the schedule.
            season (str): The season for which to fetch the schedule.
            endpoint (str): The API endpoint for fetching the schedule.

        Returns:
            dict: The JSON response from the API.
        """
        endpoint = f"{self.base_url}/{year}/{season}/schedule.json?api_key={self.api_key}"
        response = requests.get(endpoint, headers=HEADERS)
        file_path = f"{self.json_dir}/game_schedule.json"
        data = response.json()

        with open(file_path, 'w') as file:
            json.dump(response.json(), file)
        print("Data saved to game_schedule.json")

        # Extract all game_id values and store them in a list
        game_ids = [game['id'] for week in data.get('weeks', []) for game in week.get('games', [])]

        # Loop through the game_ids list and call fetch_game_statistics for each game_id
        time.sleep(2)
        for game_id in game_ids:
            self.fetch_game_statistics(game_id)

    def fetch_game_statistics(self, game_id):
        """
        Fetches the game statistics for a given game ID.

        Args:
            game_id (int): The ID of the game for which to fetch statistics.
            endpoint (str): The API endpoint for fetching the statistics.

        Returns:
            dict: The JSON response from the API.
        """
        endpoint = f"{self.base_url}/{game_id}/statistics.json?api_key={self.api_key}"
        response = requests.get(endpoint, headers=HEADERS)

        # Incorporate game_id into the filename to ensure uniqueness
        file_path = f"{self.json_dir}/game_stats_{game_id}.json"

        with open(file_path, 'w') as file:
            json.dump(response.json(), file)

        print(f"Data saved to game_stats_{game_id}.json")
        time.sleep(2)
