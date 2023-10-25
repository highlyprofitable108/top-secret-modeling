import requests
import logging
from datetime import datetime, timedelta
import time
import json
import os
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
        self.data_dir = self.config_manager.get_config('paths', 'data_dir')
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
        file_path = f"{self.data_dir}/game_schedule.json"
        data = response.json()

        with open(file_path, 'w') as file:
            json.dump(response.json(), file)
        print("Data saved to game_schedule.json")

        # Extract all game_id values and their corresponding scheduled dates and store them in a dictionary
        game_id_schedule_map = {game['id']: game['scheduled'] for week in data.get('weeks', []) for game in week.get('games', [])}

        # Loop through the game_ids list and call fetch_game_statistics for each game_id
        time.sleep(2)

        current_date = datetime.utcnow().date()  # Get the current date in UTC
        thirty_days_ago = current_date - timedelta(days=30)  # Calculate the date 30 days ago

        for game_id, scheduled in game_id_schedule_map.items():
            game_file_path = f"{self.json_dir}/game_stats_{game_id}.json"

            # Always download if the file doesn't exist
            should_download = not os.path.exists(game_file_path)

            # If the file exists, check the "scheduled" date in the JSON
            if not should_download:
                scheduled_date = datetime.fromisoformat(scheduled).date()
                if thirty_days_ago <= scheduled_date <= current_date:  # Check if the scheduled date is within the past 30 days up to the current date
                    print("Recent game, downloading....")
                    should_download = True

            if should_download:
                self.fetch_game_statistics(game_id)
            else:
                print("Archived game. Skipping....")

    def fetch_game_statistics(self, game_id):
        """
        Fetches the game statistics for a given game ID.

        Args:
            game_id (int): The ID of the game for which to fetch statistics.
            endpoint (str): The API endpoint for fetching the statistics.

        Returns:
            dict: The JSON response from the API.
        """
        # Incorporate game_id into the filename to ensure uniqueness
        file_path = f"{self.json_dir}/game_stats_{game_id}.json"

        endpoint = f"{self.base_url}/{game_id}/statistics.json?api_key={self.api_key}"
        response = requests.get(endpoint, headers=HEADERS)

        with open(file_path, 'w') as file:
            json.dump(response.json(), file)

        print(f"Data saved to game_stats_{game_id}.json")
        time.sleep(2)
