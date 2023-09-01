import requests
import time
import json
import os
import yaml

# Load the configuration
with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# Constants
BASE_URL = config['nfl_api']['base_url']
API_KEY = config['nfl_api']['api_key']
HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

# Define base directory for data
BASE_DIR = os.path.expandvars(config['default']['base_dir'])
DATA_DIR = os.path.join(BASE_DIR, 'data')
JSON_DIR = os.path.join(DATA_DIR, 'jsons')


def fetch_game_schedule(year, season):
    endpoint = config['nfl_api']['base_url']+config['nfl_api']['endpoint_schedule'].format(year=year, season=season) + f"?api_key={API_KEY}"
    response = requests.get(endpoint, headers=HEADERS)
    file_path = os.path.join(DATA_DIR, 'game_schedule.json')
    with open(file_path, 'w') as file:
        data = response.json()
        json.dump(data, file)
        print("Data saved to game_schedule.json")
    # Extract all game_id values and store them in a list
    game_ids = [game['id'] for week in data.get('weeks', []) for game in week.get('games', [])]
    # Loop through the game_ids list and call fetch_game_statistics for each game_id
    time.sleep(2)
    for game_id in game_ids:
        fetch_game_statistics(game_id)


def fetch_game_statistics(game_id):
    endpoint = config['nfl_api']['base_url']+config['nfl_api']['endpoint_stats'].format(game_id=game_id) + f"?api_key={API_KEY}"
    response = requests.get(endpoint, headers=HEADERS)
    # Incorporate game_id into the filename to ensure uniqueness
    file_path = os.path.join(JSON_DIR, f'game_stats_{game_id}.json')
    with open(file_path, 'w') as file:
        json.dump(response.json(), file)
        print(f"Data saved to game_stats_{game_id}.json")
    time.sleep(2)

# Uncomment the below lines if you want to fetch data for specific years and seasons
fetch_game_schedule(2019, "REG")

fetch_game_schedule(2018, "REG")

# fetch_game_schedule(2017, "REG")

# fetch_game_schedule(2016, "REG")
