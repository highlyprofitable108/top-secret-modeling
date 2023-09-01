import requests
import time
import json
import os

# Constants
BASE_URL = "https://api.sportradar.us/nfl/official/trial/v7/en/games"
API_KEY = "je658qgxcjqgjkn55r78z9g8"
HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

# Define base directory for data
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
JSON_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'jsons')


def fetch_game_schedule(year, season):
    endpoint = f"{BASE_URL}/{year}/{season}/schedule.json?api_key={API_KEY}"
    response = requests.get(endpoint, headers=HEADERS)
    file_path = os.path.join(DATA_DIR, 'game_schedule.json')
    with open(file_path, 'w') as file:
        data = response.json()
        json.dump(response.json(), file)
        print("Data saved to game_schedule.json")

    # Extract all game_id values and store them in a list
    game_ids = [game['id'] for week in data.get('weeks', []) for game in week.get('games', [])]

    # Loop through the game_ids list and call fetch_game_statistics for each game_id
    time.sleep(2)
    for game_id in game_ids:
        fetch_game_statistics(game_id)


# Fetch game statistics for a specific game
def fetch_game_statistics(game_id):
    endpoint = f"{BASE_URL}/{game_id}/statistics.json?api_key={API_KEY}"
    response = requests.get(endpoint, headers=HEADERS)

    # Incorporate game_id into the filename to ensure uniqueness
    file_path = os.path.join(JSON_DIR, f'game_stats_{game_id}.json')

    with open(file_path, 'w') as file:
        json.dump(response.json(), file)
        print(f"Data saved to game_stats_{game_id}.json")
        time.sleep(2)


"""
    if response.status_code == 200:
        data = response.json()
        for team_type in ['home', 'away']:
            team_data = data['statistics'][team_type]
            # team_id = team_data['id']

            # Extracting statistics
            summary = team_data['summary']
            rushing = team_data['rushing']
            passing = team_data['passing']
            penalties = team_data['penalties']
            receiving = team_data['receiving']
            field_goals = team_data['field_goals']
            defense = team_data['defense']
            # conversions = team_data['conversions']
            first_downs = team_data['first_downs']
            interceptions = team_data['interceptions']
            efficiency = team_data['efficiency']

            # Insert data into SQLite database
            cursor.execute("INSERT OR REPLACE INTO team_stats (game_id, team_type, third_down_efficiency, fourth_down_efficiency, total_yards, penalties, penalty_yards, turnovers, punts_avg, rushing_attempts, rushing_touchdowns, longest_rush, rushing_longest_touchdown, passing_attempts, passing_completions, passing_percentage, passing_touchdowns, passing_interceptions, passing_sacks, passing_sack_yards, qb_rating, longest_pass, passing_longest_touchdown, penalty_count, penalty_yards_gained, receptions, receiving_yards, receiving_touchdowns, longest_reception, field_goals_attempted, field_goals_made, longest_field_goal_made, tackles, sacks, interceptions, forced_fumbles, two_point_attempts, two_point_made, first_downs_pass, first_downs_rush, first_downs_penalty, interceptions_thrown, interceptions_returned, interceptions_touchdowns, redzone_attempts, redzone_successes, goal_to_go_attempts, goal_to_go_successes, third_down_attempts, third_down_successes, fourth_down_attempts, fourth_down_successes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (game_id, team_type, efficiency['thirddown']['attempts'], efficiency['thirddown']['successes'], summary['total_yards'], penalties['count'], penalties['yards'], summary['turnovers'], summary['punts_avg'], rushing['attempts'], rushing['touchdowns'], rushing['longest'], rushing['longest_touchdown'], passing['attempts'], passing['completions'], passing['percentage'], passing['touchdowns'], passing['interceptions'], passing['sacks'], passing['sack_yards'], passing['qb_rating'], passing['longest'], passing['longest_touchdown'], penalties['count'], penalties['yards'], receiving['receptions'], receiving['yards'], receiving['touchdowns'], receiving['longest'], field_goals['attempted'], field_goals['made'], field_goals['longest'], defense['tackles'], defense['sacks'], defense['interceptions'], defense['forced_fumbles'], first_downs['pass'], first_downs['rush'], first_downs['penalty'], interceptions['thrown'], interceptions['returned'], interceptions['touchdowns'], efficiency['redzone']['attempts'], efficiency['redzone']['successes'], efficiency['goal_to_go']['attempts'], efficiency['goal_to_go']['successes'], efficiency['third_down']['attempts'], efficiency['third_down']['successes'], efficiency['fourth_down']['attempts'], efficiency['fourth_down']['successes']))
            conn.commit()
    else:
        print(f"Failed to fetch data for game {game_id}. Status code: {response.status_code}")

    """


# Fetch data for the 2022 regular season's first week
# fetch_game_schedule(2019, "REG")

# fetch_game_schedule(2018, "REG")

# fetch_game_schedule(2017, "REG")

# fetch_game_schedule(2016, "REG")
