import os
import glob
import yaml
import sqlite3
import json

# Load the configuration
with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

# Define paths using the config
BASE_DIR = os.path.expandvars(config['default']['base_dir'])
DATA_DIR = os.path.join(BASE_DIR, 'data')
JSON_DIR = os.path.join(DATA_DIR, 'jsons')
DB_PATH = os.path.join(DATA_DIR, config['database']['database_name'])

# Database connection setup
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Loop through all the game_stats_* files in the JSON_DIR directory
for file_path in glob.glob(os.path.join(JSON_DIR, 'game_stats_*.json')):
    print(f"Reviewing: {file_path}")
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Insert data for seasons
    season = data.get('summary', {}).get('season', {})
    cursor.execute("INSERT OR IGNORE INTO season (id, year, type, name) VALUES (?, ?, ?, ?)", 
                (season.get('id'), season.get('year'), season.get('type'), season.get('name')))

    # Insert data for weeks
    week_data = data.get('summary', {}).get('week', {})
    cursor.execute("INSERT OR IGNORE INTO week (id, sequence, title) VALUES (?, ?, ?)", 
                (week_data.get('id'), week_data.get('sequence'), week_data.get('title')))

    for team_type in ['home', 'away']:
        team_data = data.get('statistics', {}).get(team_type, {})

        rushing_data = team_data.get('rushing', {}).get('totals', {})
        cursor.execute("INSERT OR IGNORE INTO rushing (game_id, team_id, avg_yards, attempts, touchdowns, tlost, tlost_yards, yards, longest, redzone_attempts, first_downs, broken_tackles, kneel_downs, scrambles, yards_after_contact) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                    (data.get('id'), team_data.get('id'), rushing_data.get('avg_yards'), rushing_data.get('attempts'), rushing_data.get('touchdowns'), rushing_data.get('tlost'), rushing_data.get('tlost_yards'), rushing_data.get('yards'), rushing_data.get('longest'), rushing_data.get('redzone_attempts'), rushing_data.get('first_downs'), rushing_data.get('broken_tackles'), rushing_data.get('kneel_downs'), rushing_data.get('scrambles'), rushing_data.get('yards_after_contact')))

        receiving_data = team_data.get('receiving', {}).get('totals', {})
        cursor.execute("""
            INSERT OR IGNORE INTO receiving 
            (game_id, team_id, receptions, targets, yards, avg_yards, longest, touchdowns, longest_touchdown, yards_after_catch, redzone_targets, air_yards, broken_tackles, dropped_passes, catchable_passes, yards_after_contact) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, 
        (data.get('id'), team_data.get('id'), receiving_data.get('receptions'), receiving_data.get('targets'), receiving_data.get('yards'), receiving_data.get('avg_yards'), receiving_data.get('longest'), receiving_data.get('touchdowns'), receiving_data.get('longest_touchdown'), receiving_data.get('yards_after_catch'), receiving_data.get('redzone_targets'), receiving_data.get('air_yards'), receiving_data.get('broken_tackles'), receiving_data.get('dropped_passes'), receiving_data.get('catchable_passes'), receiving_data.get('yards_after_contact')))

        passing_data = team_data.get('passing', {}).get('totals', {})
        cursor.execute("INSERT OR IGNORE INTO passing (game_id, team_id, completions, attempts, pct, yards, avg_yards, touchdowns, interceptions, longest, rating, sacks, sack_yards, air_yards, redzone_attempts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                    (data.get('id'), team_data.get('id'), passing_data.get('completions'), passing_data.get('attempts'), passing_data.get('pct'), passing_data.get('yards'), passing_data.get('avg_yards'), passing_data.get('touchdowns'), passing_data.get('interceptions'), passing_data.get('longest'), passing_data.get('rating'), passing_data.get('sacks'), passing_data.get('sack_yards'), passing_data.get('air_yards'), passing_data.get('redzone_attempts')))

        summary_data = team_data.get('summary', {})
        cursor.execute("INSERT OR IGNORE INTO team_summary (game_id, team_id, possession_time, avg_gain, safeties, turnovers, play_count, rush_plays, total_yards, fumbles, lost_fumbles, penalties, penalty_yards, return_yards) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                    (data.get('id'), team_data.get('id'), summary_data.get('possession_time'), summary_data.get('avg_gain'), summary_data.get('safeties'), summary_data.get('turnovers'), summary_data.get('play_count'), summary_data.get('rush_plays'), summary_data.get('total_yards'), summary_data.get('fumbles'), summary_data.get('lost_fumbles'), summary_data.get('penalties'), summary_data.get('penalty_yards'), summary_data.get('return_yards')))

        defense_data = team_data.get('defense', {}).get('totals', {})
        cursor.execute("INSERT OR IGNORE INTO team_defense (game_id, team_id, tackles, assists, combined, sacks, sack_yards, interceptions, passes_defended, forced_fumbles, fumble_recoveries, qb_hits, tloss, tloss_yards, safeties) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                    (data.get('id'), team_data.get('id'), defense_data.get('tackles'), defense_data.get('assists'), defense_data.get('combined'), defense_data.get('sacks'), defense_data.get('sack_yards'), defense_data.get('interceptions'), defense_data.get('passes_defended'), defense_data.get('forced_fumbles'), defense_data.get('fumble_recoveries'), defense_data.get('qb_hits'), defense_data.get('tloss'), defense_data.get('tloss_yards'), defense_data.get('safeties')))

        # First Downs
        first_downs_data = team_data.get('first_downs', {})
        cursor.execute("INSERT OR IGNORE INTO team_first_downs (game_id, team_id, total, rush, pass, penalty) VALUES (?, ?, ?, ?, ?, ?)", 
                    (data.get('id'), team_data.get('id'), first_downs_data.get('total'), first_downs_data.get('rush'), first_downs_data.get('pass'), first_downs_data.get('penalty')))
        
        # Interceptions
        interceptions_data = team_data.get('interceptions', {})
        cursor.execute("INSERT OR IGNORE INTO team_interceptions (game_id, team_id, interceptions, return_yards, return_touchdowns) VALUES (?, ?, ?, ?, ?)", 
                    (data.get('id'), team_data.get('id'), interceptions_data.get('interceptions'), interceptions_data.get('return_yards'), interceptions_data.get('return_touchdowns')))

        # Efficiency
        efficiency_data = team_data.get('efficiency', {})
        cursor.execute("""
            INSERT OR IGNORE INTO team_efficiency 
            (game_id, team_id, 
            goaltogo_attempts, goaltogo_successes, goaltogo_pct, 
            redzone_attempts, redzone_successes, redzone_pct, 
            thirddown_attempts, thirddown_successes, thirddown_pct, 
            fourthdown_attempts, fourthdown_successes, fourthdown_pct) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 
                    (data.get('id'), team_data.get('id'), 
                        efficiency_data.get('goaltogo', {}).get('attempts'), efficiency_data.get('goaltogo', {}).get('successes'), efficiency_data.get('goaltogo', {}).get('pct'), 
                        efficiency_data.get('redzone', {}).get('attempts'), efficiency_data.get('redzone', {}).get('successes'), efficiency_data.get('redzone', {}).get('pct'), 
                        efficiency_data.get('thirddown', {}).get('attempts'), efficiency_data.get('thirddown', {}).get('successes'), efficiency_data.get('thirddown', {}).get('pct'), 
                        efficiency_data.get('fourthdown', {}).get('attempts'), efficiency_data.get('fourthdown', {}).get('successes'), efficiency_data.get('fourthdown', {}).get('pct')))


# Commit the changes and close the connection
conn.commit()
cursor.close()
conn.close()


"""
        # Insert into venues table
        venue = game.get('venue', {})
        location = venue.get('location', {})
        cursor.execute("INSERT OR IGNORE INTO venues (venue_id, name, city, state, country, zip, address, capacity, surface, roof_type, sr_id, latitude, longitude) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (venue.get('id'), venue.get('name'), venue.get('city'), venue.get('state', None), venue.get('country'), venue.get('zip', None), venue.get('address'), venue.get('capacity'), venue.get('surface'), venue.get('roof_type'), venue.get('sr_id'), location.get('lat'), location.get('lng')))

        # Insert into games table
        broadcast = game.get('broadcast', {})
        cursor.execute("INSERT INTO games (game_id, status, scheduled, attendance, entry_mode, sr_id, game_type, conference_game, duration, home_team_id, away_team_id, venue_id, broadcast_network) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                       (game.get('id'), game.get('status'), game.get('scheduled'), game.get('attendance'), game.get('entry_mode'), game.get('sr_id'), game.get('game_type'), game.get('conference_game'), game.get('duration'), home_team.get('id'), away_team.get('id'), venue.get('id'), broadcast.get('network')))

        # Insert into weather table
        weather = game.get('weather', {})
        wind = weather.get('wind', {})
        cursor.execute("INSERT INTO weather (game_id, condition, humidity, temperature, wind_speed, wind_direction) VALUES (?, ?, ?, ?, ?, ?)", 
                       (game.get('id'), weather.get('condition'), weather.get('humidity'), weather.get('temp'), wind.get('speed'), wind.get('direction')))

        # Insert into scoring table and periods table
        scoring = game.get('scoring', {})
        cursor.execute("INSERT INTO scoring (game_id, home_points, away_points) VALUES (?, ?, ?)", 
                       (game.get('id'), scoring.get('home_points'), scoring.get('away_points')))
        score_id = cursor.lastrowid
        for period in scoring.get('periods', []):
            cursor.execute("INSERT INTO periods (period_id, score_id, period_type, number, sequence, home_points, away_points) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                           (period.get('id'), score_id, period.get('period_type'), period.get('number'), period.get('sequence'), period.get('home_points'), period.get('away_points')))
"""
