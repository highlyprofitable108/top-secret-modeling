from classes.config_manager import ConfigManager
import pymongo
import os
import csv
import json
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DBInserter:
    """A class to handle the insertion of data into the database."""

    def __init__(self):
        """Initializes the DBInserter with the given configuration."""
        config_manager = ConfigManager()
        config = config_manager.get_config()

        self.mongo_uri = config.get('database', {}).get('mongo_uri')
        self.db_name = config.get('database', {}).get('database_name')
        self.json_dir = config.get('paths', {}).get('json_dir')
        self.odds_dir = config.get('paths', {}).get('odds_dir')

        if not all([self.mongo_uri, self.db_name, self.json_dir]):
            logging.error("Missing necessary configurations")
            return

        self.client = pymongo.MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]

    def insert_data_from_json(self, file_path):
        """Inserts data from a JSON file into the database."""
        if os.path.exists(file_path):
            try:
                logging.info(f"Reading JSON data from file: {file_path}")
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    logging.info("Inserting game data")
                    self.insert_game_data(data)
                    logging.info("Inserting team data")
                    self.insert_team_data(data)
                    logging.info("Inserting venue data")
                    self.insert_venue_data(data)
                    logging.info("Inserting statistics data")
                    self.insert_statistics_data(data)
                    logging.info("Inserting players data")
                    self.insert_players_data(data)
                    logging.info("Inserting summary data")
                    self.insert_summary_data(data)

            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON from file: {file_path}")
            except KeyError as e:
                logging.error(f"KeyError '{e}' when processing file: {file_path}")
            except Exception as e:
                logging.error(f"Error inserting data from JSON: {e} in file: {file_path}")
        else:
            logging.error(f"Invalid file path: {file_path}")

    def insert_game_data(self, data):
        """Inserts game data into the database."""
        if data:
            games_collection = self.db['games']
            game_id = data.get('id')
            existing_game = games_collection.find_one({'id': game_id})
            if existing_game is None:
                games_collection.insert_one(data)

    def insert_team_data(self, data):
        """Inserts team data into the database."""
        teams_collection = self.db['teams']
        teams_collection.insert_one(data['summary']['home'])
        teams_collection.insert_one(data['summary']['away'])

    def insert_venue_data(self, data):
        """Inserts venue data into the database."""
        if 'venue' in data:
            venue_collection = self.db['venues']
            venue_collection.insert_one(data['venue'])

    def insert_statistics_data(self, data):
        """Inserts statistics data into the database."""
        if 'statistics' in data:
            statistics_collection = self.db['statistics']
            statistics_collection.insert_one(data['statistics']['home'])
            statistics_collection.insert_one(data['statistics']['away'])

    def insert_players_data(self, data):
        """Inserts players data into the database."""
        if 'statistics' in data:
            players_collection = self.db['players']
            for team in ['home', 'away']:
                for category, stats in data['statistics'][team].items():
                    if isinstance(stats, dict) and 'players' in stats:
                        for player in stats['players']:
                            players_collection.insert_one(player)

    def insert_summary_data(self, data):
        """Inserts summary data into the database."""
        if 'summary' in data:
            summary_collection = self.db['summary']
            summary_collection.insert_one(data['summary'])    

    def load_data_from_csv(self):
        """Loads data from a CSV file."""
        if os.path.exists(self.odds_dir):
            with open(self.odds_dir, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # skip header row
                for row in reader:
                    date, home_team, away_team, spread, total = self.add_odds_data(row)
                    self.update_odds_in_mongodb(date, home_team, away_team, spread, total)

    def add_odds_data(self, row):
        """Processes a row from the CSV and inserts it into the 'games' collection."""
        # Extract data from the row
        date = row[1]
        location = row[3]
        team_e = row[4]
        try:
            spread = float(row[5])
        except ValueError:
            spread = 0.0  # set spread to 0 if it can't be converted to float
        total = float(row[8].split(' ')[1])

        # Determine home and away teams based on the location
        if location in ['@', 'N']:
            home_team = team_e
            away_team = row[7]
        else:
            home_team = row[7]
            away_team = team_e
            spread *= -1  # invert the spread

        return date, home_team, away_team, spread, total

    def update_odds_in_mongodb(self, date, home_team, away_team, spread, total):
        """Updates the 'games' collection in MongoDB with odds data."""
        games_collection = self.db['games']

        # Convert the date from the CSV to match the format in MongoDB
        try:
            date_obj = datetime.strptime(date, '%b %d, %Y')  # Adjusted for the format 'Sep 8, 2019'
            date_str = date_obj.strftime('%Y-%m-%d')  # Convert to "YYYY-MM-DD" format
        except ValueError:
            logging.error(f"Invalid date format: {date}")
            return

        # Extract main team names
        home_team_main = home_team.split(' (')[0].split(' ')[-1] if ' (' in home_team else home_team.split(' ')[-1]
        away_team_main = away_team.split(' (')[0].split(' ')[-1] if ' (' in away_team else away_team.split(' ')[-1]

        if home_team_main == "Team":
            home_team_main = "Football Team"
        if away_team_main == "Team":
            away_team_main = "Football Team"

        # Query the 'games' collection to find a match
        one_day = timedelta(days=1)
        prev_day = (date_obj - one_day).strftime('%Y-%m-%d')
        next_day = (date_obj + one_day).strftime('%Y-%m-%d')
        # Define the query
        query = {
            '$or': [
                {'$and': [
                    {'scheduled': {'$regex': f'^{date_str}'}},  # Match only the date part
                    {'summary.home.name': {'$regex': home_team_main, '$options': 'i'}},  # Case-insensitive substring match for home team
                    {'summary.away.name': {'$regex': away_team_main, '$options': 'i'}}   # Case-insensitive substring match for away team
                ]},
                {'$and': [
                    {'scheduled': {'$regex': f'^{date_str}'}},  # Match only the date part
                    {'summary.home.name': {'$regex': away_team_main, '$options': 'i'}},  # Case-insensitive substring match for home team
                    {'summary.away.name': {'$regex': home_team_main, '$options': 'i'}}   # Case-insensitive substring match for away team
                ]},
                {'$and': [
                    {'scheduled': {'$regex': f'^{next_day}'}},  # Match only the date part
                    {'summary.home.name': {'$regex': home_team_main, '$options': 'i'}},  # Case-insensitive substring match for home team
                    {'summary.away.name': {'$regex': away_team_main, '$options': 'i'}}   # Case-insensitive substring match for away team
                ]},
                {'$and': [
                    {'scheduled': {'$regex': f'^{prev_day}'}},  # Match only the date part
                    {'summary.home.name': {'$regex': home_team_main, '$options': 'i'}},  # Case-insensitive substring match for home team
                    {'summary.away.name': {'$regex': away_team_main, '$options': 'i'}}   # Case-insensitive substring match for away team
                ]},
                {'$and': [
                    {'scheduled': {'$regex': f'^{next_day}'}},  # Match only the date part
                    {'summary.home.name': {'$regex': away_team_main, '$options': 'i'}},  # Case-insensitive substring match for home team
                    {'summary.away.name': {'$regex': home_team_main, '$options': 'i'}}   # Case-insensitive substring match for away team
                ]},
                {'$and': [
                    {'scheduled': {'$regex': f'^{prev_day}'}},  # Match only the date part
                    {'summary.home.name': {'$regex': away_team_main, '$options': 'i'}},  # Case-insensitive substring match for home team
                    {'summary.away.name': {'$regex': home_team_main, '$options': 'i'}}   # Case-insensitive substring match for away team
                ]},
            ]
        }

        game = games_collection.find_one(query)

        if game:
            # Extract summary data from the game document
            home_points = game['summary']['home']['points']
            away_points = game['summary']['away']['points']

            # Calculate covered value
            if (home_points + spread) - away_points > 0:
                covered = "Yes"
            elif (home_points + spread) - away_points < 0:
                covered = "No"
            else:
                covered = "Push"

            # Calculate total value
            if home_points + away_points > total:
                total_value = "Over"
            elif home_points + away_points < total:
                total_value = "Under"
            else:
                total_value = "Push"

            # Update the matched document with the spread, total, covered, and total_value values
            games_collection.update_one(
                {'_id': game['_id']},
                {'$set': {
                    'summary.odds.spread': spread,
                    'summary.odds.total': total,
                    'summary.odds.covered': covered,
                    'summary.odds.total_value': total_value
                }}
            )
        else:
            logging.warning(f"No match found for date: {date_str}, home_team: {home_team_main}, away_team: {away_team_main}")

    def insert_data_from_directory(self):
        """Inserts data from all JSON files in the specified directory into the database."""
        if os.path.exists(self.json_dir):
            try:
                for filename in os.listdir(self.json_dir):
                    if filename.endswith('.json'):
                        self.insert_data_from_json(os.path.join(self.json_dir, filename))
                logging.info('Data insertion complete.')
            except Exception as e:
                logging.error(f"Error inserting data from directory: {e}")
        else:
            logging.error(f"Invalid directory path: {self.json_dir}")

        self.load_data_from_csv()


if __name__ == "__main__":
    # Usage example:
    db_inserter = DBInserter()
    db_inserter.insert_data_from_directory()
