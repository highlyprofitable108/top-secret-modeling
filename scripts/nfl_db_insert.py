from classes.config_manager import ConfigManager
import pymongo
import os
import json
import logging

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

        if not all([self.mongo_uri, self.db_name, self.json_dir]):
            logging.error("Missing necessary configurations")
            return

        self.client = pymongo.MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]

    def insert_venue_if_not_exists(self, venue_data):
        """Inserts the venue data into the database if it does not already exist."""
        if venue_data:
            venue_collection = self.db['venues']
            venue_id = venue_data.get('id', None)
            if venue_id and not venue_collection.find_one({'venue_id': venue_id}):
                venue_collection.insert_one(venue_data)

    def insert_data_from_json(self, file_path):
        """Inserts data from a JSON file into the database."""
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    self.insert_game_data(data)
                    self.insert_team_data(data)
                    self.insert_venue_data(data)
                    self.insert_statistics_data(data)
                    self.insert_players_data(data)
                    self.insert_summary_data(data)
            except Exception as e:
                logging.error(f"Error inserting data from JSON: {e}")
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


if __name__ == "__main__":
    # Usage example:
    db_inserter = DBInserter()
    db_inserter.insert_data_from_directory()
