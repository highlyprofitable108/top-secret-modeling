from classes.config_manager import ConfigManager
import pymongo
import os
import json


class DBInserter:
    def __init__(self, config):
        self.mongo_uri = config['database']['mongo_uri']
        self.db_name = config['database']['database_name']
        self.json_dir = config['paths']['data_dir'] + config['paths']['json_dir']
        self.client = pymongo.MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]

    def insert_venue_if_not_exists(self, venue_data):
        venue_collection = self.db['venues']
        venue_id = venue_data.get('id', None)
        if venue_id and not venue_collection.find_one({'venue_id': venue_id}):
            venue_collection.insert_one(venue_data)

    def insert_data_from_json(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            # Get the game_id from the data
            game_id = data.get('id')
            # Check if the game_id already exists in the games collection
            games_collection = self.db['games']
            existing_game = games_collection.find_one({'id': game_id})
            if existing_game is None:
                # Insert entire JSON content into Games Collection
                games_collection.insert_one(data)
                # Insert home and away teams into Teams Collection
                teams_collection = self.db['teams']
                teams_collection.insert_one(data['summary']['home'])
                teams_collection.insert_one(data['summary']['away'])
                # Insert venue details into Venue Collection
                if 'venue' in data:
                    venue_collection = self.db['venues']
                    venue_collection.insert_one(data['venue'])
                # Insert statistics for home and away teams into Statistics Collection
                if 'statistics' in data:
                    statistics_collection = self.db['statistics']
                    statistics_collection.insert_one(data['statistics']['home'])
                    statistics_collection.insert_one(data['statistics']['away'])
                # Insert players from the statistics into Players Collection
                players_collection = self.db['players']
                # Extract and insert home team players
                if 'statistics' in data:
                    for category, stats in data['statistics']['home'].items():
                        if isinstance(stats, dict) and 'players' in stats:
                            for player in stats['players']:
                                players_collection.insert_one(player)
                # Extract and insert away team players
                for category, stats in data['statistics']['away'].items():
                    if isinstance(stats, dict) and 'players' in stats:
                        for player in stats['players']:
                            players_collection.insert_one(player)
                # Insert the summary of the game into Summary Collection
                if 'summary' in data:
                    summary_collection = self.db['summary']
                    summary_collection.insert_one(data['summary'])
            else:
                print(f"Game with ID {game_id} already exists in the database. Skipping...")

    def insert_data_from_directory(self):
        for filename in os.listdir(self.json_dir):
            if filename.endswith('.json'):
                self.insert_data_from_json(os.path.join(self.json_dir, filename))

        print('Data insertion complete.')


# Usage example:
config_manager = ConfigManager()
config = config_manager.get_config()
db_inserter = DBInserter(config)
db_inserter.insert_data_from_directory()
