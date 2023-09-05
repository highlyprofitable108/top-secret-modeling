import pymongo
import os
import json

# Connect to MongoDB
client = pymongo.MongoClient('mongodb://localhost:27017/')
db = client['nfl_db']

# Path to the JSON files
path_to_json_files = './data/jsons/'


# Helper function to insert venue data if it doesn't exist
def insert_venue_if_not_exists(venue_data):
    venue_id = venue_data.get('id', None)
    if venue_id and not venue_collection.find_one({'venue_id': venue_id}):
        venue_collection.insert_one(venue_data)


# Iterate over each JSON file in the directory
for filename in os.listdir(path_to_json_files):
    if filename.endswith('.json'):
        with open(os.path.join(path_to_json_files, filename), 'r') as file:
            data = json.load(file)

            # Get the game_id from the data
            game_id = data.get('id')

            # Check if the game_id already exists in the games collection
            games_collection = db['games']
            existing_game = games_collection.find_one({'id': game_id})

            if existing_game is None:
                # Insert entire JSON content into Games Collection
                games_collection.insert_one(data)

                # Insert home and away teams into Teams Collection
                teams_collection = db['teams']
                teams_collection.insert_one(data['summary']['home'])
                teams_collection.insert_one(data['summary']['away'])

                # Insert venue details into Venue Collection
                if 'venue' in data:
                    venue_collection = db['venues']
                    venue_collection.insert_one(data['venue'])

                # Insert statistics for home and away teams into Statistics Collection
                if 'statistics' in data:
                    statistics_collection = db['statistics']
                    statistics_collection.insert_one(data['statistics']['home'])
                    statistics_collection.insert_one(data['statistics']['away'])

                # Insert players from the statistics into Players Collection
                players_collection = db['players']

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
                    summary_collection = db['summary']
                    summary_collection.insert_one(data['summary'])
            else:
                print(f"Game with ID {game_id} already exists in the database. Skipping...")

print('Data insertion complete.')
