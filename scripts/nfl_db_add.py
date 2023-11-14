from classes.config_manager import ConfigManager
import pymongo
import os
import csv
import json
import logging
import pandas as pd
from datetime import datetime, timedelta
from classes.data_processing import DataProcessing
from classes.database_operations import DatabaseOperations


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DBInserter:
    """A class to handle the insertion of data into the database."""

    def __init__(self):
        """Initializes the DBInserter with the given configuration."""
        config_manager = ConfigManager()
        self.database_operations = DatabaseOperations()
        config = config_manager.get_config()

        self.mongo_uri = config.get('database', {}).get('mongo_uri')
        self.db_name = config.get('database', {}).get('database_name')
        self.json_dir = config.get('paths', {}).get('json_dir')
        self.odds_dir = config.get('paths', {}).get('odds_dir')
        self.TARGET_VARIABLE = config.get('constants', 'TARGET_VARIABLE')

        self.data_processing = DataProcessing(self.TARGET_VARIABLE)

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

            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON from file: {file_path}")
            except KeyError as e:
                logging.error(f"KeyError '{e}' when processing file: {file_path}")
            except Exception as e:
                logging.error(f"Error inserting data from JSON: {e} in file: {file_path}")
        else:
            logging.error(f"Invalid file path: {file_path}")

    def insert_game_data(self, data):
        """Inserts or updates game data in the database."""
        if data:
            games_collection = self.db['games']
            game_id = data.get('id')
            existing_game = games_collection.find_one({'id': game_id})

            # If the game doesn't exist in the collection, insert it
            if existing_game is None:
                games_collection.insert_one(data)
            else:
                # Check if any fields have changed
                has_changes = False
                for key, value in data.items():
                    if existing_game.get(key) != value:
                        has_changes = True
                        break

                # If there are changes, update the existing entry
                if has_changes:
                    games_collection.update_one({'id': game_id}, {'$set': data})

    def insert_team_data(self, data):
        """Insert data into teams table."""
        teams_collection = self.db['teams']
        teams_collection.insert_one(data['summary']['home'])
        teams_collection.insert_one(data['summary']['away'])

    def load_data_from_csv(self):
        """Loads data from a CSV file."""
        if os.path.exists(self.odds_dir):
            with open(self.odds_dir, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # skip header row
                for row in reader:
                    date, home_team, away_team, spread_open, spread_close, total_open, total_close, neutral, playoff = self.add_odds_data(row)
                    self.update_odds_in_mongodb(date, home_team, away_team, spread_open, spread_close, total_open, total_close)

    def add_odds_data(self, row):
        """Processes a row from the CSV and inserts it into the 'games' collection."""
        # Extract data from the row
        date = row[0]
        home_team = row[1]
        away_team = row[2]
        spread_open = float(row[8])
        spread_close = float(row[9])
        total_open = float(row[10])
        total_close = float(row[11])
        neutral = True if row[7] == "1" else False  # Assuming 1 for neutral and 0 for non-neutral
        playoff = True if row[6] == "1" else False  # Assuming 1 for playoff and 0 for non-playoff

        return date, home_team, away_team, spread_open, spread_close, total_open, total_close, neutral, playoff

    def update_odds_in_mongodb(self, date, home_team, away_team, spread_open, spread_close, total_open, total_close):
        """Updates the 'games' collection in MongoDB with odds data."""
        games_collection = self.db['games']

        # Convert the date from the CSV to match the format in MongoDB
        date_str = self._convert_date_format(date)
        if not date_str:
            return

        home_team_main = self._extract_team_name(home_team)
        away_team_main = self._extract_team_name(away_team)

        query = self._construct_query(date_str, home_team_main, away_team_main)

        game = games_collection.find_one(query)

        if game:
            self._update_game_odds(game, games_collection, spread_open, spread_close, total_open, total_close)
        else:
            logging.warning(f"No match found for date: {date_str}, home_team: {home_team_main}, away_team: {away_team_main}")

    def _convert_date_format(self, date):
        try:
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            return date_obj.strftime('%Y-%m-%d')
        except ValueError:
            logging.error(f"Invalid date format: {date}")
            return None

    def _extract_team_name(self, team_name):
        main_name = team_name.split(' (')[0].split(' ')[-1] if ' (' in team_name else team_name.split(' ')[-1]
        return main_name

    def _construct_query(self, date_str, home_team, away_team):
        one_day = timedelta(days=1)
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        prev_day = (date_obj - one_day).strftime('%Y-%m-%d')
        next_day = (date_obj + one_day).strftime('%Y-%m-%d')

        conditions = [date_str, prev_day, next_day]

        return {
            '$or': [self._query_condition(date, home_team, away_team) for date in conditions]
            + [self._query_condition(date, away_team, home_team) for date in conditions]
        }

    def _query_condition(self, date, home_team, away_team):
        return {
            '$and': [
                {'scheduled': {'$regex': f'^{date}'}},
                {'summary.home.name': {'$regex': home_team, '$options': 'i'}},
                {'summary.away.name': {'$regex': away_team, '$options': 'i'}}
            ]
        }

    def _update_game_odds(self, game, games_collection, spread_open, spread_close, total_open, total_close):
        game_date = datetime.strptime(game['scheduled'].split('T')[0], '%Y-%m-%d').date()
        current_date = datetime.now().date()
        days_difference = (game_date - current_date).days

        logging.info(f"Updating odds for game on {game_date} between {game['summary']['home']['name']} and {game['summary']['away']['name']}")

        # If the game is in the future or within the next 6 days, only update spread and total
        if days_difference >= 0 and days_difference <= 6:
            logging.info(f"Updating future game odds: spread_open={spread_open}, spread_close={spread_close}, total_open={total_open}, total_close={total_close}")
            games_collection.update_one(
                {'_id': game['_id']},
                {'$set': {
                    'summary.odds.spread_open': spread_open,
                    'summary.odds.spread_close': spread_close,
                    'summary.odds.total_open': total_open,
                    'summary.odds.total_close': total_close
                }}
            )
        else:
            home_points = game['summary']['home']['points']
            away_points = game['summary']['away']['points']

            covered = "Push" if (home_points + spread_close) - away_points == 0 else ("Yes" if (home_points + spread_close) - away_points > 0 else "No")
            total_value = "Push" if home_points + away_points == total_close else ("Over" if home_points + away_points > total_close else "Under")

            logging.info(f"Updating past game odds: spread_open={spread_open}, spread_close={spread_close}, total_open={total_open}, total_close={total_close}, covered={covered}, total_value={total_value}")

            games_collection.update_one(
                {'_id': game['_id']},
                {'$set': {
                    'summary.odds.spread_open': spread_open,
                    'summary.odds.spread_close': spread_close,
                    'summary.odds.total_open': total_open,
                    'summary.odds.total_close': total_close,
                    'summary.odds.covered': covered,
                    'summary.odds.total_value': total_value
                }}
            )

    def normalize(self, value, min_value=0, max_value=100):
        # Placeholder normalization function, should be replaced with actual normalization logic
        return (value - min_value) / (max_value - min_value) * 100

    def calculate_defensive_line_rating(self, sacks, tloss, qb_hits, hurries, forced_fumbles, knockdowns):
        # Normalize each stat to a 0-100 scale based on league max/min for the season
        # Then take a weighted average
        # Weights are hypothetical and should be determined by analysis
        weights = {'sacks': 0.25, 'tloss': 0.2, 'qb_hits': 0.2, 'hurries': 0.15, 'forced_fumbles': 0.1, 'knockdowns': 0.1}
        stats = {'sacks': sacks, 'tloss': tloss, 'qb_hits': qb_hits, 'hurries': hurries, 'forced_fumbles': forced_fumbles, 'knockdowns': knockdowns}
        
        rating = sum(weights[stat] * self.normalize(stats[stat]) for stat in stats)
        return rating

    def calculate_offensive_line_rating(self, pocket_time, sacks_allowed, hurries_allowed, knockdowns_allowed, rushing_yards_before_contact, throw_aways):
        # Normalize and invert where necessary (e.g., fewer sacks allowed is better)
        # Weights are hypothetical and should be determined by analysis
        weights = {'pocket_time': 0.2, 'sacks_allowed': 0.25, 'hurries_allowed': 0.25, 'knockdowns_allowed': 0.15, 'rushing_yards_before_contact': 0.1, 'throw_aways': 0.05}
        stats = {'pocket_time': pocket_time, 'sacks_allowed': sacks_allowed, 'hurries_allowed': hurries_allowed, 'knockdowns_allowed': knockdowns_allowed, 'rushing_yards_before_contact': rushing_yards_before_contact, 'throw_aways': throw_aways}
        
        rating = sum(weights[stat] * (100 - self.normalize(stats[stat])) if stat in ['sacks_allowed', 'hurries_allowed', 'knockdowns_allowed', 'throw_aways'] else weights[stat] * self.normalize(stats[stat]) for stat in stats)
        return rating
        
    def add_advanced_data(self, df):
        for team in ['home', 'away']:
            # Rush-Pass Ratio
            df[f'statistics.{team}.advanced.rush_pass_rate'] = df[f'statistics.{team}.rushing.totals.attempts'] / df[f'statistics.{team}.passing.totals.attempts']

            # Air Yards to Total Yards Ratio
            df[f'statistics.{team}.advanced.air_to_total_yards_rate'] = df[f'statistics.{team}.receiving.totals.air_yards'] / df[f'statistics.{team}.receiving.totals.yards']

            # Pressure Rate on QB
            df[f'statistics.{team}.advanced.pressure_rate_on_qb'] = (df[f'statistics.{team}.passing.totals.blitzes'] + df[f'statistics.{team}.passing.totals.hurries'] + df[f'statistics.{team}.passing.totals.knockdowns']) / df[f'statistics.{team}.passing.totals.attempts']

            # Tackle For Loss Rate
            df[f'statistics.{team}.advanced.tackle_for_loss_rate'] = df[f'statistics.{team}.defense.totals.tloss'] / df[f'statistics.{team}.defense.totals.tackles']

            # QB Hit Rate
            df[f'statistics.{team}.advanced.qb_hit_rate'] = df[f'statistics.{team}.defense.totals.qb_hits'] / df[f'statistics.{team}.passing.totals.attempts']

            # Drop Rate
            df[f'statistics.{team}.advanced.drop_rate'] = df[f'statistics.{team}.receiving.totals.dropped_passes'] / df[f'statistics.{team}.receiving.totals.targets']

            # Calculate Defensive Line Rating
            df[f'statistics.{team}.advanced.defensive_line_rating'] = self.calculate_defensive_line_rating(
                df[f'statistics.{team}.defense.totals.sacks'],
                df[f'statistics.{team}.defense.totals.tloss'],
                df[f'statistics.{team}.defense.totals.qb_hits'],
                df[f'statistics.{team}.defense.totals.hurries'],
                df[f'statistics.{team}.defense.totals.forced_fumbles'],
                df[f'statistics.{team}.defense.totals.knockdowns']
            )

            # Calculate Offensive Line Rating
            df[f'statistics.{team}.advanced.offensive_line_rating'] = self.calculate_offensive_line_rating(
                df[f'statistics.{team}.passing.totals.pocket_time'],
                df[f'statistics.{team}.passing.totals.sacks'],
                df[f'statistics.{team}.passing.totals.hurries'],
                df[f'statistics.{team}.passing.totals.knockdowns'],
                df[f'statistics.{team}.rushing.totals.yards'] - df[f'statistics.{team}.rushing.totals.yards_after_contact'],
                df[f'statistics.{team}.passing.totals.throw_aways']
            )

        return df

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

    def update_games_with_advanced_data(self):
        # Get the games collection from MongoDB
        df = self.database_operations.fetch_data_from_mongodb('games')
        df = self.data_processing.flatten_and_merge_data(df)

        # Execute the add_advanced_data() method
        df = self.add_advanced_data(df)

        for _, row in df.iterrows():
            game_id = row['_id']
            # Convert row to dictionary and remove _id to avoid conflicts during update
            game_data = row.to_dict()
            del game_data['_id']
            self.database_operations.update_data_in_mongodb('games', {'_id': game_id}, game_data)


if __name__ == "__main__":
    # Usage example:
    db_inserter = DBInserter()
    db_inserter.insert_data_from_directory()
    db_inserter.update_games_with_advanced_data()
