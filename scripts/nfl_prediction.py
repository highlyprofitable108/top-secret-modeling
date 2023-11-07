# Standard library imports
import io
import os
from datetime import datetime, timedelta
from importlib import reload

# Third-party imports
import joblib
import pandas as pd
from multiprocessing import Pool
from pymongo import MongoClient


# Local application imports
from classes.modeling import Modeling
from classes.config_manager import ConfigManager
from classes.data_processing import DataProcessing
from classes.data_visualization import Visualization
from classes.database_operations import DatabaseOperations
import scripts.constants

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

HOME_FIELD_ADJUST = 0
BREAK_EVEN_PROBABILITY = 0.5238  # 52.38% implied probability to break even


class NFLPredictor:
    def __init__(self):
        # Configuration and Database Connection
        self.config = ConfigManager()
        self.database_operations = DatabaseOperations()

        # Constants
        self.features = [col for col in scripts.constants.COLUMNS_TO_KEEP if 'odd' not in col]
        self.model_type = self.config.get_config('model_settings', 'model_type')

        self.data_dir = self.config.get_config('paths', 'data_dir')
        self.model_dir = self.config.get_config('paths', 'model_dir')
        self.static_dir = self.config.get_config('paths', 'static_dir')
        self.template_dir = self.config.get_config('paths', 'template_dir')

        self.TARGET_VARIABLE = self.config.get_config('constants', 'TARGET_VARIABLE')

        self.data_processing = DataProcessing(self.TARGET_VARIABLE)
        self.visualization = Visualization(self.template_dir, self.TARGET_VARIABLE)

        self.CUTOFF_DATE = self.config.get_config('constants', 'CUTOFF_DATE')

        self.MONGO_URI = self.config.get_config('database', 'mongo_uri')
        self.DATABASE_NAME = self.config.get_config('database', 'database_name')

        # Set up the logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        self.logger.addHandler(handler)

        try:
            self.client = MongoClient(self.MONGO_URI)
            self.db = self.client[self.DATABASE_NAME]
        except Exception as e:
            logging.error(f"Error connecting to MongoDB: {e}")
            raise

        # Loading the pre-trained model and the data scaler
        self.LOADED_MODEL, self.LOADED_SCALER = self.load_model_and_scaler()
        self.model = Modeling(self.LOADED_MODEL, self.LOADED_SCALER, HOME_FIELD_ADJUST, self.static_dir)

    def file_cleanup(self):  # your method signature
        # Delete the existing CSV files if they exist
        for filename in ['combined_sampled_data.csv', 'simulation_results.csv']:
            file_path = os.path.join(self.static_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)

    def set_game_details(self, home_team, away_team, date):
        """Set the game details for the current simulation."""
        self.logger.info(f"Setting game details for {home_team} vs {away_team} on {date}")

        # Replace "Football Team" or "Redskins" with "Commanders"
        if home_team in ["Football Team", "Redskins"]:
            home_team = "Commanders"
        if away_team in ["Football Team", "Redskins"]:
            away_team = "Commanders"

        self.home_team = home_team
        self.away_team = away_team
        # Convert the date to a datetime object and store only the "yyyy-mm-dd" part
        if isinstance(date, str):
            self.date = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S%z').strftime('%Y-%m-%d')
        else:
            self.date = date.strftime('%Y-%m-%d')

    def load_model_and_scaler(self):
        try:
            model = joblib.load(os.path.join(self.model_dir, 'trained_nfl_model.pkl'))
            print(model)
            scaler = joblib.load(os.path.join(self.model_dir, 'data_scaler.pkl'))
            return model, scaler
        except FileNotFoundError as e:
            logging.error(f"Error loading files: {e}")
            return None, None

    def get_historical_data(self, random_subset=None, get_current=False, adhoc=False, date=None):
        """
        Prepare historical data for simulations.

        :param random_subset: Number of random games to fetch. If None, fetch all games.
        :return: DataFrame containing game data.
        """
        df = self.database_operations.fetch_data_from_mongodb('games')
        historical_df = self.data_processing.flatten_and_merge_data(df)

        # Convert the 'scheduled' column to datetime format
        historical_df['scheduled'] = pd.to_datetime(historical_df['scheduled'])

        # Filter out games scheduled before "09-01-2015"
        historical_df = historical_df[historical_df['scheduled'] >= self.CUTOFF_DATE]

        columns_to_check = [col for col in historical_df.columns if col.startswith('summary.odds.spread')]
        historical_df = historical_df.dropna(subset=columns_to_check, how='any')

        # Extract necessary columns
        columns_to_extract = [
            'id',
            'scheduled',
            'summary.home.id',
            'summary.home.name',
            'summary.away.id',
            'summary.away.name',
            'summary.home.points',
            'summary.away.points'
        ]
        # Extracting all columns under 'summary.odds'
        odds_columns = [col for col in historical_df.columns if col.startswith('summary.odds.')]
        columns_to_extract.extend(odds_columns)

        game_data = historical_df[columns_to_extract]

        # Convert the 'scheduled' column to datetime format and strip time information
        game_data['scheduled'] = pd.to_datetime(game_data['scheduled']).dt.date

        # If get_current is True, filter games for the next week
        if get_current:
            today = datetime.now().date()
            one_week_from_now = today + timedelta(days=6)
            game_data = game_data[(game_data['scheduled'] >= today) & (game_data['scheduled'] <= one_week_from_now)]

        # If random_subset is specified, fetch a random subset of games
        elif random_subset:
            game_data = game_data.sample(n=random_subset, replace=True)

        return game_data

    def analyze_and_log_results(self, simulation_results, most_likely_outcome, home_team, away_team):
        # Analyze simulation results
        self.logger.info("Analyzing Simulation Results...")
        range_of_outcomes, standard_deviation, confidence_interval = self.model.analyze_simulation_results(simulation_results)

        # Create a buffer to capture log messages
        log_capture_buffer = io.StringIO()

        # Set up the logger to write to the buffer
        log_handler = logging.StreamHandler(log_capture_buffer)
        log_handler.setLevel(logging.INFO)
        self.logger.addHandler(log_handler)

        # User-friendly output
        self.logger.info("---------------------------")
        self.logger.info(f"Game: {away_team} at {home_team}")
        self.logger.info("---------------------------")
        self.logger.info(f"Expected target value: {range_of_outcomes[0]:.2f} to {range_of_outcomes[1]:.2f} points.")
        # self.logger.info(f"95% Confidence Interval: {confidence_interval[0]:.2f} to {confidence_interval[1]:.2f} for {home_team} projected spread.")
        self.logger.info(f"Most likely target value: {most_likely_outcome:.2f} for {home_team} projected spread.")
        self.logger.info(f"Standard deviation of target values: {standard_deviation:.2f} for {home_team} projected spread.")
        self.logger.info("")  # Add an empty line for separation

        # Retrieve the log messages from the buffer
        log_contents = log_capture_buffer.getvalue()
        combined_output = log_contents

        return combined_output

    def simulate_games(self, num_simulations=1000, random_subset=None, date=None, get_current=False, adhoc=False, matchups=None):
        """Predicts the target value using Monte Carlo simulation and visualizes the results."""
        self.logger.info("Starting prediction process...")
        reload(scripts.constants)

        # Fetch data only for the selected teams from the weekly_ranks collection
        df = self.database_operations.fetch_data_from_mongodb('weekly_ranks')

        if adhoc and matchups:
            # Filter out matchups where both teams are "None" or None
            valid_matchups = [(home, away) for home, away in matchups if home not in [None, "None"] and away not in [None, "None"]]

            # If it's an adhoc simulation with custom matchups, prepare the historical_df accordingly
            historical_df = pd.DataFrame(valid_matchups, columns=['summary.home.name', 'summary.away.name'])
            historical_df['scheduled'] = date

            columns_to_enable = [
                'id',
                'summary.home.id',
                'summary.away.id',
                'summary.home.points',
                'summary.away.points',
                'summary.odds.spread_close',
            ]

            # Create a new DataFrame with the desired columns set to None
            df_none = pd.DataFrame(columns=columns_to_enable)
            for column in columns_to_enable:
                df_none[column] = None

            # Update the historical_df with the new DataFrame
            historical_df = historical_df.combine_first(df_none)
        else:
            historical_df = self.get_historical_data(random_subset, get_current, adhoc, date)

        # Lists to store simulation results and actual results for each game
        all_simulation_results = []
        all_actual_results = []
        params_list = []
        # Lists to store home and away team names
        home_teams = []
        away_teams = []

        self.file_cleanup()

        # Prepare data for each game
        for _, row in historical_df.iterrows():
            home_team = row['summary.home.name']
            away_team = row['summary.away.name']
            home_team = self.data_processing.replace_team_name(home_team)
            away_team = self.data_processing.replace_team_name(away_team)

            self.set_game_details(home_team, away_team, row['scheduled'])
            game_prediction_df = self.data_processing.prepare_data(df, self.features, home_team, away_team, self.date)
            params_list.append((game_prediction_df, self.data_processing.get_standard_deviation(df), self.model))

            # Append the home and away team names to the lists
            home_teams.append(home_team)
            away_teams.append(away_team)

        with Pool() as pool:
            args_list = [(param, num_simulations) for param in params_list]
            results = pool.map(run_simulation_wrapper, args_list)

        # Process results for each game
        for idx, (simulation_results, most_likely_outcome) in enumerate(results):
            output = self.analyze_and_log_results(simulation_results, most_likely_outcome, home_teams[idx], away_teams[idx])

            perceived_value_for_game = BREAK_EVEN_PROBABILITY
            value_opportunity = self.visualization.visualize_value_opportunity(simulation_results, perceived_value_for_game, idx+1)
            self.logger.info(f"Value Opportunity for {self.home_team} vs {self.away_team}: {value_opportunity:.2f}")

            # Store simulation results and actual results for evaluation
            all_simulation_results.append(simulation_results)
            row = historical_df.iloc[idx]
            if row['summary.home.points'] is not None and row['summary.away.points'] is not None:
                actual_difference = (row['summary.home.points'] - row['summary.away.points']) * (-1)
            else:
                actual_difference = None
            all_actual_results.append(actual_difference)

            # Visualize simulation results
            self.visualization.visualize_simulation_results(simulation_results, most_likely_outcome, output, idx+1)

        self.visualization.generate_value_opportunity_page(len(historical_df))
        self.visualization.generate_simulation_distribution_page(len(historical_df))

        # Check if all_actual_results has data
        if all_actual_results:
            # Identify indices of rows that have None in all_actual_results
            indices_with_none = [i for i, result in enumerate(all_actual_results) if result is None]

            # Drop rows with None values from all_actual_results
            all_actual_results = [result for i, result in enumerate(all_actual_results) if i not in indices_with_none]

            # Drop corresponding rows from all_simulation_results
            sims_with_results = [result for i, result in enumerate(all_simulation_results) if i not in indices_with_none]

            # Only run this if all_actual_results still has data after dropping rows with None values
            if all_actual_results:
                # After the loop, compare the simulated results to the actual results
                self.visualization.compare_simulated_to_actual(sims_with_results, all_actual_results)

        # Evaluate the betting recommendation and expected value
        results_df, recommendation_accuracy, average_ev, actual_value = self.visualization.evaluate_and_recommend(all_simulation_results, historical_df)

        if (num_simulations >= 1000) and (random_subset is not None) and (random_subset >= 100):
            # Get the current time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

            # Insert 'current_time' as the first column
            results_df.insert(0, 'current_time', current_time)

            # Insert 'self.model_type' as the second column
            results_df.insert(1, 'model_type', self.model_type)

            # Insert 'self.TARGET_VARIABLE' as the third column
            results_df.insert(2, 'target_variable', self.TARGET_VARIABLE)

            # Add 'self.features_list' as the last column
            results_df['features_list'] = [self.features] * len(results_df)

            # Create a unique game identifier by concatenating date, home_team, and away_team
            results_df['game_id'] = results_df['Date'].dt.strftime('%Y-%m-%d') + '_' + results_df['Home Team']

            bet_results = results_df.to_dict('records')
            collection_name = "bet_results"

            self.database_operations.insert_data_into_mongodb(collection_name, bet_results)

        self.logger.info(f"Recommendation Accuracy: {recommendation_accuracy:.2f}%")
        self.logger.info(f"Average Expected Value: {average_ev:.2f}%")
        self.logger.info(f"Actual Results: ${actual_value:.2f}")


def run_simulation_wrapper(args):
    return run_simulation(*args)


def run_simulation(params, num_simulations):
    game_prediction_df, standard_deviation, model = params
    return model.monte_carlo_simulation(game_prediction_df, standard_deviation, num_simulations)


if __name__ == "__main__":
    predictor = NFLPredictor()
    # predictor.simulate_games(100, 8)
    # Call other methods of the predictor as needed
