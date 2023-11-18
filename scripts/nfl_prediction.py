# Standard library imports
import os
import pytz
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
from classes.sim_visualization import SimVisualization
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
        self.sim_visualization = SimVisualization(self.template_dir, self.TARGET_VARIABLE)

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
            logging.info(f"Error connecting to MongoDB: {e}")
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
            scaler = joblib.load(os.path.join(self.model_dir, 'data_scaler.pkl'))
            return model, scaler
        except FileNotFoundError as e:
            logging.info(f"Error loading files: {e}")
            return None, None

    def get_historical_data(self, random_subset=None, get_current=False):
        df = self.database_operations.fetch_data_from_mongodb('games')
        historical_df = self.data_processing.flatten_and_merge_data(df)

        # Convert the 'scheduled' column to datetime format
        historical_df['scheduled'] = pd.to_datetime(historical_df['scheduled'])

        # Get current date
        current_date = datetime.now(pytz.utc)

        # Filter out rows where 'scheduled' date is more than 7 days in the future
        seven_days_ahead = current_date + timedelta(days=7)
        historical_df = historical_df[historical_df['scheduled'] <= seven_days_ahead]

        # Filter out games scheduled before CUTOFF_DATE
        historical_df = historical_df[historical_df['scheduled'] >= self.CUTOFF_DATE]

        # Check for and drop rows with NaN values in 'summary.odds.spread' columns
        columns_to_check = [col for col in historical_df.columns if col.startswith('summary.odds.spread')]
        historical_df = historical_df.dropna(subset=columns_to_check, how='any')

        # Extract necessary columns
        columns_to_extract = [
            'id', 'scheduled', 'summary.home.id', 'summary.home.name',
            'summary.away.id', 'summary.away.name', 'summary.home.points', 'summary.away.points'
        ]
        odds_columns = [col for col in historical_df.columns if col.startswith('summary.odds.')]
        columns_to_extract.extend(odds_columns)

        game_data = historical_df[columns_to_extract]

        # Convert the 'scheduled' column to datetime format and strip time information
        game_data['scheduled'] = pd.to_datetime(game_data['scheduled']).dt.date

        # Additional filters based on get_current and random_subset
        if get_current:
            today = datetime.now().date()
            one_week_from_now = today + timedelta(days=6)
            game_data = game_data[(game_data['scheduled'] >= today) & (game_data['scheduled'] <= one_week_from_now)]
        elif random_subset:
            # Drop rows with any NaN values in the final dataset
            game_data = game_data.dropna(how='any')

            game_data = game_data.sample(n=random_subset, replace=True)

        return game_data

    def analyze_and_log_results(self, simulation_results, home_team, away_team):
        # Analyze simulation results
        self.logger.info("Starting analysis of simulation results.")

        range_of_outcomes, standard_deviation, confidence_interval = self.sim_visualization.analyze_simulation_results(simulation_results)

        # Log detailed results
        self.logger.info(f"Game: {away_team} at {home_team}")
        self.logger.info(f"Outcome Range: {range_of_outcomes}")
        self.logger.info(f"Standard Deviation: {standard_deviation}")
        self.logger.info(f"95% Confidence Interval: {confidence_interval}")

        return f"Analysis Complete: {away_team} at {home_team}, Outcome Range: {range_of_outcomes}, Standard Deviation: {standard_deviation}, 95% Confidence Interval: {confidence_interval}"

    def simulate_games(self, num_simulations=1000, random_subset=None, get_current=False):
        """Predicts the target value using Monte Carlo simulation and visualizes the results."""
        self.logger.info("Starting prediction process...")
        reload(scripts.constants)

        # Fetch data only for the selected teams from the team_aggregated_metrics collection
        df = self.database_operations.fetch_data_from_mongodb('team_aggregated_metrics')
        historical_df = self.get_historical_data(random_subset, get_current)
        # Lists to store simulation results and actual results for each game
        all_simulation_results = []
        all_actual_results = []
        params_list = []

        self.file_cleanup()

        # Prepare data for each game
        for _, row in historical_df.iterrows():
            home_team = row['summary.home.name']
            away_team = row['summary.away.name']

            home_team = self.data_processing.replace_team_name(home_team)
            away_team = self.data_processing.replace_team_name(away_team)

            self.set_game_details(home_team, away_team, row['scheduled'])
            game_prediction_df = self.data_processing.prepare_data(df, self.features, home_team, away_team, self.date)

            # Diagnostic check: Ensure game_prediction_df is not empty
            if game_prediction_df.empty:
                logging.info(f"Game prediction DataFrame is empty for {home_team} vs {away_team}")
                continue

            params_list.append((game_prediction_df, self.model, home_team, away_team))

        with Pool() as pool:
            args_list = [(param, num_simulations) for param in params_list]

            # Diagnostic check: Ensure args_list is not empty
            if not args_list:
                logging.info("args_list for multiprocessing pool is empty.")
                return

            results = pool.map(run_simulation_wrapper, args_list)

        # Process results for each game
        for idx, (simulation_results, home_team, away_team) in enumerate(results):

            self.analyze_and_log_results(simulation_results, home_team, away_team)

            # LOOK INTO THIS FOR PREDICTIONS
            # perceived_value_for_game = BREAK_EVEN_PROBABILITY
            # value_opportunity = self.visualization.visualize_value_opportunity(simulation_results, perceived_value_for_game, idx+1)

            # Store simulation results and actual results for evaluation
            all_simulation_results.append(simulation_results)
            row = historical_df.iloc[idx]
            if row['summary.home.points'] is not None and row['summary.away.points'] is not None:
                actual_difference = (row['summary.home.points'] - row['summary.away.points']) * (-1)
            else:
                actual_difference = None
            all_actual_results.append(actual_difference)

        # Evaluate the betting recommendation and expected value
        self.sim_visualization.evaluate_and_recommend(all_simulation_results, historical_df, get_current)


def run_simulation_wrapper(args):
    return run_simulation(*args)


def run_simulation(params, num_simulations):
    game_prediction_df, model, home_team, away_team = params
    return model.monte_carlo_simulation(game_prediction_df, home_team, away_team, num_simulations)


if __name__ == "__main__":
    predictor = NFLPredictor()
    # predictor.simulate_games(100, 8)
    # Call other methods of the predictor as needed
