# Standard library imports
import os
import pytz
from datetime import datetime, timedelta
from importlib import reload

# Third-party imports
import joblib
import multiprocessing
import pandas as pd
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

# Determine the percentage of CPU cores to use (e.g., 50%)
cpu_usage_percentage = 0.88

# Calculate the number of processes to use based on the percentage
available_cores = multiprocessing.cpu_count()
num_processes = max(1, int(available_cores * cpu_usage_percentage))

# Constants
HOME_FIELD_ADJUST = 0
BREAK_EVEN_PROBABILITY = 0.5238  # 52.38% implied probability to break even


class NFLPredictor:
    def __init__(self):
        """
        Initializes the NFLPredictor class.

        Responsibilities:
        - Sets up configuration and database connections.
        - Initializes various class attributes for model predictions.
        - Loads the pre-trained model and scaler.
        - Sets up logging.

        Notes:
        - Placeholder for future feature: Consider adding more dynamic handling of feature selection.
        """
        # Configuration and Database Connection
        self.config = ConfigManager()
        self.database_operations = DatabaseOperations()

        # Constants
        self.features = [col for col in scripts.constants.COLUMNS_TO_KEEP if 'odd' not in col]
        self.model_type = self.config.get_config('model_settings', 'model_type')

        # Directory paths
        self.data_dir = self.config.get_config('paths', 'data_dir')
        self.model_dir = self.config.get_config('paths', 'model_dir')
        self.static_dir = self.config.get_config('paths', 'static_dir')
        self.template_dir = self.config.get_config('paths', 'template_dir')

        # Target variable and cutoff date
        self.TARGET_VARIABLE = self.config.get_config('constants', 'TARGET_VARIABLE')
        self.CUTOFF_DATE = self.config.get_config('constants', 'CUTOFF_DATE')

        # Class instances
        self.data_processing = DataProcessing(self.TARGET_VARIABLE)
        self.sim_visualization = SimVisualization(self.template_dir, self.TARGET_VARIABLE)

        # MongoDB setup
        self.MONGO_URI = self.config.get_config('database', 'mongo_uri')
        self.DATABASE_NAME = self.config.get_config('database', 'database_name')

        # Logger setup
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        self.logger.addHandler(handler)

        # Connect to MongoDB
        try:
            self.client = MongoClient(self.MONGO_URI)
            self.db = self.client[self.DATABASE_NAME]
        except Exception as e:
            self.logger.info(f"Error connecting to MongoDB: {e}")
            raise

        # Load model and scaler
        self.LOADED_MODEL, self.LOADED_SCALER = self.load_model_and_scaler()
        self.model = Modeling(self.LOADED_MODEL, self.LOADED_SCALER, HOME_FIELD_ADJUST, self.static_dir)

    def file_cleanup(self):
        """
        Cleans up existing CSV files in the static directory.

        This method removes specific files to ensure that outdated data is not used in new predictions.

        Notes:
        - Placeholder for future feature: Extend this to handle more file types and dynamic file naming.
        """
        # Delete existing CSV files
        for filename in ['combined_sampled_data.csv', 'simulation_results.csv']:
            file_path = os.path.join(self.static_dir, filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    self.logger.info(f"Deleted file: {filename}")
                except Exception as e:
                    self.logger.error(f"Error deleting file {filename}: {e}")

    def set_game_details(self, home_team, away_team, date):
        """
        Sets the game details for the current simulation.

        Parameters:
        home_team (str): The name of the home team.
        away_team (str): The name of the away team.
        date (str or datetime): The date of the game.
        """
        self.logger.info(f"Setting game details for {home_team} vs {away_team} on {date}")

        # Team name update
        team_name_updates = {"Football Team": "Commanders", "Redskins": "Commanders"}
        home_team = team_name_updates.get(home_team, home_team)
        away_team = team_name_updates.get(away_team, away_team)

        self.home_team = home_team
        self.away_team = away_team

        # Date conversion
        if isinstance(date, str):
            self.date = datetime.strptime(date, '%Y-%m-%dT%H:%M:%S%z').strftime('%Y-%m-%d')
        else:
            self.date = date.strftime('%Y-%m-%d')

    def load_model_and_scaler(self):
        """
        Loads the pre-trained model and data scaler from the model directory.

        Returns:
        Tuple: Containing the loaded model and scaler, or (None, None) if files are not found.
        """
        try:
            model = joblib.load(os.path.join(self.model_dir, 'trained_nfl_model.pkl'))
            scaler = joblib.load(os.path.join(self.model_dir, 'data_scaler.pkl'))
            return model, scaler
        except FileNotFoundError as e:
            self.logger.info(f"Error loading files: {e}")
            return None, None

    def get_historical_data(self, random_subset=None, get_current=False):
        """
        Fetches historical game data and applies various filters.

        Parameters:
        random_subset (int, optional): Number of random records to fetch. If None, fetch all records.
        get_current (bool, optional): Whether to get only current games within a week.

        Returns:
        DataFrame: Filtered historical game data.

        Notes:
        - Placeholder for future feature: Extend filtering capabilities based on new criteria.
        """
        df = self.database_operations.fetch_data_from_mongodb('games')
        historical_df = self.data_processing.flatten_and_merge_data(df)

        # Date conversions and filtering
        historical_df['scheduled'] = pd.to_datetime(historical_df['scheduled'])
        current_date = datetime.now(pytz.utc)
        seven_days_ahead = current_date + timedelta(days=7)
        historical_df = historical_df[historical_df['scheduled'] <= seven_days_ahead]
        historical_df = historical_df[historical_df['scheduled'] >= self.CUTOFF_DATE]

        # NaN value filtering
        columns_to_check = [col for col in historical_df.columns if col.startswith('summary.odds.spread')]
        historical_df = historical_df.dropna(subset=columns_to_check, how='any')

        # Column extraction
        columns_to_extract = [
            'id', 'scheduled', 'summary.home.id', 'summary.home.name',
            'summary.away.id', 'summary.away.name', 'summary.home.points', 'summary.away.points'
        ]
        odds_columns = [col for col in historical_df.columns if col.startswith('summary.odds.')]
        columns_to_extract.extend(odds_columns)
        game_data = historical_df[columns_to_extract]

        # Date format adjustment
        game_data['scheduled'] = pd.to_datetime(game_data['scheduled']).dt.date

        # Additional filtering based on provided parameters
        if get_current:
            today = datetime.now().date()
            one_week_from_now = today + timedelta(days=6)
            game_data = game_data[(game_data['scheduled'] >= today) & (game_data['scheduled'] <= one_week_from_now)]
        elif random_subset:
            game_data = game_data.dropna(how='any')
            game_data = game_data.sample(n=random_subset, replace=True)

        return game_data

    def analyze_and_log_results(self, simulation_results, home_team, away_team):
        """
        Analyzes the simulation results and logs detailed outcomes.

        Parameters:
        simulation_results (list): The results of the game simulations.
        home_team (str): The home team's name.
        away_team (str): The away team's name.

        Returns:
        str: A summary string of the analysis results.

        Notes:
        - Calculates the range of outcomes, standard deviation, and confidence interval.
        - Logs the detailed results for each game.
        """
        self.logger.info("Starting analysis of simulation results.")

        # Analysis of simulation results
        range_of_outcomes, standard_deviation, confidence_interval = self.sim_visualization.analyze_simulation_results(simulation_results)

        # Log results
        self.logger.info(f"Game: {away_team} at {home_team}")
        self.logger.info(f"Outcome Range: {range_of_outcomes}")
        self.logger.info(f"Standard Deviation: {standard_deviation}")
        self.logger.info(f"95% Confidence Interval: {confidence_interval}")

        return (f"Analysis Complete: {away_team} at {home_team}, "
                f"Outcome Range: {range_of_outcomes}, "
                f"Standard Deviation: {standard_deviation}, "
                f"95% Confidence Interval: {confidence_interval}")

    def simulate_games(self, num_simulations=1000, random_subset=None, get_current=False):
        """
        Simulates NFL games using Monte Carlo methods and evaluates the results.

        Parameters:
        num_simulations (int): Number of simulations to run per game.
        random_subset (int, optional): Number of random records to fetch. If None, fetch all records.
        get_current (bool, optional): Whether to get only current games within a week.

        Notes:
        - Retrieves and prepares data for simulation.
        - Utilizes multiprocessing to speed up the simulation process.
        - Analyzes and logs the results of each simulation.
        - Evaluates betting recommendations and expected values.
        """
        self.logger.info("Starting prediction process...")
        reload(scripts.constants)

        # Data retrieval and preparation
        df = self.database_operations.fetch_data_from_mongodb('team_aggregated_metrics')
        historical_df = self.get_historical_data(random_subset, get_current)

        # Initialize lists for results
        all_simulation_results = []
        all_actual_results = []
        params_list = []

        # File cleanup before simulation
        self.file_cleanup()

        # Prepare data for each game
        for _, row in historical_df.iterrows():
            home_team, away_team = row['summary.home.name'], row['summary.away.name']
            home_team, away_team = self.data_processing.replace_team_name(home_team), self.data_processing.replace_team_name(away_team)

            self.set_game_details(home_team, away_team, row['scheduled'])
            game_prediction_df = self.data_processing.prepare_data(df, self.features, home_team, away_team, self.date)

            # Diagnostic check for data availability
            if game_prediction_df.empty:
                self.logger.info(f"Game prediction DataFrame is empty for {home_team} vs {away_team}")
                continue

            params_list.append((game_prediction_df, self.model, home_team, away_team))

        # Multiprocessing for simulation
        with multiprocessing.Pool(processes=num_processes) as pool:
            args_list = [(param, num_simulations) for param in params_list]

            # Diagnostic check for argument list
            if not args_list:
                self.logger.info("args_list for multiprocessing pool is empty.")
                return

            results = pool.map(run_simulation_wrapper, args_list)

        # Processing and analyzing results
        for idx, (simulation_results, home_team, away_team) in enumerate(results):
            self.logger.info(f"Processing game {idx + 1}/{len(results)}: {home_team} vs {away_team}")
            self.analyze_and_log_results(simulation_results, home_team, away_team)
            all_simulation_results.append(simulation_results)

            # Process and store actual results
            row = historical_df.iloc[idx]
            actual_difference = (row['summary.home.points'] - row['summary.away.points']) * (-1) if row['summary.home.points'] is not None and row['summary.away.points'] is not None else None
            all_actual_results.append(actual_difference)
            self.logger.info(f"Processing game {idx + 1}/{len(results)} complete.")

        # Evaluate and recommend based on simulations
        self.logger.info("Evaluating results...")

        self.sim_visualization.evaluate_and_recommend(all_simulation_results, historical_df, get_current)


def run_simulation_wrapper(args):
    """
    Wrapper function for run_simulation to facilitate multiprocessing.

    Parameters:
    args (tuple): Arguments to be passed to the run_simulation function.

    Returns:
    The result of the run_simulation function.

    Notes:
    - This function is a helper to enable the use of multiprocessing with arguments.
    """
    return run_simulation(*args)


def run_simulation(params, num_simulations):
    """
    Runs the Monte Carlo simulation for a given game.

    Parameters:
    params (tuple): A tuple containing the game prediction DataFrame, the model, and team names.
    num_simulations (int): Number of simulations to perform.

    Returns:
    The result of the Monte Carlo simulation.

    Notes:
    - The function is designed to be used with multiprocessing for efficiency.
    """
    game_prediction_df, model, home_team, away_team = params
    return model.monte_carlo_simulation(game_prediction_df, home_team, away_team, num_simulations)


# Main execution block
if __name__ == "__main__":
    """
    Main execution block for running NFL game predictions.
    """
    predictor = NFLPredictor()
    # Additional code for triggering prediction process
    # Example: predictor.simulate_games(num_simulations=1000)
