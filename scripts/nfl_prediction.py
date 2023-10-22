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

HOME_FIELD_ADJUST = -2.7


class NFLPredictor:
    def __init__(self):
        # Configuration and Database Connection
        self.config = ConfigManager()
        self.data_processing = DataProcessing()
        self.database_operations = DatabaseOperations()

        # Constants
        self.data_dir = self.config.get_config('paths', 'data_dir')
        self.model_dir = self.config.get_config('paths', 'model_dir')

        self.template_dir = self.config.get_config('paths', 'template_dir')
        self.visualization = Visualization(self.template_dir)

        self.MONGO_URI = self.config.get_config('database', 'mongo_uri')
        self.DATABASE_NAME = self.config.get_config('database', 'database_name')

        # Initialize logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        try:
            self.client = MongoClient(self.MONGO_URI)
            self.db = self.client[self.DATABASE_NAME]
        except Exception as e:
            logging.error(f"Error connecting to MongoDB: {e}")
            raise

        # Loading the pre-trained model and the data scaler
        self.LOADED_MODEL, self.LOADED_SCALER = self.load_model_and_scaler()
        self.model = Modeling(self.LOADED_MODEL, self.LOADED_SCALER, HOME_FIELD_ADJUST)

    def set_game_details(self, home_team, away_team, date):
        """Set the game details for the current simulation."""

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
            logging.error(f"Error loading files: {e}")
            return None, None

    def get_historical_data(self, random_subset=None, get_current=True, adhoc=False):
        """
        Prepare historical data for simulations.

        :param random_subset: Number of random games to fetch. If None, fetch all games.
        :return: DataFrame containing game data.
        """
        df = self.database_operations.fetch_data_from_mongodb('games')
        historical_df = self.data_processing.flatten_and_merge_data(df)

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
            game_data = game_data.sample(n=random_subset)

        elif adhoc:
            print("NEED TO DEVELOP STILL")

        return game_data

    def main(self):
        """Predicts the target value using Monte Carlo simulation and visualizes the results."""
        self.logger.info("Starting prediction process...")

        reload(scripts.constants)
        features = [col for col in scripts.constants.COLUMNS_TO_KEEP if 'odd' not in col]

        # Fetch data only for the selected teams from the weekly_ranks collection
        df = self.database_operations.fetch_data_from_mongodb('weekly_ranks')

        historical_df = self.get_historical_data(random_subset=60)

        # Lists to store simulation results and actual results for each game
        all_simulation_results = []
        all_actual_results = []
        params_list = []

        # Loop over each game in the historical data
        for _, row in historical_df.iterrows():
            # Set the game details for the current simulation
            self.set_game_details(row['summary.home.name'], row['summary.away.name'], row['scheduled'])

            game_prediction_df = self.data_processing.prepare_data(df, features, self.home_team, self.away_team, self.date)
            params_list.append((game_prediction_df, self.data_processing.get_standard_deviation(df), self.model))

        # Use Pool to run simulations in parallel
        with Pool() as pool:
            results = pool.map(run_simulation, params_list)
        pool.close()
        pool.join()

        # Process these results in a separate loop:
        for idx, (simulation_results, most_likely_outcome) in enumerate(results):
            row = historical_df.iloc[idx]

            # Run Monte Carlo simulations
            self.logger.info(f"Running Simulations for {self.home_team} vs {self.away_team} on {self.date}...")
            simulation_results, most_likely_outcome = self.model.monte_carlo_simulation(game_prediction_df, self.data_processing.get_standard_deviation(df))

            # Analyze simulation results
            self.logger.info("Analyzing Simulation Results...")
            range_of_outcomes, standard_deviation, confidence_interval = self.model.analyze_simulation_results(simulation_results)

            # Store simulation results and actual results for evaluation
            all_simulation_results.append(most_likely_outcome)
            actual_difference = (row['summary.home.points'] - row['summary.away.points'])*(-1)
            all_actual_results.append(actual_difference)

            # Create a buffer to capture log messages
            log_capture_buffer = io.StringIO()

            # Set up the logger to write to the buffer
            log_handler = logging.StreamHandler(log_capture_buffer)
            log_handler.setLevel(logging.INFO)
            self.logger.addHandler(log_handler)

            # User-friendly output
            self.logger.info(f"Expected target value for {self.away_team} at {self.home_team}: {range_of_outcomes[0]:.2f} to {range_of_outcomes[1]:.2f} points.")
            self.logger.info(f"95% Confidence Interval: {confidence_interval[0]:.2f} to {confidence_interval[1]:.2f} for {self.home_team} projected spread.")
            self.logger.info(f"Most likely target value: {most_likely_outcome:.2f} for {self.home_team} projected spread.")
            self.logger.info(f"Standard deviation of target values: {standard_deviation:.2f} for {self.home_team} projected spread.")

            # Retrieve the log messages from the buffer
            log_contents = log_capture_buffer.getvalue()
            explanation = self.model.analysis_explanation(range_of_outcomes, confidence_interval, most_likely_outcome, standard_deviation)
            combined_output = log_contents + "\n\n" + explanation

            # Visualize simulation results
            self.visualization.visualize_simulation_results(simulation_results, most_likely_outcome, combined_output)

        # After the loop, compare the simulated results to the actual results
        self.visualization.compare_simulated_to_actual(all_simulation_results, all_actual_results)

        # Evaluate the betting recommendation and expected value
        recommendation_accuracy, average_ev, actual_value = self.visualization.evaluate_and_recommend(all_simulation_results, historical_df)
        self.logger.info(f"Recommendation Accuracy: {recommendation_accuracy:.2f}%")
        self.logger.info(f"Average Expected Value: {average_ev:.2f}%")
        self.logger.info(f"Actual Results: ${actual_value:.2f}")


def run_simulation(params):
    game_prediction_df, standard_deviation, model = params
    return model.monte_carlo_simulation(game_prediction_df, standard_deviation)


if __name__ == "__main__":
    predictor = NFLPredictor()
    predictor.main()
    # Call other methods of the predictor as needed
