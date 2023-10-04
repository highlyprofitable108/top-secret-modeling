from classes.config_manager import ConfigManager
from classes.data_processing import DataProcessing
from classes.database_operations import DatabaseOperations
from .constants import COLUMNS_TO_KEEP
import os
import joblib
from pymongo import MongoClient
import time
import random
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import mode, gaussian_kde

class NFLPredictor:
    def __init__(self):
        # Configuration and Database Connection
        self.config = ConfigManager()
        self.data_processing = DataProcessing()
        self.database_operations = DatabaseOperations()

        # Constants
        self.data_dir = self.config.get_config('paths', 'data_dir')
        self.model_dir = self.config.get_config('paths', 'model_dir')
        self.MONGO_URI = self.config.get_config('database', 'mongo_uri')
        self.DATABASE_NAME = self.config.get_config('database', 'database_name')
        self.client = MongoClient(self.MONGO_URI)
        self.db = self.client[self.DATABASE_NAME]

        # Define the feature columns by removing the target variable from COLUMNS_TO_KEEP
        self.feature_columns = [col for col in COLUMNS_TO_KEEP if col != 'scoring_differential']

        # Loading the pre-trained model and the data scaler
        self.LOADED_MODEL, self.LOADED_SCALER = self.load_model_and_scaler()

    def load_model_and_scaler(self):
        try:
            model = joblib.load(os.path.join(self.model_dir, 'trained_nfl_model.pkl'))
            scaler = joblib.load(os.path.join(self.model_dir, 'data_scaler.pkl'))
            return model, scaler
        except FileNotFoundError as e:
            print(f"Error loading files: {e}")
            # Handle the error appropriately or return None values
            return None, None
        
    def get_team_data(self, alias, df):
        """Fetches data for a specific team based on the alias from the provided DataFrame."""
        return df[df['alias'] == alias]

    def rename_columns(self, data, prefix):
        """Renames columns with a specified prefix."""
        columns_to_rename = [col.replace('statistics_home.', '') for col in COLUMNS_TO_KEEP if col.startswith('statistics_home.')]
        rename_dict = {col: f"{prefix}{col}" for col in columns_to_rename}
        return data.rename(columns=rename_dict)

    def merge_data(self, home_data, away_data, home_data_stddev, away_data_stddev):
        """Merges data and standard deviation data for both home and away teams."""
        return pd.concat([home_data, home_data_stddev, away_data, away_data_stddev], axis=1)

    def filter_and_scale_data(self, merged_data):
        """Filters the merged data using the feature columns and scales the numeric features."""
        merged_data = merged_data[self.feature_columns]
        merged_data[self.feature_columns] = self.LOADED_SCALER.transform(merged_data[self.feature_columns])
        return merged_data
    
    def monte_carlo_simulation(self, df, num_simulations=2500):
        print("Starting Monte Carlo Simulation...")
        df = self.data_processing.handle_prediction_values(df)

        simulation_results = []

        start_time = time.time()
        with tqdm(total=num_simulations, ncols=100) as pbar:  # Initialize tqdm with total number of simulations
            for _ in range(num_simulations):
                sampled_df = df.copy()
                for column in sampled_df.columns:
                    if column + '_stddev' in sampled_df.columns:
                        mean_value = df[column].iloc[0]
                        stddev_value = df[column + '_stddev'].iloc[0]
                        sampled_value = np.random.normal(mean_value, stddev_value)
                        sampled_df[column] = sampled_value

                # TODO: Get updated list excluding NULLs

                # Filter columns and scale numeric features
                sampled_df = sampled_df[self.feature_columns]
                sampled_df = self.data_processing.handle_null_values(sampled_df)
                sampled_df[self.feature_columns] = self.LOADED_SCALER.transform(sampled_df[self.feature_columns])

                prediction = self.LOADED_MODEL.predict(sampled_df)
                simulation_results.append(prediction[0])

                pbar.update(1)  # Increment tqdm progress bar
                if time.time() - start_time > 10:
                    pbar.set_postfix_str("Running simulations...")
                    start_time = time.time()

        # After obtaining simulation_results
        kernel = gaussian_kde(simulation_results)
        most_likely_outcome = simulation_results[np.argmax(kernel(simulation_results))]

        print("Monte Carlo Simulation Completed!")
        return simulation_results, most_likely_outcome

    def analyze_simulation_results(self, simulation_results):
        """Analyzes the simulation results to compute the range of outcomes, standard deviation, and the most likely outcome."""

        # Sort the simulation results
        sorted_results = sorted(simulation_results)

        # Filter out the extreme x% of results on either end (keeping central y%)
        lower_bound = int(0.1 * len(sorted_results))
        upper_bound = int(0.9 * len(sorted_results))
        filtered_results = sorted_results[lower_bound:upper_bound]

        # Calculate the range of outcomes based on the filtered results
        range_of_outcomes = (min(filtered_results), max(filtered_results))

        # Calculate the standard deviation based on the filtered results
        standard_deviation = np.std(filtered_results)

        return range_of_outcomes, standard_deviation

    def visualize_simulation_results(self, simulation_results, most_likely_outcome):
        """Visualizes the simulation results using a histogram and calculates the rounded most likely outcome."""

        # Plot the histogram of the simulation results using seaborn
        plt.figure(figsize=(10, 6))
        sns.histplot(simulation_results, bins=50, kde=True, color='blue')
        plt.axvline(most_likely_outcome, color='red', linestyle='--')
        plt.title("Monte Carlo Simulation Results")
        plt.xlabel("Scoring Differential")
        plt.ylabel("Density")
        plt.grid()
        # plt.show()

        # Calculate the rounded most likely outcome
        rounded_most_likely_outcome = (round(most_likely_outcome * 2) / 2) * (-1)

        # Format the rounded most likely outcome with a leading + sign for positive values
        if rounded_most_likely_outcome > 0:
            formatted_most_likely_outcome = f"+{rounded_most_likely_outcome:.2f}"
        else:
            formatted_most_likely_outcome = f"{rounded_most_likely_outcome:.2f}"

        return formatted_most_likely_outcome

    def main(self):
        """Predicts the scoring differential using Monte Carlo simulation and visualizes the results."""
        # Fetch the teams collection to create a mapping of team_alias to team_id
        teams_df = self.database_operations.fetch_data_from_mongodb('teams')
        team_alias_to_id = dict(zip(teams_df['alias'], teams_df['id']))

        print("Available Team Aliases:")
        aliases = list(team_alias_to_id.keys())
        for i in range(0, len(aliases), 6):
            print("\t".join(aliases[i:i+6]))

        # Prompt user for input
        home_team_alias = input("Enter home team alias: ")
        while home_team_alias not in team_alias_to_id:
            print("Invalid team alias. Please enter a valid team alias from the list above.")
            home_team_alias = input("Enter home team alias: ")

        away_team_alias = input("Enter away team alias: ")
        while away_team_alias not in team_alias_to_id:
            print("Invalid opponent team alias. Please enter a valid team alias from the list above.")
            away_team_alias = input("Enter away team alias: ")

        # Fetch data only for the selected teams from the team_aggregated_metrics collection
        df = self.database_operations.fetch_data_from_mongodb('team_aggregated_metrics')

        home_team_data = self.get_team_data(home_team_alias, df)
        away_team_data = self.get_team_data(away_team_alias, df)

        # Rename columns and merge data
        home_team_data = self.rename_columns(home_team_data.reset_index(drop=True), "statistics_home.")
        away_team_data = self.rename_columns(away_team_data.reset_index(drop=True), "statistics_away.")

        # Identify and rename standard deviation columns
        home_team_data_stddev = home_team_data.filter(like='stddev').rename(columns=lambda x: "statistics_home." + x)
        away_team_data_stddev = away_team_data.filter(like='stddev').rename(columns=lambda x: "statistics_away." + x)

        # Create matchup
        merged_data = self.merge_data(home_team_data, away_team_data, home_team_data_stddev, away_team_data_stddev)

        # Run Monte Carlo simulations
        print("\nRunning Simulations...")
        simulation_results, most_likely_outcome = self.monte_carlo_simulation(merged_data)

        # Analyze simulation results
        print("Analyzing Simulation Results...")
        range_of_outcomes, standard_deviation = self.analyze_simulation_results(simulation_results)

        # Visualize simulation results
        formatted_most_likely_outcome = self.visualize_simulation_results(simulation_results, most_likely_outcome)

        # User-friendly output
        print("\nPrediction Results:")
        print(f"Based on our model, the expected scoring differential for {home_team_alias} against {away_team_alias} is between {range_of_outcomes[0]:.2f} and {range_of_outcomes[1]:.2f} points.")
        print(f"The most likely scoring differential is approximately {most_likely_outcome:.2f} points.")
        print(f"The standard deviation of the scoring differentials is approximately {standard_deviation:.2f} points.\n")
        print(f"The projected spread on this game should be {home_team_alias} {formatted_most_likely_outcome}.\n")


if __name__ == "__main__":
    predictor = NFLPredictor()
    predictor.main()
    # Call other methods of the predictor as needed
