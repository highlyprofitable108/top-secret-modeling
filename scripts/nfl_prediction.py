# Standard library imports
import io
import os
import sys
import time
from datetime import datetime, timedelta
from importlib import reload

# Third-party imports
import joblib
import mpld3

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import gaussian_kde, t
from pymongo import MongoClient
from tqdm import tqdm


# Local application imports
from classes.config_manager import ConfigManager
from classes.data_processing import DataProcessing
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

    def prepare_data(self, df, features):
        """Prepare data for simulation."""
        # Create a dictionary with column names as keys and arithmetic operations as values
        feature_operations = {col.rsplit('_', 1)[0]: col.rsplit('_', 1)[1] for col in features}

        # Extract unique base column names from the features list
        base_column_names = set(col.rsplit('_', 1)[0] for col in features)

        # Filter the home_team_data and away_team_data DataFrames to retain only the necessary columns
        home_features = self.get_team_data(df, self.home_team)[list(base_column_names.intersection(df.columns))].reset_index(drop=True)
        away_features = self.get_team_data(df, self.away_team)[list(base_column_names.intersection(df.columns))].reset_index(drop=True)
        print(home_features)
        # Initialize an empty DataFrame for the results
        game_prediction_df = pd.DataFrame()

        # Iterate over the columns using the dictionary
        for col, operation in feature_operations.items():
            if operation == "difference":
                game_prediction_df[col + "_difference"] = home_features[col] - away_features[col]
            else:
                game_prediction_df[col + "_ratio"] = home_features[col] / away_features[col]

        # Handle potential division by zero issues
        game_prediction_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return game_prediction_df

    def get_team_data(self, df, team):
        """Fetches data for a specific team based on the alias from the provided DataFrame."""
        # Convert the 'update_date' column to a datetime object and extract only the "yyyy-mm-dd" part
        print(team)
        df['update_date'] = pd.to_datetime(df['update_date']).dt.strftime('%Y-%m-%d')
        condition = df['update_date'] <= self.date

        # Filter the DataFrame based on the team name and the date condition
        filtered_df = df[(df['name'].str.lower() == team.lower()) & condition]
        print(filtered_df.columns)
        # Get the index of the row with the most recent update_date
        idx = filtered_df['update_date'].idxmax()

        # Return the row with the most recent update_date
        return df.loc[[idx]]

    def get_standard_deviation(self, df):
        """Fetches the standard deviation for each column in the provided DataFrame."""
        # Exclude non-numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        # Compute the standard deviation for each numeric column
        standard_deviation_df = numeric_df.std().to_frame().transpose()

        # Return the standard deviation DataFrame
        return standard_deviation_df

    def rename_columns(self, data, prefix):
        """Renames columns with a specified prefix."""
        reload(scripts.constants)
        columns_to_rename = [col.replace('ranks_home_', '') for col in scripts.constants.COLUMNS_TO_KEEP if col.startswith('ranks_home_')]
        rename_dict = {col: f"{prefix}{col}" for col in columns_to_rename}
        return data.rename(columns=rename_dict)

    def filter_and_scale_data(self, merged_data):
        """Filters the merged data using the feature columns and scales the numeric features."""
        # Reload constants
        reload(scripts.constants)

        # Filter the DataFrame
        filtered_data = merged_data[merged_data['update_date'] == self.date]

        # Filter columns with stripping whitespaces and exclude 'scoring_differential'
        columns_to_filter = [col for col in scripts.constants.COLUMNS_TO_KEEP if col.strip() in map(str.strip, filtered_data.columns) and col.strip() != 'scoring_differential']

        filtered_data = filtered_data[columns_to_filter]
        filtered_data[columns_to_filter] = self.LOADED_SCALER.transform(filtered_data[columns_to_filter])
        return filtered_data

    def monte_carlo_simulation(self, df, standard_deviation_df, num_simulations=2500):
        logging.info(df.head())
        logging.info("Starting Monte Carlo Simulation...")

        simulation_results = []

        # List to store sampled_df for each iteration
        sampled_data_list = []

        start_time = time.time()
        with tqdm(total=num_simulations, ncols=100) as pbar:  # Initialize tqdm with total number of simulations
            for _ in range(num_simulations):
                sampled_df = df.copy()
                for column in sampled_df.columns:
                    base_column = column.replace('_difference', '').replace('_ratio', '')
                    if base_column in standard_deviation_df.columns:
                        mean_value = df[column].iloc[0]
                        stddev_value = standard_deviation_df[base_column].iloc[0]
                        sampled_value = np.random.normal(mean_value, stddev_value)
                        sampled_df[column] = sampled_value

                # Append sampled_df to the list
                sampled_data_list.append(sampled_df)

                modified_df = sampled_df.dropna(axis=1, how='any')
                scaled_df = self.LOADED_SCALER.transform(modified_df)

                try:
                    prediction = self.LOADED_MODEL.predict(scaled_df)
                    adjusted_prediction = prediction[0] + HOME_FIELD_ADJUST
                    simulation_results.append(adjusted_prediction)
                except Exception as e:
                    logging.error(f"Error during prediction: {e}")

                pbar.update(1)  # Increment tqdm progress bar
                if time.time() - start_time > 10:
                    pbar.set_postfix_str("Running simulations...")
                    start_time = time.time()

        # After obtaining simulation_results
        kernel = gaussian_kde(simulation_results)
        most_likely_outcome = simulation_results[np.argmax(kernel(simulation_results))]

        # Save simulation_results to a CSV file
        combined_sampled_data = pd.concat(sampled_data_list, axis=0, ignore_index=True)
        combined_sampled_data.to_csv('combined_sampled_data.csv', index=False)
        pd.DataFrame(simulation_results, columns=['Simulation_Result']).to_csv('simulation_results.csv', index=False)

        logging.info("Monte Carlo Simulation Completed!")
        return simulation_results, most_likely_outcome

    def compute_confidence_interval(self, data, confidence=0.95):
        """Compute the confidence interval for a given dataset."""
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), np.std(a)/np.sqrt(n)
        h = se * t.ppf((1 + confidence) / 2., n-1)
        return m-h, m+h

    def analyze_simulation_results(self, simulation_results):
        """
        Analyzes the simulation results to compute the range of outcomes, 
        standard deviation, and the most likely outcome.
        """

        # Constants to define the bounds for filtering the results
        LOWER_PERCENTILE = 0.1
        UPPER_PERCENTILE = 0.9

        # Calculate the lower and upper bounds based on percentiles
        lower_bound_value = np.percentile(simulation_results, LOWER_PERCENTILE * 100)
        upper_bound_value = np.percentile(simulation_results, UPPER_PERCENTILE * 100)

        # Filter the results based on the calculated bounds
        filtered_results = [result for result in simulation_results if lower_bound_value <= result <= upper_bound_value]

        # Save filtered_results to a CSV file
        pd.DataFrame(filtered_results, columns=['Filtered_Result']).to_csv('filtered_results.csv', index=False)

        # Calculate the range of outcomes based on the filtered results
        range_of_outcomes = (min(filtered_results), max(filtered_results))

        # Calculate the standard deviation based on the filtered results
        standard_deviation = np.std(filtered_results)

        # Calculate confidence intervals
        confidence_interval = self.compute_confidence_interval(simulation_results)

        return range_of_outcomes, standard_deviation, confidence_interval

    def analysis_explanation(self, range_of_outcomes, confidence_interval, most_likely_outcome, standard_deviation):
        explanation = """
            Let's imagine we're trying to guess how many candies are in a big jar!

            We think there might be between {low_guess:.2f} and {high_guess:.2f} candies.
            We're pretty sure (like, 95% sure!) that the number of candies is between {low_confidence:.2f} and {high_confidence:.2f}.
            Our best guess is that there are {most_likely:.2f} candies in the jar.
            Our guesses are kind of {spread_out}. That number is {std_dev:.2f}.

            Now, let's see if we're right!
        """.format(
            low_guess=range_of_outcomes[0],
            high_guess=range_of_outcomes[1],
            low_confidence=confidence_interval[0],
            high_confidence=confidence_interval[1],
            most_likely=most_likely_outcome,
            spread_out="all over the place" if standard_deviation > 2 else "close together",
            std_dev=standard_deviation
        )
        return explanation

    def visualize_simulation_results(self, simulation_results, most_likely_outcome, output, bins=50):
        """
        Visualizes the simulation results using a histogram and calculates 
        the rounded most likely outcome.
        """

        # Plot using Plotly
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=simulation_results, name='Simulation Results', nbinsx=bins))
        fig.add_shape(type="line", x0=most_likely_outcome, x1=most_likely_outcome, y0=0, y1=1, yref='paper', line=dict(color="Red"))
        fig.update_layout(title="Monte Carlo Simulation Results", xaxis_title="Target Value", yaxis_title="Density")

        # Convert the Plotly figure to HTML
        plotly_html_string = fig.to_html(full_html=False, include_plotlyjs='cdn')

        # Format the output for better readability
        formatted_output = f"""
        <div style="background-color: #f7f7f7; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
            <h2>Simulation Summary:</h2>
            <pre style="font-size: 16px; color: #333;">{output}</pre>
        </div>
        """

        # Combine the formatted output and the Plotly HTML
        full_html = plotly_html_string + formatted_output

        # Path to save the visualization as an HTML file
        feature_importance_path = os.path.join(self.template_dir, 'simulation_distribution.html')

        # Save the combined HTML
        try:
            with open(feature_importance_path, "w") as f:
                f.write(full_html)
        except IOError as e:
            # Handle potential file write issues
            print(f"Error writing to file: {e}")

        # Calculate the rounded most likely outcome
        rounded_most_likely_outcome = (round(most_likely_outcome * 2) / 2)

        # Format the rounded most likely outcome with a leading + sign for positive values
        if rounded_most_likely_outcome > 0:
            formatted_most_likely_outcome = f"+{rounded_most_likely_outcome:.2f}"
        else:
            formatted_most_likely_outcome = f"{rounded_most_likely_outcome:.2f}"

        return formatted_most_likely_outcome

    def compare_simulated_to_actual(self, simulation_results, actual_results):
        model_covered = 0

        for idx in range(len(simulation_results)):
            actual_difference = actual_results[idx]
            predicted_difference = simulation_results[idx]

            if actual_difference > predicted_difference:
                model_covered += 1

        model_cover_rate = model_covered / len(simulation_results) * 100

        return model_cover_rate

    def evaluate_and_recommend(self, simulation_results, historical_df):
        historical_df = historical_df.reset_index(drop=True)
        correct_recommendations = 0
        total_bets = 0
        total_ev = 0

        # Create a DataFrame to store the results
        results_df = pd.DataFrame(columns=['Date', 'Home Team', 'Home Points', 'Vegas Odds', 'Modeling Odds', 'Away Team', 'Away Points', 'Recommended Bet', 'Result with Spread', 'Actual Covered', 'Bet Outcome', 'Expected Value (%)', 'Actual Value ($)'])

        for idx, row in historical_df.iterrows():
            actual_home_points = row['summary.home.points']
            actual_away_points = row['summary.away.points']
            home_team = row['summary.home.name']
            away_team = row['summary.away.name']
            spread_odds = row['summary.odds.spread']
            date = row['scheduled']

            predicted_difference = simulation_results[idx]

            # Check if the game has actual results or if it's a future game
            if pd.isna(actual_home_points) or pd.isna(actual_away_points):
                # Future game
                actual_covered = ""
                bet_outcome = ""
                actual_value = np.nan
                actual_difference = np.nan
            else:
                # Past game with actual results
                actual_difference = (actual_home_points + spread_odds) - actual_away_points

                # Determine who actually covered based on spread odds
                if actual_difference < 0:
                    actual_covered = "away"
                elif actual_difference > 0:
                    actual_covered = "home"
                else:
                    actual_covered = "push"

                # Check historical performance
                if recommended_bet == actual_covered:
                    correct_recommendations += 1
                    total_bets += 1
                    bet_outcome = "Win"
                    actual_value = 100  # Profit from a winning bet
                elif recommended_bet == "push" or actual_covered == "push":
                    bet_outcome = "Push"
                    actual_value = 0  # No profit, no loss from a push
                else:
                    total_bets += 1
                    bet_outcome = "Loss"
                    actual_value = -110  # Loss from a losing bet

            # Recommendation based on model
            reccommendation_calc = spread_odds - predicted_difference
            if reccommendation_calc < 0:
                recommended_bet = "away"
            elif reccommendation_calc > 0:
                recommended_bet = "home"
            else:
                recommended_bet = "push"  # or any default recommendation

            # Calculate the model's implied probability
            if recommended_bet == "home":
                model_probability = 1 / (1 + 10 ** (predicted_difference / 10))
            else:
                model_probability = 1 / (1 + 10 ** (-predicted_difference / 10))

            # Calculate the Vegas implied probability
            if recommended_bet == "home":
                vegas_probability = 1 / (1 + 10 ** (spread_odds / 10))
            else:
                vegas_probability = 1 / (1 + 10 ** (-spread_odds / 10))

            # Calculate expected value (adjusted for model vs. Vegas odds)
            ev_fraction = model_probability - vegas_probability  # This is the EV as a fraction of the bet

            # Express EV as a percentage of the bet
            ev_percentage = ev_fraction * 100

            total_ev += ev_percentage

            # Check historical performance
            if recommended_bet == actual_covered:
                correct_recommendations += 1
                total_bets += 1
                bet_outcome = "Win"
                actual_value = 100  # Profit from a winning bet
            elif recommended_bet == "push" or actual_covered == "push":
                bet_outcome = "Push"
                actual_value = 0  # No profit, no loss from a push
            else:
                total_bets += 1
                bet_outcome = "Loss"
                actual_value = -110  # Loss from a losing bet

            # Append to results DataFrame
            new_row = pd.Series({
                'Date': date,
                'Home Team': home_team,
                'Home Points': actual_home_points,
                'Vegas Odds': spread_odds,
                'Modeling Odds': predicted_difference,
                'Away Team': away_team,
                'Away Points': actual_away_points,
                'Recommended Bet': recommended_bet,
                'Result with Spread': actual_difference,
                'Actual Covered': actual_covered,
                'Bet Outcome': bet_outcome,
                'Expected Value (%)': ev_percentage,
                'Actual Value ($)': actual_value
            })

            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

        # Round all values in the DataFrame to 2 decimals
        results_df = results_df.round(2)

        # Save results to CSV
        results_df.to_csv('betting_recommendation_results.csv', index=False)

        # Calculate the total actual value from all bets
        total_actual_value = results_df['Actual Value ($)'].sum()

        recommendation_accuracy = correct_recommendations / total_bets * 100
        average_ev = total_ev / total_bets
        average_ev_percent = (average_ev / 110) * 100

        return recommendation_accuracy, average_ev_percent, total_actual_value

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

        # Loop over each game in the historical data
        for _, row in historical_df.iterrows():
            # Set the game details for the current simulation
            self.set_game_details(row['summary.home.name'], row['summary.away.name'], row['scheduled'])

            game_prediction_df = self.prepare_data(df, features)

            # Run Monte Carlo simulations
            self.logger.info(f"Running Simulations for {self.home_team} vs {self.away_team} on {self.date}...")
            simulation_results, most_likely_outcome = self.monte_carlo_simulation(game_prediction_df, self.get_standard_deviation(df))

            # Analyze simulation results
            self.logger.info("Analyzing Simulation Results...")
            range_of_outcomes, standard_deviation, confidence_interval = self.analyze_simulation_results(simulation_results)

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
            explanation = self.analysis_explanation(range_of_outcomes, confidence_interval, most_likely_outcome, standard_deviation)
            combined_output = log_contents + "\n\n" + explanation

            # Visualize simulation results
            self.visualize_simulation_results(simulation_results, most_likely_outcome, combined_output)

        # After the loop, compare the simulated results to the actual results
        self.compare_simulated_to_actual(all_simulation_results, all_actual_results)

        # Evaluate the betting recommendation and expected value
        recommendation_accuracy, average_ev, actual_value = self.evaluate_and_recommend(all_simulation_results, historical_df)
        self.logger.info(f"Recommendation Accuracy: {recommendation_accuracy:.2f}%")
        self.logger.info(f"Average Expected Value: {average_ev:.2f}%")
        self.logger.info(f"Actual Results: ${actual_value:.2f}")


if __name__ == "__main__":
    predictor = NFLPredictor()
    predictor.main()
    # Call other methods of the predictor as needed
