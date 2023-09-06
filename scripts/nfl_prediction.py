from classes.config_manager import ConfigManager
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


# Configuration and Database Connection
config = ConfigManager()
base_dir = config.get_config('default', 'base_dir')
data_dir = base_dir + config.get_config('paths', 'data_dir')
model_dir = base_dir + config.get_config('paths', 'model_dir')
MONGO_URI = config.get_config('database', 'mongo_uri')  # Moved to config file
DATABASE_NAME = config.get_config('database', 'database_name')  # Moved to config file
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

# Constants
# Define the feature columns by removing the target variable from COLUMNS_TO_KEEP
feature_columns = [col for col in COLUMNS_TO_KEEP if col != 'scoring_differential']

# Loading the pre-trained model and the data scaler
try:
    LOADED_MODEL = joblib.load(os.path.join(model_dir, 'trained_nfl_model.pkl'))
    LOADED_SCALER = joblib.load(os.path.join(model_dir, 'data_scaler.pkl'))
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    # Exit the script or handle the error appropriately


# ENCODER = joblib.load(os.path.join(model_dir, 'data_encoder.pkl'))
# ENCODED_COLUMNS_TRAIN = joblib.load(os.path.join(model_dir, 'encoded_columns.pkl'))
def time_to_minutes(time_str):
    """Convert time string 'MM:SS' to minutes as a float."""
    minutes, seconds = map(int, time_str.split(':'))
    return minutes + seconds / 60


def fetch_data_from_mongodb(collection_name):
    """Fetch data from a MongoDB collection."""
    try:
        cursor = db[collection_name].find()
        df = pd.DataFrame(list(cursor))
        return df
    except Exception as e:
        print(f"Error fetching data from MongoDB: {e}")
        # Handle the error appropriately (e.g., return an empty DataFrame or exit the script)
        return pd.DataFrame()


def flatten_and_merge_data(df):
    """Flatten the nested MongoDB data."""
    try:
        # Flatten each category and store in a list
        dataframes = []
        for column in df.columns:
            if isinstance(df[column][0], dict):
                flattened_df = pd.json_normalize(df[column])
                flattened_df.columns = [
                    f"{column}_{subcolumn}" for subcolumn in flattened_df.columns
                ]
                dataframes.append(flattened_df)

        # Merge flattened dataframes
        merged_df = pd.concat(dataframes, axis=1)
        return merged_df
    except Exception as e:
        print(f"Error flattening and merging data: {e}")
        # Handle the error appropriately (e.g., return the original DataFrame or an empty DataFrame)
        return pd.DataFrame()

# TODO: 3. Implement data validation checks to ensure the data fetched from the database meets the expected format and structure.
# TODO: 4. Consider adding functionality to handle different data types more efficiently during the flattening process.


def load_and_process_data():
    """Load data from MongoDB and process it."""
    df = fetch_data_from_mongodb("games")
    processed_df = flatten_and_merge_data(df)

    # Convert columns with list values to string
    for col in processed_df.columns:
        if processed_df[col].apply(type).eq(list).any():
            processed_df[col] = processed_df[col].astype(str)

    # Convert necessary columns to numeric types
    numeric_columns = ['summary_home.points', 'summary_away.points']
    processed_df[numeric_columns] = processed_df[numeric_columns].apply(
        pd.to_numeric, errors='coerce'
    )

    # Drop rows with missing values in either 'summary_home_points' or 'summary_away_points'
    processed_df.dropna(subset=numeric_columns, inplace=True)

    # Check if necessary columns are present and have numeric data types
    if all(col in processed_df.columns and pd.api.types.is_numeric_dtype(
        processed_df[col]
    ) for col in numeric_columns):
        processed_df['scoring_differential'] = processed_df[
            'summary_home.points'
        ] - processed_df[
            'summary_away.points'
        ]
        print("Computed 'scoring_differential' successfully.")
    else:
        print(
            "Unable to compute due to unsuitable data types."
        )

    # Drop games if 'scoring_differential' key does not exist
    if 'scoring_differential' not in processed_df.columns:
        print("'scoring_differential' key does not exist. Dropping games.")
        return pd.DataFrame()  # Return an empty dataframe
    else:
        return processed_df

# TODO: 5. Implement more robust error handling and data validation to ensure the 'scoring_differential' is computed correctly.
# TODO: 6. Consider adding a logging mechanism to track the data processing steps and potential issues.


def time_to_minutes(time_str):
    """Convert time string 'MM:SS' to minutes as a float."""
    print(time_str)
    try:
        minutes, seconds = map(int, time_str.split(':'))
        return minutes + seconds / 60
    except ValueError:
        print(f"Invalid time format: {time_str}. Unable to convert to minutes.")
        return None  # or return a default value

# TODO: 7. Consider adding error handling to manage incorrect time formats.


def monte_carlo_simulation(df, num_simulations=1000):
    print("Starting Monte Carlo Simulation...")
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

            # Filter columns and scale numeric features
            sampled_df = sampled_df[feature_columns]
            sampled_df[feature_columns] = LOADED_SCALER.transform(sampled_df[feature_columns])

            prediction = LOADED_MODEL.predict(sampled_df)
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


def predict_scoring_differential():
    # Fetch the teams collection to create a mapping of team_alias to team_id
    teams_df = fetch_data_from_mongodb('teams')  # Adjusted to use your new method
    team_alias_to_id = dict(zip(teams_df['alias'], teams_df['id']))

    # Display available team aliases in a 6 by X grid
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
    df = fetch_data_from_mongodb('team_aggregated_metrics')  # Adjusted to use your new method

    home_team_data = df[df['alias'] == home_team_alias]
    away_team_data = df[df['alias'] == away_team_alias]

    def rename_columns(data, prefix):
        columns_to_rename = [
            'summary.possession_time',
            'passing.totals.rating',
            'efficiency.redzone.successes',
            'summary.turnovers',
            'summary.avg_gain',
            'rushing.totals.redzone_attempts',
            'efficiency.goaltogo.successes',
            'defense.totals.sacks',
            'efficiency.thirddown.successes',
            'defense.totals.qb_hits'
        ]

        rename_dict = {col: f"{prefix}{col}" for col in columns_to_rename}
        return data.rename(columns=rename_dict)

    home_team_data = rename_columns(home_team_data.reset_index(drop=True), "statistics_home.")
    away_team_data = rename_columns(away_team_data.reset_index(drop=True), "statistics_away.")

    # Step 1: Identify and rename standard deviation columns
    home_team_data_stddev = home_team_data.filter(like='stddev').rename(columns=lambda x: "statistics_home." + x)
    away_team_data_stddev = away_team_data.filter(like='stddev').rename(columns=lambda x: "statistics_away." + x)

    # Step 2: Merge data including the standard deviation columns
    merged_data = pd.concat([home_team_data, home_team_data_stddev, away_team_data, away_team_data_stddev], axis=1)

    # Step 3: Modify feature_columns list to include standard deviation columns
    feature_columns = [col for col in COLUMNS_TO_KEEP if col != 'scoring_differential']
    feature_columns += [col for col in merged_data.columns if 'stddev' in col]

    # Step 4: Filter merged_data using the modified feature_columns list
    merged_data = merged_data[feature_columns]

    # Adding missing columns with default values
    for column in COLUMNS_TO_KEEP:
        if column not in merged_data.columns:
            merged_data[column] = 0  # or another appropriate default value

    # Run Monte Carlo simulations
    print("\nRunning Simulations...")
    simulation_results, most_likely_outcome = monte_carlo_simulation(merged_data)

    sorted_results = sorted(simulation_results)

    # Filter out the 10% of results
    sorted_results = sorted(simulation_results)
    lower_bound = int(0.1 * len(sorted_results))
    upper_bound = int(1 * len(sorted_results))
    filtered_results = sorted_results[lower_bound:upper_bound]

    # Analyze simulation results
    print("Analyzing Simulation Results...")
    range_of_outcomes = (min(filtered_results) + 2.5, max(filtered_results) + 2.5)
    standard_deviation = np.std(filtered_results)
    # most_likely_outcome = np.mean(filtered_results) + 2.5

    # Inside the predict_scoring_differential function, uncomment the visualization section and adjust as follows:
    plt.figure(figsize=(10, 6))
    sns.histplot(simulation_results, bins=50, kde=True, color='blue')
    plt.axvline(most_likely_outcome, color='red', linestyle='--')
    plt.title("Monte Carlo Simulation Results")
    plt.xlabel("Scoring Differential")
    plt.ylabel("Density")
    plt.grid()
    plt.show()

    # Calculate the highest occurring value on the histogram
    # max_value_index = np.argmax(hist_values)
    # highest_occurring_value = ((bins[max_value_index] + bins[max_value_index + 1]) / 2)

    # Round the highest occurring value to the nearest half point (0.5 increments)
    # rounded_highest_occurring_value = round(highest_occurring_value) + 2.5
    rounded_most_likely_outcome = ((round(most_likely_outcome * 2) / 2)  + 2.5) * (-1)
    # Add a leading + sign for positive values
    if rounded_most_likely_outcome > 0:
        formatted_most_likely_outcome = f"+{rounded_most_likely_outcome:.2f}"
    else:
        formatted_most_likely_outcome = f"{rounded_most_likely_outcome:.2f}"

    # User-friendly output
    print("\nPrediction Results:")
    print(f"Based on our model, the expected scoring differential for {home_team_alias} against {away_team_alias} is between {range_of_outcomes[0]:.2f} and {range_of_outcomes[1]:.2f} points.")
    print(f"The most likely scoring differential is approximately {most_likely_outcome:.2f} points.")
    print(f"The standard deviation of the scoring differentials is approximately {standard_deviation:.2f} points.\n")
    print(f"The projected spread on this game should be {home_team_alias} {formatted_most_likely_outcome}.\n")


# Usage
predict_scoring_differential()
