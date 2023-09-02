from classes.config_manager import ConfigManager
from classes.database_handler import DatabaseHandler
import os
import time
import random
import joblib
import pandas as pd
import numpy as np

config = ConfigManager()

# Get individual components
base_dir = config.get_config('default', 'base_dir')
data_dir = base_dir + config.get_config('paths', 'data_dir')
model_dir = base_dir + config.get_config('paths', 'model_dir')
database_name = config.get_config('database', 'database_name')

# Construct the full path
db_path = os.path.join(data_dir, database_name)

# Establish database connection
db_handler = DatabaseHandler(db_path)
# Constants
NUMERICAL_FEATURES = ['rating', 'rush_plays', 'avg_pass_yards', 'attempts', 'tackles_made', 'redzone_attempts', 'redzone_successes', 'turnovers', 'sack_rate', 'play_diversity_ratio', 'turnover_margin', 'pass_success_rate', 'rush_success_rate']
CATEGORICAL_FEATURES = ['opponent_team_id', 'home_or_away']
LOADED_MODEL = joblib.load(os.path.join(model_dir, 'trained_nfl_model.pkl'))
LOADED_SCALER = joblib.load(os.path.join(model_dir, 'data_scaler.pkl'))
ENCODER = joblib.load(os.path.join(model_dir, 'data_encoder.pkl'))
ENCODED_COLUMNS_TRAIN = joblib.load(os.path.join(model_dir, 'encoded_columns.pkl'))


def monte_carlo_simulation(df, num_simulations=10000):
    print("Starting Monte Carlo Simulation...")
    simulation_results = []

    # List of funny messages
    funny_messages = [
        "Simulating... Did you hear about the mathematician who’s afraid of negative numbers? He'll stop at nothing to avoid them!",
        "Crunching numbers... Why did the student do multiplication problems on the floor? The teacher told him not to use tables.",
        "Still working... Why was the equal sign so humble? Because she realized she wasn't less than or greater than anyone else!",
        "Hang tight... Why did the two fours skip lunch? Because they already eight!",
        "Almost there... Why did the student wear glasses in math class? To improve di-vision!"
    ]

    start_time = time.time()
    for _ in range(num_simulations):
        sampled_df = df.copy()
        for column in df.columns:
            if column + '_stddev' in df.columns:
                mean_value = df[column].iloc[0]
                stddev_value = df[column + '_stddev'].iloc[0]
                sampled_value = np.random.normal(mean_value, stddev_value)
                sampled_df[column] = sampled_value
        # Exclude 'scoring_differential' from the sampled DataFrame

        # print(f"Sampled Data at iteration {_}:\n", sampled_df)

        # Preprocess the sampled DataFrame (encoding, scaling, etc.)
        encoded_features = ENCODER.transform(sampled_df[CATEGORICAL_FEATURES])
        df_encoded = pd.DataFrame(encoded_features, columns=ENCODER.get_feature_names_out(CATEGORICAL_FEATURES))
        sampled_df = pd.concat([sampled_df.drop(columns=CATEGORICAL_FEATURES), df_encoded], axis=1)
        for col in ENCODED_COLUMNS_TRAIN:
            if col not in sampled_df.columns:
                sampled_df[col] = 0
        sampled_df = sampled_df[ENCODED_COLUMNS_TRAIN]

        # Exclude 'scoring_differential' from scaling
        features_to_scale = [feature for feature in NUMERICAL_FEATURES if feature != 'scoring_differential']
        sampled_df[features_to_scale] = LOADED_SCALER.transform(sampled_df[features_to_scale])

        sampled_df = sampled_df.drop(columns=['scoring_differential'])
        prediction = LOADED_MODEL.predict(sampled_df)
        simulation_results.append(prediction[0])

        # print(f"Prediction at iteration {_}: {prediction[0]}")

        # Check if 10 seconds have passed
        if time.time() - start_time > 10:
            print(random.choice(funny_messages))
            start_time = time.time()

    print("Monte Carlo Simulation Completed!")
    return simulation_results

def predict_scoring_differential():
    conn = db_handler.connect()

    # Fetch the teams table to create a mapping of team_alias to team_id
    teams_df = pd.read_sql_query("SELECT team_id, alias FROM teams", conn)
    team_alias_to_id = dict(zip(teams_df['alias'], teams_df['team_id']))

    # Display available team aliases in a 6 by X grid
    print("Available Team Aliases:")
    aliases = list(team_alias_to_id.keys())
    for i in range(0, len(aliases), 6):
        print("\t".join(aliases[i:i+6]))

    # Prompt user for input
    team_alias = input("Enter team alias: ")
    while team_alias not in team_alias_to_id:
        print("Invalid team alias. Please enter a valid team alias from the list above.")
        team_alias = input("Enter team alias: ")

    # Fetch data only for the selected team from the team_aggregated_metrics table
    query = f"SELECT * FROM team_aggregated_metrics WHERE alias = '{team_alias}'"
    df = pd.read_sql_query(query, conn)
    df = df.rename(columns={'normalized_power_rank': 'power_rank'})

    opponent_alias = input("Enter opponent team alias: ")
    while opponent_alias not in team_alias_to_id:
        print("Invalid opponent team alias. Please enter a valid team alias from the list above.")
        opponent_alias = input("Enter opponent team alias: ")

    home_or_away = input("Enter location (Home/Away): ")
    while home_or_away not in ['Home', 'Away']:
        print("Invalid location. Please enter 'Home' or 'Away'.")
        home_or_away = input("Enter location (Home/Away): ")

    # Create a DataFrame with user input
    input_data = {
        'team_id': [team_alias_to_id[team_alias]],
        'opponent_team_id': [team_alias_to_id[opponent_alias]],
        'home_or_away': [home_or_away]
    }
    input_df = pd.DataFrame(input_data)

    # Update the main DataFrame with the input data
    for col in input_df.columns:
        df[col] = input_df[col].iloc[0]

    # Run Monte Carlo simulations
    print("\nRunning Simulations...")
    simulation_results = monte_carlo_simulation(df)

    sorted_results = sorted(simulation_results)

    # Filter out the top and bottom 10% of results
    sorted_results = sorted(simulation_results)
    lower_bound = int(0.1 * len(sorted_results))
    upper_bound = int(0.9 * len(sorted_results))
    filtered_results = sorted_results[lower_bound:upper_bound]

    # Analyze simulation results
    print("Analyzing Simulation Results...")
    range_of_outcomes = (min(filtered_results), max(filtered_results))
    most_likely_outcome = np.mean(filtered_results)
    print(f"Range of Score Differentials: {range_of_outcomes}")
    print(f"Most Likely Score Differential: {most_likely_outcome}")

    # User-friendly output
    print("\nPrediction Results:")
    print(f"Based on our model, the expected scoring differential for {team_alias} against {opponent_alias} is between {range_of_outcomes[0]:.2f} and {range_of_outcomes[1]:.2f} points.")
    print(f"The most likely scoring differential is approximately {most_likely_outcome:.2f} points.\n")


# Usage
predict_scoring_differential()
