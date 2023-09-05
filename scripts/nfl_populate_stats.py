from pymongo import MongoClient
from .constants import COLUMNS_TO_KEEP
from classes.config_manager import ConfigManager
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

config = ConfigManager()

# Get individual components
base_dir = config.get_config('default', 'base_dir')
data_dir = base_dir + config.get_config('paths', 'data_dir')
model_dir = base_dir + config.get_config('paths', 'model_dir')
feature_columns = [col for col in COLUMNS_TO_KEEP if col != 'scoring_differential']
LOADED_MODEL = joblib.load(os.path.join(model_dir, 'trained_nfl_model.pkl'))
MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "nfl_db"
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

# Get the current date
today = datetime.today().date()

# Calculate the date for two years ago
two_years_ago = (today - pd.Timedelta(days=730)).strftime('%Y-%m-%d')


# Fetching data from MongoDB (modify to fetch data based on date)
def fetch_data_from_mongodb(collection_name):
    cursor = db[collection_name].find()
    df = pd.DataFrame(list(cursor))
    return df


def flatten_and_merge_data(df):
    """Flatten the nested MongoDB data."""
    # Flatten each category and store in a list
    dataframes = []
    for column in df.columns:
        if isinstance(df[column][0], dict):
            flattened_df = pd.json_normalize(df[column])
            flattened_df.columns = [
                f"{column}_{subcolumn}" for subcolumn in flattened_df.columns
            ]
            dataframes.append(flattened_df)
        else:
            dataframes.append(df[[column]])

    # Merge flattened dataframes
    merged_df = pd.concat(dataframes, axis=1)

    return merged_df


# TODO: 3. Implement data validation checks to ensure the data fetched from the database meets the expected format and structure.
# TODO: 4. Consider adding functionality to handle different data types more efficiently during the flattening process.


def load_and_process_data():
    """Load data from MongoDB and process it."""
    df = fetch_data_from_mongodb("games")
    teams_df = fetch_data_from_mongodb("teams")
    processed_df = flatten_and_merge_data(df)
    processed_teams_df = flatten_and_merge_data(teams_df)


    # Convert columns with list values to string
    for col in processed_df.columns:
        if processed_df[col].apply(type).eq(list).any():
            processed_df[col] = processed_df[col].astype(str)

    # Convert necessary columns to numeric types
    numeric_columns = ['summary_home.points', 'summary_away.points']
    processed_df[numeric_columns] = processed_df[numeric_columns].apply(
        pd.to_numeric, errors='coerce'
    )

    # Drop rows with missing values in eithers
    # 'summary_home_points' or 'summary_away_points'
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
        return processed_df, processed_teams_df

# TODO: 5. Implement more robust error handling and data validation to ensure the 'scoring_differential' is computed correctly.
# TODO: 6. Consider adding a logging mechanism to track the data processing steps and potential issues.


def time_to_minutes(time_str):
    """Convert time string 'MM:SS' to minutes as a float."""
    minutes, seconds = map(int, time_str.split(':'))
    return minutes + seconds / 60

# TODO: 7. Consider adding error handling to manage incorrect time formats.


processed_df, processed_teams_df = load_and_process_data()
processed_teams_df = processed_teams_df.drop_duplicates(subset='id')

# Convert time strings to minutes (apply this to the relevant columns)
processed_df['statistics_home.summary.possession_time'] = processed_df[
    'statistics_home.summary.possession_time'
].apply(time_to_minutes)
processed_df['statistics_away.summary.possession_time'] = processed_df[
    'statistics_away.summary.possession_time'
].apply(time_to_minutes)

processed_df['game_date'] = pd.to_datetime(processed_df['scheduled'])

today = datetime.today()

# Calculate days since the game was played
processed_df['game_date'] = processed_df['game_date'].dt.tz_localize(None)
processed_df['days_since_game'] = (today - processed_df['game_date']).dt.days

# Drop all games played more than 104 weeks (728 days) ago
max_days_since_game = 104 * 7
df = processed_df[processed_df['days_since_game'] <= max_days_since_game]


# Exponential decay function
def decay_weight(days):
    if days <= 42:  # Last 6 weeks
        lambda_val = 0.005
    elif days <= 126:  # Last 18 weeks
        lambda_val = 0.01
    elif days <= 280:  # Last 40 weeks
        lambda_val = 0.02
    elif days <= 392:  # Last 56 weeks
        lambda_val = 0.03
    else:  # Beyond 56 weeks, up to 104 weeks
        lambda_val = 0.04

    return np.exp(-lambda_val * days)


df['weight'] = df['days_since_game'].apply(decay_weight)

# List of metrics to aggregate
metrics = feature_columns

# Split Metrics List
metrics_home = [metric for metric in metrics if metric.startswith('statistics_home')]
metrics_away = [metric for metric in metrics if metric.startswith('statistics_away')]

# If it's a different type of model, you might need to use a different method to get the feature importances
feature_importances = LOADED_MODEL.feature_importances_

# Create Weights Dictionary
# Assuming feature_importances is a list of weights corresponding to the metrics in feature_columns
weights = dict(zip(feature_columns, feature_importances))

# Separate Home and Away Data
df_home = df[['summary_home.id', 'summary_home.name', 'game_date', 'days_since_game', 'weight'] + metrics_home]
df_away = df[['summary_away.id', 'summary_away.name', 'game_date', 'days_since_game', 'weight'] + metrics_away]

# Compute Power Ranks
def compute_power_rank(row, metrics):
    power_rank = 0
    for metric in metrics:
        weight = weights.get(metric)
        power_rank += weight * row[metric]
    return power_rank


# Correcting the SettingWithCopyWarning
df_home.loc[:, 'power_rank'] = df_home.apply(lambda row: compute_power_rank(row, metrics_home), axis=1)
df_away.loc[:, 'power_rank'] = df_away.apply(lambda row: compute_power_rank(row, metrics_away), axis=1)

# Remove the 'summary_home.' and 'statistics_home.' prefixes from the column names in df_home
df_home.columns = df_home.columns.str.replace('summary_home.', '').str.replace('statistics_home.', '')

# Remove the 'summary_away.' and 'statistics_away.' prefixes from the column names in df_away
df_away.columns = df_away.columns.str.replace('summary_away.', '').str.replace('statistics_away.', '')

# Concatenate the dataframes to create a single dataframe with all team performances
df = pd.concat([df_home, df_away], ignore_index=True)

print("Step 8: Concatenating DataFrames and Normalizing Power Rank Values")
print("DF Head after concatenation and normalization:")
print(df)

# Determine the week number based on the Tuesday-to-Monday window
df['week_number'] = (df['game_date'] - pd.Timedelta(days=1)).dt.isocalendar().week


# Normalize the power_rank values within each week
def normalize_within_week(group):
    min_rank = group['power_rank'].min()
    max_rank = group['power_rank'].max()
    group['power_rank'] = (group['power_rank'] - min_rank) / (max_rank - min_rank)  # Assign normalized values to power_rank column
    return group


df = df.groupby(['week_number']).apply(normalize_within_week)

# Sort the dataframe by name and power_rank
df = df.sort_values(by=['power_rank'])

# Remove the 'statistics_home.' and 'statistics_away.' prefixes from the metrics list
cleaned_metrics = [metric.replace('statistics_home.', '').replace('statistics_away.', '') for metric in metrics]

# Create a list for columns_for_power_rank_table with the cleaned metrics list
columns_for_power_rank_table = ['id', 'name', 'game_date', 'days_since_game', 'weight', 'power_rank'] + cleaned_metrics
power_rank_df = df[columns_for_power_rank_table]

# Drop the power_rank collection if it exists
db.power_rank.drop()

# Save to a new collection in the database
power_rank_df.reset_index(inplace=True, drop=True)
db.power_rank.insert_many(power_rank_df.to_dict('records'))

df['weighted_power_rank'] = df['power_rank'] * df['weight']

# Weighted aggregation of power rank
aggregated_power_rank = df.groupby('id')['weighted_power_rank'].sum() / df.groupby('id')['weight'].sum()

# Normalize the aggregated power rank values
min_aggregated_power_rank = aggregated_power_rank.min()
max_aggregated_power_rank = aggregated_power_rank.max()

normalized_aggregated_power_rank = (aggregated_power_rank - min_aggregated_power_rank) / (max_aggregated_power_rank - min_aggregated_power_rank)

# Weighted aggregation of power rank
aggregated_power_rank = df.groupby('id')['weighted_power_rank'].sum() / df.groupby('id')['weight'].sum()

# Normalize the aggregated power rank values
min_aggregated_power_rank = aggregated_power_rank.min()
max_aggregated_power_rank = aggregated_power_rank.max()

normalized_aggregated_power_rank = (aggregated_power_rank - min_aggregated_power_rank) / (max_aggregated_power_rank - min_aggregated_power_rank)

# Weighted aggregation of power rank
aggregated_power_rank = df.groupby('id')['weighted_power_rank'].sum() / df.groupby('id')['weight'].sum()

# Normalize the aggregated power rank values
min_aggregated_power_rank = aggregated_power_rank.min()
max_aggregated_power_rank = aggregated_power_rank.max()
normalized_aggregated_power_rank = (aggregated_power_rank - min_aggregated_power_rank) / (max_aggregated_power_rank - min_aggregated_power_rank)

# Weighted aggregation for other metrics
aggregated_data = {}
for metric in cleaned_metrics:
    weighted_metric_name = metric + '_weighted'
    df[weighted_metric_name] = df[metric] * df['weight']
    aggregated_data[metric] = df.groupby('id')[weighted_metric_name].sum() / df.groupby('id')['weight'].sum()

# Add aggregated power rank and normalized power rank to the aggregated data
df_aggregated_power_rank = pd.DataFrame({'power_rank': aggregated_power_rank, 'normalized_power_rank': normalized_aggregated_power_rank})
aggregated_df = pd.DataFrame(aggregated_data).merge(df_aggregated_power_rank, left_index=True, right_index=True)

# Compute standard deviation for each metric across all teams
stddev_data = {}
for metric in cleaned_metrics:
    stddev_data[metric + '_stddev'] = df.groupby('id')[metric].std()

# Convert the standard deviation data to a DataFrame
df_stddev = pd.DataFrame(stddev_data)

# Merge the standard deviation DataFrame with the aggregated_df DataFrame
aggregated_df = aggregated_df.reset_index()
processed_teams_df['id'] = processed_teams_df['id'].astype(str)
aggregated_df['id'] = aggregated_df['id'].astype(str)
aggregated_df = aggregated_df.merge(df_stddev, left_on='id', right_index=True)
print(aggregated_df.head())
print(processed_teams_df.head())
aggregated_df = aggregated_df.merge(processed_teams_df, left_on='id', right_on='id', how='left')

# Drop the team_aggregated_metrics collection if it exists
db.team_aggregated_metrics.drop()

# Save to a new collection in the database
aggregated_df.reset_index(inplace=True)
db.team_aggregated_metrics.insert_many(aggregated_df.to_dict('records'))

print('Aggregated metrics saved to team_aggregated_metrics collection.')
