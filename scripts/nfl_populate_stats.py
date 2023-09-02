from classes.config_manager import ConfigManager
from classes.database_handler import DatabaseHandler
import os
import pandas as pd
import numpy as np
from datetime import datetime

config = ConfigManager()

# Get individual components
base_dir = config.get_config('default', 'base_dir')
data_dir = base_dir + config.get_config('paths', 'data_dir')
database_name = config.get_config('database', 'database_name')

# Construct the full path
db_path = os.path.join(data_dir, database_name)

# Establish database connection
db_handler = DatabaseHandler(db_path)

# Connect to the SQLite database
conn = db_handler.connect()

# Get the current date
today = datetime.today().date()

# Calculate the date for two years ago
two_years_ago = (today - pd.Timedelta(days=730)).strftime('%Y-%m-%d')

# Fetch the consolidated data from the last 2 years
query = f"SELECT * FROM consolidated AS c LEFT JOIN consolidated_advanced a ON c.team_id = a.team_id AND DATE(c.game_date) = DATE(a.game_date) WHERE c.game_date > '{two_years_ago}'"
df = pd.read_sql_query(query, conn)

df['game_date'] = pd.to_datetime(df['game_date'])

today = datetime.today()

# Calculate days since the game was played
df['game_date'] = df['game_date'].dt.tz_localize(None)
df['days_since_game'] = (today - df['game_date']).dt.days


# Exponential decay function
def decay_weight(days, lambda_val=0.03):
    return np.exp(-lambda_val * days)


df['weight'] = df['days_since_game'].apply(decay_weight)

# List of metrics to aggregate
metrics = [
            'rating', 'rush_plays', 'avg_pass_yards', 'attempts',
            'tackles_made', 'redzone_successes', 'turnovers',
            'redzone_attempts', 'scoring_differential', 'sack_rate',
            'play_diversity_ratio', 'turnover_margin',
            'pass_success_rate', 'rush_success_rate'
            ]

# Drop columns not in metrics
df = df[['team_id', 'game_date', 'days_since_game', 'weight'] + metrics]

# Assign weights based on provided values
weights = {
    'rating': 0.154203,
    'rush_plays': 0.066077,
    'avg_pass_yards': 0.095021,
    'attempts': -0.026576,
    'tackles_made': -0.033095,
    'redzone_successes': 0.039976,
    'turnovers': 0.009306,
    'redzone_attempts': 0.021393,
    'scoring_differential': 0,
    'sack_rate': 0.063763,
    'play_diversity_ratio': 0.063636,
    'turnover_margin': 0.267512,
    'pass_success_rate': 0.037030,
    'rush_success_rate': -0.062814
}


def compute_power_rank(row):
    power_rank = 0
    for metric in metrics:
        weight = weights.get(metric)
        power_rank += weight * row[metric]
    return power_rank


df['power_rank'] = df.apply(compute_power_rank, axis=1)

# Determine the week number based on the Tuesday-to-Monday window
df['week_number'] = (df['game_date'] - pd.Timedelta(days=1)).dt.isocalendar().week


# Normalize the power_rank values within each week
def normalize_within_week(group):
    min_rank = group['power_rank'].min()
    max_rank = group['power_rank'].max()
    group['power_rank'] = (group['power_rank'] - min_rank) / (max_rank - min_rank)  # Assign normalized values to power_rank column
    return group


df = df.groupby(['week_number']).apply(normalize_within_week)

# Sort the dataframe by team_id and power_rank
df = df.sort_values(by=['team_id', 'power_rank'])

# Select the desired columns for the power_rank table
columns_for_power_rank_table = ['team_id', 'game_date', 'days_since_game', 'weight', 'power_rank'] + metrics
power_rank_df = df[columns_for_power_rank_table]

# Drop the power_rank table if it exists
conn.execute("DROP TABLE IF EXISTS power_rank")

# Create the new power_rank table
power_rank_df.to_sql('power_rank', conn, index=False)

# Drop the 3 highest and 3 lowest power_rank games for each team
df = df.groupby('team_id').apply(lambda x: x.iloc[3:-3]).reset_index(drop=True)

df['weighted_power_rank'] = df['power_rank'] * df['weight']

# Weighted aggregation of power rank
aggregated_power_rank = df.groupby('team_id')['weighted_power_rank'].sum() / df.groupby('team_id')['weight'].sum()

# Normalize the aggregated power rank values
min_aggregated_power_rank = aggregated_power_rank.min()
max_aggregated_power_rank = aggregated_power_rank.max()

normalized_aggregated_power_rank = (aggregated_power_rank - min_aggregated_power_rank) / (max_aggregated_power_rank - min_aggregated_power_rank)

# Weighted aggregation for other metrics
aggregated_data = {}
for metric in metrics:
    weighted_metric_name = metric + '_weighted'
    df[weighted_metric_name] = df[metric] * df['weight']
    aggregated_data[metric] = df.groupby('team_id')[weighted_metric_name].sum() / df.groupby('team_id')['weight'].sum()

# Add aggregated power rank and normalized power rank to the aggregated data
df_aggregated_power_rank = pd.DataFrame({'power_rank': aggregated_power_rank, 'normalized_power_rank': normalized_aggregated_power_rank})
aggregated_df = pd.DataFrame(aggregated_data).merge(df_aggregated_power_rank, left_index=True, right_index=True)

# Fetch alias from the teams table
teams_df = pd.read_sql_query('SELECT team_id, alias FROM teams', conn)
aggregated_df = aggregated_df.reset_index().merge(teams_df, on='team_id', how='left')

# Re-order columns
cols = ['team_id', 'alias', 'normalized_power_rank'] + metrics
aggregated_df = aggregated_df[cols]

# Compute standard deviation for each metric across all teams
stddev_data = {}
for metric in metrics:
    stddev_data[metric + '_stddev'] = df.groupby('team_id')[metric].std()

# Convert the standard deviation data to a DataFrame
df_stddev = pd.DataFrame(stddev_data)

# Merge the standard deviation DataFrame with the aggregated_df DataFrame
aggregated_df = aggregated_df.merge(df_stddev, left_on='team_id', right_index=True)

# Drop the table if it exists
conn.execute("DROP TABLE IF EXISTS team_aggregated_metrics")

# Save to a new table in the database
aggregated_df.to_sql('team_aggregated_metrics', conn, index=False)

print('Aggregated metrics saved to team_aggregated_metrics table.')
