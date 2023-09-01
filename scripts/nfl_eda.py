import sqlite3
import sweetviz as sv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import yaml

# Load the configuration
with open(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'), 'r') as stream:
    config = yaml.safe_load(stream)

# Define base for data
BASE_DIR = os.path.expandvars(config['default']['base_dir'])
DATA_DIR = os.path.join(BASE_DIR, 'data')
DATABASE_PATH = os.path.join(DATA_DIR, config['database']['database_name'])
TARGET_VARIABLE = 'scoring_differential'


def load_and_process_data():
    """
    Load data from the SQLite database and process it.

    Returns:
    DataFrame: The loaded data.
    """
    with sqlite3.connect(DATABASE_PATH) as db_conn:
        # Get the current date
        today = datetime.today().date()

        # Calculate the date for two years ago
        two_years_ago = (today - pd.Timedelta(days=730)).strftime('%Y-%m-%d')

        # Fetch the consolidated data from the last 2 years
        query = f"SELECT * FROM consolidated AS c LEFT JOIN consolidated_advanced a ON c.team_id = a.team_id AND DATE(c.game_date) = DATE(a.game_date) WHERE c.game_date > '{two_years_ago}'"
        df = pd.read_sql_query(query, db_conn)

        # Convert columns to numeric where applicable
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')

        # List of metrics to aggregate
        metrics = [
            'rating', 'rush_plays', 'avg_pass_yards', 'attempts',
            'tackles_made', 'redzone_successes', 'turnovers',
            'redzone_attempts', 'scoring_differential', 'sack_rate',
            'play_diversity_ratio', 'turnover_margin',
            'pass_success_rate', 'rush_success_rate'
            ]
        df = df[metrics]

    return df


def plot_numerical(df, column):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.show()


def plot_box(df, column):
    plt.figure(figsize=(10, 6))
    sns.boxplot(df[column])
    plt.title(f"Boxplot of {column}")
    plt.show()


def print_correlations(correlations, n):
    print(f"Correlations with {TARGET_VARIABLE}:")
    for feature, correlation in correlations[:n].items():
        print(f"{feature}: {correlation:.4f}")


def print_missing_values(df):
    missing_values = df.isnull().sum()
    missing_values_percent = 100 * df.isnull().sum() / len(df)
    missing_df = pd.concat([missing_values, missing_values_percent], axis=1)
    missing_df = missing_df.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'}
    )
    missing_df = missing_df[missing_df.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False
    )
    print(missing_df)


def main():
    df = load_and_process_data()

    # Generate the report
    report = sv.analyze(df)
    report.show_html(os.path.join(DATA_DIR, 'nfl_eda_report.html'))


if __name__ == "__main__":
    main()
