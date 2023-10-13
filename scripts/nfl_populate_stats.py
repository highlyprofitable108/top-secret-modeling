from datetime import datetime
from pymongo import MongoClient
import scripts.constants
import plotly.express as px
import pandas as pd
import numpy as np
import os
import joblib
import logging
from importlib import reload
from classes.config_manager import ConfigManager
from classes.data_processing import DataProcessing
from classes.database_operations import DatabaseOperations
import scripts.constants

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize ConfigManager, DatabaseOperations, and DataProcessing
config = ConfigManager()
db_operations = DatabaseOperations()
data_processing = DataProcessing()

# Fetch configurations using ConfigManager
data_dir = config.get_config('paths', 'data_dir')
model_dir = config.get_config('paths', 'model_dir')
database_name = config.get_config('database', 'database_name')


class StatsCalculator:
    def __init__(self, date=None):
        # Initialize ConfigManager, DatabaseOperations, and DataProcessing
        self.config = ConfigManager()
        self.database_operations = DatabaseOperations()
        self.data_processing = DataProcessing()
        self.constants = scripts.constants.COLUMNS_TO_KEEP

        # Fetch configurations using ConfigManager
        self.data_dir = self.config.get_config('paths', 'data_dir')
        self.model_dir = self.config.get_config('paths', 'model_dir')
        self.static_dir = self.config.get_config('paths', 'static_dir')
        self.template_dir = self.config.get_config('paths', 'template_dir')
        self.database_name = self.config.get_config('database', 'database_name')
        self.feature_columns = [col for col in self.constants]
        self.LOADED_MODEL = joblib.load(os.path.join(self.model_dir, 'trained_nfl_model.pkl'))

        # Convert the string date to a datetime object
        # self.date_obj = datetime.strptime(self.date, '%Y-%m-%d')

    def load_and_process_data(self):
        """
        This function handles the loading and initial processing of data.
        :param database_operations: An instance of DatabaseOperations class
        :param data_processing: An instance of DataProcessing class
        :return: Two dataframes containing the loaded and processed data
        """
        try:
            # Step 1: Fetch data from the database
            games_df = self.database_operations.fetch_data_from_mongodb("pre_game_data")

            # Step 2: Perform initial data processing (like flattening the data)
            processed_games_df = data_processing.flatten_and_merge_data(games_df)

            # Convert columns with list values to string
            for col in processed_games_df.columns:
                if processed_games_df[col].apply(type).eq(list).any():
                    processed_games_df[col] = processed_games_df[col].astype(str)

            # Additional data processing steps (as per the original script)
            # processed_games_df = data_processing.calculate_scoring_differential(processed_games_df)

            # Drop games if 'scoring_differential' key does not exist
            if 'odds_spread' not in processed_games_df.columns:
                print("'odds_spread' key does not exist. Dropping games.")
                return pd.DataFrame()

            return processed_games_df
        except Exception as e:
            print(f"Error in load_and_process_data function: {e}")
            return None, None

    def transform_data(self, processed_df, feature_columns):
        """
        This function handles further data transformation.
        :param processed_df: A dataframe containing processed games data
        :param feature_columns: A list of feature columns to be used in the model
        :return: A tuple containing two dataframes with further transformed data
        """
        try:
            processed_df = self.data_processing.handle_null_values(processed_df)

            # Split metrics list into home and away metrics
            metrics_home = [metric for metric in feature_columns if metric.startswith('ranks_home')]
            metrics_away = [metric for metric in feature_columns if metric.startswith('ranks_away')]

            # Separate home and away data
            df_home = processed_df[['home_id', 'home_name', 'ranks_home_update_date'] + metrics_home]
            df_away = processed_df[['away_id', 'away_name', 'ranks_away_update_date'] + metrics_away]
            df_home = df_home.rename(columns={'ranks_home_update_date': 'update_date'})
            df_away = df_away.rename(columns={'ranks_away_update_date': 'update_date'})
            return df_home, df_away
        except Exception as e:
            print(f"Error in transform_data function: {e}")
            return None, None

    def calculate_power_rank(self, df_home, df_away, metrics_home, metrics_away, weights):
        """
        This function calculates the power rank for each team based on the transformed data.
        :param df_home: A dataframe containing transformed home data
        :param df_away: A dataframe containing transformed away data
        :param metrics_home: A list of metrics for home teams
        :param metrics_away: A list of metrics for away teams
        :param weights: A dictionary containing weights for each metric
        :return: A dataframe containing the power rank and other details for each team, or an empty dataframe if an error occurs
        """
        try:
            # Define a nested function to compute power rank for a row
            def compute_power_rank(row, metrics):
                power_rank = 0
                for metric in metrics:
                    weight = weights.get(metric)

                    # Check if weight is NaN (missing) and replace it with 0
                    if pd.isna(weight):
                        weight = 0  # Replace NaN with 0

                    metric_value = row[metric]

                    # Check if metric_value is NaN (missing) and replace it with a default value (e.g., 0)
                    if pd.isna(metric_value):
                        metric_value = 0  # Replace NaN with 0 or any other appropriate default value

                    power_rank += weight * metric_value
                return power_rank

            # Calculate power rank for home and away data
            df_home.loc[:, 'power_rank'] = df_home.apply(lambda row: compute_power_rank(row, metrics_home), axis=1)
            df_away.loc[:, 'power_rank'] = df_away.apply(lambda row: compute_power_rank(row, metrics_away), axis=1)

            # Remove prefixes from column names
            df_home.columns = df_home.columns.str.replace('ranks_home_', '')
            df_away.columns = df_away.columns.str.replace('ranks_away_', '')
            df_home.columns = df_home.columns.str.replace('home_', '')
            df_away.columns = df_away.columns.str.replace('away_', '')

            # Concatenate home and away dataframes to create a single dataframe
            df = pd.concat([df_home, df_away], ignore_index=True)

            # Determine the week number based on the Tuesday-to-Monday window
            df['week_number'] = (df['update_date'] - pd.Timedelta(days=1)).dt.isocalendar().week

            return df
        except Exception as e:
            print(f"Error in calculate_power_rank function: {e}")
            return pd.DataFrame()  # Return an empty dataframe if an error occurs

    def normalize_data(self, df):
        """
        This function aggregates and normalizes the data based on various metrics and saves the results to a MongoDB collection.
        :param df: A dataframe containing the power rank and other details for each team
        :param cleaned_metrics: A list of cleaned metrics (without prefixes)
        :param database_operations: An instance of the DatabaseOperations class
        :return: A dataframe containing aggregated and normalized data
        """
        try:
            # Normalize the power_rank values within each week
            df = df[df['name'].isin(['NFC', 'AFC']) == False].copy()

            def normalize_within_week(group):
                min_rank = group['power_rank'].min()
                max_rank = group['power_rank'].max()
                group['normalized_power_rank'] = (group['power_rank'] - min_rank) / (max_rank - min_rank)
                return group

            df = df.groupby(['week_number']).apply(normalize_within_week)

            # Sort the dataframe by update_date (descending) and then by power_rank (ascending)
            df = df.sort_values(by=['update_date', 'normalized_power_rank'], ascending=[False, False])

            # Rearrange columns to place 'normalized_power_rank' and 'power_rank' after 'update_date'
            columns = df.columns.tolist()
            columns.remove('power_rank')
            columns.remove('normalized_power_rank')
            idx = columns.index('update_date')
            columns = columns[:idx+1] + ['normalized_power_rank', 'power_rank'] + columns[idx+1:]
            df = df[columns]

            return df  # The final aggregated and normalized dataframe
        except Exception as e:
            print(f"Error in normalize_data function: {e}")
            return None

    def insert_aggregated_data_into_database(self, aggregated_df,):
        """
        Inserts the aggregated and normalized data into a MongoDB collection.
        :param aggregated_df: A dataframe containing aggregated and normalized data
        :param database_operations: An instance of the DatabaseOperations class
        """
        try:
            # Drop the collection if it exists
            if 'weekly_ranks' in self.database_operations.db.list_collection_names():
                self.database_operations.db.weekly_ranks.drop()

            # Insert the aggregated data into the collection
            aggregated_df.reset_index(inplace=True, drop=True)
            self.database_operations.insert_data_into_mongodb('weekly_ranks', aggregated_df.to_dict('records'))

            # Saving the report to a CSV file
            aggregated_df = aggregated_df.sort_values(by='normalized_power_rank', ascending=False)
            power_ranks_report_path = os.path.join(self.static_dir, 'power_ranks.csv')
            aggregated_df.to_csv(power_ranks_report_path)
            print("Aggregated data inserted into MongoDB successfully.")
        except Exception as e:
            print(f"Error inserting aggregated data into MongoDB: {e}")

    def fetch_team_ranking_metrics(self):
        """Fetch weekly_ranks from MongoDB and return as a DataFrame."""
        client = MongoClient()
        db = client[self.database_name]
        collection = db['weekly_ranks']
        df = pd.DataFrame(list(collection.find()))
        return df

    def generate_interactive_htmls(self, df, date=None):
        """Generate an interactive HTML visualization based on the DataFrame."""
        # Create the interactive line chart
        fig = px.line(df, x='update_date', y='normalized_power_rank', color='name', title='Normalized Power Rank by Update Date for Each Team')

        # Save the plot
        filename = 'team_power_rank.html'
        fig.write_html(os.path.join(self.template_dir, filename))

        # If date is provided, find the prior Tuesday
        if date:
            given_date = pd.to_datetime(date)
            # Calculate the difference between the given date and the previous Tuesday (1 represents Tuesday in pandas)
            days_since_last_tuesday = (given_date.weekday() - 1) % 7
            # Subtract the difference from the given date to get the date of the prior Tuesday
            recent_date = given_date - pd.Timedelta(days=days_since_last_tuesday)
        else:
            # Find the most recent update_date that contains at least 30 rows
            date_counts = df.groupby('update_date').size()
            print(date_counts)
            valid_dates = date_counts[date_counts >= 30].index
            recent_date = valid_dates.max()

        recent_df = df[df['update_date'] == recent_date]

        # Create the interactive bar chart
        fig = px.bar(recent_df, x='name', y='normalized_power_rank', title=f'Normalized Power Rank for {recent_date}')

        # Save the plot
        filename = 'normalized_power_ranks.html'
        fig.write_html(os.path.join(self.template_dir, filename))

        # Optionally, save the plot to an HTML file
        # fig.write_html("bar_chart.html")

    def main(self):    # Load and process data
        processed_games_df = self.load_and_process_data()

        # Reload constants
        reload(scripts.constants)

        # Filter columns with stripping whitespaces and exclude 'scoring_differential'
        columns_to_filter = [col for col in self.constants]

        if processed_games_df is not None:
            # Transform data
            df_home, df_away = self.transform_data(processed_games_df, columns_to_filter)

            if df_home is not None and df_away is not None:
                # Calculate power rank
                metrics_home = [metric for metric in columns_to_filter if metric.startswith('ranks_home_')]
                metrics_away = [metric for metric in columns_to_filter if metric.startswith('ranks_away_')]
                feature_importances = self.LOADED_MODEL.feature_importances_
                weights = dict(zip(columns_to_filter, feature_importances))
                df = self.calculate_power_rank(df_home, df_away, metrics_home, metrics_away, weights)

                if df is not None:
                    normalized_df = self.normalize_data(df)
                    # Insert aggregated data into MongoDB
                    self.insert_aggregated_data_into_database(normalized_df)
                else:
                    print("Error in calculating power rank. Exiting script.")
            else:
                print("Error in data transformation. Exiting script.")
        else:
            print("Error in data loading and processing. Exiting script.")

        # After inserting aggregated data into MongoDB
        df = self.fetch_team_ranking_metrics()
        self.generate_interactive_htmls(df)


if __name__ == "__main__":
    nfl_stats = StatsCalculator()
    nfl_stats.main()