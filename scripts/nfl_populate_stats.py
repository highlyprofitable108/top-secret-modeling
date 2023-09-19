from datetime import datetime
import pandas as pd
import numpy as np
import os
import joblib
from classes.config_manager import ConfigManager
from classes.data_processing import DataProcessing
from classes.database_operations import DatabaseOperations
from .constants import COLUMNS_TO_KEEP


class StatsCalculator:
    def __init__(self):
        # Initialize ConfigManager, DatabaseOperations, and DataProcessing
        self.config = ConfigManager()
        self.database_operations = DatabaseOperations()
        self.data_processing = DataProcessing()

        # Fetch configurations using ConfigManager
        self.data_dir = self.config.get_config('paths', 'data_dir')
        self.model_dir = self.config.get_config('paths', 'model_dir')
        self.static_dir = self.config.get_config('paths', 'static_dir')
        self.database_name = self.config.get_config('database', 'database_name')
        self.feature_columns = [col for col in COLUMNS_TO_KEEP if col != 'scoring_differential']
        self.LOADED_MODEL = joblib.load(os.path.join(self.model_dir, 'trained_nfl_model.pkl'))

        # Get the current date
        self.today = datetime.today().date()

        # Calculate the date for two years ago
        self.two_years_ago = (self.today - pd.Timedelta(days=730)).strftime('%Y-%m-%d')

    def load_and_process_data(self, database_operations, data_processing):
        """
        This function handles the loading and initial processing of data.
        :param database_operations: An instance of DatabaseOperations class
        :param data_processing: An instance of DataProcessing class
        :return: Two dataframes containing the loaded and processed data
        """
        try:
            # Step 1: Fetch data from the database
            games_df = database_operations.fetch_data_from_mongodb("games")
            teams_df = database_operations.fetch_data_from_mongodb("teams")
            teams_df.loc[teams_df['name'] == 'Football Team', 'name'] = 'Commanders'

            if games_df.empty or teams_df.empty:
                print("No data fetched from the database.")
                return None, None

            # Step 2: Perform initial data processing (like flattening the data)
            processed_games_df = data_processing.flatten_and_merge_data(games_df)
            processed_teams_df = data_processing.flatten_and_merge_data(teams_df)

            # Convert columns with list values to string
            for col in processed_games_df.columns:
                if processed_games_df[col].apply(type).eq(list).any():
                    processed_games_df[col] = processed_games_df[col].astype(str)

            # Step 3: Additional data processing steps (as per the original script)
            processed_games_df = data_processing.calculate_scoring_differential(processed_games_df)

            # Drop games if 'scoring_differential' key does not exist
            if 'scoring_differential' not in processed_games_df.columns:
                print("'scoring_differential' key does not exist. Dropping games.")
                return pd.DataFrame(), processed_teams_df  # Return an empty dataframe for games data

            # Remove duplicates from teams data
            processed_teams_df = processed_teams_df.drop_duplicates(subset='id')

            return processed_games_df, processed_teams_df
        except Exception as e:
            print(f"Error in load_and_process_data function: {e}")
            return None, None

    def transform_data(self, processed_df, processed_teams_df, data_processing, feature_columns):
        """
        This function handles further data transformation.
        :param processed_df: A dataframe containing processed games data
        :param processed_teams_df: A dataframe containing processed teams data
        :param data_processing: An instance of DataProcessing class
        :param feature_columns: A list of feature columns to be used in the model
        :return: A tuple containing two dataframes with further transformed data
        """
        try:
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

            # Remove duplicates from teams data
            processed_teams_df = processed_teams_df.drop_duplicates(subset='id')

            # Convert time strings to minutes
            processed_df['statistics_home.summary.possession_time'] = processed_df['statistics_home.summary.possession_time'].apply(data_processing.time_to_minutes)
            processed_df['statistics_away.summary.possession_time'] = processed_df['statistics_away.summary.possession_time'].apply(data_processing.time_to_minutes)

            # Convert scheduled column to datetime and calculate days since the game was played
            processed_df['game_date'] = pd.to_datetime(processed_df['scheduled'])
            today = datetime.today()
            processed_df['game_date'] = processed_df['game_date'].dt.tz_localize(None)
            processed_df['days_since_game'] = (today - processed_df['game_date']).dt.days

            # Drop all games played more than 104 weeks (728 days) ago
            max_days_since_game = 104 * 7
            df = processed_df[processed_df['days_since_game'] <= max_days_since_game]

            # Apply exponential decay function to calculate weights
            df.loc[:, 'weight'] = df['days_since_game'].apply(decay_weight)

            # Split metrics list into home and away metrics
            metrics_home = [metric for metric in feature_columns if metric.startswith('statistics_home')]
            metrics_away = [metric for metric in feature_columns if metric.startswith('statistics_away')]

            # Separate home and away data
            df_home = df[['summary_home.id', 'summary_home.name', 'game_date', 'days_since_game', 'weight'] + metrics_home]
            df_away = df[['summary_away.id', 'summary_away.name', 'game_date', 'days_since_game', 'weight'] + metrics_away]

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
            df_home.columns = df_home.columns.str.replace('summary_home.', '').str.replace('statistics_home.', '')
            df_away.columns = df_away.columns.str.replace('summary_away.', '').str.replace('statistics_away.', '')

            # Concatenate home and away dataframes to create a single dataframe
            df = pd.concat([df_home, df_away], ignore_index=True)

            # Determine the week number based on the Tuesday-to-Monday window
            df['week_number'] = (df['game_date'] - pd.Timedelta(days=1)).dt.isocalendar().week

            return df
        except Exception as e:
            print(f"Error in calculate_power_rank function: {e}")
            return pd.DataFrame()  # Return an empty dataframe if an error occurs

    def aggregate_and_normalize_data(self, df, cleaned_metrics, database_operations, processed_teams_df):
        """
        This function aggregates and normalizes the data based on various metrics and saves the results to a MongoDB collection.
        :param df: A dataframe containing the power rank and other details for each team
        :param cleaned_metrics: A list of cleaned metrics (without prefixes)
        :param database_operations: An instance of the DatabaseOperations class
        :param processed_teams_df: A dataframe containing processed teams data
        :return: A dataframe containing aggregated and normalized data
        """
        try:
            # Normalize the power_rank values within each week
            def normalize_within_week(group):
                min_rank = group['power_rank'].min()
                max_rank = group['power_rank'].max()
                group['power_rank'] = (group['power_rank'] - min_rank) / (max_rank - min_rank)
                return group

            df = df.groupby(['week_number']).apply(normalize_within_week)

            # Sort the dataframe by name and power_rank
            df = df.sort_values(by=['power_rank'])

            # Create a list for columns_for_power_rank_table with the cleaned metrics list
            columns_for_power_rank_table = ['id', 'name', 'game_date', 'days_since_game', 'weight', 'power_rank'] + cleaned_metrics
            power_rank_df = df[columns_for_power_rank_table]

            # Drop the power_rank collection if it exists
            if 'power_rank' in database_operations.db.list_collection_names():
                database_operations.db.power_rank.drop()

            # Save to a new collection in the database
            power_rank_df.reset_index(inplace=True, drop=True)
            database_operations.insert_data_into_mongodb('power_rank', power_rank_df.to_dict('records'))

            df['weighted_power_rank'] = df['power_rank'] * df['weight']

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
            aggregated_df = aggregated_df.merge(processed_teams_df, left_on='id', right_on='id', how='left')

            return aggregated_df  # The final aggregated and normalized dataframe
        except Exception as e:
            print(f"Error in aggregate_and_normalize_data function: {e}")
            return None

    def insert_aggregated_data_into_database(self, aggregated_df, database_operations):
        """
        Inserts the aggregated and normalized data into a MongoDB collection.
        :param aggregated_df: A dataframe containing aggregated and normalized data
        :param database_operations: An instance of the DatabaseOperations class
        """
        try:
            # Drop the collection if it exists
            if 'team_aggregated_metrics' in database_operations.db.list_collection_names():
                database_operations.db.team_aggregated_metrics.drop()

            # Insert the aggregated data into the collection
            aggregated_df.reset_index(inplace=True, drop=True)
            database_operations.insert_data_into_mongodb('team_aggregated_metrics', aggregated_df.to_dict('records'))
            
            # Saving the report to a CSV file
            aggregated_df = aggregated_df.sort_values(by='normalized_power_rank', ascending=False)
            power_ranks_report_path = os.path.join(self.static_dir, 'power_ranks.csv')
            aggregated_df.to_csv(power_ranks_report_path)
            print("Aggregated data inserted into MongoDB successfully.")
        except Exception as e:
            print(f"Error inserting aggregated data into MongoDB: {e}")

    def main(self):    # Load and process data
        processed_games_df, processed_teams_df = self.load_and_process_data(self.database_operations, self.data_processing)

        if processed_games_df is not None and processed_teams_df is not None:
            # Transform data
            df_home, df_away = self.transform_data(processed_games_df, processed_teams_df, self.data_processing, self.feature_columns)

            if df_home is not None and df_away is not None:
                # Calculate power rank
                metrics_home = [metric for metric in self.feature_columns if metric.startswith('statistics_home')]
                metrics_away = [metric for metric in self.feature_columns if metric.startswith('statistics_away')]
                feature_importances = self.LOADED_MODEL.feature_importances_
                weights = dict(zip(self.feature_columns, feature_importances))
                df = self.calculate_power_rank(df_home, df_away, metrics_home, metrics_away, weights)

                if df is not None:
                    # Aggregate and normalize data
                    cleaned_metrics = [metric.replace('statistics_home.', '').replace('statistics_away.', '') for metric in self.feature_columns]
                    aggregated_df = self.aggregate_and_normalize_data(df, cleaned_metrics, self.database_operations, processed_teams_df)

                    if aggregated_df is not None:
                        # Insert aggregated data into MongoDB
                        self.insert_aggregated_data_into_database(aggregated_df, self.database_operations)

                        # Further script implementation here, where you can use aggregated_df for analysis
                    else:
                        print("Error in aggregating and normalizing data. Exiting script.")
                else:
                    print("Error in calculating power rank. Exiting script.")
            else:
                print("Error in data transformation. Exiting script.")
        else:
            print("Error in data loading and processing. Exiting script.")


if __name__ == "__main__":
    nfl_stats = StatsCalculator()
    nfl_stats.main()
