# Standard library imports
import os
from datetime import datetime, timedelta
from dateutil.parser import parse
from importlib import reload
from pymongo import MongoClient

# Third-party imports
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# Local module imports
from classes.config_manager import ConfigManager
from classes.data_processing import DataProcessing
from classes.database_operations import DatabaseOperations
import scripts.constants


class StatsCalculator:
    def __init__(self, date=None):
        # Initialize ConfigManager, DatabaseOperations, and DataProcessing
        self.config = ConfigManager()
        self.database_operations = DatabaseOperations()
        self.data_processing = DataProcessing()

        # Fetch configurations using ConfigManager
        try:
            self.data_dir = self.config.get_config('paths', 'data_dir')
            self.model_dir = self.config.get_config('paths', 'model_dir')
            self.static_dir = self.config.get_config('paths', 'static_dir')
            self.template_dir = self.config.get_config('paths', 'template_dir')
            self.database_name = self.config.get_config('database', 'database_name')
        except Exception as e:
            raise ValueError(f"Error fetching configurations: {e}")

        # Define feature columns, excluding 'scoring_differential'
        self.feature_columns = [col for col in scripts.constants.COLUMNS_TO_KEEP if col != 'scoring_differential']

        # Load the trained model
        try:
            self.LOADED_MODEL = joblib.load(os.path.join(self.model_dir, 'trained_nfl_model.pkl'))
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

        # Get the current date or use the provided date
        self.date = date if date else datetime.today().strftime('%Y-%m-%d')

        # Convert the string date to a datetime object for further operations
        self.date_obj = datetime.strptime(self.date, '%Y-%m-%d')

        # Calculate the date for two years ago using timedelta
        self.two_years_ago = (self.date_obj - timedelta(days=730)).strftime('%Y-%m-%d')

    def set_date(self, date):
        self.date = date

    def load_and_process_data(self):
        """
        This function handles the loading and initial processing of data.
        :param database_operations: An instance of DatabaseOperations class
        :param data_processing: An instance of DataProcessing class
        :return: Two dataframes containing the loaded and processed data
        """
        try:
            # Step 1: Fetch data from the database
            games_df = self.database_operations.fetch_data_from_mongodb("games")
            teams_df = self.database_operations.fetch_data_from_mongodb("teams")
            teams_df.loc[teams_df['name'] == 'Football Team', 'name'] = 'Commanders'

            if games_df.empty or teams_df.empty:
                print("No data fetched from the database.")
                return None, None

            # Step 2: Perform initial data processing (like flattening the data)
            processed_games_df = self.data_processing.flatten_and_merge_data(games_df)
            processed_teams_df = self.data_processing.flatten_and_merge_data(teams_df)

            # Convert columns with list values to string
            for col in processed_games_df.columns:
                if processed_games_df[col].apply(type).eq(list).any():
                    processed_games_df[col] = processed_games_df[col].astype(str)

            # Step 3: Additional data processing steps (as per the original script)
            processed_games_df = self.data_processing.calculate_scoring_differential(processed_games_df)

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

    def transform_data(self, processed_df, processed_teams_df, feature_columns):
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
                weeks_since_game = days // 7  # Convert days to weeks

                if weeks_since_game == 0:  # Within the first week
                    return 1.0
                elif weeks_since_game == 1:  # After 1 week
                    lambda_val = 0.005
                elif weeks_since_game <= 6:  # Up to 6 weeks
                    lambda_val = 0.01
                elif weeks_since_game <= 18:  # Up to 18 weeks
                    lambda_val = 0.02
                elif weeks_since_game <= 40:  # Up to 40 weeks
                    lambda_val = 0.03
                else:  # Beyond 40 weeks
                    lambda_val = 0.04

                return np.exp(-lambda_val * 7 * weeks_since_game)  # Decay based on total weeks

            # Remove duplicates from teams data
            processed_teams_df = processed_teams_df.drop_duplicates(subset='id')

            # Convert time strings to minutes
            processed_df['statistics_home.summary.possession_time'] = processed_df['statistics_home.summary.possession_time'].apply(self.data_processing.time_to_minutes)
            processed_df['statistics_away.summary.possession_time'] = processed_df['statistics_away.summary.possession_time'].apply(self.data_processing.time_to_minutes)

            # Convert scheduled column to datetime
            processed_df['game_date'] = pd.to_datetime(processed_df['scheduled'])

            # Use the date_obj directly since it's already a datetime object
            sim_date = self.date

            processed_df['game_date'] = processed_df['game_date'].dt.tz_localize(None)
            processed_df['days_since_game'] = (sim_date - processed_df['game_date']).dt.days

            # Drop all games played more than 104 weeks (728 days) ago
            max_days_since_game = 104 * 7
            two_year_df = processed_df[processed_df['days_since_game'] <= max_days_since_game]

            # Drop all games that haven't been played yet
            df = two_year_df[two_year_df['days_since_game'] >= 0].copy()

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

    def aggregate_and_normalize_data(self, df, cleaned_metrics, processed_teams_df):
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
            df = df[df['name'].isin(['NFC', 'AFC']) == False].copy()

            # Add the update_date column and populate it with self.date
            df.loc[:, 'update_date'] = self.date

            def normalize_within_week(group):
                min_rank = group['power_rank'].min()
                max_rank = group['power_rank'].max()
                group['power_rank'] = (group['power_rank'] - min_rank) / (max_rank - min_rank)
                return group

            df = df.groupby(['week_number']).apply(normalize_within_week)

            # Sort the dataframe by name and power_rank
            df = df.sort_values(by=['power_rank'])

            # Create a list for columns_for_power_rank_table with the cleaned metrics list
            columns_for_power_rank_table = ['update_date', 'id', 'name', 'game_date', 'days_since_game', 'weight', 'power_rank'] + cleaned_metrics
            ranks_df = df[columns_for_power_rank_table]

            # Get the columns that are not unique
            non_unique_columns = ranks_df.columns[ranks_df.columns.duplicated(keep=False)].to_list()

            checked = []  # To keep track of columns we've already checked

            for col in non_unique_columns:
                if col not in checked:
                    # Get all columns with the same name
                    duplicate_cols = ranks_df.columns[ranks_df.columns == col].to_list()

                    # If there are more than one column with the same name
                    if len(duplicate_cols) > 1:
                        # Compare the values of the columns
                        if ranks_df[duplicate_cols[0]].equals(ranks_df[duplicate_cols[1]]):
                            # Drop one of the columns if they are identical
                            ranks_df = ranks_df.drop(columns=duplicate_cols[0])

                    checked.append(col)

            # Save to a new collection in the database
            ranks_df.reset_index(inplace=True, drop=True)
            self.database_operations.insert_data_into_mongodb('power_rank', ranks_df.to_dict('records'))

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

    def insert_aggregated_data_into_database(self, aggregated_df):
        """
        Inserts the aggregated and normalized data into a MongoDB collection.
        :param aggregated_df: A dataframe containing aggregated and normalized data
        :param database_operations: An instance of the DatabaseOperations class
        """
        try:
            # Insert the aggregated data into the collection
            aggregated_df.reset_index(inplace=True, drop=True)
            aggregated_df.loc[:, 'update_date'] = self.date
            if '_id' in aggregated_df.columns:
                del aggregated_df['_id']

            self.database_operations.insert_data_into_mongodb('team_aggregated_metrics', aggregated_df.to_dict('records'))

            # Saving the report to a CSV file
            aggregated_df = aggregated_df.sort_values(by='normalized_power_rank', ascending=False)
            power_ranks_report_path = os.path.join(self.static_dir, 'power_ranks.csv')
            aggregated_df.to_csv(power_ranks_report_path)
            print("Aggregated data inserted into MongoDB successfully.")
        except Exception as e:
            print(f"Error inserting aggregated data into MongoDB: {e}")

    def create_pre_game_data_collection(self):
        games_df = self.database_operations.fetch_data_from_mongodb("games")
        ranks_df = self.database_operations.fetch_data_from_mongodb("team_aggregated_metrics")

        # Drop games before September 01, 2019
        games_df = games_df[games_df['scheduled'] >= '2019-09-01']

        # Create an empty list to store the pre-game data
        pre_game_data_list = []

        for _, game in games_df.iterrows():
            game_data = {
                'scheduled': game.get('scheduled'),
                'home': game['summary']['home'],
                'away': game['summary']['away'],
                'odds': game['summary'].get('odds', {})
            }

            # Convert game['scheduled'] to a tz-naive datetime object
            game_scheduled_naive = parse(game['scheduled']).replace(tzinfo=None)
            print(f"Processing game scheduled on: {game_scheduled_naive}")

            # For home team
            home_team_id = game['summary']['home']['id']
            filtered_home_rank = ranks_df[
                (ranks_df['id'] == home_team_id) & (ranks_df['update_date'] < game_scheduled_naive)
            ].sort_values(by='update_date', ascending=False)

            print(f"Number of rows for home team {home_team_id}: {len(filtered_home_rank)}")

            if not filtered_home_rank.empty:
                home_rank = filtered_home_rank.iloc[0]
                game_data['ranks_home'] = home_rank.to_dict()
            else:
                print(f"No rank data found for home team {home_team_id} before {game_scheduled_naive}")

            # For away team
            away_team_id = game['summary']['away']['id']
            filtered_away_rank = ranks_df[
                (ranks_df['id'] == away_team_id) & (ranks_df['update_date'] < game_scheduled_naive)
            ].sort_values(by='update_date', ascending=False)

            print(f"Number of rows for away team {away_team_id}: {len(filtered_away_rank)}")

            if not filtered_away_rank.empty:
                away_rank = filtered_away_rank.iloc[0]
                game_data['ranks_away'] = away_rank.to_dict()
            else:
                print(f"No rank data found for away team {away_team_id} before {game_scheduled_naive}")

            # Print unique team IDs and update dates in ranks_df for debugging
            print("Unique team IDs in ranks_df:", ranks_df['id'].unique())
            print("Unique update dates in ranks_df:", ranks_df['update_date'].unique())

            pre_game_data_list.append(game_data)

        # Convert the pre-game data list to a DataFrame
        pre_game_data_df = pd.DataFrame(pre_game_data_list)

        # Insert the pre-game data DataFrame into MongoDB
        self.database_operations.insert_data_into_mongodb("pre_game_data", pre_game_data_df)

        print("Pre-game data inserted into MongoDB successfully.")

    def fetch_team_aggregated_metrics(self):
        """Fetch team_aggregated_metrics from MongoDB and return as a DataFrame."""
        client = MongoClient()
        db = client[self.database_name]
        collection = db['team_aggregated_metrics']

        # Find the most recent update_date
        latest_date_record = collection.find_one(sort=[('update_date', -1)])
        if not latest_date_record:
            return pd.DataFrame()  # Return an empty DataFrame if no records found

        latest_date = latest_date_record['update_date']

        # Fetch records with the most recent update_date
        df = pd.DataFrame(list(collection.find({'update_date': latest_date})))

        return df

    def generate_interactive_html(self, df, date=None):
        """Generate an interactive HTML visualization based on the DataFrame."""
        # Sort the dataframe by 'normalized_power_rank' in descending order
        df_sorted = df.sort_values(by='normalized_power_rank', ascending=False)

        # Get today's date
        today_date = datetime.today().strftime('%Y-%m-%d')

        # If date is not None and does not equal today's date, save the file as 'team_power_rank_sim.html'
        if date is None or date == today_date:
            filename = 'team_power_rank.html'

            # Generate a bar chart of normalized_power_rank by team name using the sorted dataframe
            fig = px.bar(df_sorted, x='name', y='normalized_power_rank', title='Normalized Power Rank by Team')
            fig.write_html(os.path.join(self.template_dir, filename))

        filename = 'team_power_rank_sim.html'

        # Generate a bar chart of normalized_power_rank by team name using the sorted dataframe
        fig = px.bar(df_sorted, x='name', y='normalized_power_rank', title='Normalized Power Rank by Team on Sim Date')
        fig.write_html(os.path.join(self.template_dir, filename))

    def clear_temp_tables(self):
        # Drop the power_rank collection if it exists
        if 'power_rank' in self.database_operations.db.list_collection_names():
            self.database_operations.db.power_rank.drop()

    def clear_team_metrics(self):
        # Drop the collection if it exists
        if 'team_aggregated_metrics' in self.database_operations.db.list_collection_names():
            self.database_operations.db.team_aggregated_metrics.drop()

        # Drop the collection if it exists
        if 'pre_game_data' in self.database_operations.db.list_collection_names():
            self.database_operations.db.pre_game_data.drop()

    def process_data(self):
        processed_games_df, processed_teams_df = self.load_and_process_data()
        if processed_games_df is None or processed_teams_df is None:
            raise ValueError("Error in data loading and processing.")
        return processed_games_df, processed_teams_df

    def transform_and_calculate_power_rank(self, processed_games_df, processed_teams_df, columns_to_filter):
        df_home, df_away = self.transform_data(processed_games_df, processed_teams_df, columns_to_filter)
        if df_home is None or df_away is None:
            raise ValueError("Error in data transformation.")

        metrics_home = [metric for metric in columns_to_filter if metric.startswith('statistics_home')]
        metrics_away = [metric for metric in columns_to_filter if metric.startswith('statistics_away')]
        feature_importances = self.LOADED_MODEL.feature_importances_
        weights = dict(zip(columns_to_filter, feature_importances))
        df = self.calculate_power_rank(df_home, df_away, metrics_home, metrics_away, weights)
        if df is None:
            raise ValueError("Error in calculating power rank.")
        return df

    def aggregate_data(self, df, cleaned_metrics, processed_teams_df):
        aggregated_df = self.aggregate_and_normalize_data(df, cleaned_metrics, processed_teams_df)
        if aggregated_df is None:
            raise ValueError("Error in aggregating and normalizing data.")
        return aggregated_df

    def generate_ranks(self):
        try:
            processed_games_df, processed_teams_df = self.process_data()

            # Reload constants
            reload(scripts.constants)

            columns_to_filter = [col for col in scripts.constants.COLUMNS_TO_KEEP if col.strip() in map(str.strip, processed_games_df.columns) and col.strip() != 'scoring_differential']

            df = self.transform_and_calculate_power_rank(processed_games_df, processed_teams_df, columns_to_filter)
            cleaned_metrics = [metric.replace('statistics_home.', '').replace('statistics_away.', '') for metric in columns_to_filter]
            aggregated_df = self.aggregate_data(df, cleaned_metrics, processed_teams_df)
            self.insert_aggregated_data_into_database(aggregated_df)
            self.clear_temp_tables()
            df = self.fetch_team_aggregated_metrics()
            self.generate_interactive_html(aggregated_df, self.date)
        except ValueError as e:
            print(e)


def generate_tuesdays_list(date_obj):
    """
    Generate a list of Tuesdays from 09/01/2019 to the given end date, excluding Tuesdays from March through August.

    Parameters:
    - date_obj (datetime): The end date for generating the list.

    Returns:
    - List[datetime]: List of Tuesdays.
    """
    start_date_obj = datetime(2019, 9, 1)
    tuesdays_list = []

    # Adjust the start date to the first Tuesday on or after 09/01/2019
    while start_date_obj.weekday() != 1:  # 1 represents Tuesday
        start_date_obj += timedelta(days=1)

    # Append Tuesdays to the list until the end date, skipping March through August
    while start_date_obj <= date_obj:
        if start_date_obj.month not in range(3, 9):  # 3 to 9 represents March to August
            tuesdays_list.append(start_date_obj)
        start_date_obj += timedelta(weeks=1)

    return tuesdays_list


if __name__ == "__main__":
    nfl_stats = StatsCalculator()
    nfl_stats.clear_team_metrics()
    nfl_stats.clear_temp_tables()
    date_obj = datetime.today()
    tuesdays_list = generate_tuesdays_list(date_obj)
    for tuesday in tuesdays_list:
        print(tuesday)
        nfl_stats.set_date(tuesday)         
        nfl_stats.generate_ranks()
    nfl_stats.create_pre_game_data_collection()
