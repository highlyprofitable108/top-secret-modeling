# Standard library imports
from datetime import datetime, timedelta
from dateutil.parser import parse
import logging

# Third-party imports
import pandas as pd
import numpy as np

# Local module imports
from classes.config_manager import ConfigManager
from classes.data_processing import DataProcessing
from classes.database_operations import DatabaseOperations
import scripts.all_columns

# Set up logging
logging.basicConfig(level=logging.INFO)


class WeightedStatsAvg:
    def __init__(self, date=None):
        # Initialize ConfigManager, DatabaseOperations, and DataProcessing
        self.config = ConfigManager()
        self.database_operations = DatabaseOperations()

        # Fetch configurations using ConfigManager
        try:
            # Fetch constants using ConfigManager
            self.TWO_YEARS_IN_DAYS = self.config.get_constant('TWO_YEARS_IN_DAYS')
            self.MAX_DAYS_SINCE_GAME = self.config.get_constant('MAX_DAYS_SINCE_GAME')
            self.BASE_COLUMNS = eval(self.config.get_constant('BASE_COLUMNS'))  # Using eval to get the list from the script
            self.AWAY_PREFIX = self.config.get_constant('AWAY_PREFIX')
            self.HOME_PREFIX = self.config.get_constant('HOME_PREFIX')
            self.GAMES_DB_NAME = self.config.get_constant('GAMES_DB_NAME')
            self.TEAMS_DB_NAME = self.config.get_constant('TEAMS_DB_NAME')
            self.PREGAME_DB_NAME = self.config.get_constant('PREGAME_DB_NAME')
            self.RANKS_DB_NAME = self.config.get_constant('RANKS_DB_NAME')
            self.CUTOFF_DATE = datetime.strptime(self.config.get_constant('CUTOFF_DATE'), '%Y-%m-%d')
            self.END_DATE = datetime.strptime(self.config.get_constant('END_DATE'), '%Y-%m-%d')
            self.CUTOFF_DATE_STR = self.config.get_constant('CUTOFF_DATE')
            self.END_DATE_STR = self.config.get_constant('END_DATE')
            self.TARGET_VARIABLE = self.config.get_constant('TARGET_VARIABLE')

            self.data_processing = DataProcessing(self.TARGET_VARIABLE)

            self.data_dir = self.config.get_config('paths', 'data_dir')
            self.model_dir = self.config.get_config('paths', 'model_dir')
            self.static_dir = self.config.get_config('paths', 'static_dir')
            self.template_dir = self.config.get_config('paths', 'template_dir')
            self.database_name = self.config.get_config('database', 'database_name')
        except Exception as e:
            raise ValueError(f"Error fetching configurations: {e}")

        # Define feature columns, excluding 'scoring_differential'
        self.feature_columns = [col for col in scripts.all_columns.ALL_COLUMNS]

        # Set the date and update dependent attributes
        self.set_date(date if date else datetime.today().strftime('%Y-%m-%d'))

    def load_additional_tables(self):
        """Run the entire data processing pipeline."""
        try:
            # Add advanced analytics
            # Rename the games collection games_base


            # Clear existing metrics and temporary tables
            self.clear_team_metrics()
            self.clear_temp_tables()

            tuesdays_list = self.data_processing.generate_weekdays_list(self.CUTOFF_DATE, self.END_DATE)

            # Loop through each Tuesday and generate ranks
            for index, tuesday in enumerate(tuesdays_list):
                logging.info(f"Processing data for: {tuesday}")
                tuesday_str = tuesday.strftime('%Y-%m-%d')
                self.set_date(tuesday_str)

                # Ensure these methods exist in your WeightedStatsAvg class
                self.generate_ranks()

                # If not on the last iteration of the loop
                if index != len(tuesdays_list) - 1:
                    logging.info("Clearing temporary tables...")
                    self.clear_temp_tables()

            # Create pre-game data collection after processing all dates
            self.clear_pregame_metrics()
            self.create_pre_game_data_collection()

        except Exception as e:
            logging.error(f"An error occurred: {e}")
    
    def clear_team_metrics(self):
        """Drop team_aggregated_metrics and pre_game_data collections if they exist."""
        if self.RANKS_DB_NAME in self.database_operations.db.list_collection_names():
            self.database_operations.db.team_aggregated_metrics.drop()

    def clear_pregame_metrics(self):
        if self.PREGAME_DB_NAME in self.database_operations.db.list_collection_names():
            self.database_operations.db.pre_game_data.drop()

    def clear_temp_tables(self):
        """Drop the power_rank collection if it exists."""
        if 'power_rank' in self.database_operations.db.list_collection_names():
            self.database_operations.db.power_rank.drop()

    def set_date(self, date):
        """Set the date and update dependent attributes."""
        self.date = date
        self.date_obj = datetime.strptime(self.date, '%Y-%m-%d')
        self.two_years_ago = (self.date_obj - timedelta(days=self.TWO_YEARS_IN_DAYS)).strftime('%Y-%m-%d')

    def generate_ranks(self):
        """
        Process, transform, and insert data into the database to generate ranks.
        """
        try:
            logging.info("Processing data...")
            processed_games_df, processed_teams_df = self.process_data()

            logging.info("Generating columns for data transformation...")
            final_columns = self.generate_final_columns()

            logging.info("Transforming and calculating data...")
            df = self.transform_and_calculate(processed_games_df, processed_teams_df, final_columns)
            cleaned_metrics = self.clean_metrics(final_columns)

            logging.info("Aggregating data...")
            aggregated_df = self.aggregate_data(df, cleaned_metrics, processed_teams_df)

            logging.info("Inserting aggregated data into database...")
            self.insert_aggregated_data_into_database(aggregated_df)

            # Uncomment the following lines if you want to fetch team aggregated metrics and generate interactive HTML
            # logging.info("Fetching team aggregated metrics and generating interactive HTML...")
            # df = self.fetch_team_aggregated_metrics()
            # self.generate_interactive_html(aggregated_df, self.date)

        except Exception as e:
            logging.error(f"Error encountered: {e}")

    def process_data(self):
        """
        This function handles the loading and initial processing of data.
        :param database_operations: An instance of DatabaseOperations class
        :param data_processing: An instance of DataProcessing class
        :return: Two dataframes containing the loaded and processed data
        """
        try:

            games_df = self.database_operations.fetch_data_from_mongodb(self.GAMES_DB_NAME)
            teams_df = self.database_operations.fetch_data_from_mongodb(self.TEAMS_DB_NAME)
            teams_df.loc[teams_df['name'] == 'Football Team', 'name'] = 'Commanders'

            if games_df.empty or teams_df.empty:
                logging.error("No data fetched from the database.")
                return None, None

            processed_games_df = self.data_processing.flatten_and_merge_data(games_df)
            processed_teams_df = self.data_processing.flatten_and_merge_data(teams_df)

            # Convert columns with list values to string
            for col in processed_games_df.columns:
                if processed_games_df[col].apply(type).eq(list).any():
                    processed_games_df[col] = processed_games_df[col].astype(str)

            # Remove duplicates from teams data
            processed_teams_df = processed_teams_df.drop_duplicates(subset='id')

            return processed_games_df, processed_teams_df
        except Exception as e:
            logging.error(f"Error in load_and_process_data function: {e}")
            return None, None

    def generate_final_columns(self):
        """
        Generate the final list of columns for data transformation.

        :return: A list of column names
        """
        columns_with_away_prefix = [self.AWAY_PREFIX + col for col in self.BASE_COLUMNS]
        columns_with_home_prefix = [self.HOME_PREFIX + col for col in self.BASE_COLUMNS]

        return columns_with_away_prefix + columns_with_home_prefix

    def remove_duplicates_from_teams(self, processed_teams_df):
        """
        Remove duplicate entries from the processed teams dataframe.

        :param processed_teams_df: A DataFrame containing processed teams data.
        :return: A DataFrame with duplicates removed.
        """
        return processed_teams_df.drop_duplicates(subset='id')

    def convert_time_strings_to_minutes(self, processed_df):
        """
        Convert time strings to minutes in the processed dataframe.

        :param processed_df: A DataFrame containing processed data.
        :return: A DataFrame with time strings converted to minutes.
        """
        possession_time_home = f"{self.HOME_PREFIX}summary.possession_time"
        possession_time_away = f"{self.AWAY_PREFIX}summary.possession_time"

        processed_df[possession_time_home] = processed_df[possession_time_home].apply(self.data_processing.time_to_minutes)
        processed_df[possession_time_away] = processed_df[possession_time_away].apply(self.data_processing.time_to_minutes)

        return processed_df

    def process_game_dates(self, processed_df):
        """
        Process game dates and calculate days since the game in the processed dataframe.

        :param processed_df: A DataFrame containing processed data.
        :return: A DataFrame with game dates processed.
        """
        processed_df['game_date'] = pd.to_datetime(processed_df['scheduled'])
        sim_date = self.date_obj
        processed_df['game_date'] = processed_df['game_date'].dt.tz_localize(None)
        processed_df['days_since_game'] = (sim_date - processed_df['game_date']).dt.days
        return processed_df

    def filter_games_based_on_date(self, processed_df):
        """
        Filter games based on the number of days since the game in the processed dataframe.

        :param processed_df: A DataFrame containing processed data.
        :return: A filtered DataFrame.
        """
        return processed_df[(processed_df['days_since_game'] <= self.MAX_DAYS_SINCE_GAME) & (processed_df['days_since_game'] >= 0)].copy()

    def decay_weight(self, days):
        """
        Calculate a weight based on exponential decay function.

        :param days: Number of days since the game.
        :return: Calculated weight.
        """
        weeks_since_game = days // 7  # Convert days to weeks

        # Within the first week: No decay
        if weeks_since_game <= 0:
            return 1.0

        # After 1 week: Gentle decay for recent performances
        elif weeks_since_game <= 4:
            lambda_val = 0.0025

        # Up to 6 weeks: Slightly more aggressive decay as the data starts to age
        elif weeks_since_game <= 6:
            lambda_val = 0.005

        # Up to 18 weeks: More aggressive decay for older data
        elif weeks_since_game <= 18:
            lambda_val = 0.01

        # Up to 40 weeks: Even more aggressive decay as the data becomes less relevant
        elif weeks_since_game <= 40:
            lambda_val = 0.015

        # Beyond 40 weeks: Most aggressive decay for very old data
        else:
            lambda_val = 0.02

        return np.exp(-lambda_val * 7 * weeks_since_game)  # Decay based on total weeks

    def calculate_weights_using_exponential_decay(self, df):
        """
        Calculate weights using exponential decay for the provided DataFrame.

        :param df: DataFrame to calculate weights for.
        :return: DataFrame with calculated weights.
        """
        df.loc[:, 'weight'] = df['days_since_game'].apply(self.decay_weight)
        return df

    def split_data_into_home_and_away(self, df, feature_columns):
        """
        Split the data into home and away dataframes based on feature columns.

        :param df: DataFrame containing data.
        :param feature_columns: List of feature columns.
        :return: DataFrames for home and away data.
        """
        metrics_home = [metric for metric in feature_columns if metric.startswith('statistics.home')]
        metrics_away = [metric for metric in feature_columns if metric.startswith('statistics.away')]
        df_home = df[['summary.home.id', 'summary.home.name', 'game_date', 'days_since_game', 'weight'] + metrics_home]
        df_home = df_home.rename(columns={
            'summary.home.id': 'id',
            'summary.home.name': 'name'
        })
        df_away = df[['summary.away.id', 'summary.away.name', 'game_date', 'days_since_game', 'weight'] + metrics_away]
        df_away = df_away.rename(columns={
            'summary.away.id': 'id',
            'summary.away.name': 'name'
        })
        return df_home, df_away

    def transform_data(self, processed_df, processed_teams_df, feature_columns):
        """
        Transform and preprocess data.

        :param processed_df: DataFrame containing processed data.
        :param processed_teams_df: DataFrame containing processed teams data.
        :param feature_columns: List of feature columns.
        :return: DataFrames for home and away data.
        """
        try:
            logging.info("Starting data transformation...")
            processed_teams_df = self.remove_duplicates_from_teams(processed_teams_df)
            processed_df = self.convert_time_strings_to_minutes(processed_df)
            processed_df = self.process_game_dates(processed_df)
            df = self.filter_games_based_on_date(processed_df)
            df = self.calculate_weights_using_exponential_decay(df)
            df_home, df_away = self.split_data_into_home_and_away(df, feature_columns)

            logging.info("Data transformation complete!")
            return df_home, df_away

        except Exception as e:
            logging.error(f"Error in transform_data function: {e}")
            return None, None

    def transform_and_calculate(self, processed_games_df, processed_teams_df, columns_to_filter):
        """
        Transforms and calculates the necessary data from the processed dataframes.

        Parameters:
        - processed_games_df (pd.DataFrame): The processed games dataframe.
        - processed_teams_df (pd.DataFrame): The processed teams dataframe.
        - columns_to_filter (list): List of columns to filter on.

        Returns:
        - pd.DataFrame: The transformed and calculated dataframe.
        """
        df_home, df_away = self.transform_data(processed_games_df, processed_teams_df, columns_to_filter)
        # Check if dataframes are empty
        if df_home.empty or df_away.empty:
            raise ValueError("Error in data transformation.")

        # Remove prefixes from column names
        df_home.columns = df_home.columns.str.replace(self.HOME_PREFIX, '')
        df_away.columns = df_away.columns.str.replace(self.AWAY_PREFIX, '')

        # Concatenate home and away dataframes to create a single dataframe
        df = pd.concat([df_home, df_away], ignore_index=True)

        # Determine the week number based on the Tuesday-to-Monday window
        df['week_number'] = (df['game_date'] - pd.Timedelta(days=1)).dt.isocalendar().week

        # Check if resulting dataframe is empty
        if df.empty:
            raise ValueError("Error in transform and calculate.")

        return df

    def clean_metrics(self, final_columns):
        return [metric.replace(self.HOME_PREFIX, '').replace(self.AWAY_PREFIX, '') for metric in final_columns]

    def normalize_power_rank(self, df):
        """
        Filters out rows where the 'name' column has values from the Pro Bowl.

        :param df: DataFrame containing the power rank and other details for each team
        :return: DataFrame after normalization
        """
        return df[df['name'].isin(['NFC', 'AFC', 'Team Irvin', 'Team Rie']) == False].copy()

    def calculate_weighted_metrics(self, df, cleaned_metrics):
        """
        Calculate the weighted values for each metric.

        Args:
            df (pd.DataFrame): The dataframe containing the data.
            cleaned_metrics (list): List of metrics to calculate weighted values for.

        Returns:
            pd.DataFrame: DataFrame with additional columns for the weighted metrics.
        """
        weighted_data = {}
        for metric in cleaned_metrics:
            weighted_metric_name = metric + '_weighted'
            weighted_data[weighted_metric_name] = df[metric] * df['weight']

        # Convert the dictionary to a DataFrame
        weighted_df = pd.DataFrame(weighted_data)

        # Concatenate the original df with the weighted_df
        df = pd.concat([df, weighted_df], axis=1)

        return df

    def aggregate_weighted_metrics(self, df, cleaned_metrics):
        """
        Aggregate the weighted metrics.

        Args:
            df (pd.DataFrame): The dataframe containing the weighted metrics.
            cleaned_metrics (list): List of metrics to aggregate.

        Returns:
            dict: Dictionary with aggregated values for each metric.
        """
        aggregated_data = {}
        for metric in cleaned_metrics:
            weighted_metric_name = metric + '_weighted'
            aggregated_data[metric] = df.groupby('id')[weighted_metric_name].sum() / df.groupby('id')['weight'].sum()

        return aggregated_data

    def compute_standard_deviation(self, df, cleaned_metrics):
        """
        Compute the standard deviation for each metric across all teams.

        Args:
            df (pd.DataFrame): The dataframe containing the data.
            cleaned_metrics (list): List of metrics to compute standard deviation for.

        Returns:
            pd.DataFrame: DataFrame with standard deviation values for each metric.
        """
        stddev_data = {}
        for metric in cleaned_metrics:
            stddev_data[metric + '_stddev'] = df.groupby('id')[metric].std()

        return pd.DataFrame(stddev_data)

    def merge_dataframes(self, aggregated_df, df_stddev, processed_teams_df):
        """
        Merge the aggregated DataFrame with the standard deviation DataFrame and processed teams DataFrame.

        Args:
            aggregated_df (pd.DataFrame): The aggregated data.
            df_stddev (pd.DataFrame): The standard deviation data.
            processed_teams_df (pd.DataFrame): The processed teams data.

        Returns:
            pd.DataFrame: Merged DataFrame.
        """
        # Merge with standard deviation data
        aggregated_df = aggregated_df.merge(df_stddev, left_on='id', right_index=True)

        # Merge with processed teams data
        aggregated_df = aggregated_df.merge(processed_teams_df, left_on='id', right_on='id', how='left')

        return aggregated_df

    def aggregate_data(self, df, cleaned_metrics, processed_teams_df):
        """
        Aggregate and normalize data, calculate weighted metrics, and merge with teams data.

        :param df: DataFrame containing the data to be aggregated.
        :param cleaned_metrics: List of cleaned metric names.
        :param processed_teams_df: DataFrame containing processed teams data.
        :return: The final aggregated and normalized DataFrame.
        """
        try:
            df = self.normalize_power_rank(df)
            ranks_df = self.data_processing.handle_duplicate_columns(df)
            self.database_operations.insert_data_into_mongodb('power_rank', ranks_df.to_dict('records'))

            # Calculate weighted metrics
            df = self.calculate_weighted_metrics(df, cleaned_metrics)

            # Aggregate weighted metrics
            aggregated_data = self.aggregate_weighted_metrics(df, cleaned_metrics)

            df_stddev = self.compute_standard_deviation(df, cleaned_metrics)

            # Merge the standard deviation DataFrame with the aggregated_df DataFrame
            aggregated_df = pd.DataFrame(aggregated_data).reset_index()

            # Convert 'id' column data type for both DataFrames
            processed_teams_df['id'] = processed_teams_df['id'].astype(str)
            aggregated_df['id'] = aggregated_df['id'].astype(str)

            # Merge the DataFrames
            aggregated_df = self.merge_dataframes(aggregated_df, df_stddev, processed_teams_df)

            return aggregated_df  # The final aggregated and normalized dataframe

        except Exception as e:
            logging.error(f"Error in aggregate_data function: {e}")
            return None

    def insert_aggregated_data_into_database(self, aggregated_df):
        """
        Inserts the aggregated and normalized data into a MongoDB collection.
        :param aggregated_df: A dataframe containing aggregated and normalized data
        :param database_operations: An instance of the DatabaseOperations class
        """
        try:
            aggregated_df.drop_duplicates()
            aggregated_df.reset_index(inplace=True, drop=True)
            aggregated_df.loc[:, 'update_date'] = self.date
            if '_id' in aggregated_df.columns:
                del aggregated_df['_id']

            # Reorder columns to move 'update_date' to the first position
            cols = ['update_date'] + [col for col in aggregated_df if col != 'update_date']
            aggregated_df = aggregated_df[cols]

            self.database_operations.insert_data_into_mongodb(self.RANKS_DB_NAME, aggregated_df.to_dict('records'))

            logging.info("Aggregated data inserted into MongoDB successfully.")
        except Exception as e:
            logging.error(f"Error inserting aggregated data into MongoDB: {e}")

    def create_pre_game_data_collection(self):
        """
        Create and insert pre-game data into MongoDB.

        Fetches game and rank data, filters games based on date, and prepares pre-game data for insertion.

        :return: None
        """
        games_df = self.database_operations.fetch_data_from_mongodb(self.GAMES_DB_NAME)
        print(games_df)
        ranks_df = self.database_operations.fetch_data_from_mongodb(self.RANKS_DB_NAME)

        # Drop games before CUTOFF_DATE and after 
        games_df = games_df[games_df['scheduled'] >= self.CUTOFF_DATE_STR]
        games_df = games_df[games_df['scheduled'] < self.END_DATE_STR]

        # Create an empty list to store the pre-game data
        pre_game_data_list = []

        for _, game in games_df.iterrows():
            game_data = {
                'scheduled': game.get('scheduled'),
                'odds': game['summary'].get('odds', {})
            }

            # Convert game['scheduled'] to a tz-naive datetime object
            game_scheduled_naive = parse(game['scheduled']).replace(tzinfo=None)

            # Fetch rank data for teams A and B
            ranks_team_A = self.get_team_rank(game['summary']['home']['id'], ranks_df, game_scheduled_naive)
            ranks_team_B = self.get_team_rank(game['summary']['away']['id'], ranks_df, game_scheduled_naive)

            # Ensure ranks_team_A and ranks_team_B are not None before proceeding
            if ranks_team_A is not None and ranks_team_B is not None:
                # Calculate differences and ratios for each metric
                for metric in ranks_team_A.keys():
                    if metric not in ['id', 'update_date']:  # Exclude non-metric columns
                        value_A = ranks_team_A[metric]
                        value_B = ranks_team_B[metric]

                        # Ensure the values are numbers before performing arithmetic operations
                        if isinstance(value_A, (int, float)) and isinstance(value_B, (int, float)):
                            diff_key = f"{metric}_difference"
                            ratio_key = f"{metric}_ratio"
                            game_data[diff_key] = value_A - value_B
                            game_data[ratio_key] = value_A / value_B if value_B != 0 else 0

            pre_game_data_list.append(game_data)

        # Convert the pre-game data list to a DataFrame
        pre_game_data_df = pd.DataFrame(pre_game_data_list)

        # Convert the pre-game data DataFrame to a list of dictionaries
        collapsed_df = self.data_processing.collapse_dataframe(pre_game_data_df)
        collapsed_df = self.data_processing.cleanup_ranks(collapsed_df)
        pre_game_data_list_of_dicts = collapsed_df.to_dict(orient='records')

        # Insert the pre-game data list of dictionaries into MongoDB
        self.database_operations.insert_data_into_mongodb(self.PREGAME_DB_NAME, pre_game_data_list_of_dicts)

        logging.info("Pre-game data inserted into MongoDB successfully.")

    def get_team_rank(self, team_id, ranks_df, game_scheduled_naive):
        """
        Fetch the rank data for a given team before a specified date.

        Args:
            team_id (str): The ID of the team.
            ranks_df (pd.DataFrame): DataFrame containing rank data.
            game_scheduled_naive (datetime): The date before which to fetch the rank.

        Returns:
            dict: Rank data for the team before the specified date.
        """
        filtered_rank = ranks_df[
            (ranks_df['id'] == team_id) & (ranks_df['update_date'] < game_scheduled_naive.strftime('%Y-%m-%d'))
        ].sort_values(by='update_date', ascending=False)

        if not filtered_rank.empty:
            return filtered_rank.iloc[0].to_dict()
        else:
            logging.error(f"No rank data found for team {team_id} before {game_scheduled_naive}")
            return None


if __name__ == "__main__":
    nfl_weighted_stats = WeightedStatsAvg()
    nfl_weighted_stats.load_additional_tables()
