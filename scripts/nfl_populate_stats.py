import os
import logging
import joblib
import pandas as pd
from pymongo import MongoClient
import plotly.express as px
from datetime import datetime
from importlib import reload
from classes.config_manager import ConfigManager
from classes.data_processing import DataProcessing
from classes.database_operations import DatabaseOperations
import scripts.constants

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class StatsCalculator:
    def __init__(self):
        """
        Initialize the StatsCalculator class, loading configurations, constants, and the trained model.

        Initializes configuration settings, constants, class instances, and loads a pre-trained machine learning model.
        """
        self.config = ConfigManager()
        self.database_operations = DatabaseOperations()
        self.data_processing = DataProcessing()
        self.CONSTANTS = scripts.constants.COLUMNS_TO_KEEP
        self._fetch_constants_and_configs()
        self.LOADED_MODEL = joblib.load(os.path.join(self.model_dir, 'trained_nfl_model.pkl'))
        self.feature_columns = [col for col in self.CONSTANTS]

    def _fetch_constants_and_configs(self):
        """
        Fetch constants and configurations from the configuration manager.

        Fetches various constants and configuration settings from the configuration manager and initializes class attributes.
        """
        try:
            constants = [
                'TWO_YEARS_IN_DAYS', 'MAX_DAYS_SINCE_GAME', 'BASE_COLUMNS', 'AWAY_PREFIX',
                'HOME_PREFIX', 'GAMES_DB_NAME', 'TEAMS_DB_NAME', 'PREGAME_DB_NAME', 'RANKS_DB_NAME',
                'WEEKLY_RANKS_DB_NAME', 'CUTOFF_DATE', 'TARGET_VARIABLE'
            ]
            for const in constants:
                setattr(self, const, self.config.get_constant(const))
            self.CUTOFF_DATE = datetime.strptime(self.CUTOFF_DATE, '%Y-%m-%d')
            self.model_type = self.config.get_model_settings('model_type')
            self.grid_search_params = self.config.get_model_settings('grid_search')

            paths = ['data_dir', 'model_dir', 'static_dir', 'template_dir']
            for path in paths:
                setattr(self, path, self.config.get_config('paths', path))

            self.database_name = self.config.get_config('database', 'database_name')
        except Exception as e:
            raise ValueError(f"Error fetching configurations: {e}")

    def load_and_process_data(self):
        """
        This function handles the loading and initial processing of data.
        :param database_operations: An instance of DatabaseOperations class
        :param data_processing: An instance of DataProcessing class
        :return: Two dataframes containing the loaded and processed data
        """
        try:
            stats_df = self.database_operations.fetch_data_from_mongodb(self.RANKS_DB_NAME)
            processed_stats_df = self.data_processing.flatten_and_merge_data(stats_df)

            for col in processed_stats_df.columns:
                if processed_stats_df[col].apply(type).eq(list).any():
                    processed_stats_df[col] = processed_stats_df[col].astype(str)

            return processed_stats_df
        except Exception as e:
            logging.error(f"Error in load_and_process_data function: {e}")
            return None

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
            logging.error(f"Error in transform_data function: {e}")
            return None, None

    def calculate_power_rank(self, df, metrics, weights):
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

                    try:
                        power_rank += weight * metric_value
                    except TypeError as e:
                        print(f"Error multiplying for metric '{metric}' with value '{metric_value}' and weight '{weight}'")
                        raise e  # Re-raise the error to stop execution and see the traceback
                return power_rank

            # Calculate power rank for home and away data
            df.loc[:, 'power_rank'] = df.apply(lambda row: compute_power_rank(row, metrics), axis=1)

            # Ensure update_date is in datetime format
            df['update_date'] = pd.to_datetime(df['update_date'])

            # Determine the week number based on the Tuesday-to-Monday window
            df['week_number'] = (df['update_date'] - pd.Timedelta(days=1)).dt.isocalendar().week

            return df
        except Exception as e:
            logging.error(f"Error in calculate_power_rank function: {e}")
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
            logging.error(f"Error in normalize_data function: {e}")
            return None

    def insert_aggregated_data_into_database(self, aggregated_df,):
        """
        Inserts the aggregated and normalized data into a MongoDB collection.
        :param aggregated_df: A dataframe containing aggregated and normalized data
        :param database_operations: An instance of the DatabaseOperations class
        """
        try:
            # Drop the collection if it exists
            if self.WEEKLY_RANKS_DB_NAME in self.database_operations.db.list_collection_names():
                self.database_operations.db.weekly_ranks.drop()

            # Insert the aggregated data into the collection
            aggregated_df.reset_index(inplace=True, drop=True)
            self.database_operations.insert_data_into_mongodb(self.WEEKLY_RANKS_DB_NAME, aggregated_df.to_dict('records'))

            # Saving the report to a CSV file
            aggregated_df = aggregated_df.sort_values(by='normalized_power_rank', ascending=False)
            power_ranks_report_path = os.path.join(self.static_dir, 'power_ranks.csv')
            aggregated_df.to_csv(power_ranks_report_path)
            logging.info("Aggregated data inserted into MongoDB successfully.")
        except Exception as e:
            logging.error(f"Error inserting aggregated data into MongoDB: {e}")

    def fetch_team_ranking_metrics(self):
        """
        Fetch team ranking metrics from MongoDB and return as a DataFrame.

        Fetches the team ranking metrics from a MongoDB collection and returns them as a DataFrame.

        :return: DataFrame containing team ranking metrics.
        """
        client = MongoClient()
        db = client[self.database_name]
        collection = db[self.WEEKLY_RANKS_DB_NAME]
        df = pd.DataFrame(list(collection.find()))
        return df

    def generate_interactive_htmls(self, df, date=None):
        """
        Generate interactive HTML visualizations based on the DataFrame.

        Generates interactive line and bar charts based on the DataFrame data and saves them as HTML files.

        :param df: DataFrame containing team ranking metrics.
        :param date: The date for which to generate visualizations.
        """
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

            valid_dates = date_counts[date_counts >= 25].index
            recent_date = valid_dates.max()

        recent_df = df[df['update_date'] == recent_date]

        # Create the interactive bar chart
        fig = px.bar(recent_df, x='name', y='normalized_power_rank', title=f'Normalized Power Rank for {recent_date}')

        # Save the plot
        filename = 'normalized_power_ranks.html'
        fig.write_html(os.path.join(self.template_dir, filename))

        # Optionally, save the plot to an HTML file
        # fig.write_html("bar_chart.html")

    def main(self):
        """
        Main method for executing the power rank calculation and visualization pipeline.

        Orchestrates the entire process of loading, transforming, calculating power ranks, and generating visualizations.
        """
        processed_games_df = self.load_and_process_data()

        reload(scripts.constants)
        columns_to_filter = [col.replace('_difference', '').replace('_ratio', '').strip() for col in self.CONSTANTS if not col.startswith('odds.')]
        columns_to_filter.extend(['id', 'update_date', 'name'])  # Add 'id' and 'name' to the columns

        processed_games_df = processed_games_df[columns_to_filter]

        # Rearrange columns to place 'id' and 'name' at the beginning
        columns_ordered = ['id', 'update_date', 'name'] + [col for col in columns_to_filter if col not in ['id', 'update_date', 'name']]
        processed_games_df = processed_games_df[columns_ordered]

        if processed_games_df is not None:
            # Transform data
            # df_home, df_away = self.transform_data(processed_games_df, columns_to_filter)

            # Calculate power rank
            feature_importances = self.LOADED_MODEL.best_estimator_.feature_importances_
            feature_names = [col.replace('_difference', '').replace('_ratio', '') for col in self.CONSTANTS]  # <-- Corrected this line

            # Create a dictionary to store the modified feature names and their importances
            modified_importances = {}

            for name, importance in zip(feature_names, feature_importances):
                # If the modified name already exists in the dictionary, average the importances
                if name in modified_importances:
                    modified_importances[name] = (modified_importances[name] + importance) / 2
                else:
                    modified_importances[name] = importance

            weights = modified_importances  # Directly use the modified_importances dictionary as weights

            # Exclude 'id' and 'name' from the power rank calculations
            columns_for_power_rank = [col for col in columns_to_filter if col not in ['id', 'name', 'update_date']]
            df = self.calculate_power_rank(processed_games_df, columns_for_power_rank, weights)

            if df is not None:
                normalized_df = self.normalize_data(df)
                # Insert aggregated data into MongoDB
                self.insert_aggregated_data_into_database(normalized_df)
            else:
                logging.error("Error in calculating power rank. Exiting script.")
        else:
            logging.error("Error in data loading and processing. Exiting script.")

        # After inserting aggregated data into MongoDB
        df = self.fetch_team_ranking_metrics()
        self.generate_interactive_htmls(df)


if __name__ == "__main__":
    nfl_stats = StatsCalculator()
    nfl_stats.main()
