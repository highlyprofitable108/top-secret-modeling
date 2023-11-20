import pandas as pd
import numpy as np
import logging
from datetime import timedelta
from importlib import reload
import scripts.constants
import scripts.all_columns


class DataProcessing:
    """
    Class for handling data processing tasks such as flattening nested data,
    converting time strings to minutes, collapsing dataframes, renaming columns,
    and filtering and scaling data.
    """

    def __init__(self, target_variable):
        """
        Initializes the DataProcessing class.

        Parameters:
        target_variable (str): The target variable name used in data processing.
        """
        self.logger = logging.getLogger(__name__)
        self.TARGET_VARIABLE = target_variable

    def flatten_and_merge_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flattens nested MongoDB data into a pandas DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame with potentially nested data.

        Returns:
            pd.DataFrame: The flattened and merged DataFrame, or an empty DataFrame in case of an error.
        """
        try:
            dataframes = []
            for column in df.columns:
                if isinstance(df[column][0], dict):
                    flattened_df = pd.json_normalize(df[column])
                    flattened_df.columns = [f"{column}.{subcolumn}" for subcolumn in flattened_df.columns]
                    dataframes.append(flattened_df)
                else:
                    dataframes.append(df[[column]])
            merged_df = pd.concat(dataframes, axis=1)
            return merged_df
        except Exception as e:
            self.logger.error(f"Error flattening and merging data: {e}")
            return pd.DataFrame()

    def collapse_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Collapses a DataFrame with columns containing '.' into a DataFrame with nested dictionaries.

        Args:
            df (pd.DataFrame): The DataFrame to collapse.

        Returns:
            pd.DataFrame: Collapsed DataFrame with nested dictionaries.
        """
        collapsed_data = []
        for _, row in df.iterrows():
            nested_data = {}
            for col, value in row.items():
                keys = col.split('.')
                d = nested_data
                for key in keys[:-1]:
                    d = d.setdefault(key, {})
                d[keys[-1]] = value
            collapsed_data.append(nested_data)
        return pd.DataFrame(collapsed_data)

    def rename_columns(self, data, prefix):
        """
        Renames columns of a DataFrame with a specified prefix.

        Args:
            data (pd.DataFrame): The DataFrame whose columns need to be renamed.
            prefix (str): The prefix to prepend to column names.

        Returns:
            pd.DataFrame: DataFrame with renamed columns.
        """
        reload(scripts.constants)
        columns_to_rename = [col.replace('ranks_home_', '') for col in scripts.constants.COLUMNS_TO_KEEP if col.startswith('ranks_home_')]
        rename_dict = {col: f"{prefix}{col}" for col in columns_to_rename}
        return data.rename(columns=rename_dict)

    def filter_and_scale_data(self, merged_data, date):
        """
        Filters and scales data based on feature columns and a specific date.

        Args:
            merged_data (pd.DataFrame): The DataFrame to be filtered and scaled.
            date (datetime): The date to filter the DataFrame on.

        Returns:
            pd.DataFrame: Filtered and scaled DataFrame.
        """
        # Reload constants
        reload(scripts.constants)

        # Filter the DataFrame
        filtered_data = merged_data[merged_data['update_date'] == date]

        # Filter columns excluding TARGET_VARIABLE
        columns_to_filter = [col for col in scripts.constants.COLUMNS_TO_KEEP if col.strip() in map(str.strip, filtered_data.columns) and col.strip() != self.TARGET_VARIABLE]

        # Scaling the data
        filtered_data = filtered_data[columns_to_filter]
        if hasattr(self, 'LOADED_SCALER'):
            filtered_data[columns_to_filter] = self.LOADED_SCALER.transform(filtered_data[columns_to_filter])
        else:
            self.logger.warning("LOADED_SCALER not found. Data will not be scaled.")
        return filtered_data

    def get_team_data(self, df, team, date):
        """
        Fetches data for a specific team based on the alias from the provided DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing team data.
            team (str): The team name or alias.
            date (str): The cutoff date for filtering the data.

        Returns:
            pd.DataFrame: The row from the DataFrame with the most recent data for the specified team.
        """
        df['update_date'] = pd.to_datetime(df['update_date']).dt.strftime('%Y-%m-%d')
        condition = df['update_date'] <= date
        filtered_df = df[(df['name'].str.lower() == team.lower()) & condition]
        idx = filtered_df['update_date'].idxmax()
        return df.loc[[idx]]

    def replace_team_name(self, team_name):
        """
        Replaces old team names with their current names.

        Args:
            team_name (str): The original team name.

        Returns:
            str: The current team name.
        """
        if team_name in ["Redskins", "Football Team"]:
            return "Commanders"
        return team_name

    def get_standard_deviation(self, df):
        """
        Computes the standard deviation for numeric columns in a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to compute standard deviation for.

        Returns:
            pd.DataFrame: A DataFrame containing the standard deviations.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        standard_deviation_df = numeric_df.std().to_frame().transpose()
        return standard_deviation_df

    def prepare_data(self, df, features, home_team, away_team, date):
        """
        Prepares data for simulation by calculating differences and ratios.

        Args:
            df (pd.DataFrame): The DataFrame containing team data.
            features (list): List of features to process.
            home_team (str): Home team name.
            away_team (str): Away team name.
            date (str): The date for filtering data.

        Returns:
            pd.DataFrame: DataFrame prepared for simulation.
        """
        # Create a mapping of base column names to their corresponding operations (difference or ratio)
        feature_operations = {col.rsplit('_', 1)[0]: col.rsplit('_', 1)[1] for col in features}

        # Extract unique base column names from the features list
        base_column_names = set(col.rsplit('_', 1)[0] for col in features)

        # Identify columns in the DataFrame that match the base column names
        related_columns = [col for col in df.columns if any(base_col in col for base_col in base_column_names)]

        # Fetch data for home and away teams for the given date
        home_data = self.get_team_data(df, home_team, date)[related_columns].reset_index(drop=True)
        away_data = self.get_team_data(df, away_team, date)[related_columns].reset_index(drop=True)

        # Extract columns representing standard deviations
        home_stddev = home_data.filter(regex='_stddev$').copy()
        away_stddev = away_data.filter(regex='_stddev$').copy()

        # Remove standard deviation columns for subsequent operations
        home_features = home_data.drop(home_stddev.columns, axis=1)
        away_features = away_data.drop(away_stddev.columns, axis=1)

        new_columns = []
        for col, operation in feature_operations.items():
            new_col_name = col + "_difference" if operation == "difference" else col + "_ratio"
            # Compute the difference or ratio as specified in feature_operations
            new_series = pd.Series(home_features[col] - away_features[col], name=new_col_name) if operation == "difference" else pd.Series(home_features[col] / away_features[col], name=new_col_name)
            new_columns.append(new_series)

        # Combine the newly computed columns into a single DataFrame
        game_prediction_df = pd.concat(new_columns, axis=1)

        # Replace infinite values with median to avoid calculation issues
        game_prediction_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        game_prediction_df.fillna(game_prediction_df.median(), inplace=True)

        # Identify common columns in home and away stddev, and compute their absolute differences
        common_stddev_columns = home_stddev.columns.intersection(away_stddev.columns)
        stddev_difference = abs(home_stddev[common_stddev_columns] - away_stddev[common_stddev_columns])

        # Append the stddev differences to the main DataFrame
        game_prediction_df = pd.concat([game_prediction_df, stddev_difference], axis=1)

        return game_prediction_df

    def cleanup_ranks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reformats columns starting with 'ranks_' to have a nested dictionary structure.

        Args:
            df (pd.DataFrame): The DataFrame to be cleaned.

        Returns:
            pd.DataFrame: The DataFrame with cleaned 'ranks_' columns.
        """
        cleaned_data = []

        for _, row in df.iterrows():
            cleaned_row = {}

            for col, value in row.items():
                if col.startswith("ranks_") and isinstance(value, dict):
                    # For 'ranks_*' columns with nested dictionaries
                    nested_data = {}

                    for sub_col, sub_value in value.items():
                        # Check if sub-column name contains a period
                        if '.' in sub_col:
                            keys = sub_col.split('.')
                            # Nest sub-columns under 'type' if they have a hierarchy
                            type_key = keys[0]
                            sub_key = '.'.join(keys[1:])

                            if type_key not in nested_data:
                                nested_data[type_key] = {}

                            nested_data[type_key][sub_key] = sub_value
                        else:
                            nested_data[sub_col] = sub_value

                    cleaned_row[col] = nested_data
                else:
                    cleaned_row[col] = value

            cleaned_data.append(cleaned_row)

        return pd.DataFrame(cleaned_data)

    def time_to_minutes(self, time_str: str) -> float:
        """
        Converts time in 'HH:MM:SS' format to minutes.

        Args:
            time_str (str): The time string in 'HH:MM:SS' format.

        Returns:
            float: Time in minutes, or None if invalid format.
        """
        if pd.isna(time_str):
            return None

        try:
            # Splitting the time string and converting to minutes
            hours, minutes, seconds = map(int, time_str.split(':'))
            return hours * 60 + minutes + seconds / 60
        except ValueError:
            self.logger.error(f"Invalid time format: {time_str}. Unable to convert to minutes.")
            return None

    def generate_weekdays_list(self, start_date, end_date, weekday=2, excluded_months=list(range(3, 9))):
        """
        Generates a list of specific weekdays within a date range, excluding certain months.

        Args:
            start_date (datetime): The starting date of the range.
            end_date (datetime): The ending date of the range.
            weekday (int): The desired weekday (0=Monday, 1=Tuesday, ..., 6=Sunday).
            excluded_months (list): Months to exclude from the range.

        Returns:
            List[datetime]: List of dates representing the desired weekdays.
        """
        current_date = start_date
        weekdays_list = []

        # Adjust to the next desired weekday
        days_to_next_weekday = (weekday - current_date.weekday() + 7) % 7
        current_date += timedelta(days=days_to_next_weekday)

        # Loop to generate list of weekdays, excluding specified months
        while current_date <= end_date:
            if current_date.month not in excluded_months:
                weekdays_list.append(current_date)
            current_date += timedelta(weeks=1)

        return weekdays_list

    def calculate_scoring_differential(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the scoring differential between home and away teams.

        Args:
            df (pd.DataFrame): DataFrame with home and away team points.

        Returns:
            pd.DataFrame: Updated DataFrame with a new column for scoring differential.
        """
        numeric_columns = ['summary_home.points', 'summary_away.points']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        df.dropna(subset=numeric_columns, inplace=True)

        # Ensure columns are numeric before calculating differential
        if all(col in df.columns and pd.api.types.is_numeric_dtype(df[col]) for col in numeric_columns):
            df['scoring_differential'] = df['summary_home.points'] - df['summary_away.points']
            self.logger.info("Computed 'scoring_differential' successfully.")
        else:
            self.logger.error("Unable to compute due to unsuitable data types.")

        if 'scoring_differential' not in df.columns:
            self.logger.error("'scoring_differential' key does not exist. Dropping games.")
            return pd.DataFrame()
        else:
            return df

    def handle_data_types(self, df: pd.DataFrame) -> None:
        """
        Placeholder method for handling different data types during data processing.

        Args:
            df (pd.DataFrame): The DataFrame with various data types.

        Returns:
            None
        """
        # Placeholder for future implementation of data type handling logic
        pass

    def handle_null_values(self, df):
        """
        Handles null values in the DataFrame by dropping or imputing.

        Args:
            df (pd.DataFrame): DataFrame with potential null values.

        Returns:
            pd.DataFrame: DataFrame with null values handled.
        """
        try:
            if 'scheduled' in df.columns:
                # Convert 'scheduled' to datetime and remove future dates
                df['scheduled'] = pd.to_datetime(df['scheduled'])
                df['scheduled'] = df['scheduled'].dt.tz_localize(None)
                df = df[df['scheduled'] <= pd.Timestamp.now()]

            # Identify columns with high NaN count to drop
            nan_counts = df.isnull().sum()
            columns_to_drop = [col for col in nan_counts[nan_counts > 100].index.tolist() if 'odds' not in col]
            if columns_to_drop:
                logging.warning(f"Dropping columns with more than 100 NaN values: {columns_to_drop}")
                df = df.drop(columns=columns_to_drop).reset_index(drop=True)
                # Consider updating constants file here to reflect the changes

            # Impute NaN values for remaining columns
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].mode().iloc[0], inplace=True)

            return df
        except Exception as e:
            logging.error(f"Error in handle_null_values: {e}")
            return df

    def handle_duplicate_columns(self, ranks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Resolves issues with duplicate column names in the DataFrame.

        Args:
            ranks_df (pd.DataFrame): DataFrame with potential duplicate columns.

        Returns:
            pd.DataFrame: DataFrame with duplicates resolved.
        """
        # Identify and process duplicate columns
        is_duplicate = ranks_df.columns.duplicated(keep=False)
        duplicated_cols = ranks_df.columns[is_duplicate].unique()

        for col in duplicated_cols:
            if isinstance(ranks_df[col], pd.DataFrame):
                # Example strategy: take the last column if duplicates are DataFrames
                ranks_df.loc[:, col] = ranks_df[col].iloc[:, -1]

            # Drop duplicate columns except for the last occurrence
            col_locs = ranks_df.columns.get_loc(col)
            if isinstance(col_locs, slice):
                col_locs = list(range(col_locs.start, col_locs.stop))
            if isinstance(col_locs, list) and len(col_locs) > 1:
                ranks_df = ranks_df.drop(ranks_df.columns[col_locs[:-1]], axis=1)

        ranks_df = ranks_df.reset_index(drop=True)

        # Log any remaining duplicates
        remaining_duplicates = ranks_df.columns[ranks_df.columns.duplicated()]
        if len(remaining_duplicates) > 0:
            logging.info(f"Remaining duplicate columns: {remaining_duplicates}")

        return ranks_df

    def update_columns_file(self, columns_to_remove):
        """
        Updates the all_columns.py file to remove specified columns.

        Args:
            columns_to_remove (list): List of columns to be removed from the file.
        """
        # Reload the current columns file to ensure up-to-date data
        reload(scripts.all_columns)

        # Read the current file contents
        with open('./scripts/all_columns.py', 'r') as file:
            lines = file.readlines()

        # Exclude lines containing columns to remove
        new_lines = [line for line in lines if not any(col in line for col in columns_to_remove)]

        # Write the updated lines back to the file
        with open('./scripts/all_columns.py', 'w') as file:
            file.writelines(new_lines)

        # Reload the updated columns file to reflect changes
        reload(scripts.all_columns)

    def update_constants_file(self, columns_to_remove):
        """
        Updates the constants.py file to remove specified columns.

        Args:
            columns_to_remove (list): List of columns to be removed from the file.
        """
        # Reload the current constants file
        reload(scripts.constants)

        # Read the current file contents
        with open('./scripts/constants.py', 'r') as file:
            lines = file.readlines()

        # Exclude lines containing columns to remove
        new_lines = [line for line in lines if not any(col in line for col in columns_to_remove)]

        # Write the updated lines back to the file
        with open('./scripts/constants.py', 'w') as file:
            file.writelines(new_lines)

        # Reload the updated constants file to reflect changes
        reload(scripts.constants)

    def handle_prediction_values(self, df):
        """
        Handles NaN values in DataFrame for prediction purposes.

        Args:
            df (pd.DataFrame): DataFrame with prediction values.

        Returns:
            pd.DataFrame: DataFrame with NaN values handled.
        """
        # Reload constants to ensure up-to-date column information
        reload(scripts.constants)

        try:
            # Drop columns with high NaN count and impute remaining NaN values
            nan_counts = df.isnull().sum()
            columns_to_drop = nan_counts[nan_counts > 1].index.tolist()
            if columns_to_drop:
                logging.warning(f"Dropping columns with more NaN values: {columns_to_drop}")
                df = df.drop(columns=columns_to_drop).reset_index(drop=True)

            # Fill NaN values in remaining columns
            for col in df.columns:
                if np.issubdtype(df[col].dtype, np.number):
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].mode().iloc[0], inplace=True)

            return df
        except Exception as e:
            logging.error(f"Error in handle_prediction_values: {e}")
            return df

    def process_game_data(self, df):
        """
        Processes game data by flattening nested structures and calculating scoring differentials.

        Args:
            df (pd.DataFrame): DataFrame with game data.

        Returns:
            pd.DataFrame: Processed DataFrame.
        """
        processed_df = self.flatten_and_merge_data(df)
        processed_df = self.calculate_scoring_differential(processed_df)
        return processed_df

    def process_team_data(self, df):
        """
        Processes team data by flattening nested structures.

        Args:
            df (pd.DataFrame): DataFrame with team data.

        Returns:
            pd.DataFrame: Flattened DataFrame.
        """
        return self.flatten_and_merge_data(df)

    def convert_list_columns_to_string(self, df):
        """
        Converts DataFrame columns that contain lists to string type.

        Args:
            df (pd.DataFrame): DataFrame with potential list columns.

        Returns:
            pd.DataFrame: DataFrame with list columns converted to strings.
        """
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, list)).any():
                df[col] = df[col].astype(str)
        return df

    def validate_data(self, df):
        """
        Validates the DataFrame to ensure certain key columns exist.

        Args:
            df (pd.DataFrame): DataFrame to be validated.

        Returns:
            pd.DataFrame: The original DataFrame if valid, otherwise an empty DataFrame.
        """
        if 'scoring_differential' not in df.columns:
            logging.info("'scoring_differential' key does not exist. Dropping games.")
            return pd.DataFrame()  # Return an empty dataframe
        else:
            return df
