import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from importlib import reload
import scripts.constants
import scripts.all_columns


class DataProcessing:
    """
    A class to handle data processing tasks such as flattening nested data,
    converting time strings to minutes, and calculating scoring differentials.
    """

    def __init__(self, target_variable):
        """
        Initializes the DataProcessing class.
        """
        self.logger = logging.getLogger(__name__)
        self.TARGET_VARIABLE = target_variable

    def flatten_and_merge_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten the nested MongoDB data.

        Args:
            df (pd.DataFrame): The input data frame with nested data.

        Returns:
            pd.DataFrame: The flattened and merged data frame.
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
        Convert a dataframe with columns containing '.' into a dataframe with nested dictionaries.
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
        """Renames columns with a specified prefix."""
        reload(scripts.constants)
        columns_to_rename = [col.replace('ranks_home_', '') for col in scripts.constants.COLUMNS_TO_KEEP if col.startswith('ranks_home_')]
        rename_dict = {col: f"{prefix}{col}" for col in columns_to_rename}
        return data.rename(columns=rename_dict)

    def filter_and_scale_data(self, merged_data, date):
        """Filters the merged data using the feature columns and scales the numeric features."""
        # Reload constants
        reload(scripts.constants)

        # Filter the DataFrame
        filtered_data = merged_data[merged_data['update_date'] == date]

        # Filter columns with stripping whitespaces and exclude TARGET_VARIABLE
        columns_to_filter = [col for col in scripts.constants.COLUMNS_TO_KEEP if col.strip() in map(str.strip, filtered_data.columns) and col.strip() != self.TARGET_VARIABLE]

        filtered_data = filtered_data[columns_to_filter]
        filtered_data[columns_to_filter] = self.LOADED_SCALER.transform(filtered_data[columns_to_filter])
        return filtered_data

    def get_team_data(self, df, team, date):
        """Fetches data for a specific team based on the alias from the provided DataFrame."""
        # Convert the 'update_date' column to a datetime object and extract only the "yyyy-mm-dd" part
        df['update_date'] = pd.to_datetime(df['update_date']).dt.strftime('%Y-%m-%d')
        condition = df['update_date'] <= date

        # Filter the DataFrame based on the team name and the date condition
        filtered_df = df[(df['name'].str.lower() == team.lower()) & condition]
        # Get the index of the row with the most recent update_date
        idx = filtered_df['update_date'].idxmax()

        # Return the row with the most recent update_date
        return df.loc[[idx]]

    def replace_team_name(self, team_name):
        if team_name in ["Redskins", "Football Team"]:
            return "Commanders"
        return team_name

    def get_standard_deviation(self, df):
        """Fetches the standard deviation for each column in the provided DataFrame."""
        # Exclude non-numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        # Compute the standard deviation for each numeric column
        standard_deviation_df = numeric_df.std().to_frame().transpose()

        # Return the standard deviation DataFrame
        return standard_deviation_df

    def prepare_data(self, df, features, home_team, away_team, date):
        """Prepare data for simulation."""

        # Create a dictionary with column names as keys and arithmetic operations as values
        feature_operations = {col.rsplit('_', 1)[0]: col.rsplit('_', 1)[1] for col in features}

        # Extract unique base column names from the features list
        base_column_names = set(col.rsplit('_', 1)[0] for col in features)

        # Filter the home_team_data and away_team_data DataFrames to retain only the necessary columns
        home_features = self.get_team_data(df, home_team, date)[list(base_column_names.intersection(df.columns))].reset_index(drop=True)
        away_features = self.get_team_data(df, away_team, date)[list(base_column_names.intersection(df.columns))].reset_index(drop=True)

        # Initialize an empty DataFrame for the results
        game_prediction_df = pd.DataFrame()

        # Iterate over the columns using the dictionary
        for col, operation in feature_operations.items():
            if operation == "difference":
                game_prediction_df[col + "_difference"] = home_features[col] - away_features[col]
            else:
                game_prediction_df[col + "_ratio"] = home_features[col] / away_features[col]

        # Handle potential division by zero issues
        game_prediction_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return game_prediction_df

    def cleanup_ranks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert a dataframe that has ranks_* > type.stats to ranks_* > type > stats
        """
        cleaned_data = []

        for _, row in df.iterrows():
            cleaned_row = {}

            for col, value in row.items():
                if col.startswith("ranks_") and isinstance(value, dict):
                    # Process 'ranks_*' columns with nested dictionaries
                    nested_data = {}

                    for sub_col, sub_value in value.items():
                        if '.' in sub_col:
                            # Split the sub-column name by dot ('.') to get the hierarchy levels
                            keys = sub_col.split('.')

                            if len(keys) > 1:
                                # If the sub-column contains a period, nest it under 'type'
                                type_key = keys[0]
                                sub_key = '.'.join(keys[1:])

                                if type_key not in nested_data:
                                    nested_data[type_key] = {}

                                # Access the 'type' dictionary and set the sub-values
                                sub_nested_data = nested_data[type_key]
                                sub_nested_data[sub_key] = sub_value
                            else:
                                # If there is no period in the sub-column, keep it as-is
                                nested_data[sub_col] = sub_value
                        else:
                            # Keep non-'.' sub-columns as-is
                            nested_data[sub_col] = sub_value

                    cleaned_row[col] = nested_data
                else:
                    # If the column doesn't match the criteria, copy it as-is
                    cleaned_row[col] = value

            cleaned_data.append(cleaned_row)

        # Create a new DataFrame from the cleaned data
        cleaned_df = pd.DataFrame(cleaned_data)

        return cleaned_df

    def time_to_minutes(self, time_str: str) -> float:
        """
        Convert time string 'MM:SS' to minutes as a float.

        Args:
            time_str (str): The time string in 'MM:SS' format.

        Returns:
            float: The time in minutes.
        """
        if pd.isna(time_str):
            return None

        try:
            minutes, seconds = map(int, time_str.split(':'))
            return minutes + seconds / 60
        except ValueError:
            self.logger.error(f"Invalid time format: {time_str}. Unable to convert to minutes.")
            return None

    def generate_weekdays_list(self, start_date, end_date, weekday=1, excluded_months=list(range(3, 9))):
        """
        Generate a list of specific weekdays between start_date and end_date.

        Parameters:
        - start_date (datetime): The start date.
        - end_date (datetime): The end date.
        - weekday (int): The desired weekday (0=Monday, 1=Tuesday, ..., 6=Sunday).
        - excluded_months (list): List of months to exclude.

        Returns:
        - List[datetime]: List of desired weekdays.
        """
        current_date = start_date
        weekdays_list = []

        # Adjust the start date to the next desired weekday
        days_to_next_weekday = (weekday - current_date.weekday() + 7) % 7
        current_date += timedelta(days=days_to_next_weekday)

        # Append desired weekdays to the list until the end date, skipping excluded months
        while current_date <= end_date:
            if current_date.month not in excluded_months:
                weekdays_list.append(current_date)
            current_date += timedelta(weeks=1)

        return weekdays_list

    def calculate_scoring_differential(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the scoring differential between home and away points.

        Args:
            df (pd.DataFrame): The input data frame with score data.

        Returns:
            pd.DataFrame: The data frame with the calculated scoring differential.
        """
        numeric_columns = ['summary_home.points', 'summary_away.points']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        df.dropna(subset=numeric_columns, inplace=True)

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
        Handle different data types more efficiently during the flattening process.

        Args:
            df (pd.DataFrame): The input data frame with various data types.

        Returns:
            None
        """
        # TODO: Implement data type handling logic here
        pass

    def handle_null_values(self, df):
        """Handles null values in the dataframe by dropping columns with high NaN count and filling others with mean."""
        try:
            if 'scheduled' in df.columns:
                # Ensure 'scheduled' column is in datetime format
                df['scheduled'] = pd.to_datetime(df['scheduled'])
                df['scheduled'] = df['scheduled'].dt.tz_localize(None)

                # Drop rows where 'scheduled' is greater than the current date and time
                df = df[df['scheduled'] <= pd.Timestamp.now()]

            nan_counts = df.isnull().sum()
            columns_to_drop = [col for col in nan_counts[nan_counts > 100].index.tolist() if 'odds' not in col]
            if columns_to_drop:
                logging.warning(f"Dropping columns with more than 100 NaN values: {columns_to_drop}")
                df = df.drop(columns=columns_to_drop).reset_index(drop=True)

                # Update constants.py to remove dropped columns
                self.update_constants_file(columns_to_drop)
                self.update_columns_file(columns_to_drop)

            # Fill NaN values in remaining columns with the mean of each column
            nan_columns = nan_counts[nan_counts > 0].index.tolist()
            nan_columns = [col for col in nan_columns if col not in columns_to_drop]
            for col in nan_columns:
                if df[col].dtype == 'float64' or df[col].dtype == 'int64':  # Check if the column has a numeric data type
                    col_mean = df[col].mean()
                    df[col].fillna(col_mean, inplace=True)
                else:
                    col_most_frequent = df[col].mode().iloc[0]  # Fill non-numeric columns with the mode
                    df[col].fillna(col_most_frequent, inplace=True)

            return df
        except Exception as e:
            logging.error(f"Error in handle_null_values: {e}")
            return df

    def handle_duplicate_columns(self, ranks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle duplicate columns in the provided DataFrame.

        :param ranks_df: DataFrame with potential duplicate columns.
        :return: DataFrame with duplicate columns handled.
        """
        # Identify duplicate columns
        is_duplicate = ranks_df.columns.duplicated(keep=False)
        duplicated_cols = ranks_df.columns[is_duplicate].unique()

        for col in duplicated_cols:
            # If the column data is a DataFrame (multiple columns with the same name)
            if isinstance(ranks_df[col], pd.DataFrame):
                # Combine the data from all duplicate columns (e.g., take the mean, sum, or any other operation)
                # Here, I'm taking the last column as an example
                ranks_df.loc[:, col] = ranks_df[col].iloc[:, -1]

            # Drop all but the last occurrence of the duplicated column
            col_locs = ranks_df.columns.get_loc(col)
            if isinstance(col_locs, slice):
                col_locs = list(range(col_locs.start, col_locs.stop))
            if isinstance(col_locs, list) and len(col_locs) > 1:
                ranks_df = ranks_df.drop(ranks_df.columns[col_locs[:-1]], axis=1)

        # De-fragment the DataFrame
        ranks_df = ranks_df.reset_index(drop=True)

        # Check for any remaining duplicates
        remaining_duplicates = ranks_df.columns[ranks_df.columns.duplicated()]
        if len(remaining_duplicates) > 0:
            print(f"Remaining duplicate columns: {remaining_duplicates}")

        return ranks_df

    def update_columns_file(self, columns_to_remove):
        """Update the constants.py file to remove specified columns."""
        reload(scripts.all_columns)

        with open('./scripts/all_columns.py', 'r') as file:
            lines = file.readlines()

        # Remove lines containing columns to remove
        new_lines = [line for line in lines if not any(col in line for col in columns_to_remove)]

        with open('./scripts/all_columns.py', 'w') as file:
            file.writelines(new_lines)

        reload(scripts.all_columns)

    def update_constants_file(self, columns_to_remove):
        """Update the constants.py file to remove specified columns."""
        reload(scripts.constants)

        with open('./scripts/constants.py', 'r') as file:
            lines = file.readlines()

        # Remove lines containing columns to remove
        new_lines = [line for line in lines if not any(col in line for col in columns_to_remove)]

        with open('./scripts/constants.py', 'w') as file:
            file.writelines(new_lines)

        reload(scripts.constants)

    def handle_prediction_values(self, df):
        """Handles prediction values in the dataframe by dropping columns with high NaN count and filling others with mean."""
        reload(scripts.constants)

        try:
            nan_counts = df.isnull().sum()
            columns_to_drop = nan_counts[nan_counts > 1].index.tolist()
            if columns_to_drop:
                logging.warning(f"Dropping columns with more NaN values: {columns_to_drop}")
                df = df.drop(columns=columns_to_drop).reset_index(drop=True)

            # Fill NaN values in remaining columns with the mean of each column
            nan_columns = nan_counts[nan_counts > 0].index.tolist()
            nan_columns = [col for col in nan_columns if col not in columns_to_drop]
            for col in nan_columns:
                if np.issubdtype(df[col].dtype, np.number):  # Check if the column has a numeric data type
                    col_mean = df[col].mean()
                    df[col].fillna(col_mean, inplace=True)
                else:
                    col_most_frequent = df[col].mode().iloc[0]  # Fill non-numeric columns with the mode
                    df[col].fillna(col_most_frequent, inplace=True)

            return df
        except Exception as e:
            logging.error(f"Error in handle_prediction_values: {e}")
            return df

    def process_game_data(self, df):
        processed_df = self.flatten_and_merge_data(df)
        processed_df = self.calculate_scoring_differential(processed_df)
        return processed_df

    def process_team_data(self, df):
        return self.flatten_and_merge_data(df)

    def convert_list_columns_to_string(self, df):
        for col in df.columns:
            if df[col].apply(type).eq(list).any():
                df[col] = df[col].astype(str)
        return df

    def validate_data(self, df):
        if 'scoring_differential' not in df.columns:
            print("'scoring_differential' key does not exist. Dropping games.")
            return pd.DataFrame()  # Return an empty dataframe
        else:
            return df
