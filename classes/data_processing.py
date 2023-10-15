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

    def __init__(self):
        """
        Initializes the DataProcessing class.
        """
        self.logger = logging.getLogger(__name__)

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
                    flattened_df.columns = [f"{column}_{subcolumn}" for subcolumn in flattened_df.columns]
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
