import pandas as pd

class DataProcessing:
    def __init__(self):
        pass

    def flatten_and_merge_data(self, df):
        """Flatten the nested MongoDB data."""
        try:
            # Flatten each category and store in a list
            dataframes = []
            for column in df.columns:
                if isinstance(df[column][0], dict):
                    flattened_df = pd.json_normalize(df[column])
                    flattened_df.columns = [
                        f"{column}_{subcolumn}" for subcolumn in flattened_df.columns
                    ]
                    dataframes.append(flattened_df)

            # Merge flattened dataframes
            merged_df = pd.concat(dataframes, axis=1)
            return merged_df
        except Exception as e:
            print(f"Error flattening and merging data: {e}")
            # Handle the error appropriately (e.g., return the original DataFrame or an empty DataFrame)
            return pd.DataFrame()

    def time_to_minutes(self, time_str):
        """Convert time string 'MM:SS' to minutes as a float."""
        try:
            minutes, seconds = map(int, time_str.split(':'))
            return minutes + seconds / 60
        except ValueError:
            print(f"Invalid time format: {time_str}. Unable to convert to minutes.")
            return None  # or return a default value

    def calculate_scoring_differential(self, df):
        # Convert necessary columns to numeric types
        numeric_columns = ['summary_home.points', 'summary_away.points']
        df[numeric_columns] = df[numeric_columns].apply(
            pd.to_numeric, errors='coerce'
        )

        # Drop rows with missing values in eithers
        # 'summary_home_points' or 'summary_away_points'
        df.dropna(subset=numeric_columns, inplace=True)

        # Check if necessary columns are present and have numeric data types
        if all(col in df.columns and pd.api.types.is_numeric_dtype(
            df[col]
        ) for col in numeric_columns):
            df['scoring_differential'] = df[
                'summary_home.points'
            ] - df[
                'summary_away.points'
            ]
            print("Computed 'scoring_differential' successfully.")
        else:
            print(
                "Unable to compute due to unsuitable data types."
            )

        # Drop games if 'scoring_differential' key does not exist
        if 'scoring_differential' not in df.columns:
            print("'scoring_differential' key does not exist. Dropping games.")
            return pd.DataFrame()  # Return an empty dataframe
        else:
            return df

    def validate_data(self, df):
        """Implement data validation checks to ensure the data fetched from the database meets the expected format and structure."""
        # TODO: Implement data validation logic here
        pass

    def handle_data_types(self, df):
        """Handle different data types more efficiently during the flattening process."""
        # TODO: Implement data type handling logic here
        pass

# Usage example:
# data_processing = DataProcessing()
# data_processing.flatten_and_merge_data(df)
