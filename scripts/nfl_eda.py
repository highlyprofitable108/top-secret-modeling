import warnings
import pandas as pd
import sweetviz as sv
from pymongo import MongoClient
from constants import COLUMNS_TO_KEEP

# Suppress warning about DataFrame fragmentation
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

# MongoDB connection details
MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "nfl_db"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

TARGET_VARIABLE = 'scoring_differential'


def fetch_data_from_mongodb(collection_name):
    """Fetch data from a MongoDB collection."""
    cursor = db[collection_name].find()
    df = pd.DataFrame(list(cursor))
    return df


def flatten_and_merge_data(df):
    """Flatten the nested MongoDB data."""
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


def load_and_process_data():
    """Load data from MongoDB and process it."""
    df = fetch_data_from_mongodb("games")
    processed_df = flatten_and_merge_data(df)

    # Convert columns with list values to string
    for col in processed_df.columns:
        if processed_df[col].apply(type).eq(list).any():
            processed_df[col] = processed_df[col].astype(str)

    # Convert necessary columns to numeric types
    numeric_columns = ['summary_home.points', 'summary_away.points']
    processed_df[numeric_columns] = processed_df[numeric_columns].apply(
        pd.to_numeric, errors='coerce'
    )

    # Drop rows with missing values in eithers
    # 'summary_home_points' or 'summary_away_points'
    processed_df.dropna(subset=numeric_columns, inplace=True)

    # Check if necessary columns are present and have numeric data types
    if all(col in processed_df.columns and pd.api.types.is_numeric_dtype(
        processed_df[col]
    ) for col in numeric_columns):
        processed_df['scoring_differential'] = processed_df[
            'summary_home.points'
        ] - processed_df[
            'summary_away.points'
        ]
        print("Computed 'scoring_differential' successfully.")
    else:
        print(
            "Unable to compute due to unsuitable data types."
        )

    # Drop games if 'scoring_differential' key does not exist
    if 'scoring_differential' not in processed_df.columns:
        print("'scoring_differential' key does not exist. Dropping games.")
        return pd.DataFrame()  # Return an empty dataframe
    else:
        return processed_df


def main():
    # Load and process data
    processed_df = load_and_process_data()

    # Create a copy of the DataFrame
    df = processed_df.copy()

    # Drop columns that are not in the COLUMNS_TO_KEEP list
    df = df[COLUMNS_TO_KEEP]

    # Calculate the report
    report = sv.analyze(
        df, target_feat=TARGET_VARIABLE, pairwise_analysis='on'
    )

    # Show the report
    report.show_html('./data/nfl_eda_report.html')


if __name__ == "__main__":
    main()
