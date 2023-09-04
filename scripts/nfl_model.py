from ..classes.config_manager import ConfigManager
from pymongo import MongoClient
import requests
from datetime import datetime, timedelta
import os
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .constants import COLUMNS_TO_KEEP

config = ConfigManager()

# Get individual components
base_dir = config.get_config('default', 'base_dir')
data_dir = base_dir + config.get_config('paths', 'data_dir')
model_dir = base_dir + config.get_config('paths', 'model_dir')
database_name = config.get_config('database', 'database_name')

# MongoDB connection
MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "nfl_db"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]


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


def time_to_minutes(time_str):
    """Convert time string 'MM:SS' to minutes as a float."""
    minutes, seconds = map(int, time_str.split(':'))
    return minutes + seconds / 60


def train_and_evaluate(X_train, y_train, X_test, y_test, X_blind, y_blind):
    # Initialize and train the Linear Regression model
    model = RandomForestRegressor(n_estimators=100, random_state=108)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')

    # Load the blind test data
    df_blind = pd.read_csv(os.path.join(data_dir, 'blind_test_data.csv'))

    # Separate features and target
    X_blind = df_blind.drop(columns=['scoring_differential'])
    y_blind = df_blind['scoring_differential']

    # Predict using the trained model
    y_pred_blind = model.predict(X_blind)

    # Evaluate the model's performance
    mae_blind = mean_absolute_error(y_blind, y_pred_blind)
    mse_blind = mean_squared_error(y_blind, y_pred_blind)
    r2_blind = r2_score(y_blind, y_pred_blind)

    print(f'Mean Absolute Error on Blind Test Data: {mae_blind}')
    print(f'Mean Squared Error on Blind Test Data: {mse_blind}')
    print(f'R^2 Score on Blind Test Data: {r2_blind}')

    return model


def preprocess_nfl_data(df):
    """
    # Feature Selection
    numerical_features = ['rating', 'rush_plays', 'avg_pass_yards', 'attempts',
                          'tackles_made', 'redzone_attempts',
                          'redzone_successes', 'turnovers',
                          'sack_rate', 'play_diversity_ratio',
                          'turnover_margin', 'pass_success_rate',
                          'rush_success_rate'
                          ]
    categorical_features = ['opponent_team_id', 'home_or_away']

    # One-Hot Encode Categorical Features
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded_features = encoder.fit_transform(df[categorical_features])
    df_encoded = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    df = pd.concat([df.drop(columns=categorical_features), df_encoded], axis=1)
    encoded_columns = df.columns
    """

    # Drop rows with any NaN values
    df.dropna(inplace=True)

    # Normalize/Standardize Numerical Features
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)

    # Split the Data
    X = df.drop('scoring_differential', axis=1)  # Features
    y = df['scoring_differential']  # Target

    print(df)

    # Splitting into train and temp sets (80% train, 20% temp)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=108)

    # Splitting the temp set into test and blind_test sets (50% test, 50% blind_test)
    X_test, X_blind_test, y_test, y_blind_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=108)

    # Save the Blind Test Set to a File
    blind_test_data = pd.concat([X_blind_test, y_blind_test], axis=1)
    blind_test_data.to_csv(os.path.join(data_dir, 'blind_test_data.csv'), index=False)

    # Print the shape of the data
    print(f"Shape of the training data: {X_train.shape}")
    print(f"Shape of the test data: {X_test.shape}")
    print(f"Shape of the blind test data: {X_blind_test.shape}")

    # Print the first few rows of the training data for inspection
    print("\nFirst few rows of the training data:")
    print(X_train.head())

    return X_train, y_train, X_test, y_test, X_blind_test, y_blind_test, scaler, encoder, encoded_columns


# Usage
def main():
    # Load and process data
    processed_df = load_and_process_data()

    # Create a copy of the DataFrame
    df = processed_df.copy()

    # Drop columns that are not in the COLUMNS_TO_KEEP list
    df = df[COLUMNS_TO_KEEP]

    # Convert time strings to minutes (apply this to the relevant columns)
    df['statistics_home.summary.possession_time'] = df[
        'statistics_home.summary.possession_time'
    ].apply(time_to_minutes)
    df['statistics_away.summary.possession_time'] = df[
        'statistics_away.summary.possession_time'
    ].apply(time_to_minutes)

    X_train, y_train, X_test, y_test, X_blind_test, y_blind_test, scaler, encoder, encoded_columns = preprocess_nfl_data(df)
    model = train_and_evaluate(X_train, y_train, X_test, y_test, X_blind_test, y_blind_test)

    # Save the model and related files to the models directory
    joblib.dump(model, os.path.join(model_dir, 'trained_nfl_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'data_scaler.pkl'))
    joblib.dump(encoder, os.path.join(model_dir, 'data_encoder.pkl'))
    joblib.dump(encoded_columns, os.path.join(model_dir, 'encoded_columns.pkl'))

    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(importance_df)


if __name__ == "__main__":
    main()
