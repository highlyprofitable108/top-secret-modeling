from classes.config_manager import ConfigManager
from .constants import COLUMNS_TO_KEEP
import os
import joblib
import pandas as pd
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# TODO: 1. Implement error handling to manage potential issues during database connections or data processing.
# TODO: 2. Introduce logging to keep track of the operations and potential errors during the script execution.

# Configuration and Database Connection
config = ConfigManager()
base_dir = config.get_config('default', 'base_dir')
data_dir = base_dir + config.get_config('paths', 'data_dir')
model_dir = base_dir + config.get_config('paths', 'model_dir')
database_name = config.get_config('database', 'database_name')
MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "nfl_db"
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

# TODO: 3. Implement data validation checks to ensure the data fetched from the database meets the expected format and structure.
# TODO: 4. Consider adding functionality to handle different data types more efficiently during the flattening process.


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

# TODO: 5. Implement more robust error handling and data validation to ensure the 'scoring_differential' is computed correctly.
# TODO: 6. Consider adding a logging mechanism to track the data processing steps and potential issues.


def time_to_minutes(time_str):
    """Convert time string 'MM:SS' to minutes as a float."""
    minutes, seconds = map(int, time_str.split(':'))
    return minutes + seconds / 60

# TODO: 7. Consider adding error handling to manage incorrect time formats.


def preprocess_nfl_data(df):
    # TODO: Revisit feature selection and encoding section
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

    # Separate the target variable
    X = df.drop('scoring_differential', axis=1)  # Features
    y = df['scoring_differential']  # Target

    # Check for NaN values and handle them before splitting
    if X.isnull().sum().sum() > 0:
        # TODO: Handle NaN values appropriately here
        pass

    # Split the Data
    # TODO: Consider implementing stratified split here to maintain the distribution of the target variable in both train and test sets

    # Splitting into train and temp sets (80% train, 20% temp)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=108)

    # Splitting the temp set into test and blind_test sets (50% test, 50% blind_test)
    X_test, X_blind_test, y_test, y_blind_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=108)

    # Apply scaling after splitting the data to prevent data leakage
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_blind_test = scaler.transform(X_blind_test)

    # TODO: Implement cross-validation here for a more robust evaluation of the model

    # Save the Blind Test Set to a File
    # Convert numpy arrays back to dataframes before concatenating
    X_blind_test = pd.DataFrame(X_blind_test, columns=df.columns.drop('scoring_differential'))
    blind_test_data = pd.concat([X_blind_test, y_blind_test.reset_index(drop=True)], axis=1)
    blind_test_data.to_csv(os.path.join(data_dir, 'blind_test_data.csv'), index=False)

    # Print the shape of the data
    print(f"Shape of the training data: {X_train.shape}")
    print(f"Shape of the test data: {X_test.shape}")
    print(f"Shape of the blind test data: {X_blind_test.shape}")

    # Print the first few rows of the training data for inspection
    print("\nFirst few rows of the training data:")
    X_train = pd.DataFrame(X_train, columns=df.columns.drop('scoring_differential'))
    print(X_train.head())

    return X_train, y_train, X_test, y_test, X_blind_test, y_blind_test, scaler
    # encoder, encoded_columns


def train_and_evaluate(X_train, y_train, X_test, y_test, X_blind_test, y_blind_test):
    """
    This function trains the model on the training data and evaluates it on the test data and blind test data.
    """
    # Convert numpy arrays back to dataframes to preserve feature names
    feature_columns = [col for col in COLUMNS_TO_KEEP if col != 'scoring_differential']   
    
    X_train_df = pd.DataFrame(X_train, columns=feature_columns)
    X_test_df = pd.DataFrame(X_test, columns=feature_columns)
    X_blind_test_df = pd.DataFrame(X_blind_test, columns=feature_columns)

    # Initialize the model
    model = RandomForestRegressor(random_state=108)

    # Train the model
    model.fit(X_train_df, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test_df)

    # Calculate performance metrics on the test data
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("R^2 Score:", r2)

    # Make predictions on the blind test data
    y_pred_blind = model.predict(X_blind_test_df)

    # Calculate performance metrics on the blind test data
    mae_blind = mean_absolute_error(y_blind_test, y_pred_blind)
    mse_blind = mean_squared_error(y_blind_test, y_pred_blind)
    r2_blind = r2_score(y_blind_test, y_pred_blind)

    print("Mean Absolute Error on Blind Test Data:", mae_blind)
    print("Mean Squared Error on Blind Test Data:", mse_blind)
    print("R^2 Score on Blind Test Data:", r2_blind)

    return model


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

    X_train, y_train, X_test, y_test, X_blind_test, y_blind_test, scaler = preprocess_nfl_data(df)
    model = train_and_evaluate(X_train, y_train, X_test, y_test, X_blind_test, y_blind_test)

    # Save the model and related files to the models directory
    joblib.dump(model, os.path.join(model_dir, 'trained_nfl_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'data_scaler.pkl'))
    # joblib.dump(encoder, os.path.join(model_dir, 'data_encoder.pkl'))
    # joblib.dump(encoded_columns, os.path.join(model_dir, 'encoded_columns.pkl'))

    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(importance_df)


if __name__ == "__main__":
    main()
