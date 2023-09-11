from classes.config_manager import ConfigManager
from classes.database_operations import DatabaseOperations
from classes.data_processing import DataProcessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import numpy as np
import pandas as pd
from .constants import COLUMNS_TO_KEEP
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize ConfigManager, DatabaseOperations, and DataProcessing
config = ConfigManager()
db_operations = DatabaseOperations()
data_processing = DataProcessing()

# Fetch configurations using ConfigManager
data_dir = config.get_config('paths', 'data_dir')
model_dir = config.get_config('paths', 'model_dir')
database_name = config.get_config('database', 'database_name')


def load_and_process_data():
    """Load and process data."""
    try:
        # Fetch data from MongoDB using DatabaseOperations class
        collection_name = "games"
        df = db_operations.fetch_data_from_mongodb(collection_name)
        # Flatten and merge data using DataProcessing class
        df = data_processing.flatten_and_merge_data(df)
        # Compute scoring differential using DataProcessing class
        df = data_processing.calculate_scoring_differential(df)
        # Keep only the columns specified in COLUMNS_TO_KEEP
        df = df[COLUMNS_TO_KEEP]
        # Convert time strings to minutes (apply this to the relevant columns)
        if 'statistics_home.summary.possession_time' in df.columns:
            df['statistics_home.summary.possession_time'] = df['statistics_home.summary.possession_time'].apply(data_processing.time_to_minutes)
        if 'statistics_away.summary.possession_time' in df.columns:
            df['statistics_away.summary.possession_time'] = df['statistics_away.summary.possession_time'].apply(data_processing.time_to_minutes)
        # Drop games if 'scoring_differential' key does not exist
        if 'scoring_differential' not in df.columns:
            logging.warning("'scoring_differential' key does not exist. Dropping games.")
            return pd.DataFrame() # Return an empty dataframe
        else:
            return df
    except Exception as e:
        logging.error(f"Error in load_and_process_data: {e}")
        return pd.DataFrame()


def preprocess_nfl_data(df):
    """Preprocess NFL data."""
    try:
        nan_counts = df.isnull().sum()
        columns_to_drop = nan_counts[nan_counts > 0].index.tolist()
        if columns_to_drop:
            logging.warning(f"Dropping columns with more than 100 NaN values: {columns_to_drop}")
            df = df.drop(columns=columns_to_drop).reset_index(drop=True)

        # Fill NaN values in remaining columns with the mean of each column
        nan_columns = nan_counts[nan_counts > 0].index.tolist()
        nan_columns = [col for col in nan_columns if col not in columns_to_drop]
        for col in nan_columns:
            col_mean = df[col].mean()
            df[col].fillna(col_mean, inplace=True)

        # Update the feature_columns list to reflect the changes
        feature_columns = [col for col in COLUMNS_TO_KEEP if col != 'scoring_differential' and col not in columns_to_drop]

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

        return X_train, y_train, X_test, y_test, X_blind_test, y_blind_test, scaler, feature_columns
        # encoder, encoded_columns
    except Exception as e:
        logging.error(f"Error in preprocess_nfl_data: {e}")
        return pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series(), None


def train_and_evaluate(X_train, y_train, X_test, y_test, X_blind_test, y_blind_test, feature_columns):
    """
    This function trains the model on the training data and evaluates it on the test data and blind test data.
    """
    try:
        # Convert numpy arrays back to dataframes to preserve feature names
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
        logging.info(f"Mean Absolute Error: {mae}")
        logging.info(f"Mean Squared Error: {mse}")
        logging.info(f"R^2 Score: {r2}")
        # Make predictions on the blind test data
        y_pred_blind = model.predict(X_blind_test_df)
        # Calculate performance metrics on the blind test data
        mae_blind = mean_absolute_error(y_blind_test, y_pred_blind)
        mse_blind = mean_squared_error(y_blind_test, y_pred_blind)
        r2_blind = r2_score(y_blind_test, y_pred_blind)
        logging.info(f"Mean Absolute Error on Blind Test Data: {mae_blind}")
        logging.info(f"Mean Squared Error on Blind Test Data: {mse_blind}")
        logging.info(f"R^2 Score on Blind Test Data: {r2_blind}")
        return model
    except Exception as e:
        logging.error(f"Error in train_and_evaluate: {e}")
        return None


def main():
    """Main function."""
    try:
        # Load and process data
        processed_df = load_and_process_data()
        # Preprocess data
        X_train, y_train, X_test, y_test, X_blind_test, y_blind_test, scaler, feature_columns = preprocess_nfl_data(processed_df)
        # Train and evaluate model
        model = train_and_evaluate(X_train, y_train, X_test, y_test, X_blind_test, y_blind_test, feature_columns)
        # Save the model and related files to the models directory
        joblib.dump(model, os.path.join(model_dir, 'trained_nfl_model.pkl'))
        joblib.dump(scaler, os.path.join(model_dir, 'data_scaler.pkl'))
        feature_importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': feature_importances
        })
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        logging.info(importance_df)
    except Exception as e:
        logging.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()
