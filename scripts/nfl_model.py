import os
import joblib
import logging
import pandas as pd
from importlib import reload
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import scripts.constants
from classes.modeling import Modeling
from classes.config_manager import ConfigManager
from classes.data_processing import DataProcessing
from classes.database_operations import DatabaseOperations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class NFLModel:
    def __init__(self):
        """
        Initialize the NFLModel class, loading configurations and constants.

        Initializes configuration settings, constants, and necessary class instances.
        """
        self.config = ConfigManager()
        self.database_operations = DatabaseOperations()
        self.data_processing = DataProcessing()
        self.modeling = Modeling()

        self._fetch_constants_and_configs()

    # Helper Methods
    def _fetch_constants_and_configs(self):
        """
        Fetch constants and configurations from the configuration manager.

        Fetches various constants and configuration settings from the configuration manager.
        """
        try:
            constants = [
                'TWO_YEARS_IN_DAYS', 'MAX_DAYS_SINCE_GAME', 'BASE_COLUMNS', 'AWAY_PREFIX', 
                'HOME_PREFIX', 'GAMES_DB_NAME', 'TEAMS_DB_NAME', 'RANKS_DB_NAME', 'CUTOFF_DATE',
                'TARGET_VARIABLE'
            ]
            for const in constants:
                setattr(self, const, self.config.get_constant(const))
            self.BASE_COLUMNS = eval(self.BASE_COLUMNS)
            self.CUTOFF_DATE = datetime.strptime(self.CUTOFF_DATE, '%Y-%m-%d')
            self.model_type = self.config.get_model_settings('model_type')
            self.grid_search_params = self.config.get_model_settings('grid_search')

            paths = ['data_dir', 'model_dir', 'static_dir', 'template_dir']
            for path in paths:
                setattr(self, path, self.config.get_config('paths', path))

            self.database_name = self.config.get_config('database', 'database_name')
        except Exception as e:
            raise ValueError(f"Error fetching configurations: {e}")

    # Data Processing Methods
    def load_and_process_data(self):
        """
        Load and preprocess data from MongoDB.

        Loads data from MongoDB, preprocesses it, and filters based on specified criteria.

        :return: Preprocessed DataFrame.
        """
        logging.info("Loading and processing data...")

        try:
            collection_name = "pre_game_data"
            df = self.database_operations.fetch_data_from_mongodb(collection_name)
            df = self.data_processing.flatten_and_merge_data(df)
            df['scheduled'] = pd.to_datetime(df['scheduled']).dt.tz_localize(None)
            df = df[df['scheduled'] < pd.Timestamp.now().normalize()]

            reload(scripts.constants)
            columns_to_filter = [
                col for col in scripts.constants.COLUMNS_TO_KEEP if col.strip() in map(str.strip, df.columns)
            ]
            df = df[columns_to_filter]

            if self.TARGET_VARIABLE not in df.columns:
                logging.warning("self.TARGET_VARIABLE key does not exist. Dropping games.")
                return pd.DataFrame()
            return df
        except Exception as e:
            logging.error(f"Error in load_and_process_data: {e}")
            return pd.DataFrame()

    def preprocess_nfl_data(self, df):
        """
        Preprocess NFL data for model training.

        Handles null values, splits data into features and target variable, scales features, and prepares blind test data.

        :param df: Preprocessed DataFrame.
        :return: Tuple containing processed data and related objects.
        """
        logging.info("Preprocessing NFL data...")

        try:
            df = self.data_processing.handle_null_values(df)
            feature_columns = [col for col in df.columns if col != self.TARGET_VARIABLE]
            X = df.drop(self.TARGET_VARIABLE, axis=1)
            y = df[self.TARGET_VARIABLE]

            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=108)
            X_test, X_blind_test, y_test, y_blind_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=108)

            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            X_blind_test = scaler.transform(X_blind_test)

            blind_test_data = pd.concat([pd.DataFrame(X_blind_test, columns=df.columns.drop(self.TARGET_VARIABLE)),
                                         y_blind_test.reset_index(drop=True)], axis=1)
            blind_test_data.to_csv(os.path.join(self.data_dir, 'blind_test_data.csv'), index=False)

            return X_train, y_train, X_test, y_test, X_blind_test, y_blind_test, scaler, feature_columns
        except Exception as e:
            logging.error(f"Error in preprocess_nfl_data: {e}")
            return tuple([pd.DataFrame() for _ in range(6)]) + (None, [])

    def main(self):
        """
        Main method for executing the model training and evaluation pipeline.

        This method orchestrates the entire model training and evaluation process, including data preprocessing,
        model training, evaluation, and logging of results.
        """
        try:
            processed_df = self.load_and_process_data()
            X_train, y_train, X_test, y_test, X_blind_test, y_blind_test, scaler, feature_columns = self.preprocess_nfl_data(processed_df)

            # Train and evaluate model
            model = self.modeling.train_and_evaluate(X_train, y_train, X_test, y_test, X_blind_test, y_blind_test, feature_columns, self.model_type, self.grid_search_params)

            joblib.dump(model, os.path.join(self.model_dir, 'trained_nfl_model.pkl'))
            joblib.dump(scaler, os.path.join(self.model_dir, 'data_scaler.pkl'))

            # Ensure the model is a RandomForestRegressor before accessing feature_importances_
            if hasattr(model.best_estimator_, 'feature_importances_'):
                # Fetching additional details
                best_score = model.best_score_
                best_params = model.best_params_
                model_name = type(model.best_estimator_).__name__

                # Constructing importance dataframe
                importance_data = {
                    'Feature': feature_columns,
                    'Importance': model.best_estimator_.feature_importances_
                }
                importance_df = pd.DataFrame(importance_data)
                importance_df = importance_df.sort_values(by='Importance', ascending=False)
                importance_df['Rank'] = range(1, len(importance_df) + 1)
                importance_df['Cumulative Importance'] = importance_df['Importance'].cumsum()

                # Logging details
                logging.info(f"Best Model: {model_name}")
                logging.info(f"Best Parameters: {best_params}")
                logging.info(f"Best Score: {best_score:.4f}")
                logging.info("Feature Importances:")
                logging.info(importance_df)

            else:
                logging.error("The best model from GridSearchCV doesn't support feature_importances_")
        except Exception as e:
            logging.error(f"Error in main: {e}")


if __name__ == "__main__":
    nfl_model = NFLModel()
    nfl_model.main()
