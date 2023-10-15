import os
import joblib
import logging
import pandas as pd
from importlib import reload
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scripts.constants
from classes.config_manager import ConfigManager
from classes.data_processing import DataProcessing
from classes.database_operations import DatabaseOperations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class NFLModel:
    def __init__(self):
        self.config = ConfigManager()
        self.database_operations = DatabaseOperations()
        self.data_processing = DataProcessing()

        self._fetch_constants_and_configs()

    def _fetch_constants_and_configs(self):
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

    def load_and_process_data(self):
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

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, X_blind_test, y_blind_test, feature_columns):
        """
        This function trains the model on the training data and evaluates it on the test data and blind test data.
        """
        logging.info("Training and evaluating the model...")

        try:
            # Convert numpy arrays back to dataframes to preserve feature names
            X_train_df = pd.DataFrame(X_train, columns=feature_columns)

            # Train the model using the factory method
            model = self.train_model(X_train_df, y_train)
            model.fit(pd.DataFrame(X_train, columns=feature_columns), y_train)

            for dataset, dataset_name in zip([(X_test, y_test), (X_blind_test, y_blind_test)], ['Test Data', 'Blind Test Data']):
                X_df, y_data = dataset
                y_pred = model.predict(pd.DataFrame(X_df, columns=feature_columns))
                mae = mean_absolute_error(y_data, y_pred)
                mse = mean_squared_error(y_data, y_pred)
                r2 = r2_score(y_data, y_pred)
                logging.info(f"Performance on {dataset_name}: MAE: {mae}, MSE: {mse}, R^2: {r2}")

            return model
        except Exception as e:
            logging.error(f"Error in train_and_evaluate: {e}")
            return None

    def train_model(self, X, y):
        """Train a model based on the type settings in the configuration."""
        logging.info(f"Training model of type: {self.model_type}")

        model_type = self.model_type
        if model_type == "random_forest":
            return self.train_random_forest(X, y)
        # Add other model training methods here as needed
        else:
            raise ValueError(f"The model type '{model_type}' specified in the config is not supported.")

    def train_random_forest(self, X, y):
        """Train a RandomForestRegressor with hyperparameter tuning."""
        logging.info("Training RandomForestRegressor with hyperparameter tuning...")


        # Figure out how to filter and input from config.yaml
        param_grid = {
            'n_estimators': [100], 
            'max_depth': [None, 10],
            # Add more hyperparameters here as required
        }
        model = GridSearchCV(RandomForestRegressor(random_state=108), param_grid, cv=3, verbose=2)
        model.fit(X, y)
        return model

    def main(self):
        try:
            processed_df = self.load_and_process_data()
            X_train, y_train, X_test, y_test, X_blind_test, y_blind_test, scaler, feature_columns = self.preprocess_nfl_data(processed_df)

            # Train and evaluate model
            model = self.train_and_evaluate(X_train, y_train, X_test, y_test, X_blind_test, y_blind_test, feature_columns)

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
