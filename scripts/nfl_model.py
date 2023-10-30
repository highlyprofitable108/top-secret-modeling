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
from classes.data_visualization import Visualization
from classes.database_operations import DatabaseOperations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class NFLModel:
    def __init__(self):
        """Initialize the NFLDataAnalyzer class, loading configurations and constants."""
        self.config = ConfigManager()
        self.database_operations = DatabaseOperations()
        self.modeling = Modeling()

        self._fetch_constants_and_configs()
        self.visualization = Visualization(self.template_dir, self.TARGET_VARIABLE)

    def _fetch_constants_and_configs(self):
        """Fetch constants and configurations from the configuration manager."""
        try:
            constants = [
                'TWO_YEARS_IN_DAYS', 'MAX_DAYS_SINCE_GAME', 'BASE_COLUMNS', 'AWAY_PREFIX', 
                'HOME_PREFIX', 'GAMES_DB_NAME', 'TEAMS_DB_NAME', 'PREGAME_DB_NAME', 'RANKS_DB_NAME', 'CUTOFF_DATE',
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

            self.TARGET_VARIABLE = self.config.get_config('constants', 'TARGET_VARIABLE')
            self.database_name = self.config.get_config('database', 'database_name')

            self.data_processing = DataProcessing(self.TARGET_VARIABLE)

        except Exception as e:
            raise ValueError(f"Error fetching configurations: {e}")

    # Processing Methods
    def load_data(self, collection_name):
        """Loads data from the specified MongoDB collection."""
        try:
            logging.info(f"Loading data from collection: {collection_name}")
            return self.database_operations.fetch_data_from_mongodb(collection_name)
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return pd.DataFrame()

    def process_data(self, df):
        """Processes the data by flattening, merging, and calculating scoring differential."""
        try:
            df = self.data_processing.flatten_and_merge_data(df)
            df = df.dropna(subset=[self.TARGET_VARIABLE])  # Remove rows where self.TARGET_VARIABLE is NaN

            # Descriptive Statistics (Integration of suggestion 2)
            descriptive_stats = df.describe()
            descriptive_stats.to_csv(os.path.join(self.static_dir, 'descriptive_statistics.csv'))
            return df
        except Exception as e:
            logging.error(f"Error processing data: {e}")
            return pd.DataFrame()

    def filter_columns(self, df):
        """Filters the dataframe to keep only the necessary columns."""
        try:
            # Filter columns with stripping whitespaces
            columns_to_filter = [col for col in scripts.constants.COLUMNS_TO_KEEP if col.strip() in map(str.strip, df.columns)]

            # Check if any columns are filtered
            if not columns_to_filter:
                logging.error("No matching columns found.")
                return pd.DataFrame()

            return df[columns_to_filter].copy()
        except Exception as e:
            logging.error(f"Error filtering columns: {e}")
            return pd.DataFrame()

    def load_and_process_data(self, collection_name):
        """Loads and processes data from the specified MongoDB collection."""
        df = self.load_data(collection_name)
        df = self.process_data(df)
        df = self.filter_columns(df)
        df = self.data_processing.handle_null_values(df)
        return df

    # EDA Generation and Data Collection Methods
    def generate_eda_report(self, df):
        """Generates an EDA report with various analyses and saves it as image files."""
        try:
            logging.info("Generating EDA report")

            # Paths to save the generated plots
            descriptive_stats_path = self.visualization.generate_descriptive_statistics(df)
            data_quality_report_path = self.visualization.generate_data_quality_report(df)

            return descriptive_stats_path, data_quality_report_path

        except Exception as e:
            logging.error(f"Error in generate_eda_report: {e}")
            return None, None, None, None, None

    def plot_feature_importance(self, df, model):
        """Plots the feature importance for tree-based models and coefficients for linear models."""
        try:
            X = df.drop(columns=[self.TARGET_VARIABLE])
            y = df[self.TARGET_VARIABLE]

            # Check if the model is a result of GridSearchCV or RandomizedSearchCV
            if hasattr(model, 'best_estimator'):
                estimator = model.best_estimator_
            else:
                estimator = model

            # Check if the model is tree-based and has feature_importances_ attribute
            if hasattr(estimator, 'feature_importances_'):
                # Extract and standardize feature importances
                importances = estimator.feature_importances_ / estimator.feature_importances_.sum()
                importance_type = 'Importance'
            # Check if the model is linear and has coef_ attribute
            elif hasattr(estimator, 'coef_'):
                importances = estimator.coef_
                importance_type = 'Coefficient'
            else:
                logging.error("The model doesn't support feature_importances_ or coef_")
                return None

            # Identify top features based on importance or coefficients
            top_importance_features, top_correlation_features = self.modeling.identify_top_features(X, y, importances)

            # Create a DataFrame for feature importance or coefficient visualization
            feature_importance_df = self.prepare_feature_importance_df(X.columns, importances, top_importance_features, top_correlation_features, importance_type)

            # Visualize feature importance or coefficients
            feature_importance_path = self.visualization.visualize_feature_importance(feature_importance_df)

            # Create Heat Map
            heatmap_path = self.visualization.plot_interactive_correlation_heatmap(df, importances)

            logging.info(f"Best model score: {model.best_score_}")

            return feature_importance_path, heatmap_path
        except Exception as e:
            logging.error(f"Error generating feature importance plot: {e}")
            return None

    def prepare_feature_importance_df(self, feature_names, importances, top_importance_features, top_correlation_features, importance_type):
        """Prepare a DataFrame for feature importance or coefficient visualization."""
        def determine_color(feature):
            if feature in top_importance_features and feature in top_correlation_features:
                return 'Important and Related'
            elif feature in top_importance_features:
                return 'Important'
            elif feature in top_correlation_features:
                return 'Related to Target'
            else:
                return 'Just Data'

        feature_importance_df = pd.DataFrame({'Feature': feature_names, importance_type: importances})
        feature_importance_df['Highlight'] = feature_importance_df['Feature'].apply(determine_color)
        return feature_importance_df.sort_values(by=importance_type, ascending=False)

    # Data Processing Methods
    def preprocess_nfl_data(self, df):
        """
        Preprocess NFL data for model training.

        Handles null values, splits data into features and target variable, scales features, and prepares blind test data.

        :param df: Preprocessed DataFrame.
        :return: Tuple containing processed data and related objects.
        """
        logging.info("Preprocessing NFL data...")

        try:
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
        """Main method to load data, generate EDA report, and train the model."""
        try:
            logging.info("Starting main method")
            reload(scripts.constants)

            collection_name = self.PREGAME_DB_NAME
            df = self.load_and_process_data(collection_name)
            # Generate EDA report
            self.generate_eda_report(df)

            # Train and evaluate model
            X_train, y_train, X_test, y_test, X_blind_test, y_blind_test, scaler, feature_columns = self.preprocess_nfl_data(df)
            model = self.modeling.train_and_evaluate(X_train, y_train, X_test, y_test, X_blind_test, y_blind_test, feature_columns, self.model_type, self.grid_search_params)
            joblib.dump(model, os.path.join(self.model_dir, 'trained_nfl_model.pkl'))
            joblib.dump(scaler, os.path.join(self.model_dir, 'data_scaler.pkl'))

            # Check if the model is a result of GridSearchCV or RandomizedSearchCV
            if hasattr(model, 'best_estimator_'):
                estimator = model.best_estimator_
            else:
                estimator = model

            # Ensure the model is a RandomForestRegressor before accessing feature_importances_
            if hasattr(estimator, 'feature_importances_'):
                # Fetching additional details
                best_score = model.best_score_
                best_params = model.best_params_
                model_name = type(estimator).__name__

                # Constructing importance dataframe
                importance_data = {
                    'Feature': feature_columns,
                    'Importance': estimator.feature_importances_
                }
                importance_df = pd.DataFrame(importance_data)
                importance_df = importance_df.sort_values(by='Importance', ascending=False)
                importance_df['Rank'] = range(1, len(importance_df) + 1)
                importance_df['Cumulative Importance'] = importance_df['Importance'].cumsum()
                self.plot_feature_importance(df, model)

                # shap_values, explainer = self.modeling.compute_shap_values(estimator, X_train, self.model_type)
                # X = pd.DataFrame(X_train, columns=feature_columns)
                # self.visualization.visualize_shap_summary(shap_values, explainer, X)

                # Logging details
                logging.info(f"Best Model: {model_name}")
                logging.info(f"Best Parameters: {best_params}")
                logging.info(f"Best Score: {best_score:.4f}")
                logging.info("Feature Importances:")
                logging.info(importance_df)
            elif hasattr(estimator, 'coef_'):
                coefficients = estimator.coef_

                # Create a DataFrame for feature coefficients
                coef_data = {
                    'Feature': feature_columns,
                    'Coefficient': coefficients
                }
                coef_df = pd.DataFrame(coef_data)

                # Sort by the absolute value of the coefficient for importance
                coef_df = coef_df.sort_values(by='Coefficient', key=abs, ascending=False)

                # Optionally, visualize the coefficients
                self.visualization.visualize_coefficients(coef_df)

                # Logging details
                logging.info("Feature Coefficients:")
                logging.info(coef_df)

            else:
                logging.error("The best model from GridSearchCV doesn't support feature_importances_")
        except Exception as e:
            logging.error(f"Error in main: {e}")


if __name__ == "__main__":
    analyzer = NFLModel()
    analyzer.main()
