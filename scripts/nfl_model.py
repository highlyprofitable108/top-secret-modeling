import os
import joblib
import logging
import pandas as pd
from importlib import reload
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# Importing custom modules and classes
import scripts.constants
from classes.modeling import Modeling
from classes.config_manager import ConfigManager
from classes.data_processing import DataProcessing
from classes.model_visualization import ModelVisualization
from classes.database_operations import DatabaseOperations

# Configure logging for better debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class NFLModel:
    def __init__(self):
        """
        Initialize the NFLModel class.

        This class is responsible for orchestrating various components of the NFL data analysis model.
        It loads configurations, initializes database operations, and sets up the modeling environment.
        """

        # Configuration and database operations setup
        self.config = ConfigManager()
        self.database_operations = DatabaseOperations()
        self.modeling = Modeling()

        # Load constants and configurations
        self._fetch_constants_and_configs()

        # Visualization component initialization
        # Consider adding a check or a try-except block to ensure template_dir and TARGET_VARIABLE are properly initialized
        self.model_results = ModelVisualization(self.template_dir, self.TARGET_VARIABLE)

    def _fetch_constants_and_configs(self):
        """
        Private method to fetch and load constants and configurations.

        This method ensures that all necessary configurations and constants are loaded and available
        for the model. Consider extending this method to handle dynamic configuration updates.
        """
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

    def load_data(self, collection_name):
        """
        Loads data from the specified MongoDB collection.

        Parameters:
        collection_name (str): The name of the MongoDB collection to load data from.

        Returns:
        DataFrame: The loaded data as a Pandas DataFrame, or an empty DataFrame in case of an error.
        """
        try:
            logging.info(f"Loading data from collection: {collection_name}")
            return self.database_operations.fetch_data_from_mongodb(collection_name)
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return pd.DataFrame()

    def process_data(self, df):
        """
        Processes the data by flattening, merging, and calculating scoring differential.

        Parameters:
        df (DataFrame): The DataFrame to process.

        Returns:
        DataFrame: The processed DataFrame, or an empty DataFrame in case of an error.

        Notes:
        - Consider adding functionality for additional data transformations if required.
        """
        try:
            df = self.data_processing.flatten_and_merge_data(df)
            df = df.dropna(subset=[self.TARGET_VARIABLE])  # Remove rows where self.TARGET_VARIABLE is NaN

            # Descriptive Statistics
            descriptive_stats = df.describe()
            descriptive_stats.to_csv(os.path.join(self.static_dir, 'descriptive_statistics.csv'))
            return df
        except Exception as e:
            logging.error(f"Error processing data: {e}")
            return pd.DataFrame()

    def filter_columns(self, df):
        """
        Filters the dataframe to keep only the necessary columns.

        Parameters:
        df (DataFrame): The DataFrame to filter.

        Returns:
        DataFrame: The filtered DataFrame, or an empty DataFrame in case of an error.

        Notes:
        - Placeholder for future feature: add options for dynamic column filtering.
        """
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
        """
        Integrates the processes of loading and processing data from the specified MongoDB collection.

        Parameters:
        collection_name (str): The name of the MongoDB collection to load data from.

        Returns:
        DataFrame: The loaded and processed DataFrame, or an empty DataFrame in case of an error.

        Notes:
        - Placeholder for future feature: add parameter customization for processing steps.
        """
        df = self.load_data(collection_name)
        df = self.process_data(df)
        df = self.filter_columns(df)
        df = self.data_processing.handle_null_values(df)
        return df

    def generate_eda_report(self, df):
        """
        Generates an Exploratory Data Analysis (EDA) report from the given DataFrame.

        Parameters:
        df (DataFrame): The DataFrame for which to generate the EDA report.

        Returns:
        None: This function does not return any value but saves EDA reports as image files.

        Notes:
        - Placeholder for future feature: extend to generate additional types of analyses as needed.
        """
        try:
            logging.info("Generating EDA report")

            # Generate and save the reports
            self.model_results.generate_descriptive_statistics(df)
            self.model_results.generate_data_quality_report(df)

        except Exception as e:
            logging.error(f"Error in generate_eda_report: {e}")

    def preprocess_nfl_data(self, df):
        """
        Preprocesses NFL data for model training, including feature scaling and data splitting.

        Parameters:
        df (DataFrame): The NFL data DataFrame to preprocess.

        Returns:
        Tuple: Contains processed training and testing datasets, the scaler object, and feature columns list.

        Notes:
        - Placeholder for future feature: add parameter tuning for data splitting and scaling.
        """
        logging.info("Preprocessing NFL data...")

        try:
            feature_columns = [col for col in df.columns if col != self.TARGET_VARIABLE]
            X = df.drop(self.TARGET_VARIABLE, axis=1)
            y = df[self.TARGET_VARIABLE]

            # Split the dataset
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=108)
            X_test, X_blind_test, y_test, y_blind_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=108)

            # Scale the features
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            X_blind_test = scaler.transform(X_blind_test)

            # Prepare and save blind test data
            blind_test_data = pd.concat([pd.DataFrame(X_blind_test, columns=feature_columns), y_blind_test.reset_index(drop=True)], axis=1)
            blind_test_data.to_csv(os.path.join(self.static_dir, 'blind_test_data.csv'), index=False)

            return X_train, y_train, X_test, y_test, X_blind_test, y_blind_test, scaler, feature_columns
        except Exception as e:
            logging.error(f"Error in preprocess_nfl_data: {e}")
            return tuple([pd.DataFrame() for _ in range(6)]) + (None, [])

    def main(self):
        """
        Main method to orchestrate the process of data loading, EDA report generation, and model training.

        Steps:
        1. Load and process data from the specified MongoDB collection.
        2. Generate an EDA report.
        3. Preprocess data for model training.
        4. Train and evaluate the model.
        5. Perform model interpretation and generate reports.

        Notes:
        - Placeholder for future feature: Integrate additional steps like data leakage checks and model decision documentation.
        """
        try:
            logging.info("Starting main method")
            reload(scripts.constants)

            # Data loading and processing
            collection_name = self.PREGAME_DB_NAME
            df = self.load_and_process_data(collection_name)

            # Generate EDA report
            self.generate_eda_report(df)

            # Preprocess data and train model
            X_train, y_train, X_test, y_test, X_blind_test, y_blind_test, scaler, feature_columns = self.preprocess_nfl_data(df)
            model = self.modeling.train_and_evaluate(X_train, y_train, X_test, y_test, X_blind_test, y_blind_test, feature_columns, self.model_type, self.grid_search_params)
            joblib.dump(model, os.path.join(self.model_dir, 'trained_nfl_model.pkl'))
            joblib.dump(scaler, os.path.join(self.model_dir, 'data_scaler.pkl'))

            # Post-training analysis
            estimator = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
            coefficients = estimator.coef_
            self.model_results.calculate_feature_coefficients(feature_columns, coefficients)
            self.model_results.correlation_heatmap(df, coefficients)
            shap_html = self.model_results.model_interpretation(estimator, X_train, feature_columns)
            performance_metrics_html = self.model_results.performance_metrics_summary(estimator, X_test, y_test)

            # Placeholder for additional analyses
            # self.model_results.check_data_leakage(df, feature_columns, self.TARGET_VARIABLE)
            # self.model_results.document_model_decisions()

            # Generate consolidated report
            self.model_results.generate_consolidated_report(shap_html, performance_metrics_html)

        except Exception as e:
            logging.error(f"Error in main: {e}")


# Main execution block
if __name__ == "__main__":
    """
    Main execution block for running NFL stats modeling.
    """
    analyzer = NFLModel()
    analyzer.main()
