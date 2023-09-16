from classes.config_manager import ConfigManager
from classes.data_processing import DataProcessing
from classes.database_operations import DatabaseOperations
import os
import warnings
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

matplotlib.use('Agg')

class NFLDataAnalyzer:
    """A class to analyze NFL data and generate EDA reports."""

    def __init__(self):
        """Initializes the NFLDataAnalyzer with necessary configurations and setups."""
        self.config_manager = ConfigManager()
        self.db_operations = DatabaseOperations()
        self.data_processing = DataProcessing()
        self.target_variable = 'scoring_differential'
        self.data_dir = self.get_config('paths', 'data_dir')
        self.static_dir = self.get_config('paths', 'static_dir')
        warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

    def get_config(self, section, key):
        """Retrieves configuration values."""
        try:
            return self.config_manager.get_config(section, key)
        except Exception as e:
            logging.error(f"Error retrieving config: {e}")
            return None

    def load_data(self, collection_name):
        """Loads data from the specified MongoDB collection."""
        try:
            logging.info(f"Loading data from collection: {collection_name}")
            return self.db_operations.fetch_data_from_mongodb(collection_name)
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return pd.DataFrame()

    def process_data(self, df):
        """Processes the data by flattening, merging, and calculating scoring differential."""
        try:
            df = self.data_processing.flatten_and_merge_data(df)
            df = self.data_processing.calculate_scoring_differential(df)
            # Convert time strings to minutes (apply this to the relevant columns)
            if 'statistics_home.summary.possession_time' in df.columns:
                df['statistics_home.summary.possession_time'] = df['statistics_home.summary.possession_time'].apply(self.data_processing.time_to_minutes)
            if 'statistics_away.summary.possession_time' in df.columns:
                df['statistics_away.summary.possession_time'] = df['statistics_away.summary.possession_time'].apply(self.data_processing.time_to_minutes)
            return df
        except Exception as e:
            logging.error(f"Error processing data: {e}")
            return pd.DataFrame()

    def get_active_constants(self):
        """Dynamically load and return the latest active constants from constants.py."""
        try:
            import importlib.util
            # Load the constants module dynamically
            spec = importlib.util.spec_from_file_location("constants", "scripts/constants.py")
            constants = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(constants)
            return constants.COLUMNS_TO_KEEP
        except Exception as e:
            logging.error(f"Error loading active constants: {e}")
            return []

    def filter_columns(self, df):
        """Filters the dataframe to keep only the necessary columns."""
        try:
            return df[self.get_active_constants()]
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

    def plot_correlation_heatmap(self, df):
        """Plots a correlation heatmap for the given dataframe."""
        try:
            corr = df.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm')
            plt.title('Correlation Heatmap')
            heatmap_path = os.path.join(self.static_dir, 'heatmap.png')
            plt.savefig(heatmap_path)
            plt.close()
            return heatmap_path
        except Exception as e:
            logging.error(f"Error generating correlation heatmap: {e}")
            return None

    def plot_feature_importance(self, df):
        """Plots the feature importance using a RandomForestRegressor."""
        try:
            # Assuming that the target variable is 'scoring_differential'
            X = df.drop(columns=[self.target_variable])
            y = df[self.target_variable]

            model = RandomForestRegressor()
            model.fit(X, y)

            feature_importance = pd.Series(model.feature_importances_, index=X.columns)
            feature_importance.nlargest(10).plot(kind='barh')
            plt.title('Feature Importance')
            feature_importance_path = os.path.join(self.static_dir, 'feature_importance.png')
            plt.savefig(feature_importance_path)
            plt.close()
            return feature_importance_path
        except Exception as e:
            logging.error(f"Error generating feature importance plot: {e}")
            return None

    def generate_eda_report(self, df):
        """Generates an EDA report with various analyses and saves it as image files."""
        try:
            logging.info("Generating EDA report")
            heatmap_path = self.plot_correlation_heatmap(df)
            feature_importance_path = self.plot_feature_importance(df)
            return heatmap_path, feature_importance_path
        except Exception as e:
            logging.error(f"Error in generate_eda_report: {e}")
            return None, None

    def main(self):
        """Main method to load data and generate EDA report."""
        try:
            logging.info("Starting main method")
            collection_name = 'games'
            df = self.load_and_process_data(collection_name)
            return self.generate_eda_report(df)
        except Exception as e:
            logging.error(f"Error in main: {e}")
            return None, None
