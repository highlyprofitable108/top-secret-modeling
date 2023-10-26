import os
import logging

import pandas as pd
from importlib import reload
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from classes.modeling import Modeling
from classes.config_manager import ConfigManager
from classes.data_processing import DataProcessing
from classes.data_visualization import Visualization
from classes.database_operations import DatabaseOperations
import scripts.constants

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class NFLDataAnalyzer:
    """A class to analyze NFL data and generate EDA reports."""

    def __init__(self):
        """Initializes the NFLDataAnalyzer with necessary configurations and setups."""
        # Initialize ConfigManager, DatabaseOperations, and DataProcessing
        self.config = ConfigManager()
        self.database_operations = DatabaseOperations()
        self.data_processing = DataProcessing()
        self.modeling = Modeling()

        # Fetch constants using ConfigManager
        self.TWO_YEARS_IN_DAYS = self.config.get_constant('TWO_YEARS_IN_DAYS')
        self.MAX_DAYS_SINCE_GAME = self.config.get_constant('MAX_DAYS_SINCE_GAME')
        self.BASE_COLUMNS = eval(self.config.get_constant('BASE_COLUMNS'))  # Using eval to get the list from the script
        self.AWAY_PREFIX = self.config.get_constant('AWAY_PREFIX')
        self.HOME_PREFIX = self.config.get_constant('HOME_PREFIX')
        self.GAMES_DB_NAME = self.config.get_constant('GAMES_DB_NAME')
        self.TEAMS_DB_NAME = self.config.get_constant('TEAMS_DB_NAME')
        self.RANKS_DB_NAME = self.config.get_constant('RANKS_DB_NAME')
        self.PREGAME_DB_NAME = self.config.get_constant('PREGAME_DB_NAME')
        self.TARGET_VARIABLE = self.config.get_constant('TARGET_VARIABLE')

        self.data_dir = self.config.get_config('paths', 'data_dir')
        self.static_dir = self.config.get_config('paths', 'static_dir')
        self.template_dir = self.config.get_config('paths', 'template_dir')

        self.model_type = self.config.get_model_settings('model_type')
        self.grid_search_params = self.config.get_model_settings('grid_search')
        self.visualization = Visualization(self.template_dir, self.TARGET_VARIABLE)

    def load_and_process_data(self, collection_name):
        """Loads and processes data from the specified MongoDB collection."""
        df = self.load_data(collection_name)
        df = self.process_data(df)
        df = self.filter_columns(df)
        df = self.data_processing.handle_null_values(df)

        return df

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
            # Reload constants
            reload(scripts.constants)
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

    def generate_eda_report(self, df):
        """Generates an EDA report with various analyses and saves it as image files."""
        try:
            logging.info("Generating EDA report")

            # Paths to save the generated plots
            feature_importance_path, heatmap_path = self.plot_feature_importance(df)
            # histogram_path = self.plot_interactive_histograms(df)
            # boxplot_path = self.plot_boxplots(df)
            descriptive_stats_path = self.visualization.generate_descriptive_statistics(df)
            data_quality_report_path = self.visualization.generate_data_quality_report(df)

            return heatmap_path, feature_importance_path, descriptive_stats_path, data_quality_report_path

        except Exception as e:
            logging.error(f"Error in generate_eda_report: {e}")
            return None, None, None, None, None

    def plot_feature_importance(self, df):
        """Plots the feature importance and highlights top correlations."""
        try:
            X = df.drop(columns=[self.TARGET_VARIABLE])
            y = df[self.TARGET_VARIABLE]

            # Train the model
            model = self.modeling.train_model(X, y, self.model_type, self.grid_search_params)

            # Extract and standardize feature importances
            importances = model.best_estimator_.feature_importances_ / model.best_estimator_.feature_importances_.sum()

            # Identify top features based on importance and correlation
            top_importance_features, top_correlation_features = self.modeling.identify_top_features(X, y, importances)

            # Create a DataFrame for feature importance visualization
            feature_importance_df = self.prepare_feature_importance_df(X.columns, importances, top_importance_features, top_correlation_features)

            # Visualize feature importance
            feature_importance_path = self.visualization.visualize_feature_importance(feature_importance_df)

            # Create Heat Map
            heatmap_path = self.visualization.plot_interactive_correlation_heatmap(df, importances)

            logging.info(f"Best model score: {model.best_score_}")
            return feature_importance_path, heatmap_path

        except Exception as e:
            logging.error(f"Error generating feature importance plot: {e}")
            return None

    def prepare_feature_importance_df(self, feature_names, importances, top_importance_features, top_correlation_features):
        """Prepare a DataFrame for feature importance visualization."""
        def determine_color(feature):
            if feature in top_importance_features and feature in top_correlation_features:
                return 'Important and Related'
            elif feature in top_importance_features:
                return 'Important'
            elif feature in top_correlation_features:
                return 'Related to Target'
            else:
                return 'Just Data'

        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance_df['Highlight'] = feature_importance_df['Feature'].apply(determine_color)
        return feature_importance_df.sort_values(by='Importance', ascending=False)

    def main(self):
        """Main method to load data and generate EDA report."""
        try:
            logging.info("Starting main method")
            collection_name = self.PREGAME_DB_NAME
            df = self.load_and_process_data(collection_name)

            return self.generate_eda_report(df)
        except Exception as e:
            logging.error(f"Error in main: {e}")
            return None, None


analyzer = NFLDataAnalyzer()
analyzer.main()
