import os
import logging

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from importlib import reload
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from classes.config_manager import ConfigManager
from classes.data_processing import DataProcessing
from classes.database_operations import DatabaseOperations
import scripts.constants

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class NFLDataAnalyzer:
    """A class to analyze NFL data and generate EDA reports."""

    def __init__(self, TARGET_VARIABLE='self.TARGET_VARIABLE'):
        """Initializes the NFLDataAnalyzer with necessary configurations and setups."""
        # Initialize ConfigManager, DatabaseOperations, and DataProcessing
        self.config = ConfigManager()
        self.database_operations = DatabaseOperations()
        self.data_processing = DataProcessing()

        # Fetch constants using ConfigManager
        self.TWO_YEARS_IN_DAYS = self.config.get_constant('TWO_YEARS_IN_DAYS')
        self.MAX_DAYS_SINCE_GAME = self.config.get_constant('MAX_DAYS_SINCE_GAME')
        self.BASE_COLUMNS = eval(self.config.get_constant('BASE_COLUMNS'))  # Using eval to get the list from the script
        self.AWAY_PREFIX = self.config.get_constant('AWAY_PREFIX')
        self.HOME_PREFIX = self.config.get_constant('HOME_PREFIX')
        self.GAMES_DB_NAME = self.config.get_constant('GAMES_DB_NAME')
        self.TEAMS_DB_NAME = self.config.get_constant('TEAMS_DB_NAME')
        self.RANKS_DB_NAME = self.config.get_constant('RANKS_DB_NAME')
        self.TARGET_VARIABLE = self.config.get_constant('TARGET_VARIABLE')

        self.data_dir = self.config.get_config('paths', 'data_dir')
        self.static_dir = self.config.get_config('paths', 'static_dir')
        self.template_dir = self.config.get_config('paths', 'template_dir')

        self.model_type = self.config.get_model_settings('model_type')
        self.grid_search_params = self.config.get_model_settings('grid_search')

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
            df = df.dropna(subset=['self.TARGET_VARIABLE'])  # Remove rows where 'self.TARGET_VARIABLE' is NaN

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
            feature_importance_path = self.plot_feature_importance(df)
            heatmap_path = self.plot_interactive_correlation_heatmap(df)
            # histogram_path = self.plot_interactive_histograms(df)
            # boxplot_path = self.plot_boxplots(df)
            descriptive_stats_path = self.generate_descriptive_statistics(df)
            data_quality_report_path = self.generate_data_quality_report(df)

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
            model = self.train_model(X, y)

            # Extract and standardize feature importances
            importances = model.best_estimator_.feature_importances_ / model.best_estimator_.feature_importances_.sum()

            # Identify top features based on importance and correlation
            top_importance_features, top_correlation_features = self.identify_top_features(X, y, importances)

            # Create a DataFrame for feature importance visualization
            feature_importance_df = self.prepare_feature_importance_df(X.columns, importances, top_importance_features, top_correlation_features)

            # Visualize feature importance
            feature_importance_path = self.visualize_feature_importance(feature_importance_df)

            logging.info(f"Best model score: {model.best_score_}")
            return feature_importance_path

        except Exception as e:
            logging.error(f"Error generating feature importance plot: {e}")
            return None

    def train_model(self, X, y):
        """Train a model based on the EDA settings in the configuration."""
        eda_type = self.model_type
        if eda_type == "random forest":
            return self.train_random_forest(X, y)
        # Add other model training methods here as needed
        else:
            raise ValueError(f"The EDA type '{eda_type}' specified in the config is not supported.")

    def train_random_forest(self, X, y):
        """Train a RandomForestRegressor with hyperparameter tuning."""
        param_grid = self.grid_search_params
        model = GridSearchCV(RandomForestRegressor(), param_grid, cv=3)
        model.fit(X, y)
        return model

    def identify_top_features(self, X, y, importances):
        """Identify the top features based on importance and correlation."""
        correlations = X.corrwith(y).abs()
        top_20_percent = int(np.ceil(0.20 * len(importances)))
        top_importance_features = X.columns[importances.argsort()[-top_20_percent:]]
        top_correlation_features = correlations.nlargest(top_20_percent).index.tolist()
        return top_importance_features, top_correlation_features

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

    def visualize_feature_importance(self, feature_importance_df):
        """Visualize feature importance using Plotly."""
        fig = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importance',
                     color='Highlight', color_discrete_map={'Important': 'red', 'Related to Target': 'blue', 'Important and Related': 'purple', 'Just Data': 'gray'})
        feature_importance_path = os.path.join(self.template_dir, 'feature_importance.html')
        fig.write_html(feature_importance_path)
        return feature_importance_path

    # ENHANCE AND OPTIMIZE EDA OUTPUTS
    def plot_interactive_correlation_heatmap(self, df):
        """Plots an interactive correlation heatmap using Plotly."""
        try:
            corr = df.corr()

            # Using Plotly to create an interactive heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                hoverongaps=False,
                colorscale=[[0, "red"], [0.5, "white"], [1, "blue"]],  # Setting colorscale with baseline at 0
                zmin=-.8,  # Setting minimum value for color scale
                zmax=.8,   # Setting maximum value for color scale
                showscale=True,  # Display color scale bar
            ))

            # Adding annotations to the heatmap
            annotations = []
            for i, row in enumerate(corr.values):
                for j, value in enumerate(row):
                    annotations.append(
                        {
                            "x": corr.columns[j],
                            "y": corr.columns[i],
                            "font": {"color": "black"},
                            "text": str(round(value, 2)),
                            "xref": "x1",
                            "yref": "y1",
                            "showarrow": False
                        }
                    )
            fig.update_layout(annotations=annotations, title='Correlation Heatmap')

            heatmap_path = os.path.join(self.template_dir, 'interactive_heatmap.html')
            fig.write_html(heatmap_path)

            return heatmap_path
        except Exception as e:
            logging.error(f"Error generating interactive correlation heatmap: {e}")
            return None

    def plot_interactive_histograms(self, df):
        """Plots interactive histograms for each numerical column using Plotly."""
        histograms_dir = os.path.join(self.template_dir, 'interactive_histograms')
        os.makedirs(histograms_dir, exist_ok=True)

        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            fig = px.histogram(df, x=col, title=f'Histogram of {col}', nbins=30)
            fig.write_html(os.path.join(histograms_dir, f'{col}_histogram.html'))

        return histograms_dir

    def plot_boxplots(self, df):
        """Plots box plots for each numerical column in the dataframe."""
        try:
            # Create a directory to store all boxplot plots
            boxplots_dir = os.path.join(self.static_dir, 'boxplots')
            os.makedirs(boxplots_dir, exist_ok=True)

            # Loop through each numerical column and create a boxplot
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                fig = px.box(df, y=col, title=f'Boxplot of {col}')
                fig.write_html(os.path.join(boxplots_dir, f'{col}_boxplot.html'))

            logging.info("Boxplots generated successfully")
            return boxplots_dir

        except Exception as e:
            logging.error(f"Error generating boxplots: {e}")
            return None

    def generate_descriptive_statistics(self, df):
        """Generates descriptive statistics for each column in the dataframe and saves it as an HTML file."""
        try:
            # Generating descriptive statistics
            descriptive_stats = df.describe(include='all')

            # Transposing the DataFrame
            descriptive_stats = descriptive_stats.transpose()

            # Saving the descriptive statistics to an HTML file
            descriptive_stats_path = os.path.join(self.template_dir, 'descriptive_statistics.html')
            descriptive_stats.to_html(descriptive_stats_path, classes='table table-bordered', justify='center')

            return descriptive_stats_path
        except Exception as e:
            logging.error(f"Error generating descriptive statistics: {e}")
            return None

    def generate_data_quality_report(self, df):
        """Generates a data quality report for the dataframe and saves it as an HTML file."""
        try:
            # Initializing an empty dictionary to store data quality metrics
            data_quality_report = {}

            # Checking for missing values
            data_quality_report['missing_values'] = df.isnull().sum()

            # Checking for duplicate rows
            data_quality_report['duplicate_rows'] = df.duplicated().sum()

            # Checking data types of each column
            data_quality_report['data_types'] = df.dtypes

            # Checking for outliers using Z-score
            from scipy.stats import zscore
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            data_quality_report['outliers'] = df[numeric_cols].apply(lambda x: np.abs(zscore(x)) > 3).sum()

            # Converting the dictionary to a DataFrame
            data_quality_df = pd.DataFrame(data_quality_report)

            # Saving the data quality report to an HTML file
            data_quality_report_path = os.path.join(self.template_dir, 'data_quality_report.html')
            data_quality_df.to_html(data_quality_report_path, classes='table table-bordered', justify='center')

            return data_quality_report_path
        except Exception as e:
            logging.error(f"Error generating data quality report: {e}")
            return None

    def main(self):
        """Main method to load data and generate EDA report."""
        try:
            logging.info("Starting main method")
            collection_name = 'pre_game_data'
            df = self.load_and_process_data(collection_name)

            return self.generate_eda_report(df)
        except Exception as e:
            logging.error(f"Error in main: {e}")
            return None, None


# analyzer = NFLDataAnalyzer()
# analyzer.main()
