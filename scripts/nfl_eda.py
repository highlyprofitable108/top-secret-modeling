from classes.config_manager import ConfigManager
from classes.data_processing import DataProcessing
from classes.database_operations import DatabaseOperations
import os
import warnings
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import logging
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.inspection import permutation_importance
import shap

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

            # Descriptive Statistics (Integration of suggestion 2)
            descriptive_stats = df.describe()
            descriptive_stats.to_csv(os.path.join(self.static_dir, 'descriptive_statistics.csv'))

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
            heatmap_path = os.path.join(self.static_dir, 'heatmap.png')

            # Using seaborn to create a simpler heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=1, linecolor='black')
            plt.title('Correlation Heatmap', fontsize=16)

            # Save the plot as an HTML file
            plt.savefig(heatmap_path, format='png')

            return heatmap_path
        except Exception as e:
            logging.error(f"Error generating correlation heatmap: {e}")
            return None

    def plot_feature_importance(self, df):
        """Plots the feature importance using a RandomForestRegressor."""
        try:
            X = df.drop(columns=[self.target_variable])
            y = df[self.target_variable]

            # Hyperparameter tuning (you can add more parameters here)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
            }
            model = GridSearchCV(RandomForestRegressor(), param_grid, cv=3)
            model.fit(X, y)

            # Get feature importances
            importances = model.best_estimator_.feature_importances_
            feature_names = X.columns

            # Create a DataFrame to hold the feature names and their importance scores
            feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

            # Create a bar plot using Plotly
            fig = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h', title='Feature Importance')

            # Save the plot as an HTML file
            feature_importance_path = os.path.join(self.static_dir, 'feature_importance.html')
            fig.write_html(feature_importance_path)

            # Logging the best model's score
            logging.info(f"Best model score: {model.best_score_}")

            return feature_importance_path
        except Exception as e:
            logging.error(f"Error generating feature importance plot: {e}")
            return None

    def plot_histograms(self, df):
        """Plots histograms for each numerical column in the dataframe."""
        try:
            # Create a directory to store all histogram plots
            histograms_dir = os.path.join(self.static_dir, 'histograms')
            os.makedirs(histograms_dir, exist_ok=True)

            # Loop through each numerical column and create a histogram
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                fig = px.histogram(df, x=col, title=f'Histogram of {col}', nbins=30)
                fig.write_html(os.path.join(histograms_dir, f'{col}_histogram.html'))

            logging.info("Histograms generated successfully")
            return histograms_dir

        except Exception as e:
            logging.error(f"Error generating histograms: {e}")
            return None

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

    """
    # def plot_pairplots(self, df):
        # Your code to plot pair plots
        # pass

    # def plot_time_series(self, df):
        # Your code to plot time series
        # pass

    def plot_cluster_analysis(self, df):
        # Performs cluster analysis and plots the results.
        try:
            # Selecting a subset of columns for clustering (you can modify this)
            columns_to_cluster = df.select_dtypes(include=[float, int]).columns.tolist()
            df_cluster = df[columns_to_cluster]

            # Applying KMeans clustering
            kmeans = KMeans(n_clusters=3)  # You can change the number of clusters
            df['Cluster'] = kmeans.fit_predict(df_cluster)

            # Visualizing the clusters using a scatter plot (modify x and y to columns of your choice)
            fig = px.scatter(df, x='column1', y='column2', color='Cluster', 
                            title='Cluster Analysis', template='plotly_dark')

            # Saving the plot
            cluster_analysis_path = os.path.join(self.static_dir, 'cluster_analysis.html')
            fig.write_html(cluster_analysis_path)

            return cluster_analysis_path
        except Exception as e:
            logging.error(f"Error generating cluster analysis plot: {e}")
            return None

    def perform_anomaly_detection(self, df):
        # Your code to perform anomaly detection
        pass

    def conduct_statistical_tests(self, df):
        # Your code to conduct statistical tests
        pass
    """

    def generate_descriptive_statistics(self, df):
        """Generates descriptive statistics for each column in the dataframe and saves it as a CSV file."""
        try:
            # Generating descriptive statistics
            descriptive_stats = df.describe(include='all')

            # Saving the descriptive statistics to a CSV file
            descriptive_stats_path = os.path.join(self.static_dir, 'descriptive_statistics.csv')
            descriptive_stats.to_csv(descriptive_stats_path)

            return descriptive_stats_path
        except Exception as e:
            logging.error(f"Error generating descriptive statistics: {e}")
            return None

    def generate_data_quality_report(self, df):
        """Generates a data quality report for the dataframe and saves it as a CSV file."""
        try:
            # Initializing an empty dictionary to store data quality metrics
            data_quality_report = {}

            # Checking for missing values
            data_quality_report['missing_values'] = df.isnull().sum()

            # Checking for duplicate rows
            data_quality_report['duplicate_rows'] = df.duplicated().sum()

            # Checking data types of each column
            data_quality_report['data_types'] = df.dtypes

            # Checking for outliers using Z-score (you can use other methods as well)
            from scipy.stats import zscore
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            data_quality_report['outliers'] = df[numeric_cols].apply(lambda x: np.abs(zscore(x)) > 3).sum()

            # Converting the dictionary to a DataFrame
            data_quality_df = pd.DataFrame(data_quality_report)

            # Saving the data quality report to a CSV file
            data_quality_report_path = os.path.join(self.static_dir, 'data_quality_report.csv')
            data_quality_df.to_csv(data_quality_report_path)

            return data_quality_report_path
        except Exception as e:
            logging.error(f"Error generating data quality report: {e}")
            return None

    def generate_eda_report(self, df):
        """Generates an EDA report with various analyses and saves it as image files."""
        try:
            logging.info("Generating EDA report")

            # Paths to save the generated plots
            heatmap_path = self.plot_correlation_heatmap(df)
            feature_importance_path = self.plot_feature_importance(df)
            # histogram_path = self.plot_histograms(df)
            # boxplot_path = self.plot_boxplots(df)
            descriptive_stats_path = self.generate_descriptive_statistics(df)
            data_quality_report_path = self.generate_data_quality_report(df)

            return heatmap_path, feature_importance_path, descriptive_stats_path, data_quality_report_path

        except Exception as e:
            logging.error(f"Error in generate_eda_report: {e}")
            return None, None, None, None, None

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


# analyzer = NFLDataAnalyzer()
# analyzer.main()
