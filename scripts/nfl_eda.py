from classes.config_manager import ConfigManager
from classes.data_processing import DataProcessing
from classes.database_operations import DatabaseOperations
import warnings
import pandas as pd
import sweetviz as sv
from .constants import COLUMNS_TO_KEEP
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class NFLDataAnalyzer:
    """A class to analyze NFL data and generate EDA reports."""

    def __init__(self):
        """Initializes the NFLDataAnalyzer with necessary configurations and setups."""
        self.config_manager = ConfigManager()
        self.db_operations = DatabaseOperations()
        self.data_processing = DataProcessing()
        self.target_variable = 'scoring_differential'
        self.data_dir = self.config_manager.get_config('paths', 'data_dir')
        warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

    def load_and_process_data(self, collection_name: str) -> pd.DataFrame:
        """Loads and processes data from the specified MongoDB collection."""
        try:
            logging.info(f"Loading and processing data from collection: {collection_name}")
            # Fetch data from MongoDB
            df = self.db_operations.fetch_data_from_mongodb(collection_name)
            # Flatten and merge data
            df = self.data_processing.flatten_and_merge_data(df)
            # Calculate scoring differential
            df = self.data_processing.calculate_scoring_differential(df)
            # Keep only the columns specified in COLUMNS_TO_KEEP
            df = df[COLUMNS_TO_KEEP]
            return df
        except Exception as e:
            logging.error(f"Error in load_and_process_data: {e}")
            return pd.DataFrame()

    def generate_eda_report(self, df: pd.DataFrame):
        """Generates an EDA report using Sweetviz and saves it as an HTML file."""
        try:
            logging.info("Generating EDA report")
            # Generate EDA report using Sweetviz
            report = sv.analyze(df)
            # Create a filepath with data_dir prefix
            filepath = f"{self.data_dir}/eda_report.html"
            # Show the report
            report.show_html(filepath)
        except Exception as e:
            logging.error(f"Error in generate_eda_report: {e}")

    def main(self):
        """Main method to load data and generate EDA report."""
        try:
            logging.info("Starting main method")
            # Specify the collection name
            collection_name = 'games'
            # Load and process data
            df = self.load_and_process_data(collection_name)
            # Generate EDA report
            self.generate_eda_report(df)
        except Exception as e:
            logging.error(f"Error in main: {e}")


if __name__ == "__main__":
    analyzer = NFLDataAnalyzer()
    analyzer.main()
