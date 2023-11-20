import logging
import pandas as pd
from pymongo import MongoClient
from classes.config_manager import ConfigManager


class DatabaseOperations:
    """
    Class to manage operations such as fetching, inserting, updating, and deleting data in MongoDB.
    """

    def __init__(self):
        """
        Initializes the DatabaseOperations class with MongoDB configuration.
        """
        self.config = ConfigManager()
        self.mongo_uri = self.config.get_config('database', 'mongo_uri')
        self.database_name = self.config.get_config('database', 'database_name')
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.database_name]
        self.logger = logging.getLogger(__name__)

    def fetch_data_from_mongodb(self, collection_name: str) -> pd.DataFrame:
        """
        Fetches data from a specified MongoDB collection and converts it to a DataFrame.

        Args:
            collection_name (str): Name of the MongoDB collection.

        Returns:
            pd.DataFrame: DataFrame containing data from the collection.
        """
        try:
            # Retrieve data from MongoDB collection
            cursor = self.db[collection_name].find()
            df = pd.DataFrame(list(cursor))
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data from MongoDB: {e}")
            return pd.DataFrame()

    def insert_data_into_mongodb(self, collection_name: str, data: dict) -> None:
        """
        Inserts data into a specified MongoDB collection.

        Args:
            collection_name (str): Name of the MongoDB collection.
            data (dict): Data to be inserted into the collection.
        """
        try:
            # Insert data into MongoDB collection
            self.db[collection_name].insert_many(data)
            self.logger.info(f"Data inserted successfully into {collection_name} collection.")
        except Exception as e:
            self.logger.error(f"Error inserting data into MongoDB: {e}")

    def update_data_in_mongodb(self, collection_name: str, query: dict, new_values: dict) -> None:
        """
        Updates data in a specified MongoDB collection.

        Args:
            collection_name (str): Name of the MongoDB collection.
            query (dict): Query to identify the documents to update.
            new_values (dict): New values to be updated in the documents.
        """
        try:
            # Update data in MongoDB collection
            self.db[collection_name].update_many(query, {'$set': new_values})
            self.logger.info(f"Data updated successfully in {collection_name} collection.")
        except Exception as e:
            self.logger.error(f"Error updating data in MongoDB: {e}")

    def delete_data_from_mongodb(self, collection_name: str, query: dict) -> None:
        """
        Deletes data from a specified MongoDB collection.

        Args:
            collection_name (str): Name of the MongoDB collection.
            query (dict): Query to identify the documents to delete.
        """
        try:
            # Delete data from MongoDB collection
            self.db[collection_name].delete_many(query)
            self.logger.info(f"Data deleted successfully from {collection_name} collection.")
        except Exception as e:
            self.logger.error(f"Error deleting data from MongoDB: {e}")

    def fetch_game_data(self):
        """
        Fetches game data from the 'games' collection.

        Returns:
            pd.DataFrame: DataFrame containing game data.
        """
        return self.fetch_data_from_mongodb("games")

    def fetch_team_data(self):
        """
        Fetches team data from the 'teams' collection.

        Returns:
            pd.DataFrame: DataFrame containing team data.
        """
        return self.fetch_data_from_mongodb("teams")
