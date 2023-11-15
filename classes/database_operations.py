import pandas as pd
from pymongo import MongoClient
from classes.config_manager import ConfigManager
import logging


class DatabaseOperations:
    """
    A class to manage operations such as fetching, inserting, updating, and deleting data in a MongoDB database.
    """

    def __init__(self):
        """
        Initializes the DatabaseOperations class with configurations for MongoDB connection.
        """
        self.config = ConfigManager()
        self.mongo_uri = self.config.get_config('database', 'mongo_uri')
        self.database_name = self.config.get_config('database', 'database_name')
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.database_name]
        self.logger = logging.getLogger(__name__)

    def fetch_data_from_mongodb(self, collection_name: str) -> pd.DataFrame:
        """
        Fetch data from a MongoDB collection.

        Args:
            collection_name (str): The name of the MongoDB collection to fetch data from.

        Returns:
            pd.DataFrame: The data fetched from the MongoDB collection as a DataFrame.
        """
        try:
            cursor = self.db[collection_name].find()
            df = pd.DataFrame(list(cursor))

            return df
        except Exception as e:
            self.logger.error(f"Error fetching data from MongoDB: {e}")
            return pd.DataFrame()

    def insert_data_into_mongodb(self, collection_name: str, data: dict) -> None:
        """
        Insert data into a MongoDB collection.

        Args:
            collection_name (str): The name of the MongoDB collection to insert data into.
            data (dict): The data to insert into the MongoDB collection.

        Returns:
            None
        """
        try:
            self.db[collection_name].insert_many(data)
            self.logger.info(f"Data inserted successfully into {collection_name} collection.")
        except Exception as e:
            self.logger.error(f"Error inserting data into MongoDB: {e}")

    def update_data_in_mongodb(self, collection_name: str, query: dict, new_values: dict) -> None:
        """
        Update data in a MongoDB collection.

        Args:
            collection_name (str): The name of the MongoDB collection to update data in.
            query (dict): The query to select the documents to update.
            new_values (dict): The new values to update in the selected documents.

        Returns:
            None
        """
        try:
            self.db[collection_name].update_many(query, {'$set': new_values})
            self.logger.info(f"Data updated successfully in {collection_name} collection.")
        except Exception as e:
            self.logger.error(f"Error updating data in MongoDB: {e}")

    def delete_data_from_mongodb(self, collection_name: str, query: dict) -> None:
        """
        Delete data from a MongoDB collection.

        Args:
            collection_name (str): The name of the MongoDB collection to delete data from.
            query (dict): The query to select the documents to delete.

        Returns:
            None
        """
        try:
            self.db[collection_name].delete_many(query)
            self.logger.info(f"Data deleted successfully from {collection_name} collection.")
        except Exception as e:
            self.logger.error(f"Error deleting data from MongoDB: {e}")

    def fetch_game_data(self):
        return self.fetch_data_from_mongodb("games")

    def fetch_team_data(self):
        return self.fetch_data_from_mongodb("teams")


# Usage example:
# db_operations = DatabaseOperations()
# db_operations.fetch_data_from_mongodb('collection_name')
