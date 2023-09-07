import pandas as pd
from pymongo import MongoClient
from classes.config_manager import ConfigManager


class DatabaseOperations:
    def __init__(self):
        self.config = ConfigManager()
        self.mongo_uri = self.config.get_config('database', 'mongo_uri')
        self.database_name = self.config.get_config('database', 'database_name')
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.database_name]

    def fetch_data_from_mongodb(self, collection_name):
        """Fetch data from a MongoDB collection."""
        try:
            cursor = self.db[collection_name].find()
            df = pd.DataFrame(list(cursor))
            return df
        except Exception as e:
            print(f"Error fetching data from MongoDB: {e}")
            # Handle the error appropriately (e.g., return an empty DataFrame or exit the script)
            return pd.DataFrame()

    def insert_data_into_mongodb(self, collection_name, data):
        """Insert data into a MongoDB collection."""
        try:
            self.db[collection_name].insert_many(data)
            print(f"Data inserted successfully into {collection_name} collection.")
        except Exception as e:
            print(f"Error inserting data into MongoDB: {e}")

    def update_data_in_mongodb(self, collection_name, query, new_values):
        """Update data in a MongoDB collection."""
        try:
            self.db[collection_name].update_many(query, {'$set': new_values})
            print(f"Data updated successfully in {collection_name} collection.")
        except Exception as e:
            print(f"Error updating data in MongoDB: {e}")

    def delete_data_from_mongodb(self, collection_name, query):
        """Delete data from a MongoDB collection."""
        try:
            self.db[collection_name].delete_many(query)
            print(f"Data deleted successfully from {collection_name} collection.")
        except Exception as e:
            print(f"Error deleting data from MongoDB: {e}")

# Usage example:
# db_operations = DatabaseOperations()
# db_operations.fetch_data_from_mongodb('collection_name')
