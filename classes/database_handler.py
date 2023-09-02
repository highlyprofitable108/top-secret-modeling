import sqlite3

class DatabaseHandler:
    def __init__(self, db_path):
        self.db_path = db_path
        self.connection = None
        self.cursor = None

    def connect(self):
        if not self.connection:
            self.connection = sqlite3.connect(self.db_path)
            self.cursor = self.connection.cursor()
        return self.connection

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None
            self.cursor = None

    def execute_query(self, query, params=None):
        """
        Execute a SQL query.
        
        Args:
        - query (str): The SQL query to execute.
        - params (tuple, optional): The parameters to bind to the query.
        
        Returns:
        - None
        """
        self.connect()
        if params:
            self.cursor.execute(query, params)
        else:
            self.cursor.execute(query)
        self.connection.commit()

    def fetch_all(self, query, params=None):
        """
        Execute a SQL query and fetch all results.
        
        Args:
        - query (str): The SQL query to execute.
        - params (tuple, optional): The parameters to bind to the query.
        
        Returns:
        - list: The list of results.
        """
        self.connect()
        if params:
            self.cursor.execute(query, params)
        else:
            self.cursor.execute(query)
        return self.cursor.fetchall()

    def fetch_one(self, query, params=None):
        """
        Execute a SQL query and fetch one result.
        
        Args:
        - query (str): The SQL query to execute.
        - params (tuple, optional): The parameters to bind to the query.
        
        Returns:
        - tuple: The result.
        """
        self.connect()
        if params:
            self.cursor.execute(query, params)
        else:
            self.cursor.execute(query)
        return self.cursor.fetchone()

    def get_db_path(self):
        """
        Return the path to the database.
        """
        return self.db_path
