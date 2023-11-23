import yaml
import logging
from .all_columns import ALL_COLUMNS
from classes.config_manager import ConfigManager
import random


class ColumnSelector:
    def __init__(self):
        """
        Initializes the ColumnSelector class.

        Loads configuration and sets constants like PAGE_SIZE and TARGET_VARIABLE.
        """
        self.config = ConfigManager()
        self.PAGE_SIZE = 20
        self.TARGET_VARIABLE = self.config.get_constant('TARGET_VARIABLE')

    def display_columns(self, columns, page):
        """
        Displays a paginated list of columns.

        Parameters:
        columns (list): The list of columns to display.
        page (int): The page number to display.
        """
        start_index = page * self.PAGE_SIZE
        end_index = start_index + self.PAGE_SIZE
        for i, col in enumerate(columns[start_index:end_index], start=start_index):
            logging.info(f"{i+1}. {col}")

    def get_user_selection(self, selected_columns):
        """
        Processes the user's selection of columns.

        Parameters:
        selected_columns (list): List of selected columns or special keywords like 'all' or 'random'.

        Returns:
        list: A list of selected columns based on user input.
        """
        if 'all' in selected_columns:
            return list(ALL_COLUMNS)

        if 'random' in selected_columns:
            num = random.randint(1, len(ALL_COLUMNS))
            return random.sample(ALL_COLUMNS, num)

        return selected_columns

    def generate_constants_file(self, selected_columns):
        """
        Generates a YAML file defining constants based on selected columns.

        Parameters:
        selected_columns (list): The list of selected columns to include in the constants file.
        """
        constants = {
            'COLUMNS_TO_KEEP': [self.TARGET_VARIABLE] + selected_columns
        }

        with open("./flask_app/static/constants.yaml", "w") as file:
            yaml.dump(constants, file)


def main():
    """
    Main function for the column selection process.

    Steps:
    1. Display available columns.
    2. Get user selection for columns.
    3. Generate a constants file based on the selection.
    """
    selector = ColumnSelector()
    selector.display_columns(ALL_COLUMNS, 0)

    # Assuming `selected_columns` is fetched from a user interface or input
    selected_columns = selector.get_user_selection()  # Update this line to fetch user input
    selector.generate_constants_file(selected_columns)

    logging.info("constants.yaml file has been generated.")


if __name__ == "__main__":
    main()
