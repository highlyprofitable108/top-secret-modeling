import random
from .all_columns import ALL_COLUMNS

PAGE_SIZE = 20


def display_columns(columns, page):
    """Displays the columns with indices to the user."""
    start_index = page * PAGE_SIZE
    end_index = start_index + PAGE_SIZE
    for i, col in enumerate(columns[start_index:end_index], start=start_index):
        print(f"{i+1}. {col}")


def get_user_selection(selected_columns):
    """Receives the selected columns from the web interface and returns the selected columns."""

    # If 'all' is selected, return all columns
    if 'all' in selected_columns:
        return list(ALL_COLUMNS)

    # If 'random' is selected, return a random selection of columns
    if 'random' in selected_columns:
        num = random.randint(1, len(ALL_COLUMNS))  # You might want to change this to a fixed number or get it from the web interface
        return random.sample(ALL_COLUMNS, num)

    # Otherwise, return the selected columns
    return selected_columns


def generate_constants_file(selected_columns):
    """Generates the constants.py file with the selected columns."""
    with open("./scripts/constants.py", "w") as file:
        file.write("# constants.py\n\n")
        file.write("# List of column names to keep\n")
        file.write("COLUMNS_TO_KEEP = [\n")
        file.write('    "scoring_differential",\n')  # Ensure scoring_differential is always included
        for col in selected_columns:
            if 'summary' not in col and 'efficiency' not in col:
                # Find the last occurrence of '.' and replace it with '.totals.'
                last_dot_index = col.rfind('.')
                if last_dot_index != -1:  # Check if '.' is found in the string
                    col = col[:last_dot_index] + '.totals.' + col[last_dot_index + 1:]
            file.write(f'    "statistics_home.{col}",\n')
            file.write(f'    "statistics_away.{col}",\n')
        file.write("]\n")


def main():
    """Main function to display columns, get user selection, and generate constants file."""
    display_columns(ALL_COLUMNS, 0)
    selected_columns = get_user_selection()
    generate_constants_file(selected_columns)
    print("constants.py file has been generated.")


if __name__ == "__main__":
    main()
