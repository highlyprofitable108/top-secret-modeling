import random
from .all_columns import ALL_COLUMNS

PAGE_SIZE = 20


def display_columns(columns, page):
    """Displays the columns with indices to the user."""
    start_index = page * PAGE_SIZE
    end_index = start_index + PAGE_SIZE
    for i, col in enumerate(columns[start_index:end_index], start=start_index):
        print(f"{i+1}. {col}")


def get_user_selection():
    """Prompts the user to select columns and returns the selected columns."""
    selected_columns = set()
    page = 0
    total_pages = (len(ALL_COLUMNS) - 1) // PAGE_SIZE + 1

    while True:
        print(f"Page {page+1}/{total_pages}")
        display_columns(ALL_COLUMNS, page)
        user_input = input("Enter indices to add/remove (comma separated), 'g' to generate the file, 'q' to quit, 'all' to select all, 'random' for random selection, or n/p for next/previous page: ").strip()

        if user_input.lower() == 'n':
            page = min(page + 1, total_pages - 1)
        elif user_input.lower() == 'q':
            exit()
        elif user_input.lower() == 'g':
            break
        elif user_input.lower() == 'p':
            page = max(page - 1, 0)
        elif user_input.lower() == 'done':
            break
        elif user_input.lower() == 'all':
            selected_columns = set(ALL_COLUMNS)
        elif user_input.lower() == 'random':
            while True:
                try:
                    num = int(input("Enter the number of random columns to select: "))
                    if 0 < num <= len(ALL_COLUMNS):
                        selected_columns = set(random.sample(ALL_COLUMNS, num))
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(ALL_COLUMNS)}.")
                except ValueError:
                    print("Invalid input. Please enter a valid number.")
        else:
            indices = [int(index) - 1 for index in user_input.split(",") if index.isdigit()]
            for index in indices:
                if index < len(ALL_COLUMNS):
                    column = ALL_COLUMNS[index]
                    if column in selected_columns:
                        selected_columns.remove(column)
                    else:
                        selected_columns.add(column)

    return list(selected_columns)


def generate_constants_file(selected_columns):
    """Generates the constants.py file with the selected columns."""
    with open("./scripts/constants.py", "w") as file:
        file.write("# constants.py\n\n")
        file.write("# List of column names to keep\n")
        file.write("COLUMNS_TO_KEEP = [\n")
        file.write('    "scoring_differential",\n')  # Ensure scoring_differential is always included
        for col in selected_columns:
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
