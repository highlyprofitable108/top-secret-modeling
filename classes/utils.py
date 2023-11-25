import logging
from collections import defaultdict


def get_active_constants(feature_columns, TARGET_VARIABLE):
    active_constants = list(set(col for col in feature_columns if col != TARGET_VARIABLE))
    active_constants.sort()

    # Categorizing the constants
    categories = defaultdict(list)
    for constant in active_constants:
        constant = constant.replace(' Difference', '').replace(' Ratio', '')
        category = constant.split('.')[0]
        categories[category].append(constant)

    # Sorting the constants within each category
    def sort_key(x):
        parts = x.split('.')
        return parts[1] if len(parts) > 1 else parts[0]

    sorted_categories = {category: sorted(sub_categories, key=sort_key) for category, sub_categories in categories.items()}
    return sorted_categories


def setup_logging():
    logger = logging.getLogger('my_app')
    logger.setLevel(logging.INFO)

    # Create a file handler with 'w' mode to overwrite the existing file
    handler = logging.FileHandler('my_app.log', mode='w')
    # Or for stdout: handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Remove all handlers associated with the logger object.
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)

    logger.addHandler(handler)
    return logger
