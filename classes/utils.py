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
