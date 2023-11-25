from flask import Blueprint, jsonify, render_template, request, current_app
from scripts.nfl_stats_select import ColumnSelector
from collections import defaultdict, OrderedDict
from scripts import all_columns
import os
import yaml

stats_bp = Blueprint('stats', __name__)


def load_constants():
    static_dir = current_app.config.get('static_dir')
    with open(os.path.join(static_dir, 'constants.yaml'), 'r') as file:
        return yaml.safe_load(file)


@stats_bp.route('/columns')
def columns():
    columns = all_columns.ALL_COLUMNS
    categorized_columns = defaultdict(list)
    for column in columns:
        prefix, rest = column.split('.', 1)
        prefix_display = prefix.replace('_', ' ').title()
        formatted_column = rest.replace('_', ' ').title()
        categorized_columns[prefix_display].append((formatted_column, column))

    # Define the order for the key categories
    key_categories = ["Passing", "Rushing", "Receiving", "Summary", "Efficiency", "Defense"]

    # Order the categorized_columns dictionary based on the key categories
    ordered_categorized_columns = OrderedDict()
    for key in key_categories:
        if key in categorized_columns:
            ordered_categorized_columns[key] = categorized_columns[key]
            del categorized_columns[key]

    # Add the remaining categories in alphabetical order
    for key in sorted(categorized_columns.keys()):
        ordered_categorized_columns[key] = categorized_columns[key]

    return render_template('columns.html', ordered_categorized_columns=ordered_categorized_columns)


@stats_bp.route('/process_columns', methods=['POST'])
def process_columns():
    nfl_stats = ColumnSelector()
    selected_columns = request.form.getlist('columns')
    selected_columns = nfl_stats.get_user_selection(selected_columns)

    current_app.logger.info(selected_columns)
    ratio_columns = [
        "summary.avg_gain",
        "rushing.totals.avg_yards",
        "receiving.totals.avg_yards",
        "receiving.totals.yards_after_catch",
        "punts.totals.avg_net_yards",
        "punts.totals.avg_yards",
        "punts.totals.avg_hang_time",
        "passing.totals.cmp_pct",
        "passing.totals.rating",
        "passing.totals.avg_yards",
        "field_goals.totals.pct",
        "efficiency.goaltogo.pct",
        "efficiency.redzone.pct",
        "efficiency.thirddown.pct",
        "efficiency.fourthdown.pct"
    ]

    modified_columns = []
    for column in selected_columns:
        if column in ratio_columns:
            modified_columns.append(column + '_ratio')
        else:
            modified_columns.append(column + '_difference')

    nfl_stats.generate_constants_file(modified_columns)
    refresh_config()

    return jsonify(success=True)


@stats_bp.route('/refresh-config')
def refresh_config():
    constants = load_constants()
    TARGET_VARIABLE = current_app.config.get('TARGET_VARIABLE')
    feature_columns = list(set(constants.get('feature_columns', [])) - {TARGET_VARIABLE})
    return jsonify(feature_columns)
