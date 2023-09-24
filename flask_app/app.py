from scripts import nfl_stats_select, constants
from scripts.nfl_eda import NFLDataAnalyzer
from scripts.nfl_model import NFLModel
from scripts.nfl_populate_stats import StatsCalculator
from classes.config_manager import ConfigManager
from classes.data_processing import DataProcessing
from classes.database_operations import DatabaseOperations
from flask import Flask, render_template, request, redirect, url_for, jsonify
from collections import defaultdict
from datetime import datetime
import os
import importlib
import pandas as pd

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default_secret_key')
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Initialize ConfigManager, DatabaseOperations, and DataProcessing
config = ConfigManager()
database_operations = DatabaseOperations()
data_processing = DataProcessing()
analyzer = NFLDataAnalyzer()

# Fetch configurations using ConfigManager
data_dir = config.get_config('paths', 'data_dir')
model_dir = config.get_config('paths', 'model_dir')
database_name = config.get_config('database', 'database_name')
feature_columns = list(set(col for col in constants.COLUMNS_TO_KEEP if col != 'scoring_differential'))


def get_active_constants():
    active_constants = list(set(col.replace('statistics_home.', '') for col in constants.COLUMNS_TO_KEEP if 'scoring_differential' not in col))
    active_constants = list(set(col.replace('statistics_away.', '') for col in active_constants))
    active_constants.sort()

    importlib.reload(constants)

    # Categorizing the constants
    categories = defaultdict(list)
    for constant in active_constants:
        category = constant.split('.')[0]
        categories[category].append(constant)

    # Sorting the constants within each category
    sorted_categories = {category: sorted(sub_categories, key=lambda x: x.split('.')[1]) for category, sub_categories in categories.items()}

    return sorted_categories


@app.route('/')
def home():
    active_constants = get_active_constants()

    return render_template('home.html', active_constants=active_constants)


@app.route('/columns')
def columns():
    columns = nfl_stats_select.ALL_COLUMNS
    categorized_columns = defaultdict(list)
    for column in columns:
        column = column.replace('totals.', '').replace('kicks.', 'kicks ').replace('conversions.', 'conversions ')
        prefix, rest = column.split('.', 1)
        prefix_display = prefix.replace('_', ' ').title()
        if prefix.lower() == "efficiency":
            formatted_column = rest.replace('.', ' ').replace('_', ' ').title()
        else:
            formatted_column = rest.replace('_', ' ').title()
        categorized_columns[prefix_display].append((formatted_column, column))

    active_constants = get_active_constants()

    return render_template('columns.html', categorized_columns=categorized_columns, active_constants=active_constants)


@app.route('/get_model_update_time')
def get_model_update_time():
    model_dir = "./models"
    files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]

    update_times = {}
    for file in files:
        file_path = os.path.join(model_dir, file)
        mod_time = os.path.getmtime(file_path)
        update_times[file] = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')

    return jsonify(update_times)


@app.route('/process_columns', methods=['POST'])
def process_columns():
    selected_columns = request.form.getlist('columns')
    selected_columns = nfl_stats_select.get_user_selection(selected_columns)
    nfl_stats_select.generate_constants_file(selected_columns)

    importlib.reload(constants)

    return jsonify(success=True)


@app.route('/generate_analysis', methods=['POST'])
def generate_analysis():
    try:
        analyzer.main()

    except Exception as e:
        return jsonify(error=str(e)), 500

    return jsonify(status="success")


@app.route('/generate_model', methods=['POST'])
def generate_model():
    try:
        # Create an instance of the NFLModel class
        nfl_model = NFLModel()

        # Call the main method to generate the model
        nfl_model.main()

        # If successful, return a success message
        return jsonify(status="success"), 200
    except Exception as e:
        # If there is an error, return an error message
        return jsonify(error=str(e)), 500


@app.route('/generate_power_ranks', methods=['POST'])
def generate_power_ranks():
    try:
        # Create an instance of the NFLModel class
        nfl_stats = StatsCalculator()

        # Call the main method to generate the model
        nfl_stats.main()

        # If successful, return a success message
        return jsonify(status="success"), 200
    except Exception as e:
        # If there is an error, return an error message
        return jsonify(error=str(e)), 500


@app.route('/interactive_heatmap')
def heatmap():
    return render_template('interactive_heatmap.html')


@app.route('/feature_importance')
def feature_importance():
    return render_template('feature_importance.html')


@app.route('/descriptive_statistics')
def descriptive_statistics():
    return render_template('descriptive_statistics.html')


@app.route('/data_quality_report')
def data_quality_report():
    return render_template('data_quality_report.html')


@app.route('/view_analysis')
def view_analysis():
    return render_template('view_analysis.html')


@app.route('/view_power_ranks')
def view_power_ranks():
    df = pd.read_csv(os.path.join(app.root_path, 'static', 'power_ranks.csv'))
    return render_template('power_ranks.html', table=df.to_html(classes='table table-striped'), title='Custom Power Rankings')


if __name__ == "__main__":
    app.run(debug=True)
