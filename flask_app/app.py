from scripts import nfl_stats_select, constants
from scripts.nfl_eda import NFLDataAnalyzer
from scripts.nfl_model import NFLModel
from scripts.nfl_populate_stats import load_and_process_data, transform_data, calculate_power_rank, aggregate_and_normalize_data, insert_aggregated_data_into_database, LOADED_MODEL
from classes.config_manager import ConfigManager
from classes.data_processing import DataProcessing
from classes.database_operations import DatabaseOperations
from flask import Flask, render_template, request, redirect, url_for, jsonify
from collections import defaultdict
from datetime import datetime
import os
import importlib

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default_secret_key')

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
    active_constants = list(set(col.replace('statistics_*.', '') for col in constants.COLUMNS_TO_KEEP if 'scoring_differential' not in col))
    active_constants.sort()
    importlib.reload(constants)
    return active_constants


def load_and_process_collection_data(collection_name):
    return analyzer.load_and_process_data(collection_name)


def generate_eda_report(df):
    return analyzer.generate_eda_report(df)


def initialize_and_run_model():
    nfl_model = NFLModel()
    return nfl_model.main()


def load_and_process_stats_data(database_operations, data_processing):
    return load_and_process_data(database_operations, data_processing)


def transform_stats_data(processed_games_df, processed_teams_df, data_processing, feature_columns):
    return transform_data(processed_games_df, processed_teams_df, data_processing, feature_columns)


def calculate_power_ranking(df_home, df_away, feature_columns):
    metrics_home = [metric for metric in feature_columns if metric.startswith('statistics_home')]
    metrics_away = [metric for metric in feature_columns if metric.startswith('statistics_away')]
    feature_importances = LOADED_MODEL.feature_importances_
    weights = dict(zip(feature_columns, feature_importances))
    return calculate_power_rank(df_home, df_away, metrics_home, metrics_away, weights)


def aggregate_and_normalize_stats_data(df, feature_columns, database_operations, processed_teams_df):
    cleaned_metrics = [metric.replace('statistics_home.', '').replace('statistics_away.', '') for metric in feature_columns]
    return aggregate_and_normalize_data(df, cleaned_metrics, database_operations, processed_teams_df)


def insert_aggregated_data_to_db(aggregated_df, database_operations):
    return insert_aggregated_data_into_database(aggregated_df, database_operations)


@app.route('/')
def home():
    active_constants = get_active_constants()
    return render_template('index.html', active_constants=active_constants)


@app.route('/columns')
def columns():
    columns = nfl_stats_select.ALL_COLUMNS
    categorized_columns = defaultdict(list)
    for column in columns:
        prefix = column.split('.')[0]
        categorized_columns[prefix].append(column)

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

    return redirect(url_for('columns'))


@app.route('/generate_analysis', methods=['POST'])
def generate_analysis():
    try:
        collection_name = request.json.get('collection_name', 'games')
        df = load_and_process_collection_data(collection_name)
        generate_eda_report(df)

        initialize_and_run_model()

        processed_games_df, processed_teams_df = load_and_process_stats_data(database_operations, data_processing)

        if processed_games_df is None or processed_teams_df is None:
            raise ValueError("Error in data loading and processing.")

        df_home, df_away = transform_stats_data(processed_games_df, processed_teams_df, data_processing, feature_columns)

        if df_home is None or df_away is None:
            raise ValueError("Error in data transformation.")

        df = calculate_power_ranking(df_home, df_away, feature_columns)

        if df is None:
            raise ValueError("Error in calculating power rank.")

        aggregated_df = aggregate_and_normalize_stats_data(df, feature_columns, database_operations, processed_teams_df)

        if aggregated_df is None:
            raise ValueError("Error in aggregating and normalizing data.")

        insert_aggregated_data_to_db(aggregated_df, database_operations)

    except Exception as e:
        return jsonify(error=str(e)), 500

    return jsonify(status="success")


@app.route('/view_analysis')
def view_analysis():
    data = {
        "heatmap_path": "/static/heatmap.png",
        "feature_importance_path": "/static/feature_importance.png"
    }

    return render_template('view_analysis.html', data=data)


if __name__ == "__main__":
    app.run(debug=True)
