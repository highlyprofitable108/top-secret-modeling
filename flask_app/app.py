from scripts import nfl_stats_select, constants
from scripts.nfl_eda import NFLDataAnalyzer
from scripts.nfl_model import NFLModel
from scripts.nfl_populate_stats import load_and_process_data, transform_data, calculate_power_rank, aggregate_and_normalize_data, insert_aggregated_data_into_database, LOADED_MODEL
from classes.config_manager import ConfigManager
from classes.data_processing import DataProcessing
from classes.database_operations import DatabaseOperations
from flask import Flask, render_template, request, redirect, url_for, jsonify
# , flash, session
from collections import defaultdict
from datetime import datetime
import os
import importlib

app = Flask(__name__)
app.secret_key = '4815162342'  # Change this to a random secret key

# Initialize ConfigManager, DatabaseOperations, and DataProcessing
config = ConfigManager()
database_operations = DatabaseOperations()
data_processing = DataProcessing()
analyzer = NFLDataAnalyzer()

# Fetch configurations using ConfigManager
data_dir = config.get_config('paths', 'data_dir')
model_dir = config.get_config('paths', 'model_dir')
database_name = config.get_config('database', 'database_name')
feature_columns = [col for col in constants.COLUMNS_TO_KEEP if col != 'scoring_differential']


@app.route('/')
def home():
    # Regenerate the active constants list
    active_constants = constants.COLUMNS_TO_KEEP  # Assuming get_active_constants is a function that returns the list of active constants
    active_constants = [col.replace('statistics_*.', '') for col in active_constants if 'scoring_differential' not in col]
    active_constants = sorted(set(active_constants))

    # Reload the constants module to get the updated list of active constants
    importlib.reload(constants)

    return render_template('index.html', active_constants=active_constants)


@app.route('/columns')
def columns():
    columns = nfl_stats_select.ALL_COLUMNS
    categorized_columns = defaultdict(list)
    for column in columns:
        prefix = column.split('.')[0]
        categorized_columns[prefix].append(column)

    # Get the active constants from the constants.py file
    active_constants = constants.COLUMNS_TO_KEEP  # Assuming COLUMNS_TO_KEEP is the variable holding the list of active constants in constants.py
    active_constants = [col.replace('statistics_away.', '') for col in active_constants if 'scoring_differential' not in col]
    active_constants = [col.replace('statistics_home.', '') for col in active_constants if 'scoring_differential' not in col]
    active_constants = sorted(set(active_constants))

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
    # Call the modified get_user_selection function with the selected columns
    selected_columns = nfl_stats_select.get_user_selection(selected_columns)

    # Now, selected_columns contains the processed list of selected columns
    # You can now generate the constants file with these columns
    nfl_stats_select.generate_constants_file(selected_columns)

    # Regenerate the active constants list
    active_constants = constants.COLUMNS_TO_KEEP  # Assuming get_active_constants is a function that returns the list of active constants
    active_constants = [col.replace('statistics_*.', '') for col in active_constants if 'scoring_differential' not in col]
    active_constants = sorted(set(active_constants))

    # Reload the constants module to get the updated list of active constants
    importlib.reload(constants)

    # Redirect the user back to the columns page
    return redirect(url_for('columns'))


@app.route('/generate_analysis', methods=['POST'])
def generate_analysis():
    collection_name = request.json.get('collection_name', 'games')
    df = analyzer.load_and_process_data(collection_name)
    analyzer.generate_eda_report(df)

    # Create an instance of the NFLModel class and call the main method
    nfl_model = NFLModel()
    nfl_model.main()

    # Call the main function from nfl_populate_stats.py
    processed_games_df, processed_teams_df = load_and_process_data(database_operations, data_processing)

    if processed_games_df is not None and processed_teams_df is not None:
        df_home, df_away = transform_data(processed_games_df, processed_teams_df, data_processing, feature_columns)

        if df_home is not None and df_away is not None:
            metrics_home = [metric for metric in feature_columns if metric.startswith('statistics_home')]
            metrics_away = [metric for metric in feature_columns if metric.startswith('statistics_away')]
            feature_importances = LOADED_MODEL.feature_importances_
            weights = dict(zip(feature_columns, feature_importances))
            df = calculate_power_rank(df_home, df_away, metrics_home, metrics_away, weights)

            if df is not None:
                cleaned_metrics = [metric.replace('statistics_home.', '').replace('statistics_away.', '') for metric in feature_columns]
                aggregated_df = aggregate_and_normalize_data(df, cleaned_metrics, database_operations, processed_teams_df)

                if aggregated_df is not None:
                    insert_aggregated_data_into_database(aggregated_df, database_operations)
                    # Further script implementation here, where you can use aggregated_df for analysis
                else:
                    return jsonify(error="Error in aggregating and normalizing data."), 500
            else:
                return jsonify(error="Error in calculating power rank."), 500
        else:
            return jsonify(error="Error in data transformation."), 500
    else:
        return jsonify(error="Error in data loading and processing."), 500

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
