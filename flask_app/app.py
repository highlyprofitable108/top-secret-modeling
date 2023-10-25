from scripts import nfl_stats_select, constants
from scripts.nfl_eda import NFLDataAnalyzer
from scripts.nfl_model import NFLModel
from scripts.nfl_populate_stats import StatsCalculator
from scripts.nfl_prediction import NFLPredictor
from classes.config_manager import ConfigManager
from classes.data_processing import DataProcessing
from classes.database_operations import DatabaseOperations
from flask import Flask, render_template, request, redirect, url_for, jsonify
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta
import os
import importlib
import logging

# Set up logging at the top of your app.py
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default_secret_key')
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Initialize ConfigManager, DatabaseOperations, and DataProcessing
config = ConfigManager()
database_operations = DatabaseOperations()
data_processing = DataProcessing()
analyzer = NFLDataAnalyzer()
nfl_model = NFLModel()
nfl_power_ranks = StatsCalculator()

# Fetch configurations using ConfigManager
data_dir = config.get_config('paths', 'data_dir')
model_dir = config.get_config('paths', 'model_dir')
database_name = config.get_config('database', 'database_name')
feature_columns = list(set(col for col in constants.COLUMNS_TO_KEEP if col != 'odds.spread'))


def get_active_constants():
    active_constants = list(set(col for col in constants.COLUMNS_TO_KEEP if col != 'odds.spread'))
    active_constants.sort()

    importlib.reload(constants)

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


@app.route('/')
def home():
    active_constants = get_active_constants()

    return render_template('home.html', active_constants=active_constants)


@app.route('/columns')
def columns():
    columns = nfl_stats_select.ALL_COLUMNS
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

    active_constants = get_active_constants()

    return render_template('columns.html', ordered_categorized_columns=ordered_categorized_columns, active_constants=active_constants)


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

    print(selected_columns)
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

    nfl_stats_select.generate_constants_file(modified_columns)

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
        # Call the main method of the NFLModel instance to generate the model
        nfl_model.main()

        # Return a success message
        return jsonify(status="success"), 200
    except Exception as e:
        # If there is an error, return an error message
        return jsonify(error=str(e)), 500


@app.route('/generate_power_ranks', methods=['POST'])
def generate_power_ranks():
    try:
        nfl_power_ranks.main()

        # If successful, return a success message
        return jsonify(status="success"), 200
    except Exception as e:
        # If there is an error, return an error message
        return jsonify(error=str(e)), 500


@app.route('/sim_runner', methods=['POST'])
def sim_runner():
    try:
        # Retrieve parameters from the POST request
        action = request.form.get('action')
        logger.info(f"Action: {action}")

        # Handle num_simulations with default value of 1000
        simIterations = request.form.get('simIterations')
        num_simulations = int(simIterations) if simIterations else 1000
        logger.info(f"Number of Simulations: {num_simulations}")

        # Handle date_input with default value of today
        date_input = request.form.get('date')
        if not date_input:
            date_input = datetime.today()
        elif isinstance(date_input, str):
            date_input = datetime.strptime(date_input, '%Y-%m-%d')
        logger.info(f"Date Input: {date_input}")

        # Handle random_subset with default value of 0
        numRandomGames = request.form.get('numRandomGames', 0)
        random_subset = int(numRandomGames) if numRandomGames else 0
        logger.info(f"Random Subset: {random_subset}")

        # Convert date string to datetime object
        if not date_input:
            date_input = datetime.today()
        elif isinstance(date_input, str):
            date_input = datetime.strptime(date_input, '%Y-%m-%d')

        # Determine the closest past Tuesday
        while date_input.weekday() != 1:  # 1 represents Tuesday
            date_input -= timedelta(days=1)

        # Initialize the NFLPredictor
        nfl_sim = NFLPredictor()

        # Handle different actions
        if action == "randomHistorical":
            logger.info("Handling randomHistorical action")
            nfl_sim.simulate_games(num_simulations=num_simulations, random_subset=random_subset, date=date_input)
        elif action == "nextWeek":
            logger.info("Handling nextWeek action")
            nfl_sim.simulate_games(num_simulations=num_simulations, date=date_input, get_current=True)
        elif action == "customMatchups":
            logger.info("Handling customMatchups action")
            matchups = []
            for i in range(1, 17):  # Assuming max 16 matchups
                home_team = request.form.get(f'homeTeam{i}')
                away_team = request.form.get(f'awayTeam{i}')
                if home_team and away_team:
                    matchups.append((home_team, away_team))
            nfl_sim.simulate_games(num_simulations=num_simulations, date=date_input, adhoc=True, matchups=matchups)

        return render_template('simulator_results.html')

    except Exception as e:
        # If there is an error, log it and return an error message
        logger.error(f"Error in sim_runner: {e}")
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
    return render_template('team_power_rank.html')


@app.route('/simulator_input')
def simulator_input():
    return render_template('simulator_input.html')


@app.route('/simulator_results')
def simulator_results():
    return render_template('simulator_results.html')


@app.route('/simulation_distribution')
def simulation_distribution():
    return render_template('simulation_distribution.html')


@app.route('/view_power_ranks_sim')
def view_power_ranks_sim():
    return render_template('team_power_rank_sim.html')


if __name__ == "__main__":
    app.run(debug=True)
