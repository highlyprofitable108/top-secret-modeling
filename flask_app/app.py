from scripts import all_columns
from scripts.nfl_stats_select import ColumnSelector
from scripts.nfl_model import NFLModel
from scripts.nfl_prediction import NFLPredictor
from classes.config_manager import ConfigManager
from classes.database_operations import DatabaseOperations
from celery import Celery, chain
from flask import Flask, render_template, request, jsonify, abort
from collections import defaultdict, OrderedDict
from datetime import datetime, timedelta
import os
import yaml
import logging
import subprocess

# Set up logging at the top of your app.py
logger = logging.getLogger(__name__)
logger.propagate = False
if not logger.handlers:
    # Set the level of the logger. This is a one-time setup.
    logger.setLevel(logging.INFO)

    # Create a console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)


def restart_celery_worker():
    # Use subprocess to execute a command that sends a SIGHUP signal to the Celery worker
    subprocess.call(["pkill", "-HUP", "-f", "celery -A flask_app.app.celery worker --loglevel=info"])


def make_celery(app):
    # Initialize Celery with Flask app's configurations
    celery = Celery(app.import_name, broker=app.config['CELERY_BROKER_URL'])
    celery.conf.update(app.config)

    # Ensure that the task execution context is the same as Flask's
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default_secret_key')
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Set Celery configuration
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'  # Example for Redis
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

# Set the multiprocessing start method to 'spawn'
app.config['CELERYD_CONCURRENCY'] = 8  # Adjust the number of concurrent tasks as needed
app.config['CELERYD_POOL'] = 'prefork'

# Initialize Celery
celery = make_celery(app)

# Initialize ConfigManager, DatabaseOperations, and DataProcessing
config = ConfigManager()
database_operations = DatabaseOperations()
nfl_stats = ColumnSelector()

# Assuming 'config' is already defined and is an instance of ConfigManager
TARGET_VARIABLE = config.get_config('constants', 'TARGET_VARIABLE')
data_dir = config.get_config('paths', 'data_dir')
model_dir = config.get_config('paths', 'model_dir')
static_dir = config.get_config('paths', 'static_dir')  # Ensure you have this in your config
database_name = config.get_config('database', 'database_name')


# Global variable to hold constants
constants = {}
feature_columns = []


def get_active_constants():
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


@app.route('/')
def home():
    active_constants = get_active_constants()

    return render_template('home.html', active_constants=active_constants)


@app.route('/columns')
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
    selected_columns = nfl_stats.get_user_selection(selected_columns)

    logger.info(selected_columns)
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


@app.route('/refresh-config')
def refresh_config():
    global constants, feature_columns
    constants = load_constants()  # Reload constants from the YAML file

    feature_columns = list(set(constants.get('feature_columns', [])) - {TARGET_VARIABLE})
    return jsonify(feature_columns)


def load_constants():
    with open(os.path.join(static_dir, 'constants.yaml'), 'r') as file:
        return yaml.safe_load(file)


# Load constants at startup
constants = load_constants()
feature_columns = list(set(constants.get('feature_columns', [])) - {TARGET_VARIABLE})


@app.route('/execute_combined_task', methods=['POST'])
def execute_combined_task():
    quick_test_str = request.form.get('quick_test')
    quick_test = quick_test_str == 'true'
    task = combined_task.delay(quick_test)
    return jsonify({"status": "success", "task_id": task.id})


@app.route('/task_status/<task_id>')
def task_status(task_id):
    task = celery.AsyncResult(task_id)
    response = {"state": task.state}

    if task.state == 'SUCCESS':
        # Check if the task result contains another task ID (for chained tasks)
        if isinstance(task.result, dict) and 'task_id' in task.result:
            response['result'] = task.result
        else:
            response['status'] = 'Task completed successfully!'
    else:
        response['status'] = task.status

    return jsonify(response)


@celery.task
def combined_task(quick_test):
    # Chain the tasks together and wait for the result
    task_chain = chain(
        async_generate_model.s(),
        async_sim_runner.si(quick_test)
    )

    result = task_chain.apply_async()

    return {"status": "success", "task_id": result.id}


@celery.task
def async_generate_model():
    try:
        nfl_model = NFLModel()
        nfl_model.main()
        return {"status": "success"}
    except Exception as e:
        return {"status": "failure", "error": str(e)}


@celery.task
def async_sim_runner(quick_test):
    try:
        # Set the date to the current day
        date_input = datetime.today()

        # Determine the closest past Tuesday
        while date_input.weekday() != 1:  # 1 represents Tuesday
            date_input -= timedelta(days=1)

        # Initialize the NFLPredictor
        nfl_sim = NFLPredictor()

        # Set the number of simulations based on quick_test value
        historical_sims = 110 if quick_test else 1100
        next_week_sims = 1100 if quick_test else 11000
        random_subset = 2750 if quick_test else 27500

        # Run simulations
        run_simulations(nfl_sim, historical_sims, random_subset, False)
        run_simulations(nfl_sim, next_week_sims, None, True)

        return {"status": "success"}

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return {"status": "failure", "error": str(e)}


def run_simulations(nfl_sim, num_sims, random_subset, get_current):
    """
    Wrapper function to run simulations with error handling.
    """
    try:
        logger.info(f"Executing simulations with {num_sims} simulations")
        nfl_sim.simulate_games(num_simulations=num_sims, random_subset=random_subset, get_current=get_current)
    except Exception as e:
        logger.error(f"Error during simulation: {e}")


@app.route('/generate_model', methods=['POST'])
def generate_model():
    task = async_generate_model.delay()
    return jsonify({"status": "success", "task_id": task.id})


@app.route('/waiting')
def waiting():
    return render_template('waiting.html')


@app.route('/sim_runner', methods=['POST'])
def sim_runner():
    quick_test_str = request.form.get('quick_test')
    quick_test = quick_test_str == 'true'  # Correctly interpret the string

    task = async_sim_runner.delay(quick_test)
    return jsonify({"status": "success", "task_id": task.id})


@app.route('/sim_results')
def sim_results():
    return render_template('simulator_results.html')


@app.route('/summary')
def summary():
    return render_template('consolidated_model_report.html')


@app.route('/data_quality_report')
def data_quality_report():
    return render_template('data_quality_report.html')


@app.route('/interactive_heatmap')
def heatmap():
    return render_template('interactive_heatmap.html')


@app.route('/feature_coef')
def feature_coef():
    return render_template('feature_coef_report.html')


@app.route('/descriptive_statistics')
def descriptive_statistics():
    return render_template('descriptive_statistics.html')


@app.route('/view_analysis')
def view_analysis():
    task_id = request.args.get('task_id')
    if not task_id:
        abort(404, description="Task ID not provided")

    task = celery.AsyncResult(task_id)
    if task.state == 'SUCCESS':
        result_data = task.result
        return render_template('view_analysis.html', data=result_data)
    else:
        abort(404, description="Resource not found")


@app.route('/simulator_input')
def simulator_input():
    return render_template('simulator_input.html')


@app.route('/simulator_results')
def simulator_results():
    return render_template('simulator_results.html')


@app.route('/view_simulation_distribution')
def view_simulation_distribution():
    return render_template('simulation_distribution.html')


@app.route('/simulation_distribution_results_game_<int:game_number>.html')
def serve_simulation_file(game_number):
    return render_template(f'simulation_distribution_results_game_{game_number:04d}.html')


@app.route('/view_value_opportunity_distribution')
def view_value_opportunity_distribution():
    return render_template('value_opportunity_distribution.html')


@app.route('/value_opportunity_results_game_<int:game_number>.html')
def serve_opportunity_file(game_number):
    return render_template(f'value_opportunity_results_game_{game_number:04d}.html')


@app.route('/historical_results_backtesting')
def historical_results_backtesting():
    return render_template('historical_results_backtesting.html')


@app.route('/future_betting_recommendations')
def future_betting_recommendations():
    return render_template('future_betting_recommendations.html')


@app.route('/view_trending')
def view_trending():
    return render_template('trending_dash.html')


@app.route('/view_summary_dash')
def view_summary_dash():
    return render_template('summary_dash.html')


if __name__ == "__main__":
    app.run(debug=True)
