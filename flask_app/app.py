from scripts import nfl_stats_select, constants
from scripts.nfl_eda import NFLDataAnalyzer
from flask import Flask, render_template, request, redirect, url_for, jsonify
# , flash, session
from collections import defaultdict
import importlib

app = Flask(__name__)
app.secret_key = '4815162342'  # Change this to a random secret key

analyzer = NFLDataAnalyzer()


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
    collection_name = request.json.get('collection_name', 'games')  # Get the collection name from the JSON body, default is 'games'
    df = analyzer.load_and_process_data(collection_name)
    analyzer.generate_eda_report(df)
    return jsonify(status="success")  # Return a JSON response indicating success


@app.route('/view_analysis')
def view_analysis():
    data = {
        "heatmap_path": "/static/heatmap.png",
        "feature_importance_path": "/static/feature_importance.png"
    }

    return render_template('view_analysis.html', data=data)


if __name__ == "__main__":
    app.run(debug=True)
