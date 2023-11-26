from flask import Blueprint, render_template

# Initialize Blueprint
general_bp = Blueprint('general', __name__)


# Remaining routes and functions
@general_bp.route('/')
def home():
    return render_template('home.html')


# Remaining routes and functions
@general_bp.route('/user_guide')
def user_guide():
    return render_template('user_guide.html')


@general_bp.route('/waiting')
def waiting():
    return render_template('waiting.html')


@general_bp.route('/historical_results_backtesting')
def historical_results_backtesting():
    return render_template('historical_results_backtesting.html')


@general_bp.route('/future_betting_recommendations')
def future_betting_recommendations():
    return render_template('future_betting_recommendations.html')


@general_bp.route('/raw_data')
def raw_data():
    return render_template('raw_data.html')


@general_bp.route('/view_trending')
def view_trending():
    return render_template('trending_dash.html')


@general_bp.route('/view_summary_dash')
def view_summary_dash():
    return render_template('summary_dash.html')


@general_bp.route('/sim_results')
def sim_results():
    return render_template('simulator_results.html')


@general_bp.route('/summary')
def summary():
    return render_template('consolidated_model_report.html')


@general_bp.route('/data_quality_report')
def data_quality_report():
    return render_template('data_quality_report.html')


@general_bp.route('/interactive_heatmap')
def heatmap():
    return render_template('interactive_heatmap.html')


@general_bp.route('/feature_coef')
def feature_coef():
    return render_template('feature_coef_report.html')


@general_bp.route('/descriptive_statistics')
def descriptive_statistics():
    return render_template('descriptive_statistics.html')
