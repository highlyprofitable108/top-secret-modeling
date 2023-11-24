from flask import Blueprint, current_app, jsonify, request
from scripts.nfl_model import NFLModel
from scripts.nfl_prediction import NFLPredictor
from .app import celery, app
from datetime import datetime, timedelta


# Initialize Blueprint
model_bp = Blueprint('model', __name__)


# Celery task for model generation
@celery.task(bind=True)
def async_generate_model(self, max_depth=10):
    with app.app_context():  # Ensuring Flask context is available
        try:
            if max_depth <= 0:
                self.update_state(state='FAILURE', meta={'info': 'Max recursion depth reached'})
                current_app.logger.error("Max recursion depth reached")
                return {"status": "failure", "error": "Max recursion depth reached"}

            self.update_state(state='INITIALIZING', meta={'info': 'Initializing model generation'})
            current_app.logger.info("Initializing model generation")
            nfl_model = NFLModel()
            nfl_model.main()
            self.update_state(state='FINALIZING', meta={'info': 'Finalizing model generation'})
            current_app.logger.info("Finalizing model generation")

            # Fetch the next task ID from the current task's children
            next_task_id = self.request.children[0].id if self.request.children else None
            current_app.logger.info(f"Next task ID: {next_task_id}")

            return {"status": "success", "next_task_id": next_task_id}
        except Exception as e:
            self.update_state(state='FAILURE', meta={'info': str(e)})
            current_app.logger.error(f"Model generation failed: {e}")
            return {"status": "failure", "error": str(e)}


@model_bp.route('/generate_model', methods=['POST'])
def generate_model():
    task = async_generate_model.delay()
    return jsonify({"status": "success", "task_id": task.id})


def run_simulations(nfl_sim, num_sims, random_subset, get_current):
    """
    Wrapper function to run simulations with error handling.
    """
    try:
        current_app.logger.info(f"Executing {num_sims} simulations per game")
        nfl_sim.simulate_games(num_simulations=num_sims, random_subset=random_subset, get_current=get_current)
    except Exception as e:
        current_app.logger.error(f"Error during simulation: {e}")


# Celery task for running simulations
@celery.task(bind=True)
def async_sim_runner(self, quick_test, max_depth=10):
    with app.app_context():  # Ensuring Flask context is available
        try:
            if max_depth <= 0:
                self.update_state(state='FAILURE', meta={'info': 'Max recursion depth reached'})
                current_app.logger.error("Max recursion depth reached")
                return {"status": "failure", "error": "Max recursion depth reached"}

            self.update_state(state='INITIALIZING', meta={'info': 'Setting up simulations'})
            current_app.logger.info("Setting up simulations")

            # Set the date to the current day
            date_input = datetime.today()

            # Determine the closest past Tuesday
            while date_input.weekday() != 1:  # 1 represents Tuesday
                date_input -= timedelta(days=1)

            self.update_state(state='PROCESSING', meta={'info': 'Running historical simulations'})
            current_app.logger.info("Running historical simulations")

            # Run simulations
            run_simulations(NFLPredictor(), 110 if quick_test else 1100, 2750 if quick_test else 27500, False)

            self.update_state(state='PROCESSING', meta={'info': 'Running future predictions'})
            current_app.logger.info("Running future predictions")
            run_simulations(NFLPredictor(), 1100 if quick_test else 11000, None, True)

            self.update_state(state='FINALIZING', meta={'info': 'Finalizing simulations'})
            current_app.logger.info("Finalizing simulations")
            return {"status": "success", "next_task_id": self.request.id}
        except Exception as e:
            self.update_state(state='FAILURE', meta={'info': str(e)})
            current_app.logger.error(f"Simulation failed: {e}")
            return {"status": "failure", "error": str(e)}


@model_bp.route('/sim_runner', methods=['POST'])
def sim_runner():
    quick_test_str = request.form.get('quick_test')
    quick_test = quick_test_str == 'true'  # Correctly interpret the string

    task = async_sim_runner.delay(quick_test)
    return jsonify({"status": "success", "task_id": task.id})
