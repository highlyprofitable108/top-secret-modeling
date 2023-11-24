from flask import abort, jsonify, render_template, request
from .app_init import create_app
from classes.celery_config import make_celery
import logging

app, config = create_app()
celery = make_celery(app)

# Set configuration values in app.config
app.config['TARGET_VARIABLE'] = config.get_config('constants', 'TARGET_VARIABLE')
app.config['static_dir'] = config.get_config('paths', 'static_dir')

# Import routes after Celery initialization to avoid circular imports
from .stats_routes import stats_bp  # noqa: E402
from .model_routes import model_bp, async_generate_model, async_sim_runner  # noqa: E402
from .general_routes import general_bp  # noqa: E402

# Register the Blueprints
app.register_blueprint(stats_bp, url_prefix='/stats')
app.register_blueprint(model_bp, url_prefix='/model')
app.register_blueprint(general_bp)


@app.route('/execute_combined_task', methods=['POST'])
def execute_combined_task():
    logging.info("Executing combined task")

    quick_test_str = request.form.get('quick_test')
    quick_test = quick_test_str == 'true'
    task = combined_task.delay(quick_test)

    logging.info(f"Combined task initiated with ID: {task.id}")
    return jsonify({"status": "success", "task_id": task.id})


@app.route('/task_status/<task_id>')
def task_status(task_id):
    logging.info(f"Checking status for task ID: {task_id}")
    task = celery.AsyncResult(task_id)
    logging.info(f"Task state: {task.state}, Task info: {task.info}")

    response = {"state": task.state, "info": task.info}

    if task.state == 'SUCCESS':
        logging.info(f"Task {task_id} completed successfully.")
        # Check if there is a next task in the chain
        if task.children:
            next_task = task.children[0]
            next_task_state = next_task.state
            logging.info(f"Found next task in chain: {next_task.id} with state {next_task_state}")
            response.update({"next_task_id": next_task.id, "next_task_state": next_task_state})
        else:
            response['message'] = 'Task completed successfully!'
            logging.info(f"No more tasks in the chain. Final task {task_id} completed.")
    elif task.state in ['PROGRESS', 'INITIALIZING', 'PROCESSING', 'FINALIZING', 'FAILURE']:
        logging.info(f"Task {task_id} is in state {task.state}.")
    else:
        response['status'] = task.status
        logging.warning(f"Task {task_id} is in an unexpected state: {task.state}")

    return jsonify(response)


@celery.task
def combined_task(quick_test):
    logging.info("Starting combined task")
    try:
        # Create an instance of async_generate_model task
        async_generate_model_task = async_generate_model.s()

        # Create an instance of async_sim_runner task with quick_test as an argument
        async_sim_runner_task = async_sim_runner.si(quick_test)

        # Create a task chain by combining the two tasks
        task_chain = async_generate_model_task | async_sim_runner_task

        result = task_chain.apply_async()
        logging.info(f"Task chain initiated with result: {result}")

        if result:
            logging.info(f"Task chain initiated with result: {result}")
            # Return the ID of the first task in the chain for monitoring
            return {"status": "initiated", "task_id": result.id}
        else:
            logging.error("Task chain did not return a result.")
            return {"status": "error", "message": "Task chain initiation failed"}
    except Exception as e:
        logging.error(f"Error in combined_task: {e}")
        return {"status": "error", "message": str(e)}


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


if __name__ == "__main__":
    app.run(debug=True)
