from flask import Flask
import os
from classes.config_manager import ConfigManager


def create_app():
    # Initialize Flask app
    app = Flask(__name__)

    # Set up the secret key and other basic configurations
    app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default_secret_key')
    app.config['TEMPLATES_AUTO_RELOAD'] = True

    # Celery configuration
    app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
    app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
    app.config['CELERYD_CONCURRENCY'] = 12
    app.config['CELERYD_POOL'] = 'prefork'

    # Initialize ConfigManager, DatabaseOperations
    config = ConfigManager()

    # You can return other objects too if needed elsewhere
    return app, config
