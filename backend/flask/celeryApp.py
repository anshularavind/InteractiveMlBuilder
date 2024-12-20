from celery import Celery
from flask import Flask
from flask_cors import CORS


def make_celery(app):
    celery = Celery(
        app.name,
        broker=app.config['CELERY_BROKER_URL'],
        backend=app.config['CELERY_RESULT_BACKEND'],
        worker_pool='solo'
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

# Initialize Flask
flask_app = Flask(__name__)
CORS(flask_app, 
    resources={r"/api/*": {
        "origins": ["https://localhost:3000"],
        "methods": ["POST", "OPTIONS", "GET"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }},
    supports_credentials=True
)

# Celery configuration
flask_app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379/0',
    CELERY_RESULT_BACKEND='redis://localhost:6379/0',
    CELERY_TASK_TRACK_STARTED=True,
    CELERY_TASK_TIME_LIMIT=3600,
    CELERY_TASK_SERIALIZER='json',
    CELERY_RESULT_SERIALIZER='json',
    CELERY_ACCEPT_CONTENT=['json']
)

# Initialize Celery
celery = make_celery(flask_app)