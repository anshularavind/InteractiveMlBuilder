from celery import shared_task, Celery
from celery.signals import task_failure
from celery.result import AsyncResult
from celeryApp import celery
from ml.block_builder import BuiltModel
from ml.train import train_model
from database import interface as database
from functools import wraps
from flask import request, g, jsonify
from jwtValidation import auth0_service, json_abort
from dotenv import find_dotenv, load_dotenv
from os import environ as env
from datetime import datetime
from celeryApp import celery
import logging
import jwt
import os

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

db = database.UserDatabase()

# Create handlers
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(handler)

base_dir = os.path.dirname(os.path.abspath(__file__))

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)
AUTH0_AUDIENCE = env.get("AUTH0_AUDIENCE")


def useLogger(value):
    logger.info(value)


def check_task_completed(task_id):
    task_state = AsyncResult(task_id, app=celery).state
    return task_state == 'SUCCESS' or task_state == 'FAILURE'


def validate_token(token):
    # Decode the JWT token
    decoded_token = jwt.decode(
        token,
        options={"verify_signature": False},  # Skip signature verification
        algorithms=['RS256']
    )

    logger.info(decoded_token)
    logger.info(decoded_token.get('sub'))

    # Format the user info in Auth0 structure
    user_info = {
        'sub': decoded_token.get('sub'),
        'nickname': decoded_token.get(f'{AUTH0_AUDIENCE}/nickname'),
        'given_name': decoded_token.get(f'{AUTH0_AUDIENCE}/name'),
        'family_name': 'Unknown',
        'name': 'Unknown',
        'picture': '',
        'updated_at': datetime.utcnow().isoformat() + 'Z',
        'email': '',
        'email_verified': False
    }

    return user_info


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', None)
        if not auth_header:
            return jsonify({"error": "No authorization header"}), 401
        
        try:
            # Extract token from Bearer scheme
            token = auth_header.split()[1]
            
            # Validate token with Auth0
            user_info = validate_token(token)
            if not user_info:
                return jsonify({"error": "Invalid token"}), 401
            
            # Store user info in flask g object
            g.user_info = user_info
            return f(*args, **kwargs)
            
        except Exception as e:
            return jsonify({"error": str(e)}), 401
            
    return decorated


# @shared_task(bind=True, 
#             max_retries=3, 
#             soft_time_limit=3300,
#             time_limit=3600)
@celery.task(bind=True)
def train_model_task(self, model_config, user_uuid, model_uuid):
    try:
        # Initialize progress tracking
        total_steps = 100
        self.update_state(state='PROGRESS', 
                         meta={
                             'current': 0,
                             'total': total_steps,
                             'status': 'Initializing training...'
                         })

        # Build the model
        #make model_config json
        model = BuiltModel(model_config, user_uuid, model_uuid, db)
        logger.info(f"Model built successfully for user {user_uuid}")
        
        # Train the model
        training_result = train_model(model)

        # Save the model
        db.save_model_pt(user_uuid, model_uuid, model)

        status = 'Training completed successfully!'
        if training_result == -1:
            status = 'Training stopped by user request.'

        return {
            'current': 100,
            'total': total_steps,
            'status': status,
            'result': training_result,
            'model_uuid': model_uuid
        }

    except Exception as e:
        db.save_model_logs(user_uuid, model_uuid, str(e), 'error')

@task_failure.connect
def handle_task_failure(task_id=None, exception=None, **kwargs):
    """Handle task failures globally"""
    error_msg = f"Task {task_id} failed: {str(exception)}"
    # Add your logging logic here
    print(error_msg)

def get_task_progress(task_id):
    """Retrieve task progress information"""
    task = train_model_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'current': 0,
            'total': 100,
            'status': 'Task is waiting to start...'
        }
    elif task.state == 'PROGRESS':
        response = {
            'state': task.state,
            'current': task.info.get('current', 0),
            'total': task.info.get('total', 100),
            'status': task.info.get('status', '')
        }
    elif task.state == 'SUCCESS':
        response = {
            'state': task.state,
            'current': 100,
            'total': 100,
            'status': task.info.get('status', ''),
            'result': task.info.get('result', {})
        }
    else:
        response = {
            'state': task.state,
            'current': 0,
            'total': 100,
            'status': str(task.info)
        }
    return response

#use the token to get the user info
def get_user_info():
    token = request.headers.get('Authorization', None)
    if not token:
        return None
    token = token.split()[1]
    return validate_token(token)

# get datasets info
def get_metric_from_model(user_uuid, model_uuid):
    model_dir = db.get_model_dir(user_uuid, model_uuid)
    if not model_dir:
        return None
    with open(os.path.join(model_dir, "output.logs"), "r") as f:
        metric = f.read()

    if metric != '':
        try:
            # precision should be 4 decimal points
            return round(float(metric.split()[-1]), 8)
        except:
            return None

def get_dataset_model_list():
    dataset_model_list = []
    datasets = db.get_unique_datasets()
    for dataset in datasets:
        dataset_model_list.append({})
        dataset_model_list[-1]['name'] = dataset[0]
        dataset_model_list[-1]['metric'] = "Accuracy" if (
                    dataset[0].lower() == "cifar10" or dataset[0].lower() == "mnist") else "Mean Squared Error"
        models = db.get_models_by_dataset(dataset[0])
        # get username from uuid:
        dataset_model_list[-1]['models'] = [{"user_uuid": model[0],
                                             "model_uuid": model[1],
                                             "username": db.get_user_name(model[0]),
                                             "metric": get_metric_from_model(model[0], model[1])} for model in models]

    print(dataset_model_list)
    # sort dataset model list by metric, ascending if mean squared error, descending if accuracy
    for dataset in dataset_model_list:
        dataset['models'] = sorted(dataset['models'], key=lambda x: x['metric'] if x['metric'] is not None else -1,
                                   reverse=dataset['metric'] == "Accuracy")
    return dataset_model_list

def get_highest_rank(user_uuid):
    dataset_model_list = get_dataset_model_list()
    user_rank = None
    for dataset in dataset_model_list:
        for i, model in enumerate(dataset["models"]):
            if model["user_uuid"] == user_uuid:
                if user_rank is None:
                    user_rank = i + 1
                else:
                    user_rank = min(user_rank, i + 1)
    if user_rank is None:
        user_rank = "Unranked"

    return user_rank
