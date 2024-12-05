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
from datetime import datetime
from celeryApp import celery
import logging
import jwt

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

# Pass in celery object from server.py


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
        'nickname': decoded_token.get('https://InteractiveMlApi/nickname'),
        'given_name': decoded_token.get('https://InteractiveMlApi/name'),
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
        logger.info(f"Model built successfully for user {user_uuid}")
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
        raise self.retry(exc=e, countdown=60)

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

    