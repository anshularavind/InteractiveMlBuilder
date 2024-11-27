from celery import shared_task
from celery.signals import task_failure
from ml.block_builder import BuiltModel
from ml.train import train_model
from database import interface as db
from jwcrypto import jwe, jwk
from os import environ as env
import json
from urllib.request import urlopen

from authlib.oauth2.rfc7523 import JWTBearerTokenValidator
from authlib.jose.rfc7517.jwk import JsonWebKey

from functools import wraps
from http import HTTPStatus
from types import SimpleNamespace

from flask import request, g, jsonify

from jwtValidation import auth0_service, json_abort
import requests


def validate_token(token):
    auth0_domain = env.get("AUTH0_DOMAIN")
    """Validates the token by calling the Auth0 /userinfo endpoint."""
    url = f"https://{auth0_domain}/userinfo"
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()  # User info if the token is valid
    return None

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



@shared_task(bind=True, 
            max_retries=3, 
            soft_time_limit=3300,
            time_limit=3600)
@token_required
def train_model_task(self, username, model_config, dataset):
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
        model = BuiltModel(model_config)
        self.update_state(state='PROGRESS', 
                         meta={
                             'current': 10,
                             'total': total_steps,
                             'status': 'Model built successfully'
                         })

        # Train the model with progress updates
        training_result = train_model(model, dataset, 
                                    progress_callback=lambda p: self.update_state(
                                        state='PROGRESS',
                                        meta={
                                            'current': 10 + int(p * 80),
                                            'total': total_steps,
                                            'status': f'Training progress: {p*100:.2f}%'
                                        }
                                    ))

        # Save the model
        self.update_state(state='PROGRESS', 
                         meta={
                             'current': 90,
                             'total': total_steps,
                             'status': 'Saving model...'
                         })
        db.add_model(username, model, model_config)

        return {
            'current': 100,
            'total': total_steps,
            'status': 'Training completed successfully!',
            'result': training_result
        }

    except Exception as e:
        self.retry(exc=e, countdown=60)

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

