import sys
import os
import time
# Add the backend directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from os import environ as env
from urllib.parse import quote_plus, urlencode
import helper
from authlib.integrations.flask_client import OAuth
from dotenv import find_dotenv, load_dotenv
from flask import Flask, redirect, session, url_for, request, jsonify
from functools import wraps
import requests
import sys
import os
from flask import g
from jwtValidation import auth0_service
from routes import main_routes, logger 
from celeryApp import flask_app as app, celery
from helper import train_model_task  # Now this won't cause circular import

app.register_blueprint(main_routes)

base_dir = os.path.dirname(os.path.abspath(__file__))   

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

logger.info(celery.tasks.keys())

oauth = OAuth(app)

auth0_service.initialize(
    auth0_domain = env.get("AUTH0_DOMAIN"),
    auth0_audience = env.get("AUTH0_AUDIENCE")
)

cert_path = os.path.join(base_dir, env.get("SSL_CERT_PATH"))
key_path = os.path.join(base_dir, env.get("SSL_KEY_PATH"))
client_id = env.get("AUTH0_CLIENT_ID")
client_secret = env.get("AUTH0_CLIENT_SECRET")
auth0_domain = env.get("AUTH0_DOMAIN")

url = f"https://{env.get('AUTH0_DOMAIN')}/.well-known/openid-configuration"
response = requests.get(url)
print(response.json())

oauth.register(
    "auth0",
    client_id=client_id,
    client_secret=client_secret,
    client_kwargs={
        "scope": "openid profile email",
    },
    server_metadata_url=f'https://{auth0_domain}/.well-known/openid-configuration'
)

@app.route("/callback", methods=["GET", "POST"])
def callback():
    token = oauth.auth0.authorize_access_token()
    user_info = token.get("userinfo") or oauth.auth0.get("userinfo").json()
    session["user"] = user_info  
    return redirect(env.get("FRONTEND_REDIRECT_URI", "/"))

@app.route("/session", methods=["GET"])
def session_info():
    if "user" in session:
        return jsonify({"user": session["user"]}), 200
    return jsonify({"error": "Unauthorized"}), 401


# @app.route("/api/private")
# @helper.token_required
# def private():
#     """A valid access token is required."""
#     try:
#         return jsonify(
#             message="Hello from private endpoint",
#         )
#     except Exception as e:
#         return jsonify({"error": str(e)}), 401
    

# @celery.task()
# def square_number(number):
#     time.sleep(5)  # Simulate a long-running task
#     return number ** 2
#
# @app.route('/api/square', methods=['POST'])
# def test_celery():
#     """
#     Test route to square a number using Celery.
#     Expects JSON input: {"number": <int>}
#     """
#     data = request.get_json()
#
#     # Get the number from the request
#     number = data.get('number', None)
#     if number is None:
#         return jsonify({'error': 'Please provide a number!'}), 400
#
#     # Call the Celery task
#     task = square_number.delay(number)
#
#     # Return the task ID for tracking
#     return jsonify({'task_id': task.id}), 202


if __name__ == "__main__":
    # Get the absolute path to the directory containing server.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Convert relative paths to absolute paths
    # cert_path = os.path.join(base_dir, env.get("SSL_CERT_PATH"))
    # key_path = os.path.join(base_dir, env.get("SSL_KEY_PATH"))

    app.run(
        host="0.0.0.0",
        port=int(env.get("PORT", 4000)),
        # ssl_context=(
        #     cert_path,
        #     key_path
        # )
    )
