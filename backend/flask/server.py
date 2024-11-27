import sys
import os

# Add the backend directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from os import environ as env
from urllib.parse import quote_plus, urlencode
from authlib.integrations.flask_oauth2 import ResourceProtector

from celery import Celery
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
from routes import main_routes

base_dir = os.path.dirname(os.path.abspath(__file__))

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

def make_celery(app):
    celery = Celery(
        app.name,
        broker=app.config['CELERY_BROKER_URL'],
        backend=app.config['CELERY_RESULT_BACKEND']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

# Initialize Flask
app = Flask(__name__)

# Celery configuration
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379/0',
    CELERY_RESULT_BACKEND='redis://localhost:6379/0',
    CELERY_TASK_TRACK_STARTED=True,
    CELERY_TASK_TIME_LIMIT=3600,
    CELERY_TASK_SERIALIZER='json',
    CELERY_RESULT_SERIALIZER='json',
    CELERY_ACCEPT_CONTENT=['json']
)

app.register_blueprint(main_routes)

# Initialize Celery
celery = make_celery(app)

# ... (rest of your existing server.py code)


app.secret_key = env.get("APP_SECRET_KEY")

oauth = OAuth(app)

auth0_service.initialize(
    auth0_domain="dev-yaqhhig1025kpyz0.us.auth0.com",
    auth0_audience="https://InteractiveMlApi"
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


@app.route("/api/private")
@helper.token_required
def private():
    """A valid access token is required."""
    try:
        return jsonify(
            message="Hello from private endpoint",
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 401


if __name__ == "__main__":
    import os
    # Get the absolute path to the directory containing server.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Convert relative paths to absolute paths
    cert_path = os.path.join(base_dir, env.get("SSL_CERT_PATH"))
    key_path = os.path.join(base_dir, env.get("SSL_KEY_PATH"))

    app.run(
        host="0.0.0.0",
        port=int(env.get("PORT", 4000)),
        ssl_context=(
            cert_path,
            key_path 
        )
    )
