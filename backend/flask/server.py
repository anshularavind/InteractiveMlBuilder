import json
from os import environ as env
from urllib.parse import quote_plus, urlencode

from authlib.integrations.flask_client import OAuth
from dotenv import find_dotenv, load_dotenv
from flask import Flask, redirect, session, url_for, request, jsonify
from functools import wraps
import requests


ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)

app = Flask(__name__)
app.secret_key = env.get("APP_SECRET_KEY")

oauth = OAuth(app)

oauth.register(
    "auth0",
    client_id=env.get("AUTH0_CLIENT_ID"),
    client_secret=env.get("AUTH0_CLIENT_SECRET"),
    client_kwargs={
        "scope": "openid profile email",
    },
    server_metadata_url=f'https://{env.get("AUTH0_DOMAIN")}/.well-known/openid-configuration'
)


def validate_token(token):
    """Validates the token by calling the Auth0 /userinfo endpoint."""
    url = f"https://{env.get('AUTH0_DOMAIN')}/userinfo"
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()  # User info if the token is valid
    return None


@app.route("/login")
def login():
    return oauth.auth0.authorize_redirect(
        redirect_uri=url_for("callback", _external=True)
    )


def requires_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            session['next_url'] = request.url
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route("/callback", methods=["GET", "POST"])
def callback():
    token = oauth.auth0.authorize_access_token()
    user_info = token.get("userinfo") or oauth.auth0.get("userinfo").json()
    session["user"] = user_info  
    return redirect(env.get("FRONTEND_REDIRECT_URI", "/"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(
        "https://" + env.get("AUTH0_DOMAIN")
        + "/v2/logout?"
        + urlencode(
            {
                "returnTo": url_for("home", _external=True),
                "client_id": env.get("AUTH0_CLIENT_ID"),
            },
            quote_via=quote_plus,
        )
    )


@app.route("/session")
def session_info():
    if "user" in session:
        return jsonify({"user": session["user"]}), 200
    return jsonify({"error": "Unauthorized"}), 401


@app.route("/")
def home():
    return jsonify({"message": "API is working"}), 200


@app.route('/hello')
@requires_auth
def hello():
    return 'Hello, World!'


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(env.get("PORT", 3000)),
        ssl_context=(
            env.get("SSL_CERT_PATH"),  
            env.get("SSL_KEY_PATH")    
        )
    )
