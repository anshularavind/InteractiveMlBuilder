from http import HTTPStatus
from flask import jsonify, abort
import jwt
from dotenv import find_dotenv, load_dotenv
from os import environ as env


ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)


def json_abort(status_code, data=None):
    response = jsonify(data)
    response.status_code = status_code
    abort(response)

class Auth0Service:
    """Perform JSON Web Token (JWT) validation using PyJWT"""

    def __init__(self):
        self.issuer_url = env.get("AUTH0_DOMAIN")
        self.audience = env.get("AUTH0_AUDIENCE")
        self.algorithm = 'dir'
        self.jwks_uri = None

    def initialize(self, auth0_domain, auth0_audience):
        self.issuer_url = f'https://{auth0_domain}/'
        self.jwks_uri = f'{self.issuer_url}.well-known/jwks.json'
        self.audience = auth0_audience

    def get_signing_key(self, token):
        try:
            jwks_client = jwt.PyJWKClient(self.jwks_uri)

            return jwks_client.get_signing_key_from_jwt(token).key
        except Exception as error:
        
            json_abort(HTTPStatus.INTERNAL_SERVER_ERROR, {
                "error": "signing_key_unavailable",
                "error_description": error.__str__(),
                "message": "Unable to verify credentials"
            })

    def validate_jwt(self, token):
        print(self.audience)
        print(self.issuer_url)
        try:
            jwt_signing_key = self.get_signing_key(token)
            print("made it here")
            payload = jwt.decode(
                token,
                jwt_signing_key,
                algorithms=self.algorithm,
                audience=self.audience,
                issuer=self.issuer_url,
            )
        
        except Exception as error:
            json_abort(HTTPStatus.UNAUTHORIZED, {
                "error": "invalid_token",
                "error_description": error.__str__(),
                "message": "Bad credentials."
            })
            return

        return payload


auth0_service = Auth0Service()