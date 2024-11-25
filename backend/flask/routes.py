from flask import Blueprint, request, jsonify
from ..model_blocks import block_builder
from threading import Thread
import time
from flask import helper
import os
from ..database import interface as db
import json

main_routes = Blueprint("main_routes", __name__)

@main_routes.route("/api/define-model", methods=["POST"])
def define_model():
    data = request.json
    username = data.get("username")
    model_config = data.get("model_config")
    dataset = data.get("dataset")

    if not all([username, model_config, dataset]):
        return jsonify({"error": "Missing required parameters"}), 400

    # Check and add user to database
    if db.add_user(username):
        return jsonify({"status": "User added"}), 201

    # Start the Celery task
    task = helper.train_model_task.delay(username, model_config, dataset)

    return jsonify({
        "status": "Training started",
        "task_id": task.id
    }), 202

@main_routes.route("/api/task-status/<task_id>", methods=["GET"])
def task_status(task_id):
    return jsonify(helper.get_task_progress(task_id))


@main_routes.route("/api/list-datasets", methods=["GET"])
def datasets():
    return jsonify({"datasets": os.listdir("../datasets")}), 200

# @main_routes.route("/api/train", methods=["GET"])
# def train():
#     action = request.json.get("action")
#     params = request.json.get("params", {})
#     # Training control logic here
#     return jsonify({"status": "Training action processed"}), 200

@main_routes.route("/api/training-data", methods=["GET"])
def training_data(dataset_name):
    return jsonify({"training_data": dataset_name}), 200
# Other routes as needed...