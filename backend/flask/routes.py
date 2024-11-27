from flask import Blueprint, request, jsonify
from ml.block_builder import BuiltModel
from threading import Thread
import time
import os
from database import interface as database
import json
import helper

db = database.UserDatabase()
main_routes = Blueprint("main_routes", __name__)

@main_routes.route("/api/define-model", methods=["POST"])
@helper.token_required
def define_model():
    data = request.json
    username = data.get("username")
    model_config = data.get("model_config")
    dataset = data.get("dataset")

    # add to the jsonify for user model and dataset

    if not all([username, model_config, dataset]):
        return jsonify({"error": "Missing required parameters"}), 400

    # Check and add user to database
    isNewUser = db.add_user(username)        

    # Start the Celery task
   
    try:
        # Start the Celery task regardless of user status
        task = helper.train_model_task.delay(username, model_config, dataset)
        
        return jsonify({
            "status": "Training started",
            "task_id": task.id,
            "user_status": "new" if is_new_user else "existing"
        }), 202
        
    except Exception as e:
        return jsonify({
            "error": "Failed to start training task",
            "details": str(e)
        }), 500

@main_routes.route("/api/task-status/<task_id>", methods=["GET"])
@helper.token_required
def task_status(task_id):
    return jsonify(helper.get_task_progress(task_id))

# define a route for training the model and return true if any errors 
@main_routes.route("/api/train-model", methods=["POST"])
@helper.token_required
def train_model():
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

#route for getting the logs of the task using the getmodeldir and then reading the logs
@main_routes.route("/api/train_logs", methods=["GET"])
@helper.token_required
def train_logs(task_id):
    username = data.get("username")
    model_dir = db.get_model_dir(username, task_id)
    if not model_dir:
        return jsonify({"error": "Model not found"}), 404

    with open(os.path.join(model_dir, "output.logs"), "r") as f:
        output = f.read()

    with open(os.path.join(model_dir, "loss.logs"), "r") as f:
        loss = f.read()

    with open(os.path.join(model_dir, "error.logs"), "r") as f:
        error = f.read()

    return jsonify({
        "output": output,
        "loss": loss,
        "error": error
    }), 200

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