from flask import Blueprint, request, jsonify
from ml.block_builder import BuiltModel
from ml.train import train_model
from threading import Thread
from celeryApp import celery
import time
import os
from database import interface as database
import json
import helper
import logging

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(handler)
db = database.UserDatabase()
main_routes = Blueprint("main_routes", __name__)

#dictionary maping user to the task id
task_dict = {}

@main_routes.route("/api/define-model", methods=["POST"])
@helper.token_required
def define_model():
    try:
        data = request.json
        #username = data.get("username")
        model_config = data.get("model_config")
        dataset = data.get("dataset")

        #logger.info(f"Received request - username: {username}")
        logger.info(f"Model config: {model_config}")
        logger.info(f"Dataset: {dataset}")

        if not all([model_config, dataset]):
            return jsonify({"error": "Missing required parameters"}), 400

        # Get user UUID after adding user
        user_uuid = helper.get_user_info()["sub"]
        username = helper.get_user_info()["nickname"]
        is_new_user = db.add_user(user_uuid, username)
        
        if not user_uuid:
            return jsonify({"error": "Failed to get user ID"}), 500

        # Initialize model with user UUID and config
        model_uuid = db.init_model(user_uuid, model_config)

        # Add dataset with required parameters
        dataset_path = f"datasets/{dataset}"  # Adjust path as needed
        if not db.get_dataset(user_uuid, dataset_path):
            db.add_dataset(user_uuid, dataset, dataset_path)
        
        return jsonify({
            "status": "Model defined",
            "user_status": "new" if is_new_user else "existing",
            "model_uuid": model_uuid
        }), 202
        
    except Exception as e:
        logger.error(f"Error in define_model: {str(e)}")
        return jsonify({
            "error": "Failed to define model",
            "details": str(e)
        }), 500

@main_routes.route("/api/train", methods=["POST"])
@helper.token_required
def train():
    try:
        data = request.json
        user_uuid = helper.get_user_info()["sub"]
        model_config = data.get("model_config")
        model_uuid = db.get_model_uuid(user_uuid, model_config)
        username = helper.get_user_info()["nickname"]

        logger.info(f"Received training request - username: {username}, model_uuid: {model_uuid}")

        if not all([username, model_uuid]):
            return jsonify({"error": "Missing required parameters"}), 400

        if not user_uuid:
            return jsonify({"error": "User not found"}), 404

        # Get model configuration
        model_dir = db.get_model_dir(user_uuid, model_uuid)
        if not model_dir:
            return jsonify({"error": "Model not found"}), 404

        # Get dataset information
        dataset = db.get_dataset(user_uuid, f"datasets/{model_config['dataset']}")
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404

        try:
            logger.info(dataset)
            logger.info(f"Model config: {json.dumps(model_config)}")
            logger.info(f"user_uuid: {user_uuid}, model_uuid: {model_uuid}")
            # Start the Celery task
            task = helper.train_model_task.delay(
                model_config,
                user_uuid,
                model_uuid,
            )
            task_dict[username] = task.id

            return jsonify({
                "status": "Training started",
                "task_id": task.id,
                "model_uuid": model_uuid
            }), 202

        except Exception as train_error:
            logger.error(f"Capture error: {str(train_error)}")
            logger.error(f"Training error: {str(train_error)}")
            return jsonify({
                "error": "Training failed",
                "details": str(train_error)
            }), 500

    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        return jsonify({
            "error": "Failed to process training request",
            "details": str(e)
        }), 500

#route for getting the logs of the task using the getmodeldir and then reading the logs
@main_routes.route("/api/train-logs", methods=["POST"])
@helper.token_required
def train_logs():
    data = request.json
    user_uuid = helper.get_user_info()["sub"]
    username = helper.get_user_info()["nickname"]
    model_config = data.get("model_config")
    model_uuid = db.get_model_uuid(user_uuid, model_config)
    task_id = task_dict.get(username)
    model_dir = db.get_model_dir(user_uuid, model_uuid)
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

#stop the training task
@main_routes.route("/api/stop-training", methods=["POST"])
@helper.token_required
def stop_train():
    try:
        username = helper.get_user_info()["nickname"]
        task_id = task_dict.get(username)
        
        if not task_id:
            return jsonify({"error": "No active task found"}), 404
            
        # First attempt to revoke the task with termination signal
        celery.control.revoke(task_id, terminate=True, signal='SIGKILL')
        
        # Force kill any remaining worker processes
        import subprocess
        kill_command = "pkill -9 -f 'celery worker'"
        subprocess.run(kill_command, shell=True)
        
        logger.info(f"Task {task_id} terminated by user {username}")
        return jsonify({
            "status": "Task termination signal sent",
            "task_id": task_id
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to stop task: {str(e)}")
        return jsonify({
            "error": "Failed to stop task",
            "details": str(e)
        }), 500


#list all models
@main_routes.route("/api/models", methods=["GET"])
@helper.token_required
def get_models():
    user_uuid = helper.get_user_info()["sub"]
    if not user_uuid:
        return jsonify({"error": "User not found"}), 404

    models = db.get_models(user_uuid)
    return jsonify({"models": models}), 200