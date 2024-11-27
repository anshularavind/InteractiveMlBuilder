from flask import Blueprint, request, jsonify
from ml.block_builder import BuiltModel
from ml.train import train_model
from threading import Thread
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

@main_routes.route("/api/define-model", methods=["POST"])
@helper.token_required
def define_model():
    try:
        data = request.json
        username = data.get("username")
        model_config = data.get("model_config")
        dataset = data.get("dataset")

        logger.info(f"Received request - username: {username}")
        logger.info(f"Model config: {model_config}")
        logger.info(f"Dataset: {dataset}")

        if not all([username, model_config, dataset]):
            return jsonify({"error": "Missing required parameters"}), 400

        # Get user UUID after adding user
        is_new_user = db.add_user(username)
        user_uuid = db.get_user_uuid(username)
        
        if not user_uuid:
            return jsonify({"error": "Failed to get user ID"}), 500

        # Initialize model with user UUID and config
        model_uuid = db.init_model(user_uuid, json.dumps(model_config))
        
        # Add dataset with required parameters
        dataset_path = f"datasets/{dataset}"  # Adjust path as needed
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
        username = data.get("username")
        model_uuid = data.get("model_uuid")

        logger.info(f"Received training request - username: {username}, model_uuid: {model_uuid}")

        if not all([username, model_uuid]):
            return jsonify({"error": "Missing required parameters"}), 400

        # Get user UUID
        user_uuid = db.get_user_uuid(username)
        if not user_uuid:
            return jsonify({"error": "User not found"}), 404

        # Get model configuration
        model_dir = db.get_model_dir(user_uuid, model_uuid)
        if not model_dir:
            return jsonify({"error": "Model not found"}), 404

        # Read model config from saved file
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            model_config = json.load(f)

        # Get dataset information
        dataset = db.get_dataset(user_uuid, f"datasets/{model_config['dataset']}")
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404

        try:
            # Convert model_config to proper JSON string
            model_config_str = json.dumps(model_config) if isinstance(model_config, dict) else model_config
            
            # Start the Celery task
            task = helper.train_model_task.delay(
                model_config_str,
                user_uuid,
                model_uuid,
                dataset[2]  # dataset path
            )

            return jsonify({
                "status": "Training started",
                "task_id": task.id,
                "model_uuid": model_uuid
            }), 202

        except Exception as train_error:
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

@main_routes.route("/api/train/status/<task_id>")
@helper.token_required
def get_train_status(task_id):
    task = helper.train_model_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        return jsonify({'status': 'Pending', 'task_id': task_id})
    elif task.state == 'PROGRESS':
        return jsonify({'status': 'In progress', 'progress': task.info.get('progress', 0), 'task_id': task_id})
    elif task.state == 'SUCCESS':
        return jsonify({'status': 'Completed', 'result': task.result, 'task_id': task_id})
    elif task.state == 'FAILURE':
        return jsonify({'status': 'Failed', 'reason': str(task.info), 'task_id': task_id})
    else:
        return jsonify({'status': 'Unknown', 'task_id': task_id})

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

# @main_routes.route("/api/list-datasets", methods=["GET"])
# def datasets():
#     return jsonify({"datasets": os.listdir("../datasets")}), 200

# # @main_routes.route("/api/train", methods=["GET"])
# # def train():
# #     action = request.json.get("action")
# #     params = request.json.get("params", {})
# #     # Training control logic here
# #     return jsonify({"status": "Training action processed"}), 200

# @main_routes.route("/api/training-data", methods=["GET"])
# def training_data(dataset_name):
#     return jsonify({"training_data": dataset_name}), 200
# # Other routes as needed...