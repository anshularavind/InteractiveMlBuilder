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
from flask import send_file

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
        dataset = model_config.get("dataset")

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

        task_id, kill_model_uuid = task_dict.get(username, (None, None))
        # Kill any existing tasks
        if task_id:
            if not helper.check_task_completed(task_id):
                logger.info(f"Stopping existing task {task_id}")
                model_dir = db.get_model_dir(user_uuid, kill_model_uuid)
                stop_path = os.path.join(model_dir, "STOP")
                open(stop_path, "w").close()
                time.sleep(0.25)
                while os.path.exists(stop_path):
                    time.sleep(0.25)
                logger.info(f"Task {task_id} terminated by user {username}")
            del task_dict[username]

        # Initialize model with user UUID and config
        model_uuid = db.init_model(user_uuid, model_config)

        # Add dataset with required parameters
        # dataset_path = f"datasets/{dataset}"  # Adjust path as needed
        # if not db.get_dataset(user_uuid, dataset_path):
        #     db.add_dataset(user_uuid, dataset, dataset_path)
        db.add_dataset(user_uuid, model_uuid, dataset)
        
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
        dataset = db.get_dataset(user_uuid, model_uuid, model_config["dataset"])
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
            task_dict[username] = [task.id, model_uuid]

            return jsonify({
                "status": "Training started",
                "task_id": task.id,
                "model_uuid": model_uuid
            }), 202

        except Exception as train_error:
            logger.error(f"Capture error: {str(train_error)}")
            logger.error(f"Training error: {str(train_error)}")
            db.save_model_logs(user_uuid, model_uuid, "error", str(train_error))
            return jsonify({
                "error": "Training failed",
                "details": str(train_error)
            }), 202

    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        db.save_model_logs(user_uuid, model_uuid, "error", str(e))
        return jsonify({
            "error": "Failed to process training request",
            "details": str(e)
        }), 202

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
        task_id, model_uuid = task_dict.get(username)
        
        if not task_id:
            return jsonify({"error": "No active task found"}), 404

        # celery.control.revoke(task_id, terminate=True, signal='SIGKILL')  # First attempt to kill
        if not helper.check_task_completed(task_id):
            model_dir = db.get_model_dir(helper.get_user_info()["sub"], model_uuid)
            stop_path = os.path.join(model_dir, "STOP")
            open(stop_path, "w").close()
            while os.path.exists(stop_path):
                time.sleep(0.25)
        del task_dict[username]

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


# #list all models
# @main_routes.route("/api/models", methods=["GET"])
# @helper.token_required
# def get_models():
#     user_uuid = helper.get_user_info()["sub"]
#     if not user_uuid:
#         return jsonify({"error": "User not found"}), 404

#     models = db.get_models(user_uuid)
#     return jsonify({"models": models}), 200

@main_routes.route("/api/models", methods=["GET"])
@helper.token_required
def get_models():
    user_uuid = helper.get_user_info()["sub"]
    if not user_uuid:
        return jsonify({"error": "User not found"}), 404

    models = db.get_models(user_uuid)
    model_list = []
    for model in models:
        model_list.append({})
        model_list[-1]['model_uuid'] = model[0]
        model_list[-1]['model_config'] = json.load(open(os.path.join(model[2], "config.json")))
    return jsonify({"models": model_list}), 200

@main_routes.route("/api/download-model", methods=["POST"])
@helper.token_required
def download_model():
    try:
        data = request.json
        user_uuid = helper.get_user_info()["sub"]
        model_config = data.get("model_config")
        
        logger.info(f"Received model config: {model_config}")
        if not model_config:
            return jsonify({"error": "Model configuration is required"}), 400

        # Get all model directories for the user
        user_models_dir = db.get_models(user_uuid)
        if not user_models_dir:
            return jsonify({"error": "No models found for user"}), 404

        model_uuids = [[str(model[0]), model[2]] for model in user_models_dir]
        logger.info(f"Found models: {model_uuids}")

        # Normalize the received config and saved configs for comparison
        normalized_input_config = json.loads(json.dumps(model_config))

        # Search through all model directories to find matching config
        for model_dir in model_uuids:
            config_path = os.path.join(model_dir[1], "config.json")
            logger.info(f"Checking model at {config_path}")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                
                # Normalize saved config
                normalized_saved_config = json.loads(json.dumps(saved_config))
                logger.info(normalized_saved_config)
                
                # Compare normalized configurations
                if normalized_saved_config == normalized_input_config:
                    model_path = os.path.join(model_dir[1], "model.pt")
                    if os.path.exists(model_path):
                        logger.info(f"Found matching model at {model_path}")
                        return send_file(
                            model_path,
                            mimetype='application/octet-stream',
                            as_attachment=True,
                            download_name=f'model_{model_dir[0]}.pt'
                        )
                else:
                    logger.info(f"Config mismatch: \nSaved: {normalized_saved_config}\nReceived: {normalized_input_config}")

        return jsonify({"error": "No matching model found"}), 404

    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        return jsonify({"error": "Failed to download model"}), 500

# get all unique datasets
@main_routes.route("/api/datasets", methods=["GET"])
@helper.token_required
def get_datasets():
    dataset_model_list = helper.get_dataset_model_list()
    #drop the user_uuid from the response
    for dataset in dataset_model_list:
        for model in dataset['models']:
            model.pop('user_uuid')
    return jsonify({"datasets": dataset_model_list}), 200

# get all users
@main_routes.route("/api/users", methods=["GET"])
@helper.token_required
def get_users():
    user_list = []
    users = db.get_users()
    for user in users:
        user_list.append({})
        # user_list[-1]['user_uuid'] = user[0]
        user_list[-1]['username'] = user[1]
        user_list[-1]['num_models'] = len(db.get_models(user[0]))
        user_list[-1]['num_datasets'] = len(db.get_datasets_by_user(user[0]))
        user_list[-1]['highest_rank'] = helper.get_highest_rank(user[0])

    user_list = sorted(user_list, key=lambda x: x['num_models'], reverse=True)
    return jsonify({"users": user_list}), 200

@main_routes.route("/api/user-info", methods=["GET"])
@helper.token_required
def get_user_info():
    user_uuid = helper.get_user_info()["sub"]
    if not user_uuid:
        return jsonify({"error": "User not found"}), 404
    user_info = {}
    #user_info['user_uuid'] = user_uuid
    user_info['username'] = db.get_user_name(user_uuid)
    user_info['num_models'] = len(db.get_models(user_uuid))
    user_info['num_datasets'] = len(db.get_datasets_by_user(user_uuid))
    user_info['highest_rank'] = helper.get_highest_rank(user_uuid)
    return jsonify({"user_info": user_info}), 200

