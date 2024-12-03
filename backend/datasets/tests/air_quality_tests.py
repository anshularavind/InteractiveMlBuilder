from backend.datasets.air_quality import AirQuality
from backend.ml.block_builder import BuiltModel
from backend.ml.train import train_model
import math
import json


def init_db():
    # user_db = UserDatabase()
    # user_db.clear()
    # user_db.delete()
    # user_db = UserDatabase()
    user_db = None
    return user_db


def test_air_quality_nn_model():
    air_quality_nn_model = {
        "input": 11,
        "output": 2,
        "dataset": "AirQuality",
        "LR": "0.001",
        "batch_size": 512,
        "epochs": 2,
        "blocks": [
            {
                "block": "FcNN",
                "params": {
                    "output_size": 128,
                    "hidden_size": 256,
                    "num_hidden_layers": 2
                }
            },
            {
                "block": "FcNN",
                "params": {
                    "output_size": 64,
                    "hidden_size": 128,
                    "num_hidden_layers": 2
                }
            },
            {
                "block": "FcNN",
                "params": {
                    "output_size": 2,
                    "hidden_size": 64,
                    "num_hidden_layers": 1
                }
            }
        ]
    }

    # training the model to test basic functionality
    model = BuiltModel(air_quality_nn_model, 'test_user', 'test_model', init_db())
    result = train_model(model)
    assert not math.isnan(result), 'Model training failed'
