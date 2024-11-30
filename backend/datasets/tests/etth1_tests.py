from backend.ml.block_builder import BuiltModel
from backend.ml.train import train_model
import math

import os
os.chdir('../../..')


def init_db():
    # Placeholder for a user database initialization, if needed
    user_db = None
    return user_db


def test_etth1_nn_model():
    etth1_nn_model = '''{
        "input": 9,
        "output": 3,
        "dataset": "ETTh1",
        "LR": "0.001",
        "batch_size": 512,
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
                    "output_size": 3,
                    "hidden_size": 64,
                    "num_hidden_layers": 1
                }
            }
        ]
    }'''

    # Train the model to test basic functionality
    model = BuiltModel(etth1_nn_model, 'test_user', init_db())
    result = train_model(model, 2)  # Number of epochs set to 2 for testing
    assert not math.isnan(result), 'Model training failed'


if __name__ == '__main__':
    test_etth1_nn_model()
