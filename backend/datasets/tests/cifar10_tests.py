from backend.datasets.cifar10 import Cifar10
from backend.ml.block_builder import BuiltModel
from backend.ml.train import train_model
import math


def init_db():
    # user_db = UserDatabase()
    # user_db.clear()
    # user_db.delete()
    # user_db = UserDatabase()
    user_db = None
    return user_db


def test_cifar10_nn_model():
    cifar10_nn_model = '''{
        "input": 3072,
        "output": 10,
        "dataset": "CIFAR10",
        "LR": "0.001",
        "batch_size": 2048,
        "blocks": [
            {
                "block": "FcNN",
                "params": {
                    "output_size": 256,
                    "hidden_size": 512,
                    "num_hidden_layers": 2
                }
            },
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
                    "output_size": 10,
                    "hidden_size": 128,
                    "num_hidden_layers": 1
                }
            }
        ]
    }'''

    # training both models to test basic functionality
    model = BuiltModel(cifar10_nn_model, 'test_user', init_db(), '..')
    result = train_model(model, 2)
    assert not math.isnan(result), 'Model training failed'


def test_cifar10_cnn_model():
    cifar10_cnn_model = '''{
        "input": 3072,
        "output": 10,
        "dataset": "CIFAR10",
        "LR": "0.001",
        "batch_size": 2048,
        "blocks": [
            {
                "block": "Conv",
                "params": {
                    "out_channels": 32,
                    "kernel_size": 5,
                    "stride": 1,
                    "padding": 2
                }
            },
            {
                "block": "Pool",
                "params": {
                    "output_size": 784
                }
            },
            {
                "block": "Conv",
                "params": {
                    "out_channels": 64,
                    "kernel_size": 5,
                    "stride": 1,
                    "padding": 2
                }
            },
            {
                "block": "Pool",
                "params": {
                    "output_size": 196
                }
            },
            {
                "block": "FcNN",
                "params": {
                    "output_size": 256,
                    "hidden_size": 1024,
                    "num_hidden_layers": 1
                }
            },
            {
                "block": "FcNN",
                "params": {
                    "output_size": 10,
                    "hidden_size": 256,
                    "num_hidden_layers": 1
                }
            }
        ]
    }'''
    model = BuiltModel(cifar10_cnn_model, 'test_user', init_db(), '..')
    result = train_model(model, 2)
    assert not math.isnan(result), 'Model training failed'
