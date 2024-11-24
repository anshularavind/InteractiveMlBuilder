import json
from basic_blocks import *
import torch.nn as nn
from mnist import Mnist
from train import train_model

'''
Model Builder Input/Output Json Format:
{
    'input': <input_shape>,
    'output': <output_shape>,
    'dataset': <dataset_name>,
    'lr': <learning_rate>,
    'batch_size': <batch_size>,
    'blocks':
    [
        {'block': <block_name>, 'params': <block_params>},
        ...
    ]
}


# Output Size Notes:
    # Needed for all blocks except Tokenizer & TokenEmbedding
    # Expected to be Channels x Height x Width for multi-channel & multi-dim data
Block Params Json Format:
{
    'param1': <value1>,
    'param2': <value2>,
    'output_size': <output_size>,
    ...
}
'''

text_datasets = {'IMDB', 'Wikipedia', 'Twitter'}
image_datasets = {'MNIST': 3}
dataset_to_channels = {'MNIST': 1}
channel_classes = {Conv, AdaptivePool}

class BuiltModel(nn.Module):
    name_to_block = {'FcNN': FcNN, 'Conv': Conv, 'Pool': AdaptivePool, 'RnnLstm': BasicRnnLstm,
                     'Tokenizer': Tokenizer, 'TokenEmbedding': TokenEmbedding}
    name_to_dataset = {'MNIST': Mnist}

    def __init__(self, model_json: str):
        super(BuiltModel, self).__init__()
        self.model_json = json.loads(model_json)
        self.batch_size = int(self.model_json.get('batch_size', 64))
        self.dataset_name = self.model_json['dataset']
        self.dataset = BuiltModel.name_to_dataset[self.dataset_name](batch_size=self.batch_size)
        self.is_2d = self.dataset_name in image_datasets
        self.in_channels = dataset_to_channels.get(self.dataset_name, 1)
        self.model_blocks = self.load_model_from_json()
        self.lr = float(self.model_json['LR'])

    def forward(self, x):
        for block in self.model_blocks:
            x = block(x)
        if self.dataset in text_datasets:
            x = self.model_blocks[0].output_token(x)
        return x

    def load_model_from_json(self):
        model_blocks = nn.ModuleList()

        input_size = self.model_json['input']
        output_size = self.model_json['output']
        assert output_size == self.dataset.get_output_size(),\
            f'Output size, {output_size}, does not match dataset output size, {self.dataset.get_output_size()}'

        blocks = self.model_json['blocks']
        input_channels = self.in_channels

        for block in blocks:
            block_class = BuiltModel.name_to_block[block['block']]
            block_params = block['params']
            if block_class in channel_classes:  # Adding input_channels to block_params if it's a channel class
                block_params['in_channels'] = input_channels
                block_params['is_2d'] = self.is_2d
            block = block_class(input_size=input_size, **block_params)
            input_size = block.get_output_size()
            input_channels = block_params.get('out_channels', input_channels)  # Safely get out_channels if it exists
            model_blocks.append(block)

        assert input_size == output_size, 'Output size of last block does not match model output size'

        return model_blocks


if __name__ == '__main__':
    # json_str = '{"input": 10, "output": 10, "dataset": "MNIST", "LR": ".001", "blocks": [{"block": "FcNN", "params": {"output_size": 10, "hidden_size": 10, "num_hidden_layers": 2}}]}'
    mnist_nn_model = '''{
        "input": 784,
        "output": 10,
        "dataset": "MNIST",
        "LR": "0.001",
        "batch_size": 2048,
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
                    "output_size": 10,
                    "hidden_size": 64,
                    "num_hidden_layers": 1
                }
            }
        ]
    }'''

    mnist_cnn_model = '''{
    "input": 784,
    "output": 10,
    "dataset": "MNIST",
    "LR": "0.001",
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
                "output_size": 196
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
                "output_size": 49
            }
        },
        {
            "block": "FcNN",
            "params": {
                "output_size": 128,
                "hidden_size": 1024,
                "num_hidden_layers": 1
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

    # model = BuiltModel(mnist_nn_model)
    model = BuiltModel(mnist_cnn_model)
    train_model(model, 10)