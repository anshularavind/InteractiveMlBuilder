import json
from basic_blocks import *
import torch.nn as nn
from mnist import Mnist
from train import train_model

# TODO: Add specific channel squeezing and unsqueezing before and after CNN layer for image data
# TODO: Flatten Image data, then unflatten before cnn and re-flatten after cnn

'''
Model Builder Input_output Format, Json:
{
    'input': <input_shape>,
    'output': <output_shape>,
    'dataset': <dataset_name>,
    'lr': <learning_rate>,
    'blocks':
    [
        {'block': <block_name>, 'params': <block_params>},
        ...
    ]
}

Block Params Format, Json:
{
    'param1': <value1>,
    'param2': <value2>,
    ...
}
'''

class BuiltModel(nn.Module):
    name_to_block = {'FcNN': FcNN, 'Conv': Conv, 'Pool': AdaptivePool, 'RnnLstm': BasicRnnLstm,
                     'Tokenizer': Tokenizer, 'TokenEmbedding': TokenEmbedding}
    name_to_dataset = {'MNIST': Mnist}

    def __init__(self, model_json: str):
        super(BuiltModel, self).__init__()
        self.model_json = json.loads(model_json)
        self.dataset = BuiltModel.name_to_dataset[self.model_json['dataset']]()
        self.model_blocks = self.load_model_from_json()
        self.lr = float(self.model_json['LR'])

    def forward(self, x):
        for block in self.model_blocks:
            x = block(x)
        if self.model_type == 'text':
            x = self.model_blocks[0].output_token(x)
        return x

    def load_model_from_json(self):
        model_blocks = nn.ModuleList()

        input_size = self.model_json['input']
        output_size = self.model_json['output']
        blocks = self.model_json['blocks']

        for block in blocks:
            block_class = BuiltModel.name_to_block[block['block']]
            block_params = block['params']
            block = block_class(input_size=input_size, **block_params)
            input_size = block.get_output_size()
            model_blocks.append(block)

        assert input_size == output_size, 'Output size of last block does not match model output size'

        return model_blocks


if __name__ == '__main__':
    # json_str = '{"input": 10, "output": 10, "dataset": "MNIST", "LR": ".001", "blocks": [{"block": "FcNN", "params": {"output_size": 10, "hidden_size": 10, "num_hidden_layers": 2}}]}'
    mnist_model = '''{
    "model_type": "image",
    "input": 784,
    "output": 10,
    "dataset": "MNIST",
    "LR": "0.001",
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
    model = BuiltModel(mnist_model)
    train_model(model, 10)