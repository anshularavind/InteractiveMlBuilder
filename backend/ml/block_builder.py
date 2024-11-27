import json
import torch.nn as nn
from backend.datasets.mnist import Mnist
from backend.datasets.air_quality import AirQuality
from backend.datasets.cifar10 import Cifar10
from backend.ml.basic_blocks import *
from backend.ml.train import train_model
import os
from backend.database.interface import UserDatabase

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
channel_classes = {Conv, AdaptivePool}

class BuiltModel(nn.Module):
    name_to_block = {'FcNN': FcNN, 'Conv': Conv, 'Pool': AdaptivePool, 'RnnLstm': BasicRnnLstm,
                     'Tokenizer': Tokenizer, 'TokenEmbedding': TokenEmbedding}
    name_to_dataset = {'MNIST': Mnist, 'CIFAR10': Cifar10, 'AirQuality': AirQuality}

    def __init__(self, model_json: str, user_uuid: str, user_db: UserDatabase, rel_path_to_backend_dir: str = '../'):
        super(BuiltModel, self).__init__()
        self.model_json = json.loads(model_json)
        self.user_uuid = user_uuid
        self.model_uuid = user_db.init_model(user_uuid, model_json) if user_db else None
        self.model_dir = os.path.join(rel_path_to_backend_dir, 'database', user_db.get_model_dir(user_uuid, self.model_uuid)) \
            if user_db else None
        self.user_db = user_db

        self.batch_size = int(self.model_json.get('batch_size', 64))
        self.dataset_name = self.model_json['dataset']
        self.dataset = BuiltModel.name_to_dataset[self.dataset_name](batch_size=self.batch_size)
        self.is_2d = getattr(self.dataset, 'is_2d', False)
        self.in_channels = getattr(self.dataset, 'num_channels', 1)
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

    def add_output_logs(self, output: str):
        if not self.model_dir:
            return
        # add output to self.model_dir/output.logs
        with open(os.path.join(self.model_dir, 'output.logs'), 'a') as f:
            f.write(output + '\n')

    def add_loss_logs(self, loss: float):
        if not self.model_dir:
            return
        # add loss to self.model_dir/loss.logs
        with open(os.path.join(self.model_dir, 'loss.logs'), 'a') as f:
            f.write(str(loss) + ',')  # comma separated values

    def add_error_logs(self, error: str):
        if not self.model_dir:
            return
        # add error to self.model_dir/error.logs
        with open(os.path.join(self.model_dir, 'error.logs'), 'a') as f:
            f.write(error + '\n')


if __name__ == '__main__':

    # user_db = UserDatabase()
    # user_db.clear()
    # user_db.delete()
    # user_db = UserDatabase()
    user_db = None
