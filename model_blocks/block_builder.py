import json
from basic_blocks import *
import torch.nn as nn

'''
Model Builder Input_output Format, Json:
{
    'input': <input_shape>,
    'output': <output_shape>,
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
    name_to_block = {'FcNN': FcNN, 'Conv': Conv, 'Pool': AdaptivePool, 'RnnLstm': BasicRnnLstm}

    def __init__(self, model_json: str):
        super(BuiltModel, self).__init__()
        self.model_blocks = BuiltModel.load_model_from_json(model_json)

    def forward(self, x):
        for block in self.model_blocks:
            x = block(x)
        return x

    @staticmethod
    def load_model_from_json(json_string: str):
        model_blocks = nn.ModuleList()
        model_json = json.loads(json_string)

        input_size = model_json['input']
        output_size = model_json['output']
        blocks = model_json['blocks']

        for block in blocks:
            block_class = BuiltModel.name_to_block[block['block']]
            block_params = block['params']
            block = block_class(input_size=input_size, **block_params)
            input_size = block.get_output_size()
            model_blocks.append(block)

        assert input_size == output_size, 'Output size of last block does not match model output size'

        return model_blocks

if __name__ == '__main__':
    json_str = '{"input": 10, "output": 10, "blocks": [{"block": "FcNN", "params": {"output_size": 10, "hidden_size": 10, "num_hidden_layers": 2}}]}'
    model = BuiltModel(json_str)