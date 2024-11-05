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

name_to_block = {'FcNN': FcNN, 'Conv': Conv, 'Pool': AdaptivePool, 'RnnLstm': BasicRNNLSTM}

def load_model_from_json(json_string):
    model = nn.ModuleList()
    model_json = json.loads(json_string)

    input_size = model_json['input']
    output_size = model_json['output']
    blocks = model_json['blocks']

    for block in blocks:
        block_class = name_to_block[block['block']]
        block_params = block['params']
        block = block_class(input_size=input_size, **block_params)
        input_size = block.get_output_size()
        model.append(block)

    assert input_size == output_size, 'Output size of last block does not match model output size'

    return model
