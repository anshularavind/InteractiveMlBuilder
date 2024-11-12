import json
from basic_blocks import *
import torch.nn as nn

'''
Model Builder Input_output Format, Json:
{
    'model_type': <type>,
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
    name_to_block = {'FcNN': FcNN, 'Conv': Conv, 'Pool': AdaptivePool, 'RnnLstm': BasicRnnLstm,
                     'Tokenizer': Tokenizer, 'TokenEmbedding': TokenEmbedding}

    def __init__(self, model_json: str):
        super(BuiltModel, self).__init__()
        self.model_json = json.loads(model_json)
        self.model_type = json.loads(model_json)['model_type']
        self.model_blocks = BuiltModel.load_model_from_json(self.model_json)

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
    json_str = '{"input": 10, "output": 10, "blocks": [{"block": "FcNN", "params": {"output_size": 10, "hidden_size": 10, "num_hidden_layers": 2}}]}'
    model = BuiltModel(json_str)