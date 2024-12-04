import math
import torch
import torch.nn as nn
import torchtext

class FcNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_hidden_layers):
        super(FcNN, self).__init__()

        layers = []

        # Hidden layers
        prev_size = input_size
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.GELU())
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        # Combine layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def get_output_size(self):
        return self.network[-1].out_features


class Conv(nn.Module):
    def __init__(self, kernel_size, input_size, is_2d=False, in_channels=1, num_kernels=1, stride=1, padding=0, **kwargs):
        super(Conv, self).__init__()
        out_channels = num_kernels
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        assert stride < kernel_size, "Stride must be less than kernel size"
        assert 2 * padding < kernel_size, "Padding must be less than half the kernel size"

        self.input_size = input_size
        self.kernel_size = kernel_size
        self.is_2d = is_2d
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.activation = nn.ReLU()
        self.output_size = self.__get_output_size()

        if is_2d:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        if self.is_2d:
            assert x.size(-1) % self.in_channels == 0, "Input size must be divisible by number of channels"
            side_squared = x.size(-1) // self.in_channels
            side = math.floor(math.sqrt(side_squared))
            assert side * side == side_squared, "Image must be square"
            x = x.reshape(-1, self.in_channels, side, side)
        else:
            x = x.reshape(-1, self.in_channels, self.input_size)

        x = self.conv(x)
        x = self.activation(x)
        x = x.reshape(-1, self.get_output_size())
        return x

    # Private because only needed once, from there, output_size is stored and not recalculated
    def __get_output_size(self):
        def get_output_size_no_channels(input_size):
            side_length = (input_size + 2 * self.padding - self.kernel_size) // self.stride + 1
            return side_length

        input_no_ch = self.input_size // self.in_channels
        if self.is_2d:
            input_side_length = math.floor(math.sqrt(input_no_ch))
            side_length = get_output_size_no_channels(input_side_length)
            return side_length * side_length * self.out_channels
        else:
            side_length = get_output_size_no_channels(input_no_ch)
            return side_length * self.out_channels

    def get_output_size(self):
        return self.output_size


class AdaptivePool(nn.Module):
    def __init__(self, output_size, in_channels = 1, is_2d=False, input_size=None):
        super(AdaptivePool, self).__init__()
        self.in_channels = in_channels
        self.is_2d = is_2d
        if is_2d:
            output_size_sqrt = math.floor(math.sqrt(output_size))
            assert output_size_sqrt * output_size_sqrt == output_size, "Output size must be square"

            self.output_size = output_size_sqrt
            self.pool = nn.AdaptiveMaxPool2d(output_size=self.output_size)
        else:
            self.output_size = output_size
            self.pool = nn.AdaptiveMaxPool1d(output_size=output_size)

    def forward(self, x):
        if self.is_2d:
            assert x.size(-1) % self.in_channels == 0, "Input size must be divisible by number of channels"
            side_squared = x.size(-1) // self.in_channels
            side = math.floor(math.sqrt(side_squared))
            assert side * side == side_squared, "Image must be square"
            x = x.reshape(-1, self.in_channels, side, side)
        x = self.pool(x)
        x = x.reshape(-1, self.get_output_size())
        return x

    def get_output_size(self):
        if self.is_2d:
            return self.output_size * self.output_size * self.in_channels
        else:
            return self.output_size * self.in_channels


class BasicRnnLstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(BasicRnnLstm, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

    def get_output_size(self):
        return self.fc.out_features

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

    def forward(self, x):
        return self.embedding(x)

    def get_output_size(self):
        return self.embedding.embedding_dim

class Tokenizer(nn.Module):
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

    @staticmethod
    def yield_tokens(data):
        for _, text in data:
            yield Tokenizer.tokenizer(text)

    def __init__(self, data, max_tokens=10000):
        super(Tokenizer, self).__init__()
        self.vocab = torchtext.vocab.build_vocab_from_iterator(
            Tokenizer.yield_tokens(data), specials=['<unk>', '<pad>'], max_tokens=max_tokens
        )

    def __len__(self):
        return len(self.vocab)

    def forward(self, x):
        return self.vocab(Tokenizer.tokenizer(x))

    def output_token(self, token_id):
        return self.vocab.lookup_token(token_id)
