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
    def __init__(self, kernel_size, input_size):
        super(Conv, self).__init__()
        self.input_size = input_size
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        return self.activation(x)

    def get_output_size(self):
        return self.input_size - self.conv.kernel_size + 1


class AdaptivePool(nn.Module):
    def __init__(self, output_size):
        super(AdaptivePool, self).__init__()
        self.output_size = output_size
        self.pool = nn.AdaptiveMaxPool2d(output_size=output_size)

    def forward(self, x):
        return self.pool(x)

    def get_output_size(self):
        return self.output_size


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
