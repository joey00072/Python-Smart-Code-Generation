import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
            out = self.embed(x)
            out, (hidden, cell) = self.lstm(out.unsqueeze(1), (hidden, cell))
            out = self.fc(out.reshape(out.shape[0], -1))
            return out, (hidden, cell)
