import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
)

TURN_DROP_PROB = 0

class TinyLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h_last = h[-1]
        return self.fc(h_last).squeeze(1)
    

import torch.nn as nn

class PokemonFullNet(nn.Module):
    def __init__(self, input_layers=1632):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_layers, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, X):
        out = self.network(X)
        return out.squeeze(1)
