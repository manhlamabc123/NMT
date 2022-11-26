import torch.nn as nn
import torch
from constants import DEVICE
from hyperparameters import BATCH_SIZE

class EncoderBiGRU(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(embedding_size, hidden_size, bidirectional=True)

    def forward(self, input, hidden_state=None):
        # Input (length, BATCH_SIZE)
        output = self.dropout(self.embedding(input)) # (length, BATCH_SIZE, EMBEDDING_SIZE)
        output, hidden_state = self.gru(output, hidden_state)
        # output (length, BATCH_SIZE, HIDDEN_SIZE), hidden_state (2, BATCH_SIZE, HIDDEN_SIZE)
        return output, hidden_state

    def init_hidden(self, batch_size=BATCH_SIZE):
        return torch.zeros((2, batch_size, self.hidden_size), device=DEVICE)