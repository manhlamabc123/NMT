import torch.nn as nn
from hyperparameters import *
from constants import *
from models.components.positional_encoding import PositionalEncoding
import math
from torch import Tensor
from helper.transformer import *

class Transformer(nn.Module):
    def __init__(self, input_size, output_size, embedding_size=EMBEDDING_SIZE, batch_size=BATCH_SIZE, dropout_rate=DROPOUT_RATE, layers=LAYERS, heads=HEADS, feed_forward=FEED_FORWARD):
        super(Transformer, self).__init__()
        self.name = "transformer"
        self.input_size = input_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.layers = layers
        self.heads = heads

        self.input_embedding = nn.Embedding(input_size, embedding_size)
        self.target_embedding = nn.Embedding(output_size, embedding_size)
        self.positional_encoding = PositionalEncoding(embedding_size, dropout_rate)

        self.transformer = nn.Transformer(d_model=embedding_size, 
                                        nhead=heads,
                                        num_encoder_layers=layers, 
                                        num_decoder_layers=layers,
                                        dim_feedforward=feed_forward, 
                                        dropout=dropout_rate)

        self.linear = nn.Linear(embedding_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, input: Tensor, 
                target: Tensor, 
                input_mask: Tensor = None, 
                target_mask: Tensor = None,
                input_padding_mask: Tensor = None,
                target_padding_mask: Tensor = None,
                memory_key_padding_mask: Tensor = None):

        input_embedding = self.positional_encoding(self.input_embedding(input) * math.sqrt(self.embedding_size))
        target_embedding = self.positional_encoding(self.target_embedding(target) * math.sqrt(self.embedding_size))
        outputs = self.transformer(input_embedding, 
                                    target_embedding, 
                                    input_mask,
                                    target_mask,
                                    None,
                                    input_padding_mask,
                                    target_padding_mask,
                                    memory_key_padding_mask)
        outputs = self.linear(outputs)

        return outputs

    def translate(self, input):
        self.eval()

        y_input = torch.full((1, self.batch_size), BOS_IDX, device=DEVICE)

        for i in range(MAX_LENGTH):
            # Creating mask
            input_mask, target_mask, input_padding_mask, target_padding_mask = create_mask(input, y_input)

            # Forward
            output = self(input, y_input, input_mask, target_mask, input_padding_mask, target_padding_mask, input_padding_mask)

            # Take the max of each batch
            output = output[-1].argmax(dim=1)
            
            # Add the word to y_input
            y_input = torch.cat([y_input, output.unsqueeze(0)], dim=0)

        return y_input