import torch
import torch.nn as nn
from constants import *
from hyperparameters import *
from helper.teacher_forcing import teacher_forcing
from models.components.encoder_bigru import EncoderBiGRU
from models.components.decoder_gru import DecoderGRU

class Seq2SeqBiGRU(nn.Module):
    def __init__(self, input_size, output_size, embedding_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, batch_size=BATCH_SIZE, dropout_rate=DROPOUT_RATE):
        super(Seq2SeqBiGRU, self).__init__()
        self.name = 'seq2seq_bigru'
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate

        self.encoder = EncoderBiGRU(input_size, embedding_size, hidden_size, dropout_rate)
        self.decoder = DecoderGRU(output_size, embedding_size, 2*hidden_size, dropout_rate)

    def forward(self, input, target=None):
        if target == None:
            target_sequence_length = 20
        else:
            target_sequence_length = target.shape[0]

        decoder_outputs = torch.zeros((target_sequence_length, self.batch_size, self.decoder.output_size), device=DEVICE)

        # Input (1, BATCH_SIZE)
        _, encoder_hidden_state = self.encoder(input)
        # Hidden state (2, BATCH_SIZE, HIDDEN_STATE)

        decoder_input = torch.full((1, self.batch_size), BOS_IDX, device=DEVICE) # (1, BATCH_SIZE)

        first_half = torch.unsqueeze(encoder_hidden_state[0], dim=0) # First half of encoder_hidden_state
        second_half = torch.unsqueeze(encoder_hidden_state[0], dim=0) # Second half of encoder_hidden_state
        encoder_hidden_state = torch.cat((first_half, second_half), dim=2) # (1, BATCH_SIZE, 2*HIDDEN_STATE)

        decoder_hidden_state = encoder_hidden_state # (1, BATCH_SIZE, 2*HIDDEN_SIZE)

        for j in range(target_sequence_length):
            # Input (1, BATCH_SIZE), Hidden state (1, BATCH_SIZE, 2*HIDDEN_SIZE)
            decoder_output, decoder_hidden_state = self.decoder(decoder_input, decoder_hidden_state)
            # Output (1, BATCH_SIZE, DICTIONARY_LENGTH], Hidden state (1, BATCH_SIZE, 2*HIDDEN_SIZE)
            if self.training:
                decoder_input = teacher_forcing(decoder_output, target[j])
            else:
                decoder_input = decoder_output.argmax(2)
            decoder_outputs[j] = decoder_output.squeeze()

        return decoder_outputs