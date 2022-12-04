import torch
import torch.nn as nn
from constants import *
from hyperparameters import *
from helper.teacher_forcing import teacher_forcing
from models.components.encoder_bigru import EncoderBiGRU
from models.components.decoder_attention_gru import DecoderAttentionGRU

class Seq2SeqAttentionGRU(nn.Module):
    def __init__(self, input_size, output_size, embedding_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, batch_size=BATCH_SIZE, dropout_rate=DROPOUT_RATE):
        super(Seq2SeqAttentionGRU, self).__init__()
        self.name = 'seq2seq_attention_gru'
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate

        self.encoder = EncoderBiGRU(input_size, embedding_size, hidden_size, dropout_rate)
        self.decoder = DecoderAttentionGRU(output_size, embedding_size, hidden_size, dropout_rate, batch_size)

    def forward(self, input, target=None, attention_return=False):
        # Initialize some variables
        if target == None:
            target_sequence_length = 20
        else:
            target_sequence_length = target.shape[0]
        decoder_outputs = torch.zeros((target_sequence_length, self.batch_size, self.decoder.output_size), device=DEVICE)
        attentions = torch.zeros((input.shape[0], self.batch_size, target_sequence_length))

        # input (input_sequence_length, BATCH_SIZE)
        encoder_hidden_states, _ = self.encoder(input)
        # encoder_hidden_states (input_sequence_length, BATCH_SIZE, 2*HIDDEN_STATE)

        decoder_input = torch.full((1, self.batch_size), BOS_IDX, device=DEVICE) # (1, BATCH_SIZE)
        decoder_hidden_state = torch.zeros((1, self.batch_size, self.decoder.hidden_size), device=DEVICE) # (1, BATCH_SIZE, HIDDEN_STATE)

        for j in range(target_sequence_length):
            # Input (1, BATCH_SIZE), Hidden state (1, BATCH_SIZE, HIDDEN_SIZE)
            if attention_return:
                decoder_output, decoder_hidden_state, attention = self.decoder(decoder_input, decoder_hidden_state, encoder_hidden_states, attention_return)
            else:
                decoder_output, decoder_hidden_state = self.decoder(decoder_input, decoder_hidden_state, encoder_hidden_states, attention_return)
            # Output (1, BATCH_SIZE, output's dictionaty length), Hidden state (1, BATCH_SIZE, HIDDEN_SIZE)
            if self.training:
                decoder_input = teacher_forcing(decoder_output, target[j]) # (1, BATCH_SIZE)
            else:
                decoder_input = decoder_output.argmax(2) # (1, BATCH_SIZE)
            decoder_outputs[j] = decoder_output.squeeze()
            if attention_return == True:
                attentions[:,:,j] = attention.squeeze().cpu().detach()

        if attention_return == False:
            return decoder_outputs
        else:
            return decoder_outputs, attentions

    def get_attention(self, dataset_loader):
        attentions = []
        outputs = []

        for i, data in enumerate(dataset_loader):
            input, target = data
            input, target = input.to(DEVICE), target.to(DEVICE)

            output, attention = self(input, target, attention_return=True)
            output = torch.argmax(output, dim=2)

            attentions.append(attention)
            outputs.append(output)

        return outputs, attentions