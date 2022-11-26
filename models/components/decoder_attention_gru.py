import torch.nn as nn
import torch

class DecoderAttentionGRU(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, dropout_rate, batch_size):
        super(DecoderAttentionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_size = embedding_size 
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.energy = nn.Linear(3*hidden_size, 1)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()

        self.gru = nn.GRU(2*hidden_size + embedding_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, decoder_input, decoder_hidden_state, encoder_hidden_states):
        # decoder_input (1, BATCH_SIZE)
        # decoder_hidden_state (1, BATCH_SIZE, 2*HIDDEN_STATE)
        # encoder_hidden_states (input's sequence_length, BATCH_SIZE, 2*HIDDEN_STATE)
        sequence_length = encoder_hidden_states.shape[0]

        # Embedding
        embedding = self.dropout(self.embedding(decoder_input)) # (1, BATCH_SIZE, EMBEDDING_SIZE)

        # Calculate Attention
        decoder_hidden_state_expaned = decoder_hidden_state.repeat(sequence_length, 1, 1)
        # (1, BATCH_SIZE, HIDDEN_SIZE) -> (input's sequence_length, BATCH_SIZE, HIDDEN_SIZE)
        hidden_state = torch.cat((decoder_hidden_state_expaned, encoder_hidden_states), dim=2)
        # (input's sequence_length, BATCH_SIZE, 3*HIDDEN_SIZE)
        attention_score = torch.tanh(self.energy(hidden_state)) # (input's sequence_length, BATCH_SIZE, 1)
        attention_distribution = self.softmax(attention_score) # (input's sequence_length, BATCH_SIZE, 1)
        attention_output = torch.bmm(attention_distribution.permute(1, 2, 0), encoder_hidden_states.permute(1, 0, 2)) 
        # (BATCH_SIZE, 1, 2*HIDDEN_SIZE)
        rnn_input = torch.cat((attention_output.permute(1, 0, 2), embedding), dim=2)
        # (1, BATCH_SIZE, 2*HIDDEN_SIZE + EMBEDDING_SIZE)

        # Passing through GRU
        output, decoder_hidden_state = self.gru(rnn_input, decoder_hidden_state) 
        # (1, BATCH_SIZE, HIDDEN_SIZE)

        # Softmax Output
        output = self.linear(output) # (1, BATCH_SIZE, DICTIONARY_LENGTH)
        output = self.log_softmax(output) # (1, BATCH_SIZE, DICTIONARY_LENGTH)
        
        return output, decoder_hidden_state