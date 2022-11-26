import torch.nn as nn

class DecoderGRU(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, dropout_probability):
        super(DecoderGRU, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding_size = embedding_size
        self.dropout_probability = dropout_probability

        self.embedding = nn.Embedding(output_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, input, hidden_state):
        # Input (1, BATCH_SIZE)
        output = self.dropout(self.embedding(input)) # (1, BATCH_SIZE, EMBEDDING_SIZE)
        output, hidden_state = self.gru(output, hidden_state) 
        # (1, BATCH_SIZE, HIDDEN_STATE), (1, BATCH_SIZE, HIDDEN_STATE)
        output = self.log_softmax(self.linear(output)) 
        # (1, BATCH_SIZE, DICTIONARY_LENGTH)

        return output, hidden_state