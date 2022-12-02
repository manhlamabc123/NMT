import torch

PRINT_EVERY = 1
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 10 # Sentences max length
LINES = 50000

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# Create source and target language tokenizer.
SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'vi'