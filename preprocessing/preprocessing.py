import pandas as pd
import re,string
from underthesea import word_tokenize
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence
import warnings
warnings.filterwarnings('ignore')
from constants import *
from torch.utils.data import DataLoader
from hyperparameters import *

def preprocessing(df): 
    df["en"] = df["en"].apply(lambda ele: ele.translate(str.maketrans('', '', string.punctuation))) #   Remove punctuation
    df["vi"] = df["vi"].apply(lambda ele: ele.translate(str.maketrans('', '', string.punctuation)))  
    df["en"] = df["en"].apply(lambda ele: ele.lower()) # convert text to lowercase
    df["vi"] = df["vi"].apply(lambda ele: ele.lower())
    df["en"] = df["en"].apply(lambda ele: ele.strip()) 
    df["vi"] = df["vi"].apply(lambda ele: ele.strip()) 
    df["en"] = df["en"].apply(lambda ele: re.sub("\s+", " ", ele)) 
    df["vi"] = df["vi"].apply(lambda ele: re.sub("\s+", " ", ele))
    
    return df

def vi_tokenizer(sentence):
    tokens = word_tokenize(sentence)
    return tokens

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

def data_preprocessing() -> None:
    data_dir = "data/Kaggle/"
    en_sents = open(data_dir + 'en_sentences.txt', "r").read().splitlines()
    vi_sents = open(data_dir + 'vi_sentences.txt', "r").read().splitlines()
    raw_data = {
            "en": [line for line in en_sents[:LINES]],
            "vi": [line for line in vi_sents[:LINES]],
        }
    df = pd.DataFrame(raw_data, columns=["en", "vi"])
    df = preprocessing(df)

    # Split data to tran test set
    split_ratio = 0.9
    split = round(df.shape[0] * split_ratio)
    train = df.iloc[:split]
    train_ds = list(zip(train['en'], train['vi']))
    valid = df.iloc[split:split + int((df.shape[0] - split) / 2)]
    val_ds = list(zip(valid['en'], valid['vi']))
    test = df.iloc[split + int((df.shape[0] - split) / 2):]
    test_ds = list(zip(test['en'], test['vi']))

    print(len(train_ds), len(val_ds), len(test_ds))

    torch.save(train_ds, 'data/preprocessed/train')
    torch.save(val_ds, 'data/preprocessed/val')
    torch.save(test_ds, 'data/preprocessed/test')
    torch.save(df, 'data/preprocessed/df')

def load_data(sort=False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = torch.load('data/preprocessed/train')
    val_ds = torch.load('data/preprocessed/val')
    test_ds = torch.load('data/preprocessed/test')
    df = torch.load('data/preprocessed/df')

    if sort == True:
        train_ds.sort(key=lambda x: len(x[0]))
        val_ds.sort(key=lambda x: len(x[0]))
        test_ds.sort(key=lambda x: len(x[0]))

    # Place-holders
    token_transform = {}
    vocab_transform = {}

    token_transform[SRC_LANGUAGE] = get_tokenizer('basic_english')
    token_transform[TGT_LANGUAGE] = get_tokenizer(vi_tokenizer)

    # helper function to yield list of tokens
    def yield_tokens(data_iter: Iterable, language: str) -> List[str]:    
        for index, data_sample in data_iter:
            yield token_transform[language](data_sample[language])

    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        # Training data Iterator
        train_iter = df.iterrows()
        # Create torchtext's Vocab object
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)
    
    # Set UNK_IDX as the default index. This index is returned when the token is not found.
    # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_transform[ln].set_default_index(UNK_IDX)

    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                                vocab_transform[ln], #Numericalization
                                                tensor_transform) # Add BOS/EOS and create tensor

    # function to collate data samples into batch tesors
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        
        for src_sample, tgt_sample in batch:
            src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
            tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch

    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, drop_last=True)
    val_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, drop_last=True)
    test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader, text_transform, vocab_transform