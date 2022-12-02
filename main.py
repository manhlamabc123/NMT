import argparse
from preprocessing.preprocessing import data_preprocessing, load_data
from models.models.seq2seq_attention_gru import Seq2SeqAttentionGRU
from models.models.seq2seq_gru import Seq2SeqBiGRU
from models.models.transformer import Transformer
from constants import *
from train.train import train
from train.train_transformer import train_transformer
from helper.plot import plot_loss, plot_bleu, plot_bleu_per_sentence
from helper.translate import translate, transformer_translate

parser = argparse.ArgumentParser(description="This is just a description")
parser.add_argument('-m', '--model', action='store', help="model's name", required=False)
group = parser.add_mutually_exclusive_group()
group.add_argument('-d', '--data', action='store_true', help='data preprocessing')
group.add_argument('-t', '--train', action='store_true', help='train model')
group.add_argument('-e', '--evaluate', action='store_true', help='evalute model')
group.add_argument('-a', '--attention', action='store_true', help='attention visualize')
group.add_argument('-b', '--bleu', action='store_true', help='plot bleu')
args = parser.parse_args()

if args.data:
    print("Processing Data...\n")

    data_preprocessing()

    print("Done!\n")

if args.train:
    print("Training...\n")

    print("Load dataset...\n")
    train_set_loader, dev_set_loader, _, text_transform, vocab_transform = load_data()
    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

    print("Initialize model...\n")
    if args.model == "gru":
        model = Seq2SeqBiGRU(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE).to(DEVICE)
    elif args.model == "attention":
        model = Seq2SeqAttentionGRU(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE).to(DEVICE)
    elif args.model == "transformer":
        model = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE).to(DEVICE)
    else:
        model = Seq2SeqBiGRU(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE).to(DEVICE)
    
    print(f"Start training with {model.name}...\n")
    if args.model == "transformer":
        train_transformer(model, train_set_loader, dev_set_loader, vocab_transform)
    else:
        train(model, train_set_loader, dev_set_loader, vocab_transform)

    plot_loss(model)

    plot_bleu(model)

    print("Done!\n")

if args.evaluate:
    print("Evaluating...\n")

    print("Load dataset...\n")
    _, _, test_set_loader, text_transform, vocab_transform = load_data()
    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

    print("Initialize model...\n")
    if args.model == "gru":
        model = Seq2SeqBiGRU(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE).to(DEVICE)
    elif args.model == "attention":
        model = Seq2SeqAttentionGRU(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE).to(DEVICE)
    elif args.model == "transformer":
        model = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE).to(DEVICE)
    else:
        model = Seq2SeqBiGRU(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE).to(DEVICE)

    print("Load pre-trained model...\n")
    model.load_state_dict(torch.load(f"pre_train_model/{model.name}"))

    if args.model == "transformer":
        bleu_scr = transformer_translate(test_set_loader, model, vocab_transform)
    else:
        bleu_scr = translate(test_set_loader, model, vocab_transform)
        
    print(f"--> Bleu Score: {bleu_scr}")

if args.attention:
    print("Attention visualizing...\n")

    print("Load dataset...\n")
    _, _, test_set_loader, text_transform, vocab_transform = load_data()
    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

    print("Initialize model...\n")
    model = Seq2SeqAttentionGRU(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE).to(DEVICE)

    print("Load pre-trained model...\n")
    model.load_state_dict(torch.load(f"pre_train_model/{model.name}"))

    model.get_attention(test_set_loader)

if args.bleu:
    print("Plot bleu...\n")

    print("Sort data...\n")
    data_preprocessing(sort=True)

    print("Load dataset...\n")
    _, _, test_set_loader, text_transform, vocab_transform = load_data()
    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

    print("Initialize model...\n")
    if args.model == "gru":
        model = Seq2SeqBiGRU(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE).to(DEVICE)
    elif args.model == "attention":
        model = Seq2SeqAttentionGRU(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE).to(DEVICE)
    elif args.model == "transformer":
        model = Transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE).to(DEVICE)
    else:
        model = Seq2SeqBiGRU(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE).to(DEVICE)

    print("Load pre-trained model...\n")
    model.load_state_dict(torch.load(f"pre_train_model/{model.name}"))

    if args.model == "transformer":
        bleu_scr = transformer_translate(test_set_loader, model, vocab_transform)
    else:
        bleu_scr = translate(test_set_loader, model, vocab_transform)
        
    plot_bleu_per_sentence(model)