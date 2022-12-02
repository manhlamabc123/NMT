from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_loss(model):
    plot_train_loss = torch.load(f'graphs/data/{model.name}_train_loss')
    plot_dev_loss = torch.load(f'graphs/data/{model.name}_dev_loss')
    time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.title(f'{model.name}: Average train/dev loss per epoch')
    plt.plot(plot_train_loss, 'r', label='Train')
    plt.plot(plot_dev_loss, 'b', label='Dev')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Average loss')
    plt.legend()
    plt.xticks(np.arange(1, len(plot_train_loss)+1, 5))
    plt.savefig(f"graphs/graphs/loss_{model.name}_{time_stamp}.png")
    plt.figure().clear()

def plot_bleu(model):
    plot_train_bleu = torch.load(f'graphs/data/{model.name}_train_bleu')
    plot_dev_bleu = torch.load(f'graphs/data/{model.name}_dev_bleu')
    time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.title(f'{model.name}: Train/Dev bleu score per epoch')
    plt.plot(plot_train_bleu, 'r', label='Train')
    plt.plot(plot_dev_bleu, 'b', label='Dev')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Bleu Score')
    plt.legend()
    plt.xticks(np.arange(1, len(plot_train_bleu)+1, 5))
    plt.savefig(f"graphs/graphs/bleu_{model.name}_{time_stamp}.png")
    plt.figure().clear()

def plot_bleu_per_sentence(model):
    plot_bleu_per_sentence = torch.load('graphs/data/bleu_per_sentences')
    time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.title(f'{model.name}: Bleu score on each sentence, length increase')
    plt.bar(plot_bleu_per_sentence[:100])
    plt.xlabel('Sentence length increase')
    plt.ylabel('Bleu Score')
    plt.savefig(f"graphs/graphs/bleu_per_sentence_{model.name}_{time_stamp}.png")
    plt.figure().clear()