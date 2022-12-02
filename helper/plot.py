from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

def plot_loss(model):
    plot_train_loss = torch.load(f'graphs/data/{model.name}_train_loss')
    train_min_value = min(plot_train_loss)
    train_min_index = plot_train_loss.index(train_min_value)

    plot_dev_loss = torch.load(f'graphs/data/{model.name}_dev_loss')
    dev_min_value = min(plot_dev_loss)
    dev_min_index = plot_dev_loss.index(dev_min_value)

    time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.title(f'{model.name}: Average train/dev loss per epoch')
    plt.plot(plot_train_loss, 'r', label='Train')
    plt.plot(train_min_index, train_min_value, 'ro')
    plt.annotate(f"{train_min_value:.4f}", (train_min_index, train_min_value), verticalalignment='top')

    plt.plot(plot_dev_loss, 'b', label='Dev')
    plt.plot(dev_min_index, dev_min_value, 'bo')
    plt.annotate(f"{dev_min_value:.4f}", (dev_min_index, dev_min_value), verticalalignment='top')

    plt.xlabel('Number of Epochs')
    plt.ylabel('Average loss')
    plt.legend()
    plt.xticks(np.arange(1, len(plot_train_loss)+1, 5))
    plt.savefig(f"graphs/graphs/loss_{model.name}_{time_stamp}.png")
    plt.figure().clear()

def plot_bleu(model):
    plot_train_bleu = torch.load(f'graphs/data/{model.name}_train_bleu')
    train_max_value = max(plot_train_bleu)
    train_max_index = plot_train_bleu.index(train_max_value)

    plot_dev_bleu = torch.load(f'graphs/data/{model.name}_dev_bleu')
    dev_max_value = max(plot_dev_bleu)
    dev_max_index = plot_dev_bleu.index(dev_max_value)

    time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.title(f'{model.name}: Train/Dev bleu score per epoch')
    plt.plot(plot_train_bleu, 'r', label='Train')
    plt.plot(train_max_index, train_max_value, 'ro')
    plt.annotate(f"{train_max_value:.4f}", (train_max_index, train_max_value))

    plt.plot(plot_dev_bleu, 'b', label='Dev')
    plt.plot(dev_max_index, dev_max_value, 'bo')
    plt.annotate(f"{dev_max_value:.4f}", (dev_max_index, dev_max_value))

    plt.xlabel('Number of Epochs')
    plt.ylabel('Bleu Score')
    plt.legend()
    plt.xticks(np.arange(1, len(plot_train_bleu)+1, 5))
    plt.savefig(f"graphs/graphs/bleu_{model.name}_{time_stamp}.png")
    plt.figure().clear()

def plot_bleu_per_sentence(model):
    plot_bleu_per_sentence = torch.load('graphs/data/bleu_per_sentences')
    plot_input_lengths = torch.load("graphs/data/inputs_length")
    time_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    plt.title(f'{model.name}: Bleu score on each sentence, length increase')
    plt.plit(x=plot_bleu_per_sentence, y=plot_input_lengths, kind="bar", rot=0)
    plt.xlabel("Sentence's length")
    plt.ylabel('Bleu Score')

    plt.savefig(f"graphs/graphs/bleu_per_sentence_{model.name}_{time_stamp}.png")
    plt.figure().clear()

def plot_attention(dataset_loader, attentions, vocab_transform):
    batch_size = attentions.shape[1]
    for i, data in enumerate(dataset_loader):
        input, target = data
        attention = attentions[i]

        for batch in range(batch_size):
            plot_x = input[:, batch].cpu().detach().numpy()
            plot_y = target[:, batch].cpu().detech().numpy()
            plot_attention = attention[:, batch]
            print(plot_x, plot_y, plot_attention)

            break

        break