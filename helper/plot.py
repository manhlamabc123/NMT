from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from hyperparameters import *
from constants import *
import matplotlib.ticker as ticker

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
    plt.barh(plot_input_lengths[:30], plot_bleu_per_sentence[:30])
    plt.yticks(plot_input_lengths[:30])
    plt.ylabel("Sentence's length")
    plt.xlabel('Bleu Score')

    plt.savefig(f"graphs/graphs/bleu_per_sentence_{model.name}_{time_stamp}.png")
    plt.figure().clear()

def plot_attention(dataset_loader, outputs, attentions, vocab_transform, batch_size=BATCH_SIZE):
    for i, data in enumerate(dataset_loader):
        input, _ = data
        attention = attentions[i]
        output = outputs[i]

        for batch in range(batch_size):
            print(input[:, batch].shape, output[:, batch].shape)
            print(attention[:, batch].shape)
            plot_attention = attention[:, batch].cpu().detach().numpy()

            translated_input = " ".join(vocab_transform[SRC_LANGUAGE].lookup_tokens(list(input[:, batch].cpu().numpy()))).replace("<bos>", "").replace("<eos>", "").replace('<pad>', "").strip().split(" ")
            translated_output = " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(output[:, batch].cpu().numpy()))).replace("<bos>", "").replace("<eos>", "").replace('<pad>', "").strip().split(" ")

            fig = plt.figure()
            ax = fig.add_subplot(111)
            cax = ax.matshow(plot_attention, cmap='bone')
            fig.colorbar(cax)

            # Set up axes
            ax.set_xticklabels(translated_output, rotation=90)
            ax.set_yticklabels(translated_input)

            # Show label at every tick
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

            plt.savefig(f"graphs/graphs/attention_visual_batch-{i}_item-{batch}.png")
            plt.figure().clear()

            break

        break