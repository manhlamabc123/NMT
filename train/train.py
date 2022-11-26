import torch
import torch.nn as nn
import torch.optim as optim
from constants import *
from hyperparameters import *
from helper.evaluate import evaluate
from helper.metric import model_bleu_score
from timeit import default_timer as timer
from helper.early_stopping import EarlyStopping

def train_per_iter(train_set_loader, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.NLLLoss):
    # Initialize some variables
    final_loss = 0
    current_loss = 0

    for i, data in enumerate(train_set_loader):
        # Extract data from dataset_loader
        input, target = data # (length, BATCH_SIZE)
        input = input.to(DEVICE)
        target = target.to(DEVICE)

        # Clear cache
        optimizer.zero_grad()
        
        # Foward
        output = model(input, target) # (length, BATCH_SIZE, target's dictionary_length)

        # Compute loss
        loss = criterion(output.view(-1, output.shape[2]), target.view(target.shape[0] * target.shape[1]))

        # Update loss
        current_loss += loss.item()

        # Backward
        loss.backward()

        # Update parameters
        optimizer.step()

        

    final_loss = current_loss / len(train_set_loader)
    return final_loss

def train(model: nn.Module, train_set_loader, dev_set_loader, vocab_transform, learning_rate=LEARNING_RATE, epochs=EPOCHS, print_every=PRINT_EVERY):
    # Initialize some variables
    train_average_loss = 0
    dev_average_loss = 0
    best_dev_loss = 1_000_000.
    plot_train_loss = []
    plot_dev_loss = []
    plot_train_bleu = []
    plot_dev_bleu = []
    early_stopping = EarlyStopping(tolerance=5)
    previous_dev_loss = 10000000

    # Loss & Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss(ignore_index=PAD_IDX)

    # Training
    for epoch in range(epochs):
        # Start timer
        start_time = timer()

        # Print epoch/epochs:
        print(f"Epoch: {epoch+1}/{epochs}:")

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)   

        # Train per iteration
        train_average_loss = train_per_iter(train_set_loader, model, optimizer, criterion)
        plot_train_loss.append(train_average_loss)

        # Turn off gradient tracking cause it is not needed anymore
        model.train(False)

        # Calculate Loss on dev set
        dev_average_loss = evaluate(dev_set_loader, model, criterion)
        plot_dev_loss.append(dev_average_loss)

        # Calculate bleu on train set
        train_bleu = model_bleu_score(train_set_loader, model, vocab_transform)
        plot_train_bleu.append(train_bleu)
        dev_bleu = model_bleu_score(dev_set_loader, model, vocab_transform)
        plot_dev_bleu.append(dev_bleu)

        # End timer
        end_time = timer()

        # Print information
        if epoch % print_every == 0:
            print(f"- Loss       | Train: {train_average_loss:.4f} - Dev: {dev_average_loss:.4f}")
            print(f"- Bleu       | Train: {train_bleu:.4f} - Dev: {dev_bleu:.4f}")
            print(f"- Epoch's time: {(end_time - start_time):.3f}s")

        # Early stopping
        early_stopping(previous_dev_loss, dev_average_loss)
        if early_stopping.early_stop:
            print("Stopped training by Early stopping.")
            break

        # Tracking best performance, and save the model's state
        if dev_average_loss < best_dev_loss:
            best_dev_loss = dev_average_loss
            model_path = f'pre_train_model/{model.name}'
            torch.save(model.state_dict(), model_path)

        

    torch.save(plot_train_loss, f'graphs/data/{model.name}_train_loss')
    torch.save(plot_dev_loss, f'graphs/data/{model.name}_dev_loss')
    torch.save(plot_train_bleu, f'graphs/data/{model.name}_train_bleu')
    torch.save(plot_dev_bleu, f'graphs/data/{model.name}_dev_bleu')