from constants import DEVICE

def evaluate(dataset_loader, model, criterion):
    # Initialize some variables
    final_loss = 0
    current_loss = 0

    for i, data in enumerate(dataset_loader):
        # Extract data from dataset_loader
        input, target = data # (length, BATCH_SIZE)
        input = input.to(DEVICE)
        target = target.to(DEVICE)
        
        # Foward
        output = model(input, target) # (length, BATCH_SIZE, target's dictionary_length)

        # Compute loss
        loss = criterion(output.view(-1, output.shape[2]), target.view(target.shape[0] * target.shape[1]))

        # Update loss
        current_loss += loss.item()

    final_loss = current_loss / len(dataset_loader)
    return final_loss