from constants import *
from hyperparameters import *
from nltk.translate.bleu_score import sentence_bleu

def model_bleu_score(dataset_loader, model, vocab_transform):
    bleu_per_epoch = 0

    for i, data in enumerate(dataset_loader):
        input, target = data
        input = input.to(DEVICE)
        target = target.to(DEVICE)

        bleu_per_batch = 0

        # Forward
        output = model(input)
        output = torch.argmax(output, dim=2)

        # From Tensors to Sentences -> Calculate Bleu on sentence
        for j in range(BATCH_SIZE):
            translated_output = " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(output[:, j].cpu().numpy()))).replace("<bos>", "").replace("<eos>", "").replace('<pad>', "")
            translated_target = " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(target[:, j].cpu().numpy()))).replace("<bos>", "").replace("<eos>", "").replace('<pad>', "")
            bleu_per_tensor = sentence_bleu(translated_target, translated_output, weights=(1.0, 0, 0, 0))
            bleu_per_batch += bleu_per_tensor
        bleu_per_batch = bleu_per_batch / BATCH_SIZE
        bleu_per_epoch += bleu_per_batch

    return bleu_per_epoch / len(dataset_loader)

def transformer_bleu_score(dataset_loader, model, vocab_transform):
    bleu_per_epoch = 0

    for i, data in enumerate(dataset_loader):
        input, target = data
        input = input.to(DEVICE)
        target = target.to(DEVICE)

        bleu_per_batch = 0

        # Forward
        output = model.translate(input)

        # From Tensors to Sentences -> Calculate Bleu on sentence
        for j in range(BATCH_SIZE):

            # Translate
            translated_output = " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(output[:, j].cpu().numpy()))).replace("<bos>", "").replace("<eos>", "").replace('<pad>', "")
            translated_target = " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(target[:, j].cpu().numpy()))).replace("<bos>", "").replace("<eos>", "").replace('<pad>', "")

            # Calculate Bleu
            bleu_per_tensor = sentence_bleu(translated_target, translated_output, weights=(1.0, 0, 0, 0))
            bleu_per_batch += bleu_per_tensor

        bleu_per_batch = bleu_per_batch / BATCH_SIZE
        bleu_per_epoch += bleu_per_batch

    return bleu_per_epoch / len(dataset_loader)