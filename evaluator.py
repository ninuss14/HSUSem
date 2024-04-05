import torch
import numpy as np


def evaluate_model(model, data_loader, loss_fn):
    model.eval() #prepnutie do evaluacneho modu
    losses = []
    true_predictions = 0
    total_samples = 0
    for images, labels in data_loader:
        with torch.no_grad():
            predictions = model.forward(images.to('cpu'))
            predictions = predictions.cpu()

        loss = loss_fn(predictions, labels)

        top_p, top_class = predictions.topk(1, dim=1) #trieda s najvyssim vysledkom -> top class ktoru predikuje nasa siet
        labels = labels.view(*top_class.shape)  #konvertujeme shape
        equals = top_class == labels #porovname predikovanu a realnu hodnotu
        true_predictions += equals.type(torch.FloatTensor).sum()  # sumujeme true hodnoty
        total_samples += len(labels)

        losses.append(loss)

    return np.mean(losses), (true_predictions / total_samples)
