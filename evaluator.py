import torch
import numpy as np


def evaluate_model(model, data_loader, loss_fn):
    model.eval()
    losses = []
    true_predictions = 0
    total_samples = 0
    for images, labels in data_loader:
        with torch.no_grad():
            predictions = model.forward(images.to('cpu'))
            predictions = predictions.cpu()

        loss = loss_fn(predictions, labels)

        top_p, top_class = predictions.topk(1)
        labels = labels.view(*top_class.shape)
        equals = top_class == labels
        true_predictions = equals.type(torch.FloatTensor).sum()
        total_samples = len(labels)

        losses.append(loss)

    return np.mean(losses), (true_predictions / total_samples).item()
