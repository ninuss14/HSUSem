import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import custom_dataset
import evaluator
import loader
import neural_network

# nacitanie do tensorov
train_images, train_categories = loader.load_image_from_folder_train('dataset/Train')
test_images, test_categories = loader.load_images_from_folder_test("dataset/Test")

# spojit obrazky s kategoriami
train_dataset = custom_dataset.CustomDataset(train_images, train_categories)
test_dataset = custom_dataset.CustomDataset(test_images, test_categories)

# loadovanie batch-u dat
train_loader = DataLoader(train_dataset, 5000, True)
test_loader = DataLoader(test_dataset, 1000, False)

# trenovanie neuronovej siete
model = neural_network.FullyConvolutionalNeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), 0.01)

train_losses = []
validation_losses = []
epochs = []

for epoch in range(15):
    model.train()
    e_loss = []
    for b_images, b_labels in train_loader:
        b_images = b_images.to('cpu')
        b_labels = b_labels.to('cpu')
        pred_labels = model(b_images)
        optimizer.zero_grad()
        loss = loss_fn(pred_labels, b_labels)
        loss.backward()
        optimizer.step()
        e_loss.append(loss.item())

    train_losses.append(np.mean(e_loss))

    validation_loss, accuracy = evaluator.evaluate_model(model, test_loader, loss_fn)
    validation_losses.append(validation_loss)

    print('Epoch:', epoch, 'Train Loss:', train_losses[-1], 'Validation loss: ', validation_loss, 'Accuracy:', accuracy)

print(sum(
    parameter.numel()
    for parameter in model.parameters()
))
