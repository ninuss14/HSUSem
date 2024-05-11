import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import custom_dataset
import evaluator
import loader
import neural_network
import residual_network

# nacitanie do tensorov
train_images, train_categories = loader.load_image_from_folder_train('dataset/Train')
test_images, test_categories = loader.load_images_from_folder_test("dataset/Test")

# spojit obrazky s kategoriami
train_dataset = custom_dataset.CustomDataset(train_images, train_categories)
test_dataset = custom_dataset.CustomDataset(test_images, test_categories)

# loadovanie batch-u dat
train_loader = DataLoader(train_dataset, 2000, True)
test_loader = DataLoader(test_dataset, 200, True)

# trenovanie neuronovej siete
# model = neural_network.FullyConvolutionalNeuralNetwork()
model = residual_network.ResNet(residual_network.BasicBlock, [2, 2, 2, 2], num_classes=43)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), 0.01)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)

train_losses = []
validation_losses = []
epochs = []
accuracies = []

best_validation_loss = float('inf')
patience = 5  # Počet epoch, po ktorých sa má skoncit ak nemame lepsiu validation loss
threshold = 0.90

for epoch in range(0, 100):
    model.train()
    e_loss = []
    for b_images, b_labels in train_loader:
        b_images = b_images.to('cpu')
        b_labels = b_labels.to('cpu')
        pred_labels = model.forward(b_images)
        optimizer.zero_grad()
        loss = loss_fn(pred_labels, b_labels)
        loss.backward()
        optimizer.step()
        e_loss.append(loss.item())

    scheduler.step()
    train_losses.append(np.mean(e_loss))

    validation_loss, accuracy = evaluator.evaluate_model(model, test_loader, loss_fn)
    validation_losses.append(validation_loss)

    print('Epoch:', epoch, 'Train Loss:', train_losses[-1], 'Validation loss: ', validation_loss, 'Accuracy:', accuracy)
    epochs.append(epoch)
    accuracies.append(accuracy)

    if accuracy >= threshold:
        break

    # Early stopping
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping after epoch", epoch)
            break


# vypis poctu parametrov modelu
print(sum(
    parameter.numel()
    for parameter in model.parameters()
))

# Zobrazenie grafov
plt.plot(train_losses, label='Train Loss')
plt.plot(validation_losses, label='Validation Loss')
# Zvýraznenie červenou čiarou počtu epoch, kedy došlo k early stoppingu
plt.axvline(x=epochs[-1], color='r', linestyle='--', label='Early Stopping')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

