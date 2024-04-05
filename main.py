import matplotlib.pyplot as plt
from torch import optim
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import loader
import neural_network
import evaluator
import custom_dataset

# nacitanie do tensorov
train_images, train_categories = loader.load_image_from_folder_train('dataset/Train')
test_images, test_categories = loader.load_images_from_folder_test("dataset/Test")

train_dataset = custom_dataset.CustomDataset(train_images, train_categories)
test_dataset = custom_dataset.CustomDataset(test_images, test_categories)

# TODO navrhnut zakladnu konvolucnu neuronovu siet - vyskusat rozne architektury / rozne vrstvy
# TODO trenovanie - early stopping, skusit dynamicky znizovt/zvysovat learning rate, vyskusat rozne optimizers...
# TODO vyhodnotit spolahlivost - confusion matrix, MSE...
# TODO skusit pouzit early stopping, znizit pocet vah, skusit farebne obrazky

# plot 10tich trenovacich obrazkov z roznych kategorii
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(train_images[i * 3000].reshape(32, 32), cmap='gray')  # max je 39209
    plt.title(f"Cat.: {train_categories[i * 3000]}")
    plt.axis('off')
plt.show()

# plot 10tich testovacich obrazkov
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[i * 500].reshape(32, 32), cmap='gray')  # max je 12630
    plt.title(f"Cat.: {test_categories[i]}")
    plt.axis('off')
plt.show()

train_loader = DataLoader(train_dataset, 1000, False)
test_loader = DataLoader(test_dataset, 100, False)

# trenovanie neuronovej siete
model = neural_network.NeuralNetwork()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), 0.001)

train_losses = []
validation_losses = []
for epoch in range(10):
    model.train()
    e_loss = []
    for b_images, b_labels in train_loader:
        b_images = b_images.to('cpu')  # nemame grafiku od nvidie (cuda), musime dat potom do colabu
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
