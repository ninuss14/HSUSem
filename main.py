import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import custom_dataset
import evaluator
import loader
import neural_network

# nacitanie do tensorov
train_images, train_categories = loader.load_image_from_folder_train('dataset/Train')
test_images, test_categories = loader.load_images_from_folder_test("dataset/Test")

# data preprocessing - normalizacia
train_transforms = transforms.Compose([
    transforms.RandomRotation(10),  # pootocenie
    transforms.RandomResizedCrop(size=(32, 32), scale=(0.7, 1.0)),  # scaling
    transforms.ColorJitter(brightness=0.2),  # nahodné zosvetlenie
    transforms.Normalize(mean=[0.3193], std=[0.1605])
])

test_transforms = transforms.Compose([
    # na testovacich uz nie je dobre nahodne rotovat
    transforms.Normalize(mean=[0.3193], std=[0.1605])
])

# spojit obrazky s kategoriami
train_dataset = custom_dataset.CustomDataset(train_images, train_categories, transform=train_transforms)
test_dataset = custom_dataset.CustomDataset(test_images, test_categories, transform=test_transforms)

# loadovanie batch-u dat
train_loader = DataLoader(train_dataset, batch_size=2000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=True)

class_weights = [0.02288235, 0.00476716, 0.00497442, 0.01634453, 0.00817227, 0.0088009,
                 0.00497442, 0.00715073, 0.0088009, 0.00762745, 0.02288235, 0.00520053,
                 0.00715073, 0.03813724, 0.03813724, 0.05720586, 0.00762745, 0.00476716,
                 0.00520053, 0.00544818, 0.00953431, 0.02860293, 0.00457647, 0.00762745,
                 0.02288235, 0.03813724, 0.01634453, 0.01906862, 0.02288235, 0.02860293,
                 0.01271241, 0.05720586, 0.03813724, 0.02288235, 0.02860293, 0.03813724,
                 0.05720586, 0.01271241, 0.05720586, 0.05720586, 0.03813724, 0.03813724,
                 0.05720586]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

# vypocet priemeru a std pre transformacie, spustame len raz
# mean_value, std_value = metric_calculator.calculate_mean_std(train_loader)
# mean_value_test, std_value_test = metric_calculator.calculate_mean_std(test_loader)
#
# print(mean_value)  # 0.3193
# print(std_value)  # 0.1605
# print(mean_value_test)  # 0.3169
# print(std_value_test)  # 0.1620


# trenovanie neuronovej siete
# model = neural_network.ConvolutionalNeuralNetwork()
model = neural_network.FullyConvolutionalNeuralNetwork()
# model = residual_network.ResNet(residual_network.BasicBlock, [2, 2, 2, 2], num_classes=43)

# vypis poctu parametrov modelu
print(sum(
    parameter.numel()
    for parameter in model.parameters()
))

loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), 0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.9, step_size=2)

train_losses = []
validation_losses = []
epochs = []
accuracies = []

best_validation_loss = float('inf')
patience = 5  # Počet epoch, po ktorých sa má skoncit ak nemame lepsiu validation loss
patience_counter = 0

for epoch in range(0, 1000):
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

    print('Epoch:', epoch, 'Train Loss:', train_losses[-1], 'Validation loss: ', validation_loss, 'Accuracy:', accuracy,
          'Learning rate:', optimizer.param_groups[0]['lr'])
    epochs.append(epoch)
    accuracies.append(accuracy)

    # Early stopping
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        patience_counter = 0

        if best_validation_loss < 0.7:
            torch.save(model.state_dict(), 'best_model.pt')
            #print("Model saved with validation loss:", best_validation_loss)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping after epoch", epoch)
            break


# Early stopping graf
plt.plot(train_losses, label='Train Loss')
plt.plot(validation_losses, label='Validation Loss')
# Zvýraznenie červenou čiarou počtu epoch, kedy došlo k early stoppingu
plt.axvline(x=epochs[-1], color='r', linestyle='--', label='Early Stopping')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Confusion matrix
actual_labels, predictions = evaluator.evaluate_model_and_get_predictions(model, test_loader)
conf_matrix = confusion_matrix(actual_labels, predictions)
plt.figure(figsize=(20, 18))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(43), yticklabels=range(43))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
