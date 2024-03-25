import numpy as np
import matplotlib.pyplot as plt

import loader

train_images, categories = loader.load_image_from_folder_train('dataset/Train')
test_images = loader.load_images_from_folder_test("dataset/Test")

print(train_images.shape)
print(test_images.shape)

# plot 10tich trenovacich obrazkov z roznych kategorii
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(train_images[i * 3000].reshape(32, 32), cmap='gray')  # max je 39209
    plt.title(f"Cat.: {categories[i * 3000]}")
    plt.axis('off')
plt.show()

# plot 10tich testovacich obrazkov
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(test_images[i * 500].reshape(32, 32), cmap='gray')  # max je 12630
    plt.axis('off')
plt.show()
