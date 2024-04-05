import matplotlib.pyplot as plt
import loader

# nacitanie do tensorov
train_images, train_categories = loader.load_image_from_folder_train('dataset/Train')
test_images, test_categories = loader.load_images_from_folder_test("dataset/Test")


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
