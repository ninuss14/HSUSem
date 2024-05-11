import matplotlib.pyplot as plt


def plotTrainImages(train_images, train_categories):
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(train_images[i * 3000].reshape(32, 32), cmap='gray')  # max je 39209
        plt.title(f"Cat.: {train_categories[i * 3000]}")
        plt.axis('off')
    plt.show()


def plotTestImages(test_images, test_categories):
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(test_images[i * 500].reshape(32, 32), cmap='gray')  # max je 12630
        plt.title(f"Cat.: {test_categories[i]}")
        plt.axis('off')
    plt.show()
