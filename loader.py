import os
import cv2
import torch
from torchvision import transforms
import numpy as np


# load metoda pre trenovacie data s kategoriami v osobitnych priecinkoch
def load_image_from_folder_train(folder, image_size=(32, 32)):
    images = []
    categories = []
    # od 0 po 42, mame 43 kategorii
    for number in os.listdir(folder):
        digit_folder = os.path.join(folder, number)
        for image in os.listdir(digit_folder):
            image_path = os.path.join(digit_folder, image)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, image_size)  # obrazky nie su rovnakej velkosti
                images.append(img_resized)
                categories.append(int(number))

    images_tensor = torch.stack([transforms.ToTensor()(img) for img in images])
    categories_tensor = torch.tensor(categories)
    return images_tensor, categories_tensor


# load metoda pre testovacie data bez kategorii
def load_images_from_folder_test(folder, image_size=(32, 32)):
    images = []
    for filename in os.listdir(folder):
        image_path = os.path.join(folder, filename)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, image_size)  # obrazky nie su rovnakej velkosti
            images.append(img_resized)

    images_tensor = torch.stack([transforms.ToTensor()(img) for img in images])
    return images_tensor
