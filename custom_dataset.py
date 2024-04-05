from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Grayscale, Compose


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform  # Can be None or other transformations excluding ToTensor

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)  # Apply transformation if not None
        return image, label

    def __len__(self):
        return len(self.images)
