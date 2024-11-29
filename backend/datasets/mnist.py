import numpy
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from torch import nn
from backend.datasets.base_dataset import BaseDataset

class MNISTDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]["image"]
        label = self.data[idx]["label"]

        # Apply the transformation, if any
        if self.transform:
            image = numpy.array(image)
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            image = self.transform(image).squeeze(0)

        # flattening and getting rid of channel
        H, W = image.size()
        image = image.reshape(H * W)

        return image, label

class Mnist(BaseDataset):
    dataset = load_dataset("ylecun/mnist")
    criterion = nn.CrossEntropyLoss()
    is_2d = True
    num_channels = 1
    accuracy_descriptor = 'accuracy (%)'

    def __init__(self, batch_size=64):
        super().__init__(batch_size=batch_size)
        self.train_loader, self.test_loader = Mnist.__get_mnist_data_loaders(batch_size=batch_size)
        self.batch_size = batch_size

    def get_output_size(self):
        return len(self.dataset["train"].features["label"].names)

    def get_data_loaders(self):
        return self.train_loader, self.test_loader

    @staticmethod
    def get_eval_numbers(output, target):
        _, predicted = torch.max(output.data, 1)
        total = target.size(0)
        correct = (predicted == target).sum().item()
        return 100 * correct, total  # return percentage

    @staticmethod
    def __get_mnist_data_loaders(batch_size=64):

        # Step 3: Define transformations (normalize values to [-1, 1])
        transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = MNISTDataset(Mnist.dataset["train"], transform=transform)
        test_dataset = MNISTDataset(Mnist.dataset["test"], transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
