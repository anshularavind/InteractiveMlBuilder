import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from torch import nn
from backend.ml.train import train_model
from backend.datasets.base_dataset import BaseDataset
import math


class CIFAR10Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]["img"]
        label = self.data[idx]["label"]

        # Apply the transformation, if any
        if self.transform:
            image = np.array(image)
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            image = self.transform(image)

        C, H, W = image.size()
        image = image.reshape(C * H * W)

        return image, label

class Cifar10(BaseDataset):
    dataset = load_dataset("cifar10")
    criterion = nn.CrossEntropyLoss()
    is_2d = True
    num_channels = 3
    accuracy_descriptor = 'accuracy (%)'

    def __init__(self, batch_size=64):
        super().__init__(batch_size=batch_size)
        self.train_loader, self.test_loader = Cifar10.__get_cifar10_data_loaders(batch_size=batch_size)
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
    def __get_cifar10_data_loaders(batch_size=64):
        transform = transforms.Compose([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

        train_dataset = CIFAR10Dataset(Cifar10.dataset["train"], transform=transform)
        test_dataset = CIFAR10Dataset(Cifar10.dataset["test"], transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
