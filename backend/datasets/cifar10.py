import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from torch import nn
from backend.ml.train import train_model
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

class Cifar10:
    dataset = load_dataset("cifar10")
    criterion = nn.CrossEntropyLoss()
    num_channels = 3

    def __init__(self, batch_size=64):
        self.train_loader, self.test_loader = Cifar10.__get_cifar10_data_loaders(batch_size=batch_size)
        self.batch_size = batch_size

    def get_output_size(self):
        return len(self.dataset["train"].features["label"].names)

    @staticmethod
    def get_eval_numbers(output, target):
        _, predicted = torch.max(output.data, 1)
        total = target.size(0)
        correct = (predicted == target).sum().item()
        return correct, total

    @staticmethod
    def __get_cifar10_data_loaders(batch_size=64):
        transform = transforms.Compose([transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

        train_dataset = CIFAR10Dataset(Cifar10.dataset["train"], transform=transform)
        test_dataset = CIFAR10Dataset(Cifar10.dataset["test"], transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

if __name__ == '__main__':
    class Pre_Process(nn.Module):
        def __init__(self, num_channels=3):
            super(Pre_Process, self).__init__()
            self.num_channels = num_channels

        def forward(self, x):
            side_squared = x.size(-1) / self.num_channels
            side = math.floor(math.sqrt(side_squared))
            assert side * side == side_squared, "Image must be square"
            x = x.reshape(-1, 3, side, side)
            return x

    class ExampleModel(nn.Module):
        def __init__(self):
            super(ExampleModel, self).__init__()
            self.dataset = Cifar10()
            self.epochs = 5
            self.lr = 0.001
            self.model = nn.Sequential(
                Pre_Process(),
                nn.Conv2d(3, 32, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(1600, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )

        def forward(self, x):
            return self.model(x)

        def add_output_logs(self, log):
            print(log)

    model = ExampleModel()
    train_model(model, epochs=5)