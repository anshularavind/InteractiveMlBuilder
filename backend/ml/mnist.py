import numpy
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from torch import nn
from train import train_model
import math


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

class Mnist:
    dataset = load_dataset("ylecun/mnist")
    criterion = nn.CrossEntropyLoss()

    def __init__(self, batch_size=64):
        self.train_loader, self.test_loader = Mnist.__get_mnist_data_loaders(batch_size=batch_size)
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
    def __get_mnist_data_loaders(batch_size=64):

        # Step 3: Define transformations (normalize values to [-1, 1])
        transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = MNISTDataset(Mnist.dataset["train"], transform=transform)
        test_dataset = MNISTDataset(Mnist.dataset["test"], transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

if __name__ == '__main__':
    class Pre_Process(nn.Module):
        def __init__(self):
            super(Pre_Process, self).__init__()

        def forward(self, x):
            side_squared = x.size(-1)
            side = math.floor(math.sqrt(side_squared))
            assert side * side == side_squared, "Image must be square"
            x = x.reshape(-1, 1, side, side)
            return x

    class ExampleModel(nn.Module):
        def __init__(self):
            super(ExampleModel, self).__init__()
            self.dataset = Mnist()
            self.epochs = 5
            self.lr = 0.001
            self.model = nn.Sequential(
                Pre_Process(),
                nn.Conv2d(1, 32, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(1024, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )

        def forward(self, x):
            return self.model(x)

    model = ExampleModel()
    train_model(model, epochs=5)

