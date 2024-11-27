from ucimlrepo import fetch_ucirepo
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from torch import nn


class AirQualityDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        features = sample.drop('label').values.astype(float)
        label = sample['label'].astype(float)  # Ensure label is a float for regression

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


class AirQuality:
    criterion = nn.MSELoss()

    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self.train_loader, self.test_loader = AirQuality.__get_air_quality_data_loaders(batch_size=batch_size)

    @staticmethod
    def get_eval_numbers(output, target):
        total = target.size(0)
        mse = nn.MSELoss()(output, target).item()
        return mse, total

    @staticmethod
    def __get_air_quality_data_loaders(batch_size=64):
        # Fetch the dataset
        air_quality = fetch_ucirepo(id=360)
        X = air_quality.data.features

        # Split the dataset
        test_ratio = 0.2
        train_data, test_data = X[:int((1 - test_ratio) * len(X))], X[int(test_ratio * len(X)):]

        # Normalize the data
        mean = train_data.iloc[:, :-1].mean()
        std = train_data.iloc[:, :-1].std()
        train_data.iloc[:, :-1] = (train_data.iloc[:, :-1] - mean) / std
        test_data.iloc[:, :-1] = (test_data.iloc[:, :-1] - mean) / std

        # Create datasets
        train_dataset = AirQualityDataset(train_data)
        test_dataset = AirQualityDataset(test_data)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

if __name__ == '__main__':
    air_quality = AirQuality()
