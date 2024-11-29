from ucimlrepo import fetch_ucirepo
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from torch import nn
from backend.datasets.base_dataset import BaseDataset
import numpy as np


class AirQualityDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx].copy()
        co, no2 = sample['CO(GT)'], sample['NO2(GT)']
        other_values = sample.drop(['CO(GT)', 'NO2(GT)']).values
        return torch.tensor(other_values, dtype=torch.float32), torch.tensor([co, no2], dtype=torch.float32)


class AirQuality(BaseDataset):
    criterion = nn.MSELoss()
    is_2d = False
    num_channels = 1
    accuracy_descriptor = 'MSE (std normalized)'

    def __init__(self, batch_size=64):
        super().__init__(batch_size=batch_size)
        self.batch_size = batch_size
        self.train_loader, self.test_loader = AirQuality.__get_air_quality_data_loaders(batch_size=batch_size)

    def get_output_size(self):
        return 2

    def get_data_loaders(self):
        return self.train_loader, self.test_loader

    @staticmethod
    def get_eval_numbers(output, target):
        total = target.size(0)
        mse = nn.MSELoss()(output, target).item()
        return mse, total

    @staticmethod
    def __get_air_quality_data_loaders(batch_size=64):
        # Fetch the dataset
        air_quality = fetch_ucirepo(id=360)
        X = air_quality.data.features.drop(['Date', 'Time'], axis=1)

        # shuffling the data
        X = X.sample(frac=1, random_state=42).reset_index(drop=True)
        X = X[(X['CO(GT)'] != -200) & (X['NO2(GT)'] != -200)]  # Dropping values with missing CO or NO2 values, == -200
        X = X[(X['CO(GT)'] != np.nan) & (X['NO2(GT)'] != np.nan)]  # Dropping nan values

        # Split the dataset
        test_ratio = 0.2
        cutoff = int((1 - test_ratio) * len(X))
        train_data, test_data = X[:cutoff], X[cutoff:]

        # Normalize the data
        train_data = train_data.replace(-200, np.nan)  # Replace -200 with nan to calculate mean and std
        mean = train_data.mean()
        std = train_data.std()

        train_data = train_data.fillna(mean).copy()  # Fill nan with mean
        test_data = test_data.replace(-200, mean).copy()  # Replace missing values, -200, with mean

        train_data = (train_data - mean) / std
        test_data = (test_data - mean) / std

        # Create datasets
        train_dataset = AirQualityDataset(train_data)
        test_dataset = AirQualityDataset(test_data)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

if __name__ == '__main__':
    air_quality = AirQuality()
