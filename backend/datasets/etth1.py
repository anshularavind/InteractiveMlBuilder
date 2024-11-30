import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
from backend.datasets.base_dataset import BaseDataset

class ETTh1Dataset(Dataset):
    def __init__(self, data, forecast_size, input_size):
        self.data = data['HUFL']
        self.forecast_size = forecast_size
        self.input_size = input_size

    def __len__(self):
        return len(self.data) - self.forecast_size - self.input_size

    def __getitem__(self, idx):
        x = self.data.iloc[idx: idx + self.input_size].copy()
        y = self.data.iloc[idx + self.input_size: idx + self.input_size + self.forecast_size]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class ETTh1(BaseDataset):
    criterion = nn.MSELoss()
    is_2d = False
    num_channels = 1
    accuracy_descriptor = 'MSE'
    forecast_size = 3
    input_size = 9

    def __init__(self, batch_size=64):
        super().__init__(batch_size=batch_size)
        self.batch_size = batch_size
        self.train_loader, self.test_loader = ETTh1.__get_etth1_data_loaders(batch_size=batch_size)

    def get_output_size(self):
        return ETTh1.forecast_size  # Assuming a single target value

    def get_data_loaders(self):
        return self.train_loader, self.test_loader

    def get_eval_numbers(self, output, target):
        total = target.size(0)
        mse = nn.MSELoss()(output, target).item()
        return mse, total

    @staticmethod
    def __get_etth1_data_loaders(batch_size=64):
        # Load the dataset
        dataset_path = 'backend/datasets/local_data/ETTh1.csv'
        data = pd.read_csv(dataset_path)

        # Ensure the date column is properly formatted
        data['date'] = pd.to_datetime(data['date'], errors='coerce')


        # Shuffle the data
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split into train and test sets
        test_ratio = 0.2
        cutoff = int((1 - test_ratio) * len(data))
        train_data, test_data = data.iloc[:cutoff], data.iloc[cutoff:]

        # Create datasets
        train_dataset = ETTh1Dataset(train_data, forecast_size=ETTh1.forecast_size, input_size=ETTh1.input_size)
        test_dataset = ETTh1Dataset(test_data, forecast_size=ETTh1.forecast_size, input_size=ETTh1.input_size)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

if __name__ == '__main__':
    import os
    os.chdir('../..')
    etth1 = ETTh1()
