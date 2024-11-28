from abc import ABC, abstractmethod

class BaseDataset(ABC):
    @abstractmethod
    def __init__(self, batch_size=64):
        pass

    @abstractmethod
    def get_output_size(self):
        pass

    @abstractmethod
    def get_eval_numbers(self, output, target):
        pass

    @abstractmethod
    def get_data_loaders(self):
        pass

    @property
    @abstractmethod
    def criterion(self):
        pass

    @property
    @abstractmethod
    def is_2d(self):
        pass

    @property
    @abstractmethod
    def num_channels(self):
        pass

    @property
    @abstractmethod
    def accuracy_descriptor(self):
        pass
