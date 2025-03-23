from abc import ABC

class Dataset(ABC):

    def __getitem__(self, idx):
        pass