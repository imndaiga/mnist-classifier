import torch
from torchvision import datasets, transforms


class Dataset():
    def __init__(self, datasetName, dataDir='./data/', transform=None, autoload=True):
        self.transform = transform
        self.datasetName = datasetName
        self.dataDir = dataDir
        self.train = None
        self.test = None
        if autoload:
            self.load()

    def load(self):
        self.train = getattr(datasets, self.datasetName)(
            root=self.dataDir,
            train=True,
            download=True,
            transform=self.transform
        )
        self.test = getattr(datasets, self.datasetName)(
            root=self.dataDir,
            train=False,
            download=True,
            transform=self.transform
        )

        return self.train, self.test

    def _split(self, train_size, val_size):
        return torch.utils.data.random_split(
            self.train,
            [train_size, val_size]
        )

    def loaders(self, train_size, val_size, batch_size=20, shuffle=True):
        if not self.train or not self.test:
            return None, None, None

        train_data, val_data = self._split(train_size, val_size)

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=shuffle
        )
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=shuffle
        )
        test_loader = torch.utils.data.DataLoader(
            self.test,
            batch_size=batch_size,
            shuffle=shuffle
        )

        return (train_loader, val_loader, test_loader)
