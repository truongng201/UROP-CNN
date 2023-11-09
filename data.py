import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self):
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])
        self.train_dataset, self.validation_dataset, self.test_dataset = None, None, None
        self.train_loader, self.validation_loader, self.test_loader = None, None, None


    def __load_dataset(self, data_dir):
        self.train_dataset = torchvision.datasets.CIFAR10(
                root=data_dir, 
                train=True,
                download=False,
                transform=self.transform
        )
        self.test_dataset= torchvision.datasets.CIFAR10(
                root=data_dir, 
                train=False,
                download=False,
                transform=self.transform
        )


    def __data_loader(self, batch_size):
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        self.validation_loader = torch.utils.data.DataLoader(
            dataset=self.validation_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
    

    def __split_train_dataset(self, val_split):
        train_idx, val_idx = train_test_split(
            list(range(len(self.train_dataset))),
            test_size=val_split
        )
        self.train_dataset = torch.utils.data.Subset(self.train_dataset, train_idx)
        self.validation_dataset = torch.utils.data.Subset(self.train_dataset, val_idx)
    

    def __dataset_info(self):
        print("-"*50)
        print("--------------- Dataset Information ---------------")
        print(f"Train dataset: {self.train_loader.dataset}")
        print(f"Train batch size: {self.train_loader.batch_size}")
        print("-----------------------------------")
        print(f"Validation dataset: {self.validation_loader.dataset}")
        print(f"Validation batch size: {self.validation_loader.batch_size}")
        print("-----------------------------------")
        print(f"Test dataset: {self.test_loader.dataset}")
        print(f"Test batch size: {self.test_loader.batch_size}")
        print("-"*50)


    def execute(self, data_dir='./data', batch_size=64, val_split=0.25):
        self.__load_dataset(data_dir)
        self.__split_train_dataset(val_split)
        self.__data_loader(batch_size)
        self.__dataset_info()
        return self.train_loader, self.test_loader


