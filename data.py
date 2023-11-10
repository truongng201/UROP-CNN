import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split 
from torchvision.datasets import CIFAR10

class Dataset:
    def __init__(self):
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])
        self.dataset = None
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.validation_loader = None
        self.test_loader = None


    def __load_dataset(self, data_dir):
        self.dataset = CIFAR10(
            root=data_dir,
            train=True,
            download=False,
            transform=self.transform
        )

    
    def __split_dataset(self, train_split, validation_split, test_split):
        train_size = int(train_split * len(self.dataset))
        validation_size = int(validation_split * len(self.dataset))
        test_size = int(test_split * len(self.dataset))
        self.train_dataset, self.validation_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, validation_size, test_size]
        )


    def __data_loader(self, batch_size):
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        self.validation_loader = DataLoader(
            dataset=self.validation_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        self.test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
    

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


    def execute(self, data_dir='./data', batch_size=64, train_split=0.7, val_split=0.1, test_split=0.2):
        self.__load_dataset(data_dir)
        self.__split_dataset(train_split, val_split, test_split)
        self.__data_loader(batch_size)
        self.__dataset_info()
        return self.train_loader, self.test_loader


