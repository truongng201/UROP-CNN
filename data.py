import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class Dataset:
    def __init__(self):
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None


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
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
    

    def __dataset_info(self):
        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")
        print(f"Train loader size: {len(self.train_loader)}")
        print(f"Test loader size: {len(self.test_loader)}")


    def execute(self, data_dir, batch_size):
        self.__load_dataset(data_dir)
        self.__data_loader(batch_size)
        self.__dataset_info()
        return self.train_loader, self.test_loader


