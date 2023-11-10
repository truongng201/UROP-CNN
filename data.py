import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split 
from torchvision.datasets import CIFAR10

class Dataset:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.dataset = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None


    def __load_dataset(self, data_dir):
        self.dataset = CIFAR10(
            root=data_dir,
            download=False,
            transform=self.transform
        )

    
    def __split_dataset(self, train_split, val_split, test_split):
        train_size = int(train_split * len(self.dataset))
        val_size = int(val_split * len(self.dataset))
        test_size = int(test_split * len(self.dataset))
        self.train_dataset, self.val_dataset , self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )

    
    def __get_shape(self, dataset_loader):
        for idx, (inputs, labels) in enumerate(dataset_loader):
          inputs = np.array(inputs)
          labels = np.array(labels)
          return inputs.shape, labels.shape


    def __data_loader(self, batch_size):
        self.train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            dataset=self.val_dataset,
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
        print("--------------- Dataset Information --------------")
        print("-"*50)
        print(f"All classes : {self.dataset.classes}")
        print("       -----------------------------------       ")
        print(f"Train dataset size: {len(self.train_loader.dataset)}")
        print(f"Train batch size: {self.train_loader.batch_size}")
        print(f"Train input shape: {self.__get_shape(self.train_loader)[0]}")
        print(f"Train label shape: {self.__get_shape(self.train_loader)[1]}")
        print("       -----------------------------------       ")
        print(f"Validation dataset size: {len(self.val_loader.dataset)}")
        print(f"Validation batch size: {self.val_loader.batch_size}")
        print(f"Validation input shape: {self.__get_shape(self.val_loader)[0]}")
        print(f"Validation label shape: {self.__get_shape(self.val_loader)[1]}")
        print("       -----------------------------------       ")
        print(f"Test dataset size: {len(self.test_loader.dataset)}")
        print(f"Test batch size: {self.test_loader.batch_size}")
        print(f"Test input shape: {self.__get_shape(self.test_loader)[0]}")
        print(f"Test label shape: {self.__get_shape(self.test_loader)[1]}")
        print("-"*50)
        print("-"*50)
        print("-"*50)
        print()
        print()

    
    def __show_examples(self, img, label):
        print('Label: ', self.dataset.classes[label], "("+str(label)+")")
        plt.imshow(img.permute(1, 2, 0))


    def execute(self, data_dir='./data', batch_size=64, train_split=0.7, val_split=0.1, test_split=0.2):
        self.__load_dataset(data_dir)
        self.__split_dataset(train_split, val_split, test_split)
        self.__data_loader(batch_size)
        self.__dataset_info()
        # self.__show_examples(*self.dataset[0])
        return self.train_loader, self.val_loader, self.test_loader
