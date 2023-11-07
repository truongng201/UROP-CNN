import torchvision
import torchvision.transforms as transforms

class Dataset:
    def __init__(self):
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])
        self.train_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            download=True,
            transform=self.transform
        )
        self.test_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=False,
            download=True,
            transform=self.transform
        )
   
    def get_train_dataset(self):
        return self.train_dataset
    
    def get_test_dataset(self):
        return self.test_dataset

Dataset()
