import torchvision
import torchvision.transforms as transforms

class Dataset:
    def __init__(self):
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])

    def download(self):
        torchvision.datasets.CIFAR10(
            root='../data',
            train=True,
            download=True,
            transform=self.transform
        )
        torchvision.datasets.CIFAR10(
            root='../data',
            train=False,
            download=True,
            transform=self.transform
        )


Dataset().download()
