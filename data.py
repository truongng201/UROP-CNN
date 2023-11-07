from argparse import ArgumentParser
import builtins

import torchvision
import torchvision.transforms as transforms
import torch.utils.data.DataLoader as DataLoader


class Dataset:
    def __init__(self):
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])
        self.train_dataset = None
        self.test_dataset = None
    

    def __load_dataset(self, data_dir):
        self.train_dataset = torchvision.datasets.CIFAR10(
                root=data_dir, 
                train=True,
                download=True,
                transform=self.transform
        )
        self.test_dataset= torchvision.datasets.CIFAR10(
                root=data_dir, 
                train=False,
                download=True,
                transform=self.transform
        )


    def __data_loader(self, bach_size):
        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=bath_size,
            shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        builtins.train_loader = train_loader
        builtins.test_loader = test_loader


    def execute(self, data_dir, batch_size):
        self.load_dataset(data_dir)
        self.data_loader(batch_size)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    dataset = Dataset()
    dataset.execute(args.data_dir, args.batch_size)
    print('Done!')
