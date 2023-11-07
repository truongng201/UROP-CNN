from arg import ArguemntParser
import torchvision
import torchvision.transforms as transforms

class Dataset:
    def __init__(self):
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])

    def download(self, data_dir):
        torchvision.sets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=self.transform
        )
        torchvision.sets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=self.transform
        )


parser = ArguemntParser()
parser.add_argument("--data-dir", type=str, default="../data")
args = parser.parse_args()
Dataset().download(args.data_dir)
