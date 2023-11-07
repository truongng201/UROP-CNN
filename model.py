import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(256*256*3, 100)
        self.fc2 = nn.Linear(100, 10)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = x.view(-1, 256*256*3)
        x = F.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x
