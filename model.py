import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5) # 3 input image channel, 64 output channels, 5x5 square convolution kernel
        self.pool = nn.MaxPool2d(2, 2) # Max pooling over a (2, 2) window
        self.conv2 = nn.Conv2d(64, 128, 5) # 64 input image channel, 128 output channels, 5x5 square convolution kernel
        self.fc1 = nn.Linear(128 * 5 * 5, 512) # an affine operation: y = Wx + b
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10) # 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # Activation function: ReLU 
        x = self.pool(F.relu(self.conv2(x))) # Activation function: ReLU
        x = x.view(-1, 128 * 5 * 5) # reshape tensor
        x = F.relu(self.fc1(x)) # Activation function: ReLU
        x = F.relu(self.fc2(x)) # Activation function: ReLU
        x = self.fc3(x)
        return x

