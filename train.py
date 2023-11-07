from model import Net
from data import Dataset

import torch
import torch.optim as optim
import torch.nn as nn


epochs = 100
# Create the network
net = Net()

# Create the optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Create the loss function
criterion = nn.CrossEntropyLoss()

# Create the data loader
train_loader, test_loader = Dataset().execute('../data', 64)
print(train_loader)
# Training loop
net.train()
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item()
            ))
