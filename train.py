from argparse import ArgumentParser

from model import Net
from data import Dataset

import torch
import torch.optim as optim
import torch.nn as nn



if __name__ == '__main__':
    arg = ArgumentParser()
    arg.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    arg.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    arg.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    arg.add_argument('--data-dir', type=str, default='./data', help='path to dataset (default: ./data)')
    arg.add_argument('--log-interval', type=int, default=100, help='how many batches to wait before logging training status')
    arg.add_argument('--save-model', action='store_true', default=False, help='for saving the current model')
    arg.add_argument('--val-split', type=float, default=0.1, help='validation split (default: 0.1)')
    arg.add_argument('--train-split', type=float, default=0.7, help='train split (default: 0.7)')
    arg.add_argument('--test-split', type=float, default=0.2, help='test split (default: 0.2)')
    
args = arg.parse_args()

# Training settings
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr
data_dir = args.data_dir
log_interval = args.log_interval
save_model = args.save_model
val_split = args.val_split
train_split = args.train_split
test_split = args.test_split

# Initialize model
net = Net()

# Create the optimizer
optimizer = optim.SGD(net.parameters(), lr=lr)

# Create the loss function
criterion = nn.CrossEntropyLoss()

# Create the data loader
train_loader, test_loader = Dataset().execute(
    data_dir,
    batch_size,
    train_split,
    val_split,
    test_split
)

# Training loop
net.train()
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    accuracy = 100 * (output.argmax(1) == target).sum().item() / len(data)

    print(f"Epoch: {epoch} | Loss: {loss.item()} | Accuracy: {accuracy}")

# Save the model
if save_model:
    torch.save(net.state_dict(), "cifar10_net.pt")
