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
train_loader, test_loader, val_loader = Dataset().execute(
    data_dir,
    batch_size,
    train_split,
    val_split,
    test_split
)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=optimizer):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

# See the model's performance before training
evaluate(net, train_loader)
# Train the model
history = fit(epochs, lr, net, train_loader, val_loader)

def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');

plot_accuracies(history)
