import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.network = nn.Sequential(
            # 3 input image channel, 64 output channels, 3x3 square convolution, stride of 1, padding of 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), # 64x32x32x3 -> 64x32x32x64
            nn.ReLU(), # activation function (relu) -> 64x32x32x64
            nn.MaxPool2d(2, 2), # max pooling layer 2x2 -> 64x16x16x64
            # 64 input image channel, 128 output channels, 5x5 square convolution, stride of 1, padding of 1
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 64x16x16x64 -> 128x16x16x64
            nn.ReLU(), # activation function (relu) -> 128x16x16x64
            nn.MaxPool2d(2, 2), # max pooling layer 2x2 -> 128x8x8x64
            nn.Flatten(), # flatten -> 128x8x8x64 -> 128x4096
            nn.Linear(128 * 8 * 8, 512), # fully connected layer (128 * 8 * 8 inputs -> 512 outputs)
            nn.ReLU(), # activation function (relu) -> 128x4096 -> 128x512
            nn.Linear(512, 128), # fully connected layer (512 inputs -> 128 outputs)
            nn.ReLU(), # activation function (relu) -> 128x512 -> 128x128
            nn.Linear(128, 10) # fully connected layer (128 inputs -> 10 outputs)
        )

    
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    
    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))


    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = self.accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}


    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}


    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


    def forward(self, x):
        return self.network(x)

