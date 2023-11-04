from model import Net
from data import Dataset
# def train(model, device, train_loader, optimizer, epoch):
#     for batch_idx, (data, input) in enumerate(train_loader):
#         data, input = data.to(device), input.to(device)
#         optimizer.zero_grad() # zero the gradient buffers
#         output = model(data)
#         loss = F.nll_loss(output, input)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 100 == 0:
#             print(f'Train Epoch: {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)} ({100.*batch_idx/len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

Dataset()

