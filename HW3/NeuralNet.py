import torch
import torchvision
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import matplotlib.pyplot as plt 

n_epochs = 6
lr = 0.01
batch_size = 32
log_interval = 100

class Net(nn.Module):
    def __init__(self):
        super(Net, self,).__init__()
        # self.bias = nn.Parameter(torch.ones((28*28),1))
        self.fc1 = nn.Linear(1 * 28 * 28, 128,bias=True)
        self.fc2 = nn.Linear(128, 10,bias=False)
    
    def forward(self,x):
        # Flatten image
        # x = x + self.bias
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x


train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size, shuffle=True)


net = Net()
optimizer = optim.SGD(net.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

train_losses = []
train_accuracies = []
test_losses = []

def train(epoch):
  net.train()
  correct = 0
  train_loss = 0
  t_losses = []
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = net(data.reshape(batch_size,28*28))
    loss = loss_fn(output, target)
    train_loss += loss.item()
    t_losses.append(loss.item())
    pred = output.data.max(1, keepdim=True)[1]
    correct += pred.eq(target.data.view_as(pred)).sum()
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
    #   torch.save(net.state_dict(), '/results/model.pth')
    #   torch.save(optimizer.state_dict(), '/results/optimizer.pth')
  train_loss /= len(train_loader.dataset)
  train_losses.append(t_losses)
  train_accuracies.append(100. * correct / len(train_loader.dataset))
  print('\nTrain set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    train_loss, correct, len(train_loader.dataset),
    100. * correct / len(train_loader.dataset)))

def test():
  net.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = net(data.reshape(batch_size,28*28))
      test_loss += loss_fn(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


for epoch in range(1, n_epochs + 1):
  train(epoch)

plt.figure()
for i,t in enumerate(train_losses):
    plt.plot(range(len(t)),t,label="epoch-{}".format(i))
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss functions for each epoch")
plt.legend()
plt.show()

plt.figure()
plt.plot(range(len(train_accuracies)),train_accuracies)
plt.xlabel("Epoch #")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Over Training Epochs")
plt.show()