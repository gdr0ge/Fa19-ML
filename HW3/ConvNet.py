import torch
import torchvision
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt 
import time


n_epochs = 3
lr = 0.01
batch_size = 32
log_interval = 100

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cv1 = nn.Conv2d(1,20,3,stride=1)
        self.fc1 = nn.Linear(3380,128,bias=True)
        self.fc2 = nn.Linear(128,10,bias=False)

    def forward(self,x):
        x = F.max_pool2d(self.cv1(x),(2,2))
        x = F.relu(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


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
# optimizer = optim.SGD(net.parameters(), lr=lr)
# optimizer = optim.Adam(net.parameters(), lr=lr)
optimizer = optim.Adagrad(net.parameters(), lr=lr)
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
    output = net(data)
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
      output = net(data)
      test_loss += loss_fn(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

train_times = []
for epoch in range(1, n_epochs + 1):
    start = time.time()
    train(epoch)
    train_times.append((time.time() - start))

print("")
print("Train time avg: ",sum(train_times)/len(train_times))

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

# [73.2, 66.75,66.52,64.56]
