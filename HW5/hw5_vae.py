import torch 
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt 
import numpy as np


class VAE(nn.Module):
    '''
        Implements variational auto encoder for MNIST 
        dataset
    '''
    def __init__(self):
        super(VAE, self).__init__()
        n_input = 784
        zdim = 20

        ### Encoder
        self.layer1 = nn.Sequential(nn.Linear(n_input, 400),
                                    nn.ReLU())
        self.layerMu = nn.Sequential(nn.Linear(400, zdim))
        self.layerSig = nn.Linear(400, zdim)

        ### Decoder
        self.layer2 = nn.Sequential(nn.Linear(zdim, 400), 
                                    nn.ReLU())
        self.layer3 = nn.Sequential(nn.Linear(400, n_input), 
                                    nn.Sigmoid())

    def encode(self, x):
        x = self.layer1(x)
        return self.layerMu(x), self.layerSig(x)

    def sample(self, mu, sig):
        std = torch.exp(0.5*sig)
        n = torch.randn_like(std)
        return mu + n*std

    def decode(self, z):
        z = self.layer2(z)
        return self.layer3(z)

    def forward(self, x):
        mu, sig = self.encode(x)
        z = self.sample(mu, sig)
        out = self.decode(z)
        return (out, mu, sig)


def loss_fn(x_hat, x, mu, sig):
    '''
        Overall loss function 
    '''
    # reconstruction loss
    bce = nn.BCELoss(x_hat, x, reduction='sum')

    # KL divergence
    kl = -0.5 + torch.sum(1 + sig - mu.pow(2) - sig.exp())

    return bce + kl


def get_data():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
    write_dir = './data'
    return datasets.MNIST(root=write_dir, train=True, transform=transform, download=True)


# Get data
data = get_data():
data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
num_batches = len(data_loader)

# Initialize net
vae = VAE()

# Initialize optimizer and loss 
optimizer = optim.Adam(vae.params(), lr=0.001)
criterion = loss_fn

# Params 
num_epochs = 10
num_test_images = 16
test_images = data.train_data[:num_test_images]

# Train
avg_epoch_losses = []
for e in range(num_epochs):

    running_loss = 0
    for num_batch, (batch,_) in enumerate(data_loader):

        # Zero grad
        optimizer.zero_grad()

        # Forward pass
        sample, mu, sig = vae(batch)

        # Compute loss
        loss = criterion(sample, batch, mu, sig)

        # Updates
        loss.backward()
        running_loss += loss.item()
        optimizer.step()

    avg_loss = running_loss / num_batches
    avg_epoch_losses.append(avg_loss)

