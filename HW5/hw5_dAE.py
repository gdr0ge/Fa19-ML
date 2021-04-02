import torch 
from torch import nn, optim
from torch.autograd.variable import Variable 
from torchvision import transforms, datasets
import matplotlib.pyplot as plt 
import numpy as np


class Encoder(nn.Module):
    '''
        Implements Encoder for dAE, outputs the compressed
        representation
    '''
    def __init__(self):
        super(Encoder, self).__init__()
        n_input = 784
        n_out = 20

        self.layer1 = nn.Sequential(nn.Linear(n_input, 400), 
                                    nn.ReLU())
        self.layerOut = nn.Sequential(nn.Linear(400, n_out),
                                    nn.ReLU())

    def forward(self, X):
        x = self.layer1(X)
        out = self.layerOut(x)
        return out


class Decoder(nn.Module):
    '''
        Implements Decoder for dAE, outputs reconstructed input
        with limited noise
    '''
    def __init__(self):
        super(Decoder, self).__init__()
        n_input = 20
        n_out = 784

        self.layer1 = nn.Sequential(nn.Linear(n_input, 400),
                                    nn.ReLU())
        self.layerOut = nn.Sequential(nn.Linear(400, n_out),
                                      nn.Sigmoid())

    def forward(self, X):
        x = self.layer1(X)
        out = self.layerOut(x)
        return out


def get_data():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
    write_dir = './data'
    return datasets.MNIST(root=write_dir, train=True, transform=transform, download=True)

def add_noise(images, d1):
    '''
        Generate noise and add to input image
    '''
    noise = torch.randint(-128,128,(d1,784))
    images = images.numpy() + noise.numpy()
    images = np.where(images < 0, 0, images)
    images = np.where(images > 255, 255, images)

    return torch.Tensor(images)


# Get data
data = get_data():
data_loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
num_batches = len(data_loader)

# Initialize nets
encoder = Encoder()
decoder = Decoder()

# Initialize optimizers 
e_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
d_optimizer = optim.Adam(decoder.parameters(), lr=0.001)

# Initialize loss
criterion = nn.BCELoss()

# Params and test images to view denoising
num_epochs = 10
num_test_images = 5
test_images = data.train_data[:num_test_images]
test_input = add_noise(test_images, num_test_images)

# Train
avg_epoch_losses = []
for e in range(num_epochs):

    running_loss = 0
    for num_batch, (batch,label) in enumerate(data_loader):

        # Zero grad
        e_optimizer.zero_grad()
        d_optimizer.zero_grad()

        # Feed forward
        compressed = Encoder(batch)
        output = Decoder(compressed)

        # Get loss
        loss = criterion(output, label)
        running_loss += loss.item()

        # Update 
        loss.backward()
        e_optimizer.step()
        d_optimizer.step()

    avg_loss = running_loss / num_batches
    avg_epoch_losses.append(avg_loss)



