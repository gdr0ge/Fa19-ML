import torch 
from torch import nn, optim
from torch.autograd.variable import Variable 
from torchvision import transforms, datasets
import matplotlib.pyplot as plt 


class Discriminator(nn.Module):
    '''
        Discriminator class who's goal is to maximize 
        D(x) and minimize D(G(z))
    '''
    def __init__(self):
        super(Discriminator, self).__init__()
        n_input = 784
        n_out = 1

        self.layer1 = nn.Sequential(nn.Linear(n_input,512),
                                    nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(nn.Linear(512, 256),
                                    nn.LeakyReLU(0.2))
        self.layerOut = nn.Sequential(nn.Linear(256, n_out),
                                      nn.Sigmoid())

    def forward(self, X):
        x = self.layer1(X)
        x = self.layer2(x)
        x = self.layerOut(x)
        return x


class Generator(nn.Module):
    '''
        Generator class who's goal is to maximize
        D(G(z))
    '''
    def __init__(self):
        super(Generator, self).__init__()
        n_input = 128
        n_out = 784

        self.layer1 = nn.Sequential(nn.Linear(n_input, 256),
                                    nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(nn.Linear(256, 512),
                                    nn.LeakyReLU(0.2))
        self.layerOut = nn.Sequential(nn.Linear(512, n_out),
                                      nn.Tanh())

    def forward(self, X):
        x = self.layer1(X)
        x = self.layer2(x)
        x = self.layerOut(x)
        return x


def get_data():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
    write_dir = './data'
    return datasets.MNIST(root=write_dir, train=True, transform=transform, download=True)

def noise(d1):
    '''
        Generate noise for input to Generator
    '''
    n = Variable(torch.randn(d1, 128))
    return n

def train_d(opt, real, fake):
    '''
        Function for training the discriminator
    '''
    d1 = real.szie(0)
    opt.zero_grad()

    pred_real = discriminator(real)
    # Get error and update weights
    error_real = loss(pred_real, Variable(torch.ones(d1,1)))
    error_real.backward()

    pred_fake = discriminator(fake)
    # Get error and update weights
    error_fake = loss(pred_fake, Variable(torch.zeros(d1, 1)))
    error_fake.backward()

    opt.step()

    return (error_real + error_fake, pred_real, pred_fake)

def train_g(opt, fake):
    '''
        Function for training the generator
    '''
    d1 = fake.size(0)
    opt.zero_grad()

    pred = discriminator(fake)
    # Get error and update weights
    error = loss(pred, Variable(torch.ones(d1, 1)))
    error.backward()

    opt.step()

    return error

# Get data 
data = get_data()
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
num_batches = len(data_loader)

# Build Nets 
discriminator = Discriminator()
generator = Generator()

# Optimizer and loss
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
g_optimizer = optim.Adam(generator.parameters(), lr=0.001)
loss = nn.BCELoss()

# Testing
num_epochs = 50
num_test_images = 16
test_image_input = noise(num_test_images)

g_erros = []
d_errors = []
for e in range(num_epochs):
    for num_batch, (real,_) in enumerate(data_loader):
        d1 = real.size(0)

        ###### Discriminator
        real_data = Variable(real.view(real.size(0), 784))

        # Generate fake image
        fake_data = generator(noise(d1)).detach()

        # Train
        d_error, d_pred_real, d_pred_fake = train_d(d_optimizer, real_data, fake_data)

        ####### Generator
        fake_data = generator(noise(d1))

        g_error = train_g(g_optimizer, fake_data)

        # Log errors
        g_errors.append(g_error)
        d_errors.append(d_error)

        if (num_batch % 10 == 0):
            test_images = generator(test_image_input)
            test_images = test_images.view(test_images.size(0), 1, 28, 28).data


            





