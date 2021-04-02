import torch
import torch.nn as nn 
import torch.nn.function as F 

'''
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
'''

def Net(nn.Module):
    super(Net, self).__init__()

    self.conv1 = nn.Conv2d(1,6,3)
    self.conv2 = nn.Conv2d(6,16,3)
    # an affine operation: y = Wx + b
    self.fc1 = nn.Linear(16 * 6 * 6, 120) # 6*6 from image dimension
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2,2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # if the size is a square you only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x 

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimentsion
        num_features = 1
        for s in size:
            num_features *= s 
        return num_features


net = Net()
print(net)


# Loss Function 
input = torch.randn(1,1,32,32) # single channel 32*32
output = net(input)
target = torch.randn(10)
target = target.view(1,-1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

'''
if you call loss.backward() the whole graph is differentiated w.r.t the loss 
and all Tensors in the graph that has requires_grad=True will have their .grad 
Tensor accumulated with the gradient 
'''

net.zero_grad() # zeroes the gradient buffers of all parameters 
loss.backward()

'''
update weights with weight = weight - learning  * gradient
'''
import torch.optim as optim 

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() # Does the update 

'''
Example of training the network 
'''
for epock in range(2):

    running_loss = 0.0 
    for i, data in enumerate(data, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data 

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize 
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999: 
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')