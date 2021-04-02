# import libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd



def one_hot(y, numOfClasses):
    """
    Convert a vector into one-hot encoding matrix where that particular column value is 1 and rest 0 for that row.
    """
    y = np.asarray(y, dtype='int32')

    if len(y) > 1:
        y = y.reshape(-1)

    if not numOfClasses:
        numOfClasses = np.max(y) + 1

    yMatrix = np.zeros((len(y), numOfClasses))
    yMatrix[np.arange(len(y)), y] = 1

    return yMatrix


def get_data():
    '''
    Function to get the training and test data
    '''
    mnist_train = pd.read_csv('mnist_train.csv',header=None)
    mnist_test = pd.read_csv('mnist_test.csv',header=None)

    # Convert to numpy matrix [X:y] were X is features matrix and y is label vector
    mnist_train_data = mnist_train.values
    mnist_test_data = mnist_test.values


    # Normalize to range 0 to 1
    xtrain = mnist_train_data[:,1:].astype(np.float32)/255
    xtest = mnist_test_data[:,1:].astype(np.float32)/255

    ytrain = mnist_train_data[:,0].reshape(-1,1)
    ytest = mnist_test_data[:,0].reshape(-1,1)

    return xtrain,ytrain,xtest,ytest



def softmax(data):
  ''' 
  data is our batch matrix where each row 
  represents the linear mapping for a sample row x
  '''
  # Shift values so prob doesnt blow up
  data -= np.max(data)
  prob = (np.exp(data).T / np.sum(np.exp(data), axis=1)).T 

  return prob


def getGrad(x,y):
  '''
  Get gradient of cross entropy to shift
  weights
  '''
  prob = softmax( np.dot(x, W) )

  norm = 1 / x.shape[0]

  grad = -norm * np.dot(x.T, (y - prob)) 

  return grad

def getLoss(x,y):
  '''
  Compute the loss for the predictions and 
  the real labels (one-hot encoded)
  '''
  data = np.dot(x, W)
  p = softmax(data)

  loss = -np.log(np.max(p)) * y
  total_loss = np.sum(loss) / x.shape[0]

  return total_loss

def multi_logistic_train(xtrain,ytrain):
  '''
  Main training function that loops over all batches 
    --> Computes loss
    --> Computes gradient
    --> Shifts weights 
  '''
  global W 

  train_loss = 0
  test_loss = 0
  
  # Main loop for batches
  batch_losses = []
  for i in range(0,xtrain.shape[0],batch_size):
    x_batch = xtrain[i:i+batch_size]
    y_batch = ytrain[i:i+batch_size]

    # Encode class labels into one-hot encoding
    yEnc = one_hot(y_batch.ravel(), num_outputs)

    # Calculate cross-entropy loss
    loss = getLoss(x_batch, yEnc)

    # Calculate gradient of cross-entropy loss
    grad = getGrad(x_batch, yEnc)

    # Shift to new weights
    W -= lr * grad

    # Record loss on this batch
    batch_losses.append(loss)

  # Recode losses on training data and testing data
  train_loss = np.sum(batch_losses) / len(batch_losses)
  test_loss = getLoss(x_batch, y_batch)

  return train_loss, test_loss, batch_losses 


def multi_logistic_predict(x):
  '''
  Function to predict classifications given 
  input batch
  '''
  global W 
  
  return np.argmax(np.dot(x, W), 1)


def getAccuracy(x,y):
  '''
  Calculates the accuracy of predictions
  '''
  pred = multi_logistic_predict(x)
  pred = pred.reshape((-1,1))

  return np.mean(np.equal(y,pred))


def plot_loss(losses,kind):
  '''
  Plot loss for given losses array
  '''
  plt.figure
  plt.title("Loss Plot for {}".format(kind))
  plt.xlabel("Iteration")
  plt.ylabel("Loss")
  plt.plot(range(len(losses)),losses)

  plt.show()


"""
Overall logic
"""
# Initialize parameters
num_inputs = 784
num_outputs = 10
lr = .75
batch_size = 32

# Initialize weight matrix to random normal distribution
W = np.random.random_sample((num_inputs, num_outputs))

# Get the data for training and testing
xtrain,ytrain,xtest,ytest = get_data()

# Train the model
train_loss, test_loss, batch_losses = multi_logistic_train(xtrain,ytrain)

# Get the accuracies for the training and testing data
# after the model is trained
train_accuracy = getAccuracy(xtrain,ytrain)
test_accuracy = getAccuracy(xtest,ytest)

print("")
print("Train Loss: {} \t Test Loss: {}".format(train_loss,test_loss))
print("")
print("Train Accuracy: {} \t Test Accuracy: {}".format(train_accuracy,test_accuracy))
print("")

plot_loss(batch_losses,"Batch")


# k = 5

# X_shuffled = dict([(i,feat) for i,feat in enumerate(np.array_split(data_comb[:,:-1],k))])
# y_shuffled = dict([(i,feat) for i,feat in enumerate(np.array_split(data_comb[:,-1],k))])