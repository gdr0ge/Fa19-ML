# import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import cvxopt 


def get_next_train_valid(X_shuffled, y_shuffled, itr):
    """
    Return one validation set and concatenate the rest to 
    use as a training set
    """
    val_x, val_y = X_shuffled[itr], y_shuffled[itr][:,None]
    
    training_x = np.empty((0,X_shuffled[0].shape[1]))
    training_y = np.empty((0,1))
    
    for k,v in X_shuffled.items():
      if k != itr:
        training_x = np.concatenate((training_x,v),axis=0)
    
    for k,v in y_shuffled.items():
      if k != itr:
        training_y = np.concatenate((training_y,v.reshape(v.shape[0],1)),axis=0)
        
        
    return training_x, training_y, val_x, val_y

def get_data(split):
    '''
    Function to get the training and test data
    '''
    data = pd.read_csv('hw2data.csv',header=None)

    # Convert to numpy matrix [X:y] were X is features matrix and y is label vector
    data = data.values

    # Shuffle the data 
    np.random.shuffle(data)

    # Split data into train and test based on 
    # given split value 
    index_split = int(split * data.shape[0])
    train_set = data[:index_split]
    test_set = data[index_split:]

    # Get train and test x,y
    xtrain, ytrain = train_set[:,:-1], train_set[:,-1]
    xtest, ytest = test_set[:,:-1], test_set[:,-1]

    return xtrain,ytrain.reshape(-1,1),xtest,ytest.reshape(-1,1)


def svmfit(X,y,C):
    '''
    Function to traint the SVM model with slack variables 
    and using the cvxopt optimizer
    '''
    global w 
    global b

    D,f = X.shape

    # Gram matrix 
    K = np.zeros((D,D))
    for i in range(D):
        for j in range(D):
            K[i,j] = np.dot(X[i],X[j])

    P = cvxopt.matrix(np.outer(y,y) * K)
    q = cvxopt.matrix(-np.ones(D))
    A = cvxopt.matrix(y, (1,D))
    b = cvxopt.matrix(0.0)

    G = cvxopt.matrix(np.vstack(( np.diag(-np.ones(D)), np.identity(D) )))
    h = cvxopt.matrix(np.hstack((  np.zeros(D), np.ones(D)*C )))

    sol = cvxopt.solvers.qp(P,q,G,h,A,b)

    # Lagrange multipliers
    threshold = 1e-4
    lmult = np.array([sol['x']])

    # Support vectors 
    indices = np.arange(len(lmult))[lmult > threshold]

    lmult = lmult[lmult > threshold]
    sv_x = X[threshold]
    sv_y = y[threshold]

    # Intercept solution
    b = 0
    for n in range(len(lmult)):
        b += sv_y[n]
        b -= np.sum(lmult * sv_y*K[indices[n],sv_x])
    b /= len(lmult)

    # Weight vector using solver solutions
    w = np.zeros(f)
    for n in range(len(lmult)):
        w += lmult[n] * sv_y[n] * sv_x[n]


def svmpredict(X):
    '''
    Function to predict classification 
    '''
    return np.sign(np.dot(X,w) + b)


'''
Main Logic
'''

# The weight vector and intercept term
w = 0
b = 0
C = 0.0001

# train split percentage
split = 0.8

# Get the data
xtrain_split,ytrain_split,xtest,ytest = get_data(split)
train_comb = np.concatenate((xtrain_split,ytrain_split), axis=1)

# Perform the shuffling and folding of data
k = 10

X_shuffled = dict([(i,feat) for i,feat in enumerate(np.array_split(train_comb[:,:-1],k))])
y_shuffled = dict([(i,feat) for i,feat in enumerate(np.array_split(train_comb[:,-1],k))])

errors_train = []
errors_valid = []
for i in range(0,k):

    # Get training and validation sets
    x_train, y_train, x_valid, y_valid = get_next_train_valid(X_shuffled, y_shuffled, i)

    # Train/Optimize 
    svmfit(x_train,y_train,C)

    # Get prediction values 
    pred_train = svmpredict(x_train)
    pred_valid = svmpredict(x_valid)

    # Get errors
    error_train = np.not_equal(pred_train,y_train).astype(int).sum()
    error_valid = np.not_equal(pred_valid,y_valid).astype(int).sum()

    errors_train.append(error_train)
    errors_test.append(error_valid)

avg_train_error = sum(errors_train) / len(errors_train)
avg_valid_error = sum(errors_valid) / len(errors_valid)

print("Average error rate across folds: Train --> {} \t Test --> {}".format(avg_train_error,avg_valid_error))