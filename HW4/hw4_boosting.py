import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

from test_score import score

sns.set() 

class LinearRegression:
    def __init__(self, lr=0.0001, num_iter=100000, fit_intercept=True):
        self.lr = lr 
        self.num_iter = num_iter 
        self.fit_intercept = fit_intercept

    def __add_intercept(self, A):
        intercept = np.ones((A.shape[0], 1))
        return np.concatenate((intercept, A), axis=1)

    def __objfunc(self,x):
        b_ = np.dot(self.A,x)
        norm = np.linalg.norm( (b_ - self.b) )**2

        return .5 * norm

    def __gradient(self,x):
        tmp1 = np.dot( np.dot(self.A.T,self.A), x)
        tmp2 = np.dot(self.A.T,self.b)

        return tmp1 - tmp2

    def fit(self,A,b):
        if self.fit_intercept:
            self.A = self.__add_intercept(A)
        else:
            self.A = A 
        
        self.b = b.reshape(-1,1)
        self.theta = np.zeros((self.A.shape[1],1))
        
        self.iter_vals = []
        for i in range(self.num_iter):

            self.theta -= self.lr * self.__gradient(self.theta)

            if (i % 1000 == 0):
                z = self.__objfunc(self.theta)
                self.iter_vals.append(z)

    def predict(self,A):
        if self.fit_intercept:
            A = self.__add_intercept(A)
        pred = np.sign( np.dot(A,self.theta) )
        return pred
        
    def get_boundary(self,A):
        plot_x = np.array([min(A[:,0]), max(A[:,0])])
        plot_y = (-1/self.theta[2]) * (self.theta[1] * plot_x + self.theta[0])
        
        return plot_x,plot_y



model = LinearRegression()

