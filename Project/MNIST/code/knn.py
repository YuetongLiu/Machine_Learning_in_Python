"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y 

    def predict(self, Xtest):
       # raise NotImplementedError()
       n,d = self.X.shape
       t,d = Xtest.shape

       ytest = np.zeros(t)
       Dis = utils.euclidean_dist_squared(self.X, Xtest)


       for i in range(t):

       	    neighbor = np.argsort(Dis[:, i])
       	    ytest[i] = utils.mode(self.y[neighbor[0:self.k]])

       return ytest

     


