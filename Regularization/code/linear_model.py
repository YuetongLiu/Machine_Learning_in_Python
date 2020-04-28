import numpy as np
from numpy.linalg import solve
import findMin
from scipy.optimize import approx_fprime
import utils

class logReg:
    # Logistic Regression
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)

class logRegL2:
    # Logistic Regression L2
    def __init__(self, verbose=1, lammy=1.0, maxEvals=100):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)
        lammy = self.lammy

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + (w.T.dot(w) * lammy / 2)

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) + (lammy * w)

        return f, g

    def fit(self, X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals,X, y, verbose=self.verbose)

    def predict(self, X):
        w = self.w
        yhat = np.dot(X, w)

        return np.sign(yhat)

class logRegL1:
    # Logistic Regression L1
    def __init__(self, verbose=1, L1_lambda=1.0, maxEvals=100):
        self.verbose = verbose
        self.L1_lambda = L1_lambda
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self, X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMinL1(self.funObj, self.w, self.L1_lambda,
                                             self.maxEvals,X, y,
                                             verbose=self.verbose)

    def predict(self, X):
        w = self.w
        yhat = np.dot(X, w)

        return np.sign(yhat)


class logRegL0(logReg):
    # L0 Regularized Logistic Regression
    def __init__(self, L0_lambda=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L0_lambda = L0_lambda
        self.maxEvals = maxEvals

    def fit(self, X, y):
        n, d = X.shape
        minimize = lambda ind: findMin.findMin(self.funObj,
                                                  np.zeros(len(ind)),
                                                  self.maxEvals,
                                                  X[:, ind], y, verbose=0)
        selected = set()
        selected.add(0)
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        while minLoss != oldLoss:
            oldLoss = minLoss
            print("Epoch %d " % len(selected))
            print("Selected feature: %d" % (bestFeature))
            print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):
                if i in selected:
                    continue

                selected_new = selected | {i} # tentatively add feature "i" to the seected set
                w = np.zeros(d)
                w[list(selected_new)], loss = minimize(list(selected_new))
                self.L0 = (self.L0_lambda * np.count_nonzero(w))
                loss += self.L0
                if loss < minLoss:
                    minLoss = loss
                    bestFeature = i
                    w0 = w



                # TODO for Q2.3: Fit the model with 'i' added to the features,
                # then compute the loss and update the minLoss/bestFeature


            selected.add(bestFeature)

        self.w = np.zeros(d)
        self.w[list(selected)], _ = minimize(list(selected))


class leastSquaresClassifier:
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T@X+0.0001*np.eye(d), X.T@ytmp)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)


class logLinearClassifier(logReg):
    # Q3 - multi-classification with Logistic loss

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((d, self.n_classes))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y == i] = 1
            ytmp[y != i] = -1

            self.W[:, i], _ = findMin.findMin(self.funObj, self.W[:, i],
                                         self.maxEvals, X, ytmp,
                                         verbose=self.verbose
                                         )

    def predict(self, X):
        yhat = np.dot(X, self.W)

        return np.argmax(yhat, axis=1)



class softmaxClassifier:
    # Q3 - multi-classification with Softmax loss
    def __init__(self, verbose=1, maxEvals=400):
        self.verbose = verbose
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        n, d = X.shape
        W = np.reshape(w, (d, self.n_classes))

        Xw = X.dot(W)
        Xwy = np.zeros((n,))
        g1 = np.zeros(W.shape)
        g2 = np.zeros(W.shape)
        res = np.ones((n,))

        for i in range(n):
            Xwy[i] = Xw[i, y[i]]

        for i in np.unique(y):
            g1[:, i] = -np.sum(X[y == i], axis=0)
            den = np.sum(np.exp(Xw), axis=1)
            num = np.exp(Xw[:, i])
            res[:] = num[:]/den[:]
            g2[:, i] = res.dot(X)

        f = -np.sum(Xwy)+np.sum(np.log(np.sum(np.exp(Xw), axis=1)))

        g = g1 + g2

        return f, g.ravel()

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.w = np.zeros(d*self.n_classes)
        utils.check_gradient(self, X, y)

        self.w, _ = findMin.findMin(self.funObj, self.w,
                                    self.maxEvals,X, y,
                                    verbose = self.verbose
                                    )

        self.w = np.reshape(self.w, (d, self.n_classes))

    def predict(self, X):
        yhat = np.dot(X, self.w)

        return np.argmax(yhat, axis=1)
