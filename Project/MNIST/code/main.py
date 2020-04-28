import os
import pickle
import gzip
import argparse

from knn import KNN
import linear_model
from neural_net import NeuralNet
from svm import SVM
from cnn import CNN

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question
    
    # KNN
    if question == "1.1": 


        with open(os.path.join('..', 'data', 'mnist.pkl'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

        X, y = train_set
        Xtest, ytest = test_set


        model = KNN(k=5)
        model.fit(X, y)

        
        y_pred2 = model.predict(Xtest)
        v_error2 = np.mean(y_pred2 != ytest)
        
        print(v_error2)
    
    # linear model
    elif question == "1.2": 


        with open(os.path.join('..', 'data', 'mnist.pkl'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

        X, y = train_set
        Xtest, ytest = test_set

        model = linear_model.LinearModelGradient()
        model.fit(X, y)

        
        y_pred2 = model.predict(Xtest)
        v_error2 = np.mean(y_pred2 != ytest)

        print(v_error2)

    # SVM
    elif question == "1.3": 


        with open(os.path.join('..', 'data', 'mnist.pkl'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

        X, y = train_set
        Xtest, ytest = test_set

        model = SVM()
        model.fit(X, y)

        
        y_pred2 = model.predict(Xtest)
        v_error2 = np.mean(y_pred2 != ytest)
        
        print(v_error2)

    # MLP
    elif question == "1.4": 


        with open(os.path.join('..', 'data', 'mnist.pkl'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

        X, y = train_set
        Xtest, ytest = test_set


        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

        hidden_layer_sizes = [50]
        model = NeuralNet(hidden_layer_sizes)

        
        model.fit(X,Y)
        
        y_pred2 = model.predict(Xtest)
        v_error2 = np.mean(y_pred2 != ytest)
        
        print(v_error2)

    # CNN
    elif question == "1.5": 


        with open(os.path.join('..', 'data', 'mnist.pkl'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")

        X, y = train_set
        Xtest, ytest = test_set

        model = CNN()
        model.fit(X, y)

        
        y_pred2 = model.predict(Xtest)
        v_error2 = np.mean(y_pred2 != ytest)
        
        print(v_error2)

    else:
        print("Unknown question: %s" % question)    