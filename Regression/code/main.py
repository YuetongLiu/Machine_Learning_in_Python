
# basics
import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# our code
import linear_model
import utils

url_amazon = "https://www.amazon.com/dp/%s"

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)
    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":

        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))

        print("Number of ratings:", len(ratings))
        print("The average rating:", np.mean(ratings["rating"]))

        n = len(set(ratings["user"]))
        d = len(set(ratings["item"]))
        print("Number of users:", n)
        print("Number of items:", d)
        print("Fraction nonzero:", len(ratings)/(n*d))

        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        print(type(X))
        print("Dimensions of X:", X.shape)

    elif question == "1.1":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0
        
        # YOUR CODE HERE FOR Q1.1.1
        index=np.argmax(np.sum(X,axis=0))
        value=np.max(np.sum(X,axis=0))
        pop_item_ID=item_inverse_mapper[index]
        pop_item_ID_url=url_amazon % pop_item_ID
        #print(pop_item_ID)
        print(pop_item_ID_url)
        print('Number of total stars it gets are: '+np.str(value))

        # YOUR CODE HERE FOR Q1.1.2
        array=(X!=0).sum(axis=1)
        user=np.argmax(array)
        pop_user=user_inverse_mapper[user]
        pop_user_value=np.max(array)
        print('The user who rated the most items is: '+pop_user)
        print('The number of items he/she rated are: '+np.str(pop_user_value))

        # YOUR CODE HERE FOR Q1.1.3
        plt.figure() 
        plt.hist(X.getnnz(axis=1),bins=10)
        plt.yscale('log',nonposy='clip')
        plt.title("The number of ratings per user")
        plt.xlabel('# of ratings')
        plt.ylabel('# of users who have this rating')
        fname = os.path.join("..", "figs", "q1_1_3_1.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

        plt.figure() 
        plt.hist(X.getnnz(axis=0),bins=10)
        plt.yscale('log',nonposy='clip')
        plt.title("The number of ratings per item")
        plt.xlabel('# of ratings')
        plt.ylabel('# of items who have this rating')
        fname = os.path.join("..", "figs", "q1_1_3_2.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        
        plt.figure() 
        #plt.hist(scipy.sparse.find(X)[2],bins=10)
        plt.hist(X.data,bins=10)
        plt.title("The number of ratings themselves")
        plt.xlabel('ratings')
        plt.ylabel('# of ratings')
        fname = os.path.join("..", "figs", "q1_1_3_3.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == "1.2":
        filename = "ratings_Patio_Lawn_and_Garden.csv"
        with open(os.path.join("..", "data", filename), "rb") as f:
            ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
        X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
        X_binary = X != 0

        grill_brush = "B00CFM0P7Y"
        grill_brush_ind = item_mapper[grill_brush]
        grill_brush_vec = X[:,grill_brush_ind]

        print(url_amazon % grill_brush)

        # YOUR CODE HERE FOR Q1.2
        # euclidean similarity
        neigh=NearestNeighbors(n_neighbors=6, metric='euclidean')
        neigh.fit(np.transpose(X))
        euclidean=neigh.kneighbors(np.transpose(grill_brush_vec))
        
        print(euclidean[1].flatten())
        for i in euclidean[1].flatten():
            print(item_inverse_mapper[i])
        print('\n')
        
        #normalized similarity
        X_norm=normalize(X, norm='l2', axis=0,return_norm=False) #normalize cols of X
        neigh=NearestNeighbors(n_neighbors=6, metric='euclidean')
        neigh.fit(np.transpose(X_norm))
        normalized_euclidean=neigh.kneighbors(np.transpose(grill_brush_vec))
        
        print(normalized_euclidean[1].flatten())
        for i in normalized_euclidean[1].flatten():
            print(item_inverse_mapper[i])
        print('\n')
        
        #cosine similarity
        neigh=NearestNeighbors(n_neighbors=6, metric='cosine')
        neigh.fit(np.transpose(X))
        cosine=neigh.kneighbors(np.transpose(grill_brush_vec))
        
        print(cosine[1].flatten())
        for i in cosine[1].flatten():
            print(item_inverse_mapper[i])



        # YOUR CODE HERE FOR Q1.3
        #euclidean distance
        reviews=X[:,euclidean[1].flatten()].getnnz(axis=0)
        print(reviews)
        
        #cosine similarity
        reviews=X[:,cosine[1].flatten()].getnnz(axis=0)
        print(reviews)


    elif question == "3":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LeastSquares()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Least Squares",filename="least_squares_outliers.pdf")

    elif question == "3.1":
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # YOUR CODE HERE
        model = linear_model.WeightedLeastSquares()
        n,d=X.shape

        #setting v = 1 for the first 400 data points and v = 0.1 for the last 100 data points
        v=np.array(np.zeros([n,1]))
        v[0:399]=1
        v[400:n]=0.1
        V=np.diag(v.flatten())

        #fit to model
        model.fit(X,y,V)

        print(model.w)
        utils.test_and_plot(model,X,y,title="3.1", filename="q3_1.pdf")


    elif question == "3.3":
        # loads the data in the form of dictionary
        data = load_dataset("outliersData.pkl")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LinearModelGradient()
        model.fit(X,y)
        print(model.w)

        utils.test_and_plot(model,X,y,title="Robust (L1) Linear Regression",filename="least_squares_robust.pdf")

    elif question == "4":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # Fit least-squares model
        model = linear_model.LeastSquares()
        model.fit(X,y)

        utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, no bias",filename="least_squares_no_bias.pdf")

    elif question == "4.1":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        # YOUR CODE HERE
        # Fit LeastSquaresBias model
        model = linear_model.LeastSquaresBias()
        model.fit(X,y)

        utils.test_and_plot(model,X,y,Xtest,ytest,title="4.1",filename="q4_1.pdf")

    elif question == "4.2":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        for p in range(11):
            print("p=%d" % p)

            # YOUR CODE HERE
            model = linear_model.LeastSquaresPoly(p)
            model.fit(X,y)

            utils.test_and_plot(model,X,y,Xtest,ytest,title="4.2",filename="4_2.pdf")

    else:
        print("Unknown question: %s" % question)







