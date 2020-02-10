# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas


# sklearn imports
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier



# our code
import utils

from knn import KNN

from naive_bayes import NaiveBayes

from decision_stump import DecisionStumpErrorRate, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
from random_forest import RandomForest

from kmeans import Kmeans
from sklearn.cluster import DBSCAN

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question


    if question == "1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]        
        model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

    elif question == "1.1":
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]

        depths = np.arange(1,15)

        t = time.time()
        test_errors = np.zeros(depths.size)
        train_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X, y)
            y_error = model.predict(X_test)
            test_errors[i] = np.mean(y_error != y_test)

            y_train = model.predict(X)
            train_errors[i] = np.mean(y_train != y)


        print("scikit-learn's decision tree took %f seconds" % (time.time()-t))

        plt.plot(depths, test_errors, label="test", linestyle=":", linewidth=3)
        plt.plot(depths, train_errors, label="train", linestyle=":", linewidth=3)


        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q1_1_tree_errors.pdf")
        plt.savefig(fname)




    elif question == '1.2':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        n, d = X.shape

        X_valid = X[:int(n/2),:]
        y_valid = y[:int(n/2)]
        X_train = X[int(n/2):,:]
        y_train = y[int(n/2):]
        
        depths = np.arange(1,15)

        t = time.time()
        sklearn_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_valid)
            sklearn_tree_errors[i] = np.mean(y_pred != y_valid)
        print("scikit-learn's decision tree took %f seconds" % (time.time()-t))

        plt.plot(depths, sklearn_tree_errors, label="sklearn", linestyle=":", linewidth=3)

        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "q1_2(1)_tree_errors.pdf")
        plt.savefig(fname)



    elif question == '2.2':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
        groupnames = dataset["groupnames"]
        wordlist = dataset["wordlist"]
        print(wordlist[50])

        x_frame = pandas.DataFrame({'bin': X[500,:],'name': wordlist},
        	columns=['bin','name'])
        x_filter = x_frame[x_frame['bin']==1]
        print(x_filter)

        print(groupnames[int(y[500])])
       




    elif question == '2.3':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]

        print("d = %d" % X.shape[1])
        print("n = %d" % X.shape[0])
        print("t = %d" % X_valid.shape[0])
        print("Num classes = %d" % len(np.unique(y)))

        model = NaiveBayes(num_classes=4)
        model.fit(X, y)

        y_pred = model.predict(X_valid)
        #print(y_pred)
        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes (ours) validation error: %.3f" % v_error)

        nb = BernoulliNB()
        nb.fit(X, y)
        y_pred1 = nb.predict(X_valid)
        #print(y_pred1)
        v_error1 = np.mean(y_pred1 != y_valid)
        print("BernoulliNB validation error: %.3f" % v_error1)


    

    elif question == '3':
        with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']

        model = KNN(k=1)
        model.fit(X, y)

        y_pred1 = model.predict(X)
        y_pred2 = model.predict(Xtest)
        
        v_error1 = np.mean(y_pred1 != y)
        v_error2 = np.mean(y_pred2 != ytest)
        print(v_error1)
        print(v_error2)

        utils.plotClassifier(model,Xtest,ytest)
        plt.title("KNN")
        fname = os.path.join("..", "figs", "q3.pdf")
        plt.savefig(fname)

        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(X, y)
        utils.plotClassifier(neigh,Xtest,ytest)
        plt.title("KNeighborsClassifier")
        fname = os.path.join("..", "figs", "q3(1).pdf")
        plt.savefig(fname)



    elif question == '4':
        dataset = load_dataset('vowel.pkl')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']
        print("\nn = %d, d = %d\n" % X.shape)

        def evaluate_model(model):
            model.fit(X,y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            print("    Training error: %.3f" % tr_error)
            print("    Testing error: %.3f" % te_error)

        print("Decision tree info gain")
        evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))

        t = time.time()
        print("Random Forest")
        evaluate_model(RandomForest(max_depth=np.inf, num_trees=50))
        print("Random Forest took %f seconds" % (time.time()-t))


        t = time.time()
        print("RandomForestClassifier")
        m = RandomForestClassifier(max_depth=np.inf, random_state=0)
        m.fit(X,y)

        y_pred = m.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = m.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("    Training error: %.3f" % tr_error)
        print("    Testing error: %.3f" % te_error)
        print("RandomForestClassifier took %f seconds" % (time.time()-t))
        

        #print("scikit-learn's decision tree took %f seconds" % (time.time()-t))



    elif question == '5':
        X = load_dataset('clusterData.pkl')['X']

        model = Kmeans(k=4)
        model.fit(X)
        y = model.predict(X)
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet")

        fname = os.path.join("..", "figs", "kmeans_basic.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == '5.1':
        X = load_dataset('clusterData.pkl')['X']
        model = Kmeans(k=4)
        low = model.error(X)

        for i in range(49):
            new_model = Kmeans(k=4)
            err = new_model.error(X)
            if err < low :
                model = new_model
                low = err

        utils.plot_2dclustering(X, model['predict'](model, X))
        print("Displaying figure...")
        plt.title("K-Means on clusterData")
        plt.show()



    elif question == '5.2':
        X = load_dataset('clusterData.pkl')['X']




    elif question == '5.3':
        X = load_dataset('clusterData2.pkl')['X']

        model = DBSCAN(eps=2, min_samples=2)
        y = model.fit_predict(X)

        print("Labels (-1 is unassigned):", np.unique(model.labels_))
        
        plt.scatter(X[:,0], X[:,1], c=y, cmap="jet", s=5)
        fname = os.path.join("..", "figs", "density.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        
        plt.xlim(-25,25)
        plt.ylim(-15,30)
        fname = os.path.join("..", "figs", "density2.png")
        plt.savefig(fname)
        print("Figure saved as '%s'" % fname)
        
    else:
        print("Unknown question: %s" % question)
