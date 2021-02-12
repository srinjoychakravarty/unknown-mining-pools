import sys
import time
import timeit
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn import datasets, metrics, svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from MultiClassifier import MultiClassifier

num_cores = 1

def print_results(accuracy, time, num_cores):
    print("Accuracy: " + str(accuracy), ", Time: " +
          str(time) + " seconds" + ", cores: " + str(num_cores))

def get_multi_classifier():
    clf1 = RidgeClassifier()
    clf2 = RandomForestClassifier(n_estimators=10)
    clf3 = LinearDiscriminantAnalysis()
    clf4 = GaussianNB()
    classifier = MultiClassifier([
        clf1,
        clf2,
        clf3,
        clf4
    ])
    return classifier


def train_test_run(X, y, num_cores):
    print("Train-test serial run:")
    # Split data into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)
    start = time.time()
    classifier = get_multi_classifier()
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
    acc = accuracy_score(predicted, y_test)
    end = time.time()
    print_results(acc, end - start, num_cores)

train_test_run(X, y, num_cores)