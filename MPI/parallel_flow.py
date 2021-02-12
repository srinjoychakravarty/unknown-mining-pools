import argparse
import sys
import time
import timeit
import numpy as np
import pandas as pd
import seaborn as sn
from mpi4py import MPI
from sklearn import datasets, metrics, svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFfold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import StackingClassifier as st

def bcast_data(data):
    print(
        f'[INFO] Bcasting data from the root process ({rank})') if rank == 0 else None
    bcast_start_time = MPI.Wtime()
    data = comm.bcast(data, root=0)
    bcast_finish_time = MPI.Wtime()

    bcast_time = bcast_finish_time - bcast_start_time
    print(f'[TIME] Master process ({rank}) finished Bcasting data with time {bcast_time}') if rank == 0 else print(
        f'[TIME] Process {rank} finished receive bcasted data with time {bcast_time}')
    return data

def classify(X_train, X_test, y_train, y_test):
    
    algorithm = None    # classification
    classification_time_start = MPI.Wtime()
    if rank == 0:
        algorithm = 'ridge'
        clf0 = RidgeClassifier()
        st.fit(clf0, X_train, y_train)
        classification_output = st.predict(clf0, X_test)
        pass
    elif rank == 1:
        algorithm = 'randomForest'
        clf1 = RandomForestClassifier(n_estimators=10)
        st.fit(clf1, X_train, y_train)
        classification_output = st.predict(clf1, X_test)
        pass
    elif rank == 2:
        algorithm = 'lda'
        clf2 = LinearDiscriminantAnalysis()
        st.fit(clf2, X_train, y_train)
        classification_output = st.predict(clf2, X_test)
        pass
    elif rank == 3:
        algorithm = 'GaussianNaiveBayes'
        clf3 = GaussianNB()
        st.fit(clf3, X_train, y_train)
        classification_output = st.predict(clf3, X_test)
        pass
    classification_time_end = MPI.Wtime()
    classification_time = classification_time_end - classification_time_start
    print(
        f'[TIME] Process {rank} finished classification by {algorithm} algorithm with time: {classification_time}')
    return classification_output

def train_test(X, y):
    if rank == 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False)

        data = (X_train, X_test, y_train, y_test)
    else:
        data = None
        program_start_time = MPI.Wtime()

    X_train, X_test, y_train, y_test = bcast_data(data)

    classification_output = classify(X_train, X_test, y_train, y_test)
    outputs_from_classifications = comm.gather(classification_output)
    
    if rank == 0:   # stacking
        voted_data = st.vote(outputs_from_classifications)
        acc = accuracy_score(voted_data, y_test)
        print(f'[ACCURANCY] Final accurancy for test-train is {acc}')

comm = MPI.COMM_WORLD   # initialize MPI environment
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    print(f"[INFO] Program ran in {size} processes")

print(f"[INFO] Hello from process number {rank}")

unpickled_dataframe = pd.read_pickle("bitcoin_pickle_protocol_2")
unpickled_dataframe.drop(labels = ['stddev_output_idle_time','stddev_input_idle_time'], axis = 1, inplace = True)
X = unpickled_dataframe.drop(labels = ['is_miner','address'], axis = 1)
y = unpickled_dataframe['is_miner'].values

program_start_time = MPI.Wtime()

classification_output = train_test(X, y)

program_end_time = MPI.Wtime()
program_time = program_end_time - program_start_time

if rank == 0:
    print(f'[INFO] Stacking classifier finish work with time: {program_time}')

MPI.Finalize()  # MPI environment finalization