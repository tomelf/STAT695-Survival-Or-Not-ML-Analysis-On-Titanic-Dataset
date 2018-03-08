import numpy as np
import os
import re

from sklearn import preprocessing, svm, metrics
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.model_selection import cross_val_score

import dataloader

def main():

    X, y = dataloader.load_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1)
    print("Number of training examples: {}, testing examples: {}".format(len(X_train), len(X_test)))
    # Origianl features: PassengerId,Pclass,Name,last_name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    X_train = X_train[:,[1,6,7]]

    clf = svm.SVC(kernel='rbf', C=1, gamma='auto', probability=False)
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print("5-fold cross validation: {}, mean: {}".format(scores, np.mean(scores)))

if __name__ == "__main__":
    main()
