import numpy as np
import os
import re

from sklearn import preprocessing, svm, metrics
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.model_selection import cross_val_score

import dataloader

def main():

    X_train, y_train = dataloader.load_training_data()
    X_test = dataloader.load_testing_data()

    # print(X_train)
    # print(y_train)
    # print(X_test)

if __name__ == "__main__":
    main()
