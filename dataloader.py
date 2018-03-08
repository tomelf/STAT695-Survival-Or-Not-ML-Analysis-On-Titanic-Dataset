import numpy as np
import csv

def preprocess(data):
    return data

def load_testing_data():
    test_data_path = "dataset/test.csv"

    X_test = []
    with open(test_data_path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for idx, row in enumerate(spamreader):
            if idx > 0:
                PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked = row
                X_test.append([PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked])

    X_test = preprocess(X_test)

    return X_test

def load_training_data():
    train_data_path = "dataset/train.csv"

    X_train = []
    y_train = []
    with open(train_data_path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for idx, row in enumerate(spamreader):
            if idx > 0:
                PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked = row
                X_train.append([PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked])
                y_train.append(Survived)

    X_train = preprocess(X_train)

    return X_train, y_train
