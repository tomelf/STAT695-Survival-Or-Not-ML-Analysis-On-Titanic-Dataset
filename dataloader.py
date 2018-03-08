import numpy as np
import csv
import re
import sys

def preprocess(data):
    for idx, d in enumerate(data):
        PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked = d;
        # get last name
        name_objs = re.match( r'([^,]*)', Name)
        last_name = name_objs.group(1).strip()
        Sex = (0 if Sex == "male" else 1)
        data[idx] = [PassengerId,Pclass,Name,last_name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked]
    return np.array(data)

def preprocess2(data):
    for idx, d in enumerate(data):
        pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked,boat,body,dest = d;
        # get last name
        name_objs = re.match( r'([^,]*)', name)
        last_name = name_objs.group(1).strip()
        sex = (0 if sex == "male" else 1)
        has_sibsp = 1 if int(sibsp) > 0 else 0
        has_parch = 1 if int(parch) > 0 else 0
        data[idx] = [pclass,name,last_name,sex,age,sibsp,has_sibsp,parch,has_parch,ticket,fare,cabin,embarked,boat,body,dest]
    np_data_fare = np.array(data)[:,10]
    np_data_fare[np_data_fare==""] = "0"
    np_data_fare = np_data_fare.astype('float')
    fare_mean = np.mean(np_data_fare)
    fare_std = np.std(np_data_fare)
    group_bounds = [fare_mean-2*fare_std, fare_mean-fare_std, fare_mean, fare_mean+fare_std, fare_mean+2*fare_std]
    for idx, d in enumerate(data):
        data[idx][10] = 0 if data[idx][10]=="" else float(data[idx][10])
        fare = data[idx][10]
        fare_group = -1
        for i in range(len(group_bounds)):
            if i==0:
                b1 = 0
                b2 = group_bounds[i]
            elif i==len(group_bounds)-1:
                b1 = group_bounds[i]
                b2 = sys.float_info.max
            else:
                b1 = group_bounds[i]
                b2 = group_bounds[i+1]
            if b1<=fare<b2:
                fare_group = i
                break
        data[idx].insert(11, fare_group)
    return np.array(data)

def load_all_data():
    test_data_path = "dataset/titanic3.csv"

    X_test = []
    y_test = []
    with open(test_data_path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for idx, row in enumerate(spamreader):
            if idx > 0:
                pclass,survived,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked,boat,body,dest = row
                X_test.append([pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked,boat,body,dest])
                y_test.append(survived)

    X_test = preprocess2(X_test)

    return X_test, y_test

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
