import numpy as np
import os
import re

from sklearn import preprocessing, svm, metrics
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.model_selection import cross_val_score

import dataloader

def get_feature_index_list(str_f_list):
    str_feature_list = "pclass,name,last_name,sex,age,sibsp,has_sibsp,parch,has_parch,ticket,fare,fare_group,cabin,embarked,boat,body,dest"
    feature_list = str_feature_list.split(",")
    f_list = str_f_list.split(",")
    return [feature_list.index(f) for f in f_list]

def main():
    X, y = dataloader.load_all_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=(418/(418+891)))
    print("Number of training examples: {}, testing examples: {}".format(len(X_train), len(X_test)))
    # Origianl features: pclass,name,last_name,sex,age,sibsp,has_sibsp,parch,has_parch,ticket,fare,fare_group,cabin,embarked,boat,body,dest

    testing_feature_sets = [
        "pclass",
        "sex",
        "sibsp",
        "parch",
        "fare",
        "pclass,sex,sibsp,parch,fare",
        "pclass,sex,sibsp,parch,fare,has_sibsp,has_parch,fare_group"
    ]

    for testing_feature_set in testing_feature_sets:
        print("Used feature set: {}".format(testing_feature_set))
        f_idxes = get_feature_index_list(testing_feature_set)
        X_train_filtered = X_train[:,f_idxes]
        X_test_filtered = X_test[:,f_idxes]

        # print("Example of original: {}".format(X_train[0]))
        # print("Example of filtered: {}".format(X_train_filtered[0]))

        clf = svm.SVC(kernel='rbf', C=1, gamma='auto', probability=False)

        # scores = cross_val_score(clf, X_train_filtered, y_train, cv=5)
        # print("5-fold cross validation: mean accuracy: {}, accuracy of each folders: {}".format(np.mean(scores), scores))

        clf.fit(X_train_filtered, y_train)
        print("Training accuracy: {}, Testing accuracy: {}\n".format(
                clf.score(X_train_filtered, y_train),
                clf.score(X_test_filtered, y_test)
            )
        )

if __name__ == "__main__":
    main()
