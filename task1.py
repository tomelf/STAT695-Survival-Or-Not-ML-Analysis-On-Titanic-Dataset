import numpy as np
import os
import re

from sklearn import preprocessing, svm, metrics, tree
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.model_selection import cross_val_score
import graphviz

import dataloader

def main():
    X, y = dataloader.load_all_data()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=float(418)/(418+891))
    print("Number of training examples: {}, testing examples: {}".format(len(X_train), len(X_test)))
    # Origianl features: pclass,name,last_name,sex,age,sibsp,has_sibsp,parch,has_parch,ticket,fare,fare_group,cabin,embarked,boat,body,dest

    testing_feature_sets = [
        # "pclass",
        # "sex",
        # "sibsp",
        # "has_sibsp",
        # "parch",
        # "has_parch",
        # "fare",
        # "fare_group",
        # "last_name",
        # "pclass,last_name,sex,sibsp,parch,fare,has_sibsp,has_parch,fare_group",
        "pclass,sex,sibsp,parch,fare"
    ]

    for testing_feature_set in testing_feature_sets:
        print("Used feature set: {}".format(testing_feature_set))
        f_idxes = dataloader.get_feature_index_list(testing_feature_set)
        X_train_filtered = X_train[:,f_idxes]
        X_test_filtered = X_test[:,f_idxes]

        # clf = svm.SVC(kernel='rbf', C=1, gamma='auto', probability=False)
        
        clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=7)

        clf.fit(X_train_filtered, y_train)
        print("Training accuracy: {}, Testing accuracy: {}".format(
                clf.score(X_train_filtered, y_train),
                clf.score(X_test_filtered, y_test)
            )
        )
        y_train = np.array(y_train, dtype='int')
        y_test = np.array(y_test, dtype='int')
        y_train_predict = np.array(clf.predict(X_train_filtered), dtype='int')
        y_test_predict = np.array(clf.predict(X_test_filtered), dtype='int')
        print("Training area under ROC: {}, Testing area under ROC: {}".format(
                metrics.roc_auc_score(y_train, y_train_predict),
                metrics.roc_auc_score(y_test, y_test_predict)
            )
        )
        
        dot_data = tree.export_graphviz(clf, 
                                        out_file=None, 
                                        feature_names=(testing_feature_set.split(',')),
                                        class_names=["survived", "dead"],  
                                        filled=True, 
                                        rounded=True,
                                        impurity=True,
                                        special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render('titanic.gv', view=True)
        

if __name__ == "__main__":
    main()
