# https://scikit-learn.org/stable/modules/svm.html

from sklearn import svm
import numpy as np
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import util as util
from subprocess import call
from sklearn.model_selection import GridSearchCV
import time


def train_svm(filename, X_train, X_test, y_train, y_test, full_param=False, debug=False):
    np.random.seed(1)

    start = time.time()
    if full_param:
        param_grid = [{'kernel'      : ['linear', 'poly', 'rbf', 'sigmoid'],
                       'C'           : [0.01, 0.1, 1., 10.],
                       'degree'      : [1, 2, 3, 4, 5, 6, 7, 8],
                       'gamma'       : ['scale', 'auto'],
                       'coef0'       : [0],
                       'max_iter'    : [10, 500, 2000],
                       'shrinking'   : [True, False],
                       'probability' : [True, False],
                       'random_state': [1]
                       }]
    else:
        param_grid = [{'kernel'      : ['linear', 'rbf'],
                       'C'           : [0.01, 0.1, 1., 10.],
                       'random_state': [1]
                       }]

    knn_classifier = svm.SVC()
    grid_search = GridSearchCV(knn_classifier, param_grid, cv=10,
                               scoring='accuracy',
                               return_train_score=True, n_jobs=-1, verbose=debug)
    grid_search.fit(X_train, y_train)

    file = open("svm-" + filename + ".txt", "w")
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        file.writelines([str(mean_score), ' ', str(params), "\n"])

    file.close()

    best_params = grid_search.best_params_
    # print(best_params)
    svm_classifier = svm.SVC()
    svm_classifier.set_params(**best_params)

    svm_classifier.fit(X_train, y_train)
    train_score = svm_classifier.score(X_train, y_train)
    test_score = svm_classifier.score(X_test, y_test)

    return time.time() - start, round(train_score, 4), round(test_score, 4)

