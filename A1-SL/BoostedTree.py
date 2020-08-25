# https://scikit-learn.org/stable/modules/ensemble.html
# No a boosted tree comes with 2 APIs. the API for the booster and one for the base learner (decision tree)
# https://scikit-learn.org/stable/modules/tree.html
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import util as util
from subprocess import call
from sklearn.model_selection import GridSearchCV
import time


def train_BTree(filename, X_train, X_test, y_train, y_test, full_param=False, debug=False):
    np.random.seed(1)
    start = time.time()

    if full_param:
        param_grid = [{'base_estimator__criterion'        : ['gini', 'entropy'],
                       'base_estimator__max_depth'        : [2, 3, 4, 5, 6, 7, 8, 9, 10],
                       'base_estimator__min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                       'base_estimator__min_samples_leaf' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       'base_estimator__max_features'     : ['auto', 'log2', None],
                       'base_estimator__max_leaf_nodes'   : [2, 3, 4, 5, 6, 7, 8, 9, 10, None],
                       'base_estimator__ccp_alpha'        : [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.30, 0.35, 0.04],
                       "base_estimator__splitter"         : ["best", "random"],
                       "n_estimators"                     : [1, 50, 100, 150],
                       "learning_rate"                    : [0.1, 0.5, 1],
                       'random_state': [1]
                       }]
    else:
        param_grid = [{'base_estimator__criterion'        : ['gini', 'entropy'],
                       'base_estimator__max_depth'        : [3, 5, 7, 10],
                       'base_estimator__min_samples_split': [3, 5, 7, 10],
                       #'base_estimator__ccp_alpha'        : [0.0, 0.005, 0.015, 0.025, 0.35, 0.04],
                       "n_estimators"                     : [1, 50, 100, 150],
                       #"learning_rate"                    : [0.1, 0.5, 1],
                       'random_state': [1]
                       }]

    DTC = DecisionTreeClassifier(random_state=11, max_depth=5)
    adaTree = AdaBoostClassifier(base_estimator=DTC)


    # run grid search
    grid_search = GridSearchCV(adaTree, param_grid=param_grid, cv=10,
                               scoring='accuracy',
                               return_train_score=True, n_jobs=-1, verbose=debug)

    grid_search.fit(X_train, y_train)

    # scores = cross_val_score(tree, X_train, y_train, scoring="accuracy", cv=10)

    # print("Scores:", scores)
    # print("Mean:", scores.mean())
    file = open("adaTree-" + filename + ".txt", "w")
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        file.writelines([str(mean_score), ' ', str(params), "\n"])

    file.close()

    best_params = grid_search.best_params_
    tree_classifier = AdaBoostClassifier(base_estimator=DTC)
    tree_classifier.set_params(**best_params)

    tree_classifier.fit(X_train, y_train)
    train_score = tree_classifier.score(X_train, y_train)
    test_score = tree_classifier.score(X_test, y_test)

    return time.time() - start, round(train_score, 4), round(test_score, 4)

