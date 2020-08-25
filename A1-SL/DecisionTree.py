# https://scikit-learn.org/stable/modules/tree.html

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


def create_DT_image(Dtree):
    # TODO: Make it dynamically named
    export_graphviz(Dtree,
                    out_file="tree.dot",
                    # feature_names=tree.feature_names[2:],class_names=tree.target_names,
                    rounded=True, filled=True)

    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])


def train_DTree(filename, X_train, X_test, y_train, y_test, full_param=False, debug=False):
    np.random.seed(1)
    start = time.time()

    if full_param:
        param_grid = [{'criterion'        : ['gini', 'entropy'],
                       'max_depth'        : [2, 3, 4, 5, 6, 7, 8, 9, 10],
                       'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                       'min_samples_leaf' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       'max_features'     : ['auto', 'log2', None],
                       'max_leaf_nodes'   : [2, 3, 4, 5, 6, 7, 8, 9, 10, None],
                       'ccp_alpha'        : [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.30, 0.35, 0.04],
                       "splitter"         : ["best", "random"],
                       'random_state'     : [1]
                       }]
    else:
        param_grid = [{'criterion'        : ['gini', 'entropy'],
                       'max_depth'        : [3, 5, 7, 10],
                       'min_samples_split': [3, 5, 7, 10],
                       'ccp_alpha'        : [0.0, 0.005, 0.015, 0.025, 0.35, 0.04],
                       'random_state'     : [1]
                       }]

    tree_classifier = DecisionTreeClassifier()
    grid_search = GridSearchCV(tree_classifier, param_grid, cv=10,
                               scoring='accuracy',
                               return_train_score=True, n_jobs=-1, verbose=debug)
    grid_search.fit(X_train, y_train)


    file = open("DTree-" + filename + ".txt", "w")
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        file.writelines([str(mean_score), ' ', str(params), "\n"])

    file.close()

    best_params = grid_search.best_params_
    tree_classifier = DecisionTreeClassifier()
    tree_classifier.set_params(**best_params)

    tree_classifier.fit(X_train, y_train)
    train_score = tree_classifier.score(X_train, y_train)
    test_score = tree_classifier.score(X_test, y_test)

    return time.time() - start, round(train_score, 4), round(test_score, 4)

