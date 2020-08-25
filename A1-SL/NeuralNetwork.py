# https://scikit-learn.org/stable/modules/neural_networks_supervised.html

from sklearn.neural_network import MLPClassifier
import numpy as np
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


def train_NN(filename, X_train, X_test, y_train, y_test, full_param=False, debug=False):
    np.random.seed(1)

    start = time.time()
    if full_param:
        param_grid = [{'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                       'activation'        : ['identity', 'logistic', 'tanh', 'relu'],
                       'solver'            : ['lbfgs', 'sgd', 'adam'],
                       'alpha'             : [0.0001, 0.0],
                       'batch_size'        : ['auto'],
                       'learning_rate'     : ['constant', 'invscaling', 'adaptive'],
                       'learning_rate_init': [0.001],
                       'max_iter'          : [1000],
                       'random_state'      : [1]
                       }]
    else:
        param_grid = [{'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                       'activation'        : ['identity', 'logistic', 'tanh', 'relu'],
                       'max_iter'          : [10000],
                       'random_state'      : [1]
                       }]

    nn_classifier = MLPClassifier()
    grid_search = GridSearchCV(nn_classifier, param_grid, cv=10,
                               scoring='accuracy',
                               return_train_score=True, n_jobs=-1, verbose=debug)
    grid_search.fit(X_train, y_train)

    file = open("NN-" + filename + ".txt", "w")
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        file.writelines([str(mean_score), ' ', str(params), "\n"])

    file.close()

    best_params = grid_search.best_params_
    # print(best_params)
    nn_classifier = MLPClassifier()
    nn_classifier.set_params(**best_params)

    nn_classifier.fit(X_train, y_train)
    train_score = nn_classifier.score(X_train, y_train)
    test_score = nn_classifier.score(X_test, y_test)

    return time.time() - start, round(train_score, 4), round(test_score, 4)

