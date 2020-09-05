# https://scikit-learn.org/stable/modules/neural_networks_supervised.html

from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
import time


def train_NN(filename, X_train, X_test, y_train, y_test, full_param=False, debug=False):
    np.random.seed(1)

    start = time.time()
    if full_param:
        param_grid = [{'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                       'activation'        : ['identity', 'logistic', 'tanh', 'relu'],
                       'solver'            : [solver],  # 'lbfgs',
                       'alpha'             : [0.0001, 0.0, 0.001, 0.01, 0.1],
                       'batch_size'        : ['auto'],
                       'learning_rate_init': [0.001],
                       'max_iter'          : [1000],
                       'random_state'      : [1]
                       }]
        if solver == 'sgd':
            param_grid[0]['learning_rate'] = ['constant', 'invscaling', 'adaptive']  # Only used when solver='sgd'

    else:
        param_grid = [{'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                       'activation'        : ['identity', 'logistic', 'tanh', 'relu'],
                       'max_iter'          : [10000],
                       'early_stopping'    : [True],
                       'random_state'      : [1]
                       }]

    nn_classifier = MLPClassifier()
    grid_search = GridSearchCV(nn_classifier, param_grid, cv=10,
                               scoring='accuracy',
                               return_train_score=True, n_jobs=-1, verbose=debug)
    grid_search.fit(X_train, y_train)

    grid_search.fit(scaler.fit_transform(X_train), y_train)

    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        file.writelines([str(mean_score), ' ', str(params), "\n"])
    file.writelines(best_params)
    file.close()

    nn_classifier = MLPClassifier()
    nn_classifier.set_params(**best_params)

    nn_classifier.fit(X_train, y_train)
    train_score = nn_classifier.score(X_train, y_train)
    test_score = nn_classifier.score(X_test, y_test)

    return time.time() - start, round(train_score, 4), round(test_score, 4)

