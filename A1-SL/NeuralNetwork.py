# https://scikit-learn.org/stable/modules/neural_networks_supervised.html

from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
import time
from torch import nn

from skorch import NeuralNetClassifier


def train_NN(filename, X_train, X_test, y_train, y_test, solver='adam', full_param=False, debug=False, numFolds=10,
             njobs=-1, scalar=1):
    np.random.seed(1)

    start = time.time()
    if full_param:
        param_grid = [{'hidden_layer_sizes': [(128,), (128, 128), (128, 128, 128), (128, 128, 128, 128),
                                              (256,), (256, 256), (256, 256, 256), (256, 256, 256, 256),
                                              (8), (16), (32), (8, 8), (16, 16), (32, 32), (8, 16), (8, 32), (16, 32),
                                              (8, 8, 8),
                                              (8, 8, 16),
                                              (8, 8, 32),
                                              (8, 16, 16),
                                              (8, 16, 32),
                                              (8, 32, 32),
                                              (16, 16, 16),
                                              (16, 16, 32),
                                              (16, 32, 32),
                                              (32, 32, 32), (512), (512, 512), (512, 512, 512), (512, 512, 512, 512)],
                       'activation'        : ['identity', 'logistic', 'tanh', 'relu'],
                       'solver'            : [solver],  # 'lbfgs',
                       'alpha'             : [0.0001, 0.0, 0.001, 0.01, 0.1],
                       'batch_size'        : ['auto'],
                       'learning_rate_init': [0.001],
                       'max_iter'          : [10000],
                       'warm_start'        : [True],
                       'early_stopping'    : [True],
                       'random_state'      : [1]
                       }]
        if solver == 'sgd':
            param_grid[0]['learning_rate'] = ['constant', 'invscaling', 'adaptive']  # Only used when solver='sgd'

    else:
        param_grid = [{'hidden_layer_sizes': [(8), (16), (32), (8, 8), (16, 16), (32, 32), (8, 16), (8, 32), (16, 32),
                                              (128,), (128, 128), (128, 128, 128), (128, 128, 128, 128)],
                       # 'hidden_layer_sizes': [(512, 512),  (256, 256), (1024), (1024, 1024),], #(256, 256, 256), (256, 256, 256, 256), (512, 512, 512), (512, 512, 512, 512)],
                       'solver'            : [solver],
                       'activation'        : ['identity', 'logistic', 'tanh', 'relu'],
                       'max_iter'          : [10000],
                       'early_stopping'    : [True],
                       'random_state'      : [1]
                       }]

    nn_classifier = MLPClassifier()

    grid_search = GridSearchCV(nn_classifier, param_grid, cv=numFolds,
                               scoring='accuracy',
                               return_train_score=True, n_jobs=njobs, verbose=debug)

    scaler = StandardScaler()

    grid_search.fit(scaler.fit_transform(X_train), y_train)

    cvres = grid_search.cv_results_
    best_params = grid_search.best_params_

    file = open("ParamTests\ANN-" + solver + "-" + filename + "-" + str(scalar) + ".txt", "w")
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


def train_NN_torch(filename, X_train, X_test, y_train, y_test, solver='adam', full_param=False, debug=False,
                   numFolds=10,
                   njobs=-1):
    np.random.seed(1)

    start = time.time()
    if full_param:
        param_grid = [{'hidden_layer_sizes': [(128,), (128, 128), (128, 128, 128), (128, 128, 128, 128),
                                              (256,), (256, 256), (256, 256, 256), (256, 256, 256, 256),
                                              (8), (16), (32), (8, 8), (16, 16), (32, 32), (8, 16), (8, 32), (16, 32),
                                              (8, 8, 8),
                                              (8, 8, 16),
                                              (8, 8, 32),
                                              (8, 16, 16),
                                              (8, 16, 32),
                                              (8, 32, 32),
                                              (16, 16, 16),
                                              (16, 16, 32),
                                              (16, 32, 32),
                                              (32, 32, 32)],
                       'activation'        : ['identity', 'logistic', 'tanh', 'relu'],
                       'solver'            : [solver],  # 'lbfgs',
                       'alpha'             : [0.0001, 0.0, 0.001, 0.01, 0.1],
                       'batch_size'        : ['auto'],
                       'learning_rate_init': [0.001],
                       'max_iter'          : [10000],
                       'early_stopping'    : [True],
                       'random_state'      : [1]
                       }]
        if solver == 'sgd':
            param_grid[0]['learning_rate'] = ['constant', 'invscaling', 'adaptive']  # Only used when solver='sgd'

    else:
        param_grid = [{
            # 'hidden_layer_sizes': [(8), (16), (32), (8, 8), (16, 16), (32, 32), (8, 16), (8, 32), (16, 32), (128,), (128, 128), (128, 128, 128), (128, 128, 128, 128)],
            'hidden_layer_sizes': [(256,), (256, 256), (256, 256, 256), (256, 256, 256, 256), (512,), (512, 512),
                                   (512, 512, 512), (512, 512, 512, 512)],
            'solver'            : [solver],
            'activation'        : ['identity', 'logistic', 'tanh', 'relu'],
            'max_iter'          : [10000],
            'early_stopping'    : [True],
            'random_state'      : [1]
        }]

    net = NeuralNetClassifier(
            ClassifierModule,
            max_epochs=20,
            lr=0.1,
            verbose=0,
            optimizer__momentum=0.9,
    )
    params = {
        'lr'                 : [0.05, 0.1],
        'module__num_units'  : [10, 20],
        'module__dropout'    : [0, 0.5],
        'optimizer__nesterov': [False, True],
    }

    nn_classifier = MLPClassifier()

    grid_search = GridSearchCV(nn_classifier, param_grid, cv=numFolds,
                               scoring='accuracy',
                               return_train_score=True, n_jobs=njobs, verbose=debug)

    scaler = StandardScaler()

    grid_search.fit(scaler.fit_transform(X_train), y_train)

    cvres = grid_search.cv_results_
    best_params = grid_search.best_params_

    file = open("ParamTests\ANN-" + solver + "-" + filename + ".txt", "w")
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
