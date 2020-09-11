# https://scikit-learn.org/stable/modules/neural_networks_supervised.html

from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import time
import util


def train_NN(filename, X_train, X_test, y_train, y_test, solver='adam', full_param=False, debug=False, numFolds=10,
             njobs=-1, scalar=1, make_graphs=False, pNN={}):
    np.random.seed(1)
    algo = 'Neural Network'

    start = time.time()
    if len(pNN) == 0:
        if full_param:
            param_grid = [{'hidden_layer_sizes': [
                # (8), (16), (32),
                # (8, 8), (16, 16), (32, 32),
                # (8, 8, 8), (16, 16, 16), (32, 32, 32),
                #(128,), (128, 128), (128, 128, 128), (128, 128, 128, 128)],
                #(256,), (256, 256),
                #(512,), (512, 512)],
                (256, 256, 256), (256, 256, 256, 256),(512, 512, 512), (512, 512, 512, 512)],
                'activation'                   : ['logistic', 'tanh', 'relu'],  #'identity',
                'solver'                       : [solver],  # 'lbfgs',
                'alpha'                        : [0.0, 0.0001, 0.001, 0.01, 0.1],
                'batch_size'                   : ['auto'],
                'learning_rate_init'           : [0.001, 0.01],
                'max_iter'                     : [10000],
                'warm_start'                   : [True],
                'early_stopping'               : [True],
                'random_state'                 : [1]
            }]
            if solver == 'sgd':
                param_grid[0]['learning_rate'] = ['constant', 'invscaling', 'adaptive']  # Only used when solver='sgd'

        else:
            param_grid = [
                {'hidden_layer_sizes': [(8), (16), (32), (8, 8), (16, 16), (32, 32), (8, 16), (8, 32), (16, 32),
                                        (128,), (128, 128), (128, 128, 128), (128, 128, 128, 128)],
                 # 'hidden_layer_sizes': [(512, 512),  (256, 256), (1024), (1024, 1024),], #(256, 256, 256), (256, 256, 256, 256), (512, 512, 512), (512, 512, 512, 512)],
                 'solver'            : [solver],
                 'activation'        : ['identity', 'relu'],  # , 'logistic', 'tanh'],
                 'max_iter'          : [10000],
                 'early_stopping'    : [True],
                 'random_state'      : [1]
                 }]

        nn_classifier = MLPClassifier()

        grid_search = GridSearchCV(nn_classifier, param_grid, cv=numFolds,
                                   scoring='roc_auc_ovr_weighted',
                                   return_train_score=True, n_jobs=njobs, verbose=debug)

        grid_search.fit(X_train, y_train)

        cvres = grid_search.cv_results_
        best_params = grid_search.best_params_

        util.save_gridsearch_to_csv(cvres, algo, filename[:-4], scalar, solver)

        nn_classifier = MLPClassifier()
        nn_classifier.set_params(**best_params)
    else:
        nn_classifier = MLPClassifier()
        nn_classifier.set_params(**pNN)

    nn_classifier.fit(X_train, y_train)
    train_score = nn_classifier.score(X_train, y_train)
    test_score = nn_classifier.score(X_test, y_test)

    #ytp = nn_classifier.predict(X_train)
    #print(roc_auc_score(y_train, ytp, multi_class='ovr', average='weighted'))

    #y_pred = nn_classifier.predict(X_test)
    #print(roc_auc_score(y_test, y_pred, multi_class='ovr', average='weighted'))

    if make_graphs:
        # util.compute_roc(algo, 'n_neighbors', [1, 2, 3, 4, 5, 6, 7,8,9, 10, 15, 20, 25, 30, 35, 40, 50], X_train, y_train, X_test, y_test, knn_classifier,filename[:-4])

        # computer Model Complexity/Validation curves
        # util.compute_vc(algo, 'hidden_layer_sizes', [
        #    (8), (16), (32),
        #    (8, 8), (16, 16), (32, 32),
        #    (8, 8, 8), (16, 16, 16), (32, 32, 32),
        #    (128,), (128, 128), (128, 128, 128), (128, 128, 128, 128),
        #    (256,), (256, 256), (256, 256, 256), (256, 256, 256, 256),
        #    (512,), (512, 512), (512, 512, 512), (512, 512, 512, 512)], X_train, y_train, nn_classifier, filename[:-4], log=False, njobs=njobs)
        util.compute_vc(algo, 'activation', ['identity', 'logistic', 'tanh', 'relu'], X_train, y_train, nn_classifier,
                        filename[:-4], log=False, njobs=njobs)
        util.compute_vc(algo, 'solver', ['adam', 'sgd', 'lbfgs'], X_train, y_train, nn_classifier, filename[:-4],
                        log=False, njobs=njobs)
        util.compute_vc(algo, 'alpha',
                        [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000,
                         1000000],
                        X_train, y_train, nn_classifier, filename[:-4], njobs, log=True, njobs=njobs)

        if solver == 'sgd':
            util.compute_vc(algo, 'learning_rate', ['constant', 'invscaling', 'adaptive'], X_train, y_train,
                            nn_classifier, filename[:-4], log=False, njobs=njobs)
    return time.time() - start, round(train_score, 4), round(test_score, 4)
