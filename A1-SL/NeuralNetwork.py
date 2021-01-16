# https://scikit-learn.org/stable/modules/neural_networks_supervised.html

import numpy as np
import util
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
import time


def train_NN(filename, X_train, X_test, y_train, y_test, solver='adam', full_param=False, debug=False, numFolds=10,
             njobs=-1, scalar=1, make_graphs=False, pNN={}, nolegend=False):
    np.random.seed(1)
    algo = 'Neural Network'

    start = time.time()
    if len(pNN) == 0:
        if full_param:
            param_grid = [{'hidden_layer_sizes': [
                (8), (16), (32),
                (8, 8), (16, 16), (32, 32),
                (8, 8, 8), (16, 16, 16), (32, 32, 32),
                (128,), (128, 128), (128, 128, 128), (128, 128, 128, 128),
                (256,), (256, 256),
                (512,), (512, 512),
                (256, 256, 256), (256, 256, 256, 256), (512, 512, 512), (512, 512, 512, 512)],
                'activation'                   : ['logistic', 'tanh', 'relu'],  # 'identity',
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

    start = time.time()
    nn_classifier.fit(X_train, y_train)
    print('NN Fit Time: ', time.time() - start)
    start = time.time()

    y_prob = nn_classifier.predict_proba(X_train)
    train_score = roc_auc_score(y_train, y_prob, multi_class="ovr", average="weighted")
    print('NN Train Score Time: ', time.time() - start)

    start = time.time()

    y_prob = nn_classifier.predict_proba(X_test)
    test_score = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    print('NN Test Score Time: ', time.time() - start)

    test_class = MLPClassifier()
    test_class.set_params(**pNN)

    if make_graphs:
        # computer Model Complexity/Validation curves
        util.plot_learning_curve(nn_classifier, algo, filename[:-4], X_train, y_train, ylim=(0.0, 1.05), cv=10,
                                 n_jobs=njobs, debug=debug)
        util.compute_vc(algo, 'activation', ['identity', 'logistic', 'tanh', 'relu'],
                        X_train, y_train, X_test, y_test, nn_classifier, filename[:-4], test_class, pNN, log=False,
                        njobs=njobs, debug=debug)
        util.compute_vc(algo, 'max_iter',
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600,
                         700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], X_train,
                        y_train, X_test, y_test, nn_classifier, filename[
                                                                :-4], test_class, pNN, log=True, njobs=njobs,
                        debug=debug)
        util.compute_vc(algo, 'hidden_layer_sizes', [

            (1), (2), (4), (8), (16), (32), (64), (128), (256), (512)
        ], X_train, y_train, X_test, y_test, nn_classifier, filename[:-4], test_class, pNN, log=False, njobs=njobs,
                        debug=debug, fString=True, extraText=' 1-Layer', rotatex=True, nolegend=nolegend)


        util.compute_vc(algo, 'hidden_layer_sizes', [
            (1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512),
        ], X_train, y_train, X_test, y_test, nn_classifier, filename[:-4], test_class, pNN, log=False, njobs=njobs,
                        debug=debug, fString=True, extraText=' 2-Layer', rotatex=True, nolegend=nolegend)
        util.compute_vc(algo, 'hidden_layer_sizes', [
            (1, 1, 1), (2, 2, 2), (4, 4, 4), (8, 8, 8), (16, 16, 16), (32, 32, 32), (64, 64, 64), (128, 128, 128),
            (256, 256, 256), (512, 512, 512),
        ], X_train, y_train, X_test, y_test, nn_classifier, filename[:-4], test_class, pNN, log=False, njobs=njobs,
                        debug=debug, fString=True, extraText=' 3-Layer', rotatex=True, nolegend=nolegend)
        util.compute_vc(algo, 'hidden_layer_sizes', [
            (1, 1, 1, 1), (2, 2, 2, 2), (4, 4, 4, 4), (8, 8, 8, 8), (16, 16, 16, 16), (32, 32, 32, 32),
            (64, 64, 64, 64), (128, 128, 128, 128), (256, 256, 256, 256), (512, 512, 512, 512)
        ], X_train, y_train, X_test, y_test, nn_classifier, filename[:-4], test_class, pNN, log=False, njobs=njobs,
                        debug=debug, fString=True, extraText=' 4-Layer', rotatex=True, nolegend=nolegend)

        util.compute_vc(algo, 'solver', ['adam', 'sgd', 'lbfgs'],
                        X_train, y_train, X_test, y_test, nn_classifier, filename[:-4], test_class, log=False,
                        njobs=njobs)
        util.compute_vc(algo, 'alpha',
                        [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500,
                         1000, 5000, 10000, 100000, 1000000], X_train, y_train, X_test, y_test, nn_classifier,
                        filename[:-4], test_class, pNN, log=True, njobs=njobs, debug=debug)

        if solver == 'sgd':
            util.compute_vc(algo, 'learning_rate', ['constant', 'invscaling', 'adaptive'], X_train, y_train, X_test,
                            y_test, nn_classifier, filename[:-4], test_class, log=True, njobs=njobs)

        return time.time() - start, round(train_score, 4), round(test_score, 4)
