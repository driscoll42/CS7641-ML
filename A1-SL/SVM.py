# https://scikit-learn.org/stable/modules/svm.html

import numpy as np
import util
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import time


def train_svm(filename, X_train, X_test, y_train, y_test, solver='rbf', full_param=False, debug=False, numFolds=10,
              njobs=-1, scalar=1, make_graphs=False, pSVM={}):
    np.random.seed(1)
    algo = 'SVM'

    start = time.time()
    if len(pSVM) == 0:
        if full_param:
            param_grid = [{'kernel'      : [solver],
                           # 0.0001 - Finished for Linear
                           # 'max_iter': [-1, 10000, 100000],
                           # 'shrinking'   : [True, False], # Seems to just make things faster/slower on larger iterations, I think cutting down 2x is better
                           # 'probability' : [True, False],
                           'random_state': [1]
                           }]
            if solver == 'rbf':
                param_grid[0]['C'] = [0.001] #, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000, 100000]
                param_grid[0]['gamma'] = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000, 100000]
            elif solver == 'sigmoid':
                param_grid[0]['gamma'] = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000]
                param_grid[0]['coef0'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                param_grid[0]['C'] = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000, 100000]

            elif solver == 'poly':
                param_grid[0]['gamma'] = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000, 100000]
                param_grid[0]['degree'] = [1, 2, 3, 4, 5, 6, 7, 8]
                param_grid[0]['coef0'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                param_grid[0]['C'] = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000, 100000]
            elif solver == 'linear':
                param_grid[0]['C'] = [1.0]

        else:
            param_grid = [{'kernel'      : [solver],
                           'C'           : [0.01, 0.1, 1., 10., 100],
                           'cache_size'  : [2000],
                           'random_state': [1]
                           }]
            if solver == 'poly' or solver == 'linear':
                param_grid = [{'kernel'      : [solver],
                               'C'           : [0.001, 0.01, 0.1, 1., 10.],
                               'cache_size'  : [2000],
                               'random_state': [1]
                               }]
        svm_classifier = svm.SVC(probability=True)
        grid_search = GridSearchCV(svm_classifier, param_grid, cv=numFolds,
                                   scoring='roc_auc_ovr_weighted',
                                   return_train_score=True, n_jobs=njobs, verbose=debug)
        grid_search.fit(X_train, y_train)

        cvres = grid_search.cv_results_
        best_params = grid_search.best_params_

        util.save_gridsearch_to_csv(cvres, algo, filename[:-4], scalar, solver)

        svm_classifier = svm.SVC()
        svm_classifier.set_params(**best_params)
    else:
        svm_classifier = svm.SVC()
        svm_classifier.set_params(**pSVM)

    start = time.time()
    svm_classifier.fit(X_train, y_train)
    print('SVM Fit Time: ', time.time() - start)
    start = time.time()

    y_prob = svm_classifier.predict_proba(X_train)
    train_score = roc_auc_score(y_train, y_prob, multi_class="ovr", average="weighted")
    print('SVM Train Score Time: ', time.time() - start)

    start = time.time()

    y_prob = svm_classifier.predict_proba(X_test)
    test_score = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    print('SVM Test Score Time: ', time.time() - start)
    test_class = svm.SVC()
    test_class.set_params(**pSVM)

    if make_graphs:
        util.plot_learning_curve(svm_classifier, algo, filename[:-4], X_train, y_train, ylim=(0.0, 1.05), cv=10,
                                 n_jobs=njobs, debug=debug)
        util.compute_vc(algo, 'kernel', ['rbf', 'sigmoid', 'poly', 'linear'], X_train,
                        y_train, X_test, y_test, svm_classifier, filename[:-4], test_class, pSVM, log=False,
                        njobs=njobs, debug=debug, smalllegend=True)
        util.svm_rbf_C_Gamma_viz(X_train, y_train, pSVM, njobs, filename[:-4], train_score)

        # computer Model Complexity/Validation curves
        util.compute_vc(algo, 'kernel', ['rbf', 'sigmoid', 'poly', 'linear'], X_train,
                        y_train, X_test, y_test, svm_classifier, filename[:-4], test_class, pSVM, log=False, njobs=njobs)

        util.compute_vc(algo, 'C', [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000], X_train,
                        y_train, X_test, y_test, svm_classifier, filename[:-4], test_class, pSVM, log=True, njobs=njobs, debug=debug)
        if solver == 'rbf':
            util.compute_vc(algo, 'gamma', [0.0001, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0], X_train,
                        y_train, X_test, y_test, svm_classifier, filename[:-4], test_class, pSVM, log=True, njobs=njobs, debug=debug)
        elif solver == 'sigmoid':
            util.compute_vc(algo, 'gamma', [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], X_train,
                        y_train, X_test, y_test, svm_classifier, filename[:-4], test_class, pSVM, log=True, njobs=njobs, debug=debug)
            util.compute_vc(algo, 'coef0', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], X_train,
                        y_train, X_test, y_test, svm_classifier, filename[:-4], test_class, pSVM, log=False, njobs=njobs, debug=debug)
        elif solver == 'poly':
            util.compute_vc(algo, 'gamma', [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], X_train,
                        y_train, X_test, y_test, svm_classifier, filename[:-4], test_class, pSVM, log=True, njobs=njobs, debug=debug)
            util.compute_vc(algo, 'coef0', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], X_train,
                        y_train, X_test, y_test, svm_classifier, filename[:-4], test_class, pSVM, log=False, njobs=njobs, debug=debug)
            util.compute_vc(algo, 'degree', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], X_train,
                        y_train, X_test, y_test, svm_classifier, filename[:-4], test_class, pSVM, log=False, njobs=njobs, debug=debug)

    return time.time() - start, round(train_score, 4), round(test_score, 4)
