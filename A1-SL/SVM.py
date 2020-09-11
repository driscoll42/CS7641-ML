# https://scikit-learn.org/stable/modules/svm.html

from sklearn import svm
import numpy as np
from sklearn.model_selection import GridSearchCV
import time
import util


def train_svm(filename, X_train, X_test, y_train, y_test, solver='rbf', full_param=False, debug=False, numFolds=10,
              njobs=-1, scalar=1, make_graphs=False, pSVM={}):
    np.random.seed(1)
    algo = 'SVM'

    start = time.time()
    if len(pSVM) == 0:
        if full_param:
            param_grid = [{'kernel'      : [solver],
                           'C'           : [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000, 100000],
                           # 'max_iter': [-1, 10000, 100000],
                           # 'shrinking'   : [True, False], # Seems to just make things faster/slower on larger iterations, I think cutting down 2x is better
                           # 'probability' : [True, False],
                           'cache_size'  : [2000],
                           'random_state': [1]
                           }]
            if solver == 'rbf':
                pass
                #param_grid[0]['gamma'] = ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0,1000.0, 10000, 100000]
            elif solver == 'sigmoid':
                #param_grid[0]['gamma'] = ['auto', 0.0001, 0.001]  # , 'scale', 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000]
                param_grid[0]['coef0'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                param_grid[0]['C'] = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

            elif solver == 'poly':
                #param_grid[0]['gamma'] = ['scale', 'auto', 0.0001, 0.001] #0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]  # , 10000, 100000]
                param_grid[0]['degree'] = [1, 2, 3, 4, 5, 6, 7, 8]
                param_grid[0]['coef0'] = [0, 1, 2, 3, 4, 5 , 6, 7, 8, 9, 10]
                param_grid[0]['C'] = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000, 10000]
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

    svm_classifier.fit(X_train, y_train)
    train_score = svm_classifier.score(X_train, y_train)
    test_score = svm_classifier.score(X_test, y_test)

    if make_graphs:
        # util.compute_roc(algo, 'n_neighbors', [1, 2, 3, 4, 5, 6, 7,8,9, 10, 15, 20, 25, 30, 35, 40, 50], X_train, y_train, X_test, y_test, svm_classifier,filename[:-4])

        # computer Model Complexity/Validation curves
        util.compute_vc(algo, 'kernel', ['rbf', 'sigmoid', 'poly', 'linear'], X_train, y_train, svm_classifier,
                        filename[:-4], njobs)
        util.compute_vc(algo, 'C', [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0], X_train, #, 1000.0, 10000, 100000
                        y_train, svm_classifier, filename[:-4], log=True, njobs=njobs)
        if solver == 'rbf':
            util.compute_vc(algo, 'gamma', [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], X_train, y_train,
                            svm_classifier, filename[:-4], log=True, njobs=njobs)
        elif solver == 'sigmoid':
            util.compute_vc(algo, 'gamma', [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], X_train, y_train,
                            svm_classifier, filename[:-4], log=True, njobs=njobs)
            util.compute_vc(algo, 'coef0', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], X_train, y_train, svm_classifier,
                            filename[:-4], log=False, njobs=njobs)
        elif solver == 'poly':
            util.compute_vc(algo, 'gamma', [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], X_train, y_train,
                            svm_classifier, filename[:-4], log=True, njobs=njobs)
            util.compute_vc(algo, 'coef0', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], X_train, y_train, svm_classifier,
                            filename[:-4], log=False, njobs=njobs)
            util.compute_vc(algo, 'degree', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], X_train, y_train, svm_classifier,
                            filename[:-4], log=False, njobs=njobs)

    return time.time() - start, round(train_score, 4), round(test_score, 4)
