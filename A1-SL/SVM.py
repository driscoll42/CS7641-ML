# https://scikit-learn.org/stable/modules/svm.html

from sklearn import svm
import numpy as np
from sklearn.model_selection import GridSearchCV
import time


def train_svm(filename, X_train, X_test, y_train, y_test, solver='rbf', full_param=False, debug=False, numFolds=10, njobs=-1, scalar=1):
    np.random.seed(1)

    start = time.time()
    if full_param:
        param_grid = [{'kernel'      : [solver],
                       'C'           : [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], #, 10000, 100000],
                       # 'max_iter': [-1, 10000, 100000],
                       # 'shrinking'   : [True, False], # Seems to just make things faster/slower on larger iterations, I think cutting down 2x is better
                       # 'probability' : [True, False],
                       'cache_size'  : [1500],
                       'random_state': [1]
                       }]
        if solver == 'rbf':
            param_grid[0]['gamma'] = ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], #, 10000, 100000]
        elif solver == 'sigmoid':
            param_grid[0]['gamma'] = ['auto', 0.0001, 0.001] #, 'scale', 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000]
            param_grid[0]['coef0'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        elif solver == 'poly':
            param_grid[0]['gamma'] = ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], #, 10000, 100000]
            param_grid[0]['degree'] = [1, 2, 3, 4, 5, 6, 7, 8]
            param_grid[0]['coef0'] = [0, 1, 2, 3, 4]  # , 5 #, 6, 7, 8, 9, 10]
    else:
        param_grid = [{'kernel'      : [solver],
                       'C'           : [0.01, 0.1, 1., 10., 100, 1000],
                       'cache_size'  : [1500],
                       'random_state': [1]
                       }]

    knn_classifier = svm.SVC()
    grid_search = GridSearchCV(knn_classifier, param_grid, cv=numFolds,
                               scoring='accuracy',
                               return_train_score=True, n_jobs=njobs, verbose=debug)
    grid_search.fit(X_train, y_train)

    cvres = grid_search.cv_results_
    best_params = grid_search.best_params_

    file = open("ParamTests\SVM-" + solver + "-" + filename  + "-" + str(scalar) +  ".txt", "w")
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        file.writelines([str(mean_score), ' ', str(params), "\n"])
    file.writelines(best_params)
    file.close()

    svm_classifier = svm.SVC()
    svm_classifier.set_params(**best_params)

    svm_classifier.fit(X_train, y_train)
    train_score = svm_classifier.score(X_train, y_train)
    test_score = svm_classifier.score(X_test, y_test)

    return time.time() - start, round(train_score, 4), round(test_score, 4)
