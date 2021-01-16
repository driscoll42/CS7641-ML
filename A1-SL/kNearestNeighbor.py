# https://scikit-learn.org/stable/modules/neighbors.html
# https://www.freecodecamp.org/news/how-to-build-and-train-k-nearest-neighbors-ml-models-in-python/

import numpy as np
import util
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import time


def train_knn(filename, X_train, X_test, y_train, y_test, full_param=False, debug=False, numFolds=10, njobs=-1,
              scalar=1, make_graphs=False, pknn={}):
    np.random.seed(1)
    algo = 'k-Nearest Neighbor'

    start = time.time()
    if len(pknn) == 0:
        if full_param:
            param_grid = [{'weights'    : ['distance'],  # 'weights'    : ['uniform', 'distance'],
                           'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                           22, 23, 24, 25],  # , 60, 70, 80, 90, 100],
                           'algorithm'  : ['ball_tree', 'kd_tree', 'brute'],
                           # 'leaf_size'  : [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50],  # , 60, 70, 80, 90, 100],
                           'p'          : [1, 2, 3]
                           }]
        else:
            param_grid = [{'weights'    : ['uniform', 'distance'],
                           'n_neighbors': [1, 3, 5, 6, 8, 10],
                           'algorithm'  : ['ball_tree', 'kd_tree', 'brute'],
                           'leaf_size'  : [1, 10, 20, 30, 40, 50],
                           }]

        knn_classifier = KNeighborsClassifier()
        grid_search = GridSearchCV(knn_classifier, param_grid, cv=numFolds,
                                   scoring='roc_auc_ovr_weighted',
                                   return_train_score=True, n_jobs=njobs, verbose=debug)
        grid_search.fit(X_train, y_train)

        cvres = grid_search.cv_results_
        best_params = grid_search.best_params_

        util.save_gridsearch_to_csv(cvres, algo, filename[:-4], scalar)

        knn_classifier = KNeighborsClassifier()
        knn_classifier.set_params(**best_params)
    else:
        # Fit algo to best parameters and compute test score
        knn_classifier = KNeighborsClassifier()
        knn_classifier.set_params(**pknn)

    start = time.time()
    knn_classifier.fit(X_train, y_train)
    print('KNN Fit Time: ', time.time() - start)
    start = time.time()

    y_prob = knn_classifier.predict_proba(X_train)
    train_score = roc_auc_score(y_train, y_prob, multi_class="ovr", average="weighted")
    print('KNN Train Score Time: ', time.time() - start)

    start = time.time()
    y_prob = knn_classifier.predict_proba(X_test)
    test_score = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    print('KNN Test Score Time: ', time.time() - start)

    test_class = KNeighborsClassifier()
    test_class.set_params(**pknn)

    if make_graphs:
        # Plot Learning Curve
        util.plot_learning_curve(knn_classifier, algo, filename[:-4], X_train, y_train, ylim=(0.0, 1.05), cv=10,
                                 n_jobs=njobs, debug=debug)

        util.compute_vc(algo, 'n_neighbors',
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 50, 100, 200, 300, 400, 500, 600, 700,
                         800, 900, 1000], X_train, y_train, X_test, y_test, knn_classifier, filename[:-4], test_class,
                        pknn, log=True, njobs=njobs, debug=debug)
        util.compute_vc(algo, 'weights', ['uniform', 'distance'], X_train, y_train, X_test, y_test, knn_classifier,
                        filename[:-4], test_class, pknn, log=False, njobs=njobs, debug=debug)

        knn_classifier.set_params(**{'weights': 'uniform'})
        test_class.set_params(**{'weights': 'uniform'})
        pknn['weights'] = 'uniform'
        util.compute_vc(algo, 'n_neighbors',
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 50, 100, 200, 300, 400, 500, 600, 700,
                         800, 900, 1000], X_train, y_train, X_test, y_test, knn_classifier, filename[:-4], test_class,
                        pknn, log=True, njobs=njobs, debug=debug, extraText='uniformweight')

        util.compute_vc(algo, 'p', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], X_train, y_train, X_test, y_test, knn_classifier,
                        filename[:-4], test_class, pknn, log=False, njobs=njobs)
        util.compute_vc(algo, 'algorithm', ['ball_tree', 'kd_tree', 'brute'], X_train, y_train, X_test, y_test,
                        knn_classifier,
                        filename[:-4], test_class, pknn, log=False, njobs=njobs)


    return time.time() - start, round(train_score, 4), round(test_score, 4)
