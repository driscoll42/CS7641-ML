# https://scikit-learn.org/stable/modules/neighbors.html
# https://www.freecodecamp.org/news/how-to-build-and-train-k-nearest-neighbors-ml-models-in-python/
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
import time
import util


def train_knn(filename, X_train, X_test, y_train, y_test, full_param=False, debug=False, numFolds=10, njobs=-1,
              scalar=1, make_graphs=False, pknn={}):
    np.random.seed(1)
    algo = 'k-Nearest Neighbor'

    start = time.time()
    if len(pknn) == 0:
        if full_param:
            param_grid = [{'weights'    : ['distance'],  # 'weights'    : ['uniform', 'distance'],
                           'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],  # , 60, 70, 80, 90, 100],
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

    knn_classifier.fit(X_train, y_train)
    train_score = knn_classifier.score(X_train, y_train)
    test_score = knn_classifier.score(X_test, y_test)

    if make_graphs:
        # util.compute_roc(algo, 'n_neighbors', [1, 2, 3, 4, 5, 6, 7,8,9, 10, 15, 20, 25, 30, 35, 40, 50], X_train, y_train, X_test, y_test, knn_classifier,filename[:-4])

        # computer Model Complexity/Validation curves
        util.compute_vc(algo, 'weights', ['uniform', 'distance'], X_train, y_train, knn_classifier, filename[:-4],
                        log=False, njobs=njobs)
        util.compute_vc(algo, 'n_neighbors', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 50], X_train,
                        y_train,
                        knn_classifier, filename[:-4], log=False, njobs=njobs)
        util.compute_vc(algo, 'algorithm', ['ball_tree', 'kd_tree', 'brute'], X_train, y_train, knn_classifier,
                        filename[:-4], log=False, njobs=njobs)
        util.compute_vc(algo, 'leaf_size', [1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50], X_train, y_train,
                        knn_classifier, filename[:-4], log=False, njobs=njobs)
        util.compute_vc(algo, 'p', [1, 2, 3, 4, 5], X_train, y_train, knn_classifier, filename[:-4], log=False,
                        njobs=njobs)

    return time.time() - start, round(train_score, 4), round(test_score, 4)
