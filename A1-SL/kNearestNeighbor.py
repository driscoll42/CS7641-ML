# https://scikit-learn.org/stable/modules/neighbors.html
# https://www.freecodecamp.org/news/how-to-build-and-train-k-nearest-neighbors-ml-models-in-python/
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
import time


def train_knn(filename, X_train, X_test, y_train, y_test, full_param=False, debug=False, numFolds=10, njobs=-1, scalar=1):
    np.random.seed(1)

    start = time.time()
    if full_param:
        param_grid = [{'weights'    : ['uniform', 'distance'],
                       'n_neighbors': [1, 3, 5, 6, 8, 10, 15, 20, 25, 30, 35, 40, 50],
                       'algorithm'  : ['ball_tree', 'kd_tree', 'brute'],
                       'leaf_size'  : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50],
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
                               scoring='accuracy',
                               return_train_score=True, n_jobs=njobs, verbose=debug)
    grid_search.fit(X_train, y_train)

    cvres = grid_search.cv_results_
    best_params = grid_search.best_params_

    file = open("ParamTests\knn-" + filename  + "-" + str(scalar) + ".txt", "w")
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        file.writelines([str(mean_score), ' ', str(params), "\n"])
    file.writelines(best_params)
    file.close()

    knn_classifier = KNeighborsClassifier()
    knn_classifier.set_params(**best_params)

    knn_classifier.fit(X_train, y_train)
    train_score = knn_classifier.score(X_train, y_train)
    test_score = knn_classifier.score(X_test, y_test)

    return time.time() - start, round(train_score, 4), round(test_score, 4)
