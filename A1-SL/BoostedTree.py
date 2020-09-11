# https://scikit-learn.org/stable/modules/ensemble.html
# No a boosted tree comes with 2 APIs. the API for the booster and one for the base learner (decision tree)
# https://scikit-learn.org/stable/modules/tree.html
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
import time
import util


def train_BTree(filename, X_train, X_test, y_train, y_test, full_param=False, debug=False, numFolds=10, njobs=-1,
                scalar=1, make_graphs=False, pBTree={}):
    np.random.seed(1)
    start = time.time()
    algo = 'Boosted Tree'
    if len(pBTree) == 0:
        if full_param:
            param_grid = [{'base_estimator__criterion'     : ['gini', 'entropy'],
                           'base_estimator__max_depth'     : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100],
                           # 'base_estimator__min_samples_split': [2, 3, 5, 6, 8, 10],
                           # 'base_estimator__min_samples_leaf' : [1, 2, 3, 5, 6, 8, 10],
                           'base_estimator__max_features'  : [0.9, 1.0],  # 0.1, 0.3, 0.5,
                           'base_estimator__max_leaf_nodes': [10, 100],  # 2, 4, 5, 7,
                           'base_estimator__ccp_alpha'     : [0.0, 0.005, 0.01],
                           # 0.015, 0.02, 0.025, 0.030, 0.035, 0.04],
                           "base_estimator__splitter"      : ["best"],  # "random"],
                           "n_estimators"                  : [1, 50, 100, 150, 200, 250, 300],
                           "learning_rate"                 : [0.1, 0.5, 1],
                           'random_state'                  : [1]
                           }]
        else:
            param_grid = [{'base_estimator__criterion': ['gini', 'entropy'],
                           'base_estimator__max_depth': [3, 5, 7, 10],
                           'base_estimator__ccp_alpha': [0.0, 0.005, 0.01, 0.035],
                           # 'base_estimator__min_samples_split': [3, 5, 7, 10],
                           # 'base_estimator__ccp_alpha'        : [0.0, 0.005, 0.015, 0.025, 0.35, 0.04],
                           "n_estimators"             : [1, 50, 100, 150],
                           # "learning_rate"                    : [0.1, 0.5, 1],
                           'random_state'             : [1]
                           }]

        DTC = DecisionTreeClassifier(random_state=11)
        adaTree = AdaBoostClassifier(base_estimator=DTC)

        # run grid search
        grid_search = GridSearchCV(adaTree, param_grid=param_grid, cv=numFolds,
                                   scoring='roc_auc_ovr_weighted',
                                   return_train_score=True, n_jobs=njobs, verbose=debug)

        grid_search.fit(X_train, y_train)

        cvres = grid_search.cv_results_
        best_params = grid_search.best_params_

        util.save_gridsearch_to_csv(cvres, algo, filename[:-4], scalar)

        btree_classifier = AdaBoostClassifier(base_estimator=DTC)
        btree_classifier.set_params(**best_params)
    else:
        DTC = DecisionTreeClassifier()
        btree_classifier = AdaBoostClassifier(base_estimator=DTC)
        btree_classifier.set_params(**pBTree)

    btree_classifier.fit(X_train, y_train)
    train_score = btree_classifier.score(X_train, y_train)
    test_score = btree_classifier.score(X_test, y_test)

    if make_graphs:
        # util.compute_roc(algo, 'n_neighbors', [1, 2, 3, 4, 5, 6, 7,8,9, 10, 15, 20, 25, 30, 35, 40, 50], X_train, y_train, X_test, y_test, knn_classifier,filename[:-4])
        util.compute_vc(algo, 'base_estimator__max_depth',
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80,
                         90, 100], X_train, y_train,
                        btree_classifier, filename[:-4], log=False, njobs=njobs)
        util.compute_vc(algo, 'base_estimator__max_leaf_nodes', [2, 4, 5, 7, 10, 25, 50, 75, 100, 200, 1000, 10000, 100000, 1000000], X_train,
                        y_train, btree_classifier, filename[:-4], log=False, njobs=njobs)
        # computer Model Complexity/Validation curves
        util.compute_vc(algo, 'base_estimator__criterion', ['gini', 'entropy'], X_train, y_train, btree_classifier,
                        filename[:-4], log=False, njobs=njobs)

        util.compute_vc(algo, 'n_estimators', [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 500], X_train,
                        y_train, btree_classifier, filename[:-4], log=False, njobs=njobs)
        util.compute_vc(algo, 'learning_rate', [0.1, 0.5, 1], X_train, y_train,
                        btree_classifier, filename[:-4], log=False, njobs=njobs)
        util.compute_vc(algo, 'base_estimator__ccp_alpha',
                        [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.30, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        X_train, y_train, btree_classifier, filename[:-4], log=False, njobs=njobs)
        util.compute_vc(algo, 'base_estimator__min_samples_split', [2, 3, 5, 6, 8, 10], X_train, y_train,
                        btree_classifier, filename[:-4], log=False, njobs=njobs)
        util.compute_vc(algo, 'base_estimator__min_samples_leaf', [1, 2, 3, 5, 6, 8, 10], X_train, y_train,
                        btree_classifier, filename[:-4], log=False, njobs=njobs)
        util.compute_vc(algo, 'base_estimator__max_features', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1], X_train,
                        y_train, btree_classifier, filename[:-4], log=False, njobs=njobs)

        util.compute_vc(algo, 'base_estimator__splitter', ["best", "random"], X_train, y_train, btree_classifier,
                        filename[:-4], log=False, njobs=njobs)

    return time.time() - start, round(train_score, 4), round(test_score, 4)
