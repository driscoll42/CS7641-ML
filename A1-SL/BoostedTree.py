# https://scikit-learn.org/stable/modules/ensemble.html
# https://scikit-learn.org/stable/modules/tree.html
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

import time

import numpy as np
import util
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score


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
                           # 'base_estimator__max_features'  : [0.9, 1.0],  # 0.1, 0.3, 0.5,
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

    start = time.time()
    btree_classifier.fit(X_train, y_train)
    print('BTree Fit Time: ', time.time() - start)
    start = time.time()

    y_prob = btree_classifier.predict_proba(X_train)
    train_score = roc_auc_score(y_train, y_prob, multi_class="ovr", average="weighted")
    print('BTree Train Score Time: ', time.time() - start)
    start = time.time()

    y_prob = btree_classifier.predict_proba(X_test)
    test_score = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    print('BTree Test Score Time: ', time.time() - start)
    DTC = DecisionTreeClassifier()
    test_class = AdaBoostClassifier(base_estimator=DTC)
    test_class.set_params(**pBTree)

    if make_graphs:
        util.boost_lr_vs_nest(X_train, y_train, pBTree, njobs, filename[:-4], train_score)
        util.compute_vc(algo, 'n_estimators',
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 1000],
                        X_train, y_train, X_test, y_test, btree_classifier, filename[:-4], test_class, pBTree,
                        log=True, njobs=njobs, debug=debug, extraText='log')

        util.plot_learning_curve(btree_classifier, algo, filename[:-4], X_train, y_train, ylim=(0.0, 1.05), cv=10,
                                 n_jobs=njobs, debug=debug)

        util.compute_vc(algo, 'base_estimator__max_depth',
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80,
                         90, 100], X_train, y_train, X_test, y_test, btree_classifier, filename[:-4], test_class,
                        pBTree, log=True, njobs=njobs, debug=debug)

        util.compute_vc(algo, 'base_estimator__max_leaf_nodes',
                        [2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 75, 100, 200, 500, 1000, 10000], X_train, y_train, X_test,
                        y_test, btree_classifier, filename[:-4], test_class, pBTree, log=True, njobs=njobs)
        # computer Model Complexity/Validation curves
        util.compute_vc(algo, 'base_estimator__criterion', ['gini', 'entropy'], X_train, y_train, X_test, y_test,
                        btree_classifier, filename[:-4], test_class, pBTree, log=False, njobs=njobs)

        util.compute_vc(algo, 'n_estimators',
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 1000],
                        X_train, y_train, X_test, y_test, btree_classifier, filename[:-4], test_class, pBTree,
                        log=False, njobs=njobs, debug=debug)
        util.compute_vc(algo, 'n_estimators',
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 1000],
                        X_train, y_train, X_test, y_test, btree_classifier, filename[:-4], test_class, pBTree,
                        log=True, njobs=njobs, debug=debug, extraText='log')
        util.compute_vc(algo, 'learning_rate',
                        [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01,
                         0.05, 0.1, 0.5, 1], X_train, y_train, X_test, y_test, btree_classifier, filename[:-4],
                        test_class, pBTree, log=True, njobs=njobs, debug=debug)

        util.compute_vc(algo, 'base_estimator__ccp_alpha',
                        [0.000001, 0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009,
                         0.0001, 0.00011, 0.00012, 0.00013, 0.00014, 0.00015, 0.00016, 0.00017, 0.00018, 0.00019,
                         0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.01, 0.1, 1],
                        X_train,
                        y_train, X_test, y_test, btree_classifier, filename[:-4], test_class, pBTree, log=True,
                        njobs=njobs)
        util.compute_vc(algo, 'base_estimator__min_samples_split', [2, 3, 5, 6, 8, 10], X_train, y_train, X_test,
                        y_test, btree_classifier, filename[:-4], test_class, pBTree, log=False, njobs=njobs)
        util.compute_vc(algo, 'base_estimator__min_samples_leaf',
                        [1, 2, 3, 5, 6, 8, 10, 25, 50, 75, 100, 250, 500, 750, 1000], X_train,
                        y_train, X_test, y_test, btree_classifier, filename[:-4], test_class, pBTree, log=True,
                        njobs=njobs)
        util.compute_vc(algo, 'base_estimator__max_features',
                        [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 0.99999, 1.0], X_train, y_train, X_test, y_test,
                        btree_classifier, filename[:-4], test_class, pBTree, log=False, njobs=njobs)

        util.compute_vc(algo, 'base_estimator__splitter', ["best", "random"], X_train, y_train, X_test, y_test,
                        btree_classifier, filename[:-4], test_class, pBTree, log=False, njobs=njobs)

    return time.time() - start, round(train_score, 4), round(test_score, 4)
