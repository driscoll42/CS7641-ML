# https://scikit-learn.org/stable/modules/tree.html

from sklearn.tree import DecisionTreeClassifier, export_graphviz
import numpy as np
from subprocess import call
from sklearn.model_selection import GridSearchCV, ShuffleSplit
import time
import util


def train_DTree(filename, X_train, X_test, y_train, y_test, full_param=False, debug=False, numFolds=10, njobs=-1,
                scalar=1, make_graphs=False, pDTree={}):
    np.random.seed(1)
    algo = 'Decision Tree'
    start = time.time()
    if len(pDTree) == 0:
        if full_param:
            param_grid = [{'criterion'        : ['gini', 'entropy'],
                           'max_depth'        : [3, 5, 7, 10, 100, 1000],  # 3, 5, 7, 10, 100,\
                           'min_samples_split': [2, 3, 5, 7, 8, 10],
                           # 'min_samples_leaf' : [0.1, 0.2, 0.3, 0.5],
                           'ccp_alpha'        : [0.0, 0.00001, 0.0001, 0.001, 0.005, 0.01, 0.015],
                           'random_state'     : [1],
                           }]
        else:
            param_grid = [{'criterion'        : ['gini', 'entropy'],
                           'max_depth'        : [3, 5, 7, 10, 100, 99999999],
                           'min_samples_split': [2, 3, 5, 7, 10],
                           'ccp_alpha'        : [0, .01, .02],
                           'random_state'     : [1]
                           }]

        tree_classifier = DecisionTreeClassifier()
        grid_search = GridSearchCV(tree_classifier, param_grid, cv=numFolds,
                                   scoring='roc_auc_ovr_weighted',
                                   return_train_score=True, n_jobs=njobs, verbose=debug)
        grid_search.fit(X_train, y_train)

        cvres = grid_search.cv_results_
        best_params = grid_search.best_params_

        util.save_gridsearch_to_csv(cvres, algo, filename[:-4], scalar)
        # Fit algo to best parameters and compute test score
        tree_classifier = DecisionTreeClassifier()
        tree_classifier.set_params(**best_params)
    else:
        # Fit algo to best parameters and compute test score
        tree_classifier = DecisionTreeClassifier()
        tree_classifier.set_params(**pDTree)

    tree_classifier.fit(X_train, y_train)
    train_score = tree_classifier.score(X_train, y_train)
    test_score = tree_classifier.score(X_test, y_test)

    # util.compute_roc(algo, 'max_depth', [1, 2, 3, 4, 5, 6, 7, 8, 9], X_train, y_train, X_test, y_test, tree_classifier,filename[:-4])

    # util.compute_vc(algo, 'max_depth', [1, 2, 3, 4, 5, 6, 7, 8, 9], X_train, y_train, tree_classifier, filename[:-4])

    # cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    if make_graphs:
        # Plot DTree
        # create_DT_image(tree_classifier, X_train, filename[:-4], scalar, True)

        # Plot without pruning, need to make it again with ccp_alpha = 0
        # tree_classifier.set_params(**{'ccp_alpha': 0})
        # tree_classifier.fit(X_train, y_train)
        # create_DT_image(tree_classifier, X_train, filename[:-4], scalar, False)

        # Plot Learning Curve
        # util.plot_learning_curve(tree_classifier, algo, filename[:-4], X_train, y_train, ylim=(0.7, 1.01), cv=10,
        #                         n_jobs=njobs)

        # computer Model Complexity/Validation curves
        test_class = DecisionTreeClassifier()
        test_class.set_params(**pDTree)
        other_class = DecisionTreeClassifier()
        other_class.set_params(**pDTree)
        other_class.set_params(**{'criterion': 'entropy'})
        other_class.set_params(**{'ccp_alpha': 0.001})
        other_class.set_params(**{'min_samples_split': 2})

        util.compute_vc(algo, 'max_depth',
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
                        X_train, y_train, X_test, y_test, tree_classifier, other_class, filename[:-4], test_class, log=False, njobs=njobs)
        raise SystemExit(0)
        tree_classifier.set_params(**{'ccp_alpha': 0.0})
        test_class.set_params(**{'ccp_alpha': 0.0})

        util.compute_vc(algo, 'max_depth',
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
                        X_train, y_train, X_test, y_test, tree_classifier, filename[:-4], test_class, log=False, njobs=njobs)
        tree_classifier.set_params(**{'min_samples_split': 2})
        test_class.set_params(**{'min_samples_split': 2})

        util.compute_vc(algo, 'max_depth',
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
                        X_train, y_train, X_test, y_test, tree_classifier, filename[:-4], test_class, log=False, njobs=njobs)
        raise SystemExit(0)

        util.compute_vc(algo, 'ccp_alpha',
                        [0.000001, 0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009,
                         0.0001, 0.00011, 0.00012, 0.00013, 0.00014, 0.00015, 0.00016, 0.00017, 0.00018, 0.00019,
                         0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001],
                        # 0.01 , 0.1, 1],
                        X_train, y_train, X_test, y_test, tree_classifier, filename[:-4], pDTree, log=True, njobs=njobs)

        util.compute_vc(algo, 'criterion', ['gini', 'entropy'], X_train, y_train, X_test, y_test, tree_classifier,
                        filename[:-4], pDTree, log=False, njobs=njobs)

        util.compute_vc(algo, 'min_samples_split', [2, 3, 5, 6, 8, 10, 20, 30], X_train, y_train, X_test, y_test,
                        tree_classifier,
                        filename[:-4], pDTree, log=False, njobs=njobs)
        raise SystemExit(0)

        util.compute_vc(algo, 'min_samples_leaf', [1, 2, 3, 5, 6, 8, 10], X_train, y_train, X_test, y_test,
                        tree_classifier,
                        filename[:-4], log=False, njobs=njobs)
        util.compute_vc(algo, 'max_leaf_nodes', [2, 4, 5, 7, 10, 20, 30, 40, 50, 60, 100, 1000], X_train, y_train,
                        X_test, y_test,
                        tree_classifier,
                        filename[:-4], log=False, njobs=njobs)
        util.compute_vc(algo, 'max_features', [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0], X_train, y_train, X_test,
                        y_test,
                        tree_classifier, filename[:-4], log=False, njobs=njobs)
        util.compute_vc(algo, 'splitter', ["best", "random"], X_train, y_train, X_test, y_test, tree_classifier,
                        filename[:-4],
                        log=False, njobs=njobs)

        # Plot ROC
        util.compute_roc(algo, 'max_depth', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50], X_train, y_train, X_test, y_test,
                         tree_classifier, filename[:-4])
    return time.time() - start, round(train_score, 4), round(test_score, 4)


def create_DT_image(Dtree, X_train, filename, scalar, prune):
    if prune:
        filename = filename + '-pruned'
    export_graphviz(Dtree,
                    out_file="Images\Dtree-" + filename + "-" + str(scalar) + ".dot",
                    feature_names=list(X_train),
                    class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8'],
                    rounded=True, filled=True)

    call(['dot', '-Tpng', "Images\Dtree-" + filename + "-" + str(scalar) + ".dot", '-o',
          "Images\Dtree-" + filename + "-" + str(scalar) + ".png", '-Gdpi=600'])
