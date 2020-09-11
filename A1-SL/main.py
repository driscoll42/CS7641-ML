import DecisionTree as DTree
import kNearestNeighbor as knn
import BoostedTree as BTree
import NeuralNetwork as NN
import SVM as SVM
import time
import util


def run_sl_algos(filename, result_col, full_param=False, test_all=False, debug=False, rDTree=True, pDTree={}, rknn=True,
                 pknn={},
                 rSVM=True, pSVM={}, rNN=True, pNN={}, rBTree=True, pBTree={}, numFolds=10, njobs=-1, scalar=1,
                 make_graphs=False):
    print(filename, '-', scalar)
    start = time.time()
    X_train, X_test, y_train, y_test = util.data_load(filename, result_col, debug, scalar, make_graphs)
    print('data_load:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start)))
    min_score = 1
    max_score = 0

    if rDTree:
        runTime, train_score, test_score = DTree.train_DTree(filename, X_train, X_test, y_train, y_test, full_param,
                                                             debug, numFolds, njobs, scalar, make_graphs, pDTree)
        print('DTree: ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, test_score)
        min_score = min(min_score, test_score)
        max_score = max(max_score, test_score)

    if rknn:
        runTime, train_score, test_score = knn.train_knn(filename, X_train, X_test, y_train, y_test, full_param,
                                                         debug, numFolds, njobs, scalar, make_graphs, pknn)
        print('knn:   ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, test_score)
        min_score = min(min_score, test_score)
        max_score = max(max_score, test_score)

    if rBTree:
        runTime, train_score, test_score = BTree.train_BTree(filename, X_train, X_test, y_train, y_test, full_param,
                                                             debug, numFolds, njobs, scalar, make_graphs, pBTree)
        print('BTree: ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, test_score)

        min_score = min(min_score, test_score)
        max_score = max(max_score, test_score)

    if rNN:
        A_score, S_Score = 0, 0
        if len(pNN) > 0:
            runTime, train_score, R_Score = NN.train_NN(filename, X_train, X_test, y_train, y_test, pNN['solver'],
                                                        full_param, debug, numFolds, njobs, scalar, make_graphs, pNN)
            print('NN: ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, R_Score)
        else:
            # In 10 out of 14 data sets adam was better than sgd, so testing it first
            runTime, train_score, S_Score = NN.train_NN(filename, X_train, X_test, y_train, y_test, 'adam', full_param,
                                                        debug, numFolds, njobs, scalar, make_graphs)
            print('NN-A:  ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, S_Score)

            if test_all or S_Score < 0.6:
                runTime, train_score, A_score = NN.train_NN(filename, X_train, X_test, y_train, y_test, 'sgd',
                                                            full_param,
                                                            debug, numFolds, njobs, scalar, make_graphs)
                print('NN-S:  ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, A_score)

        overall_score = max(A_score, S_Score)
        min_score = min(min_score, overall_score)
        max_score = max(max_score, overall_score)


    if rSVM:
        L_score, R_Score, S_Score, P_Score = 0, 0, 0, 0
        if len(pSVM) > 0:
            runTime, train_score, R_Score = SVM.train_svm(filename, X_train, X_test, y_train, y_test, pSVM['kernel'],
                                                          full_param, debug, numFolds, njobs, scalar, make_graphs, pSVM)
            print('SVM: ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, R_Score)
        else:
            # In 14 data sets, the most common scoring was rbf > poly > linear > sigmoid
            runTime, train_score, R_Score = SVM.train_svm(filename, X_train, X_test, y_train, y_test, 'rbf',
                                                          full_param, debug, numFolds, njobs, scalar, make_graphs)
            print('SVM-R: ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, R_Score)
            if test_all or R_Score < 0.8:
                runTime, train_score, P_Score = SVM.train_svm(filename, X_train, X_test, y_train, y_test, 'poly',
                                                              full_param, debug, numFolds, njobs, scalar, make_graphs)
                print('SVM-P: ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, P_Score)

            if test_all or max(R_Score, P_Score) < 0.9:
                runTime, train_score, L_score = SVM.train_svm(filename, X_train, X_test, y_train, y_test, 'linear',
                                                              full_param, debug, numFolds, njobs, scalar, make_graphs)
                print('SVM-L: ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, L_score)

            if test_all or max(R_Score, P_Score, L_score) < 0.9:
                runTime, train_score, S_Score = SVM.train_svm(filename, X_train, X_test, y_train, y_test, 'sigmoid',
                                                              full_param, debug, numFolds, njobs, scalar, make_graphs)
                print('SVM-S: ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, S_Score)

        overall_score = max(L_score, R_Score, S_Score, P_Score)
        min_score = min(min_score, overall_score)
        max_score = max(max_score, overall_score)

    print('Overall Variance: ', round(max_score - min_score, 4), '\n')


if __name__ == "__main__":
    run_sl_algos(filename='Mobile_Prices.csv',
                 result_col='price_range',
                 scalar=0,
                 njobs=5,
                 full_param=True,
                 make_graphs=False,
                 test_all=True,
                 debug=True,
                 rDTree=False,
                 rknn=True,
                 rSVM=False,
                 rNN=False,
                 rBTree=False)

    run_sl_algos(filename='Chess.csv',
                 result_col='OpDepth',
                 scalar=0,
                 njobs=5,
                 full_param=True,
                 make_graphs=False,
                 test_all=False,
                 debug=True,
                 rDTree=False,
                 rknn=False,
                 rSVM=True,
                 rNN=True,
                 rBTree=True
                         )
    raise SystemExit(0)

    run_sl_algos(filename='Mobile_Prices.csv',
                 result_col='price_range',
                 scalar=0,
                 njobs=7,
                 full_param=True,
                 make_graphs=False,
                 test_all=True,
                 debug=True,
                 rDTree=False,
                 rknn=False,
                 rSVM=True,
                 rNN=False,
                 rBTree=False)

    run_sl_algos(filename='Chess.csv',
                 result_col='OpDepth',
                 scalar=0,
                 njobs=4,
                 full_param=False,
                 make_graphs=True,
                 test_all=False,
                 debug=True,
                 rDTree=True,
                 pDTree={'criterion': 'entropy', 'max_depth': 10000, 'min_samples_split': 2,
                         'ccp_alpha': 0.0, 'random_state': 1},
                 rknn=True,
                 pknn={'algorithm': 'ball_tree', 'leaf_size': 1, 'n_neighbors': 6, 'weights': 'distance'},
                 rSVM=True,
                 pSVM={'C': 100, 'cache_size': 2000, 'kernel': 'rbf', 'random_state': 1},
                 rNN=True,
                 pNN={'activation': 'relu', 'early_stopping': True, 'hidden_layer_sizes': (128, 128, 128, 128),
                      'max_iter'  : 10000, 'random_state': 1, 'solver': 'adam'},
                 rBTree=True,
                 pBTree={'base_estimator__ccp_alpha': 0, 'base_estimator__criterion': 'entropy',
                         'base_estimator__max_depth': 10, 'n_estimators': 150, 'random_state': 1
                         }, )
    # raise SystemExit(0)

    run_sl_algos(filename='Mobile_Prices.csv',
                 result_col='price_range',
                 scalar=0,
                 njobs=7,
                 full_param=False,
                 test_all=False,
                 make_graphs=True,
                 debug=True,
                 rDTree=False,
                 pDTree={'criterion'   : 'gini', 'max_depth': 7, 'min_samples_leaf': 2, 'min_samples_split': 2,
                         'random_state': 1},

                 rknn=False,
                 pknn={'algorithm': 'ball_tree', 'leaf_size': 1, 'n_neighbors': 20, 'p': 1, 'weights': 'distance'},
                 rNN=True,
                 pNN={'activation': 'relu', 'early_stopping': True, 'hidden_layer_sizes': (128),
                      'max_iter'  : 10000, 'random_state': 1, 'solver': 'adam'},
                 rSVM=False,
                 pSVM={'C'           : .0001, 'cache_size': 2000, 'coef0': 0, 'degree': 1, 'gamma': 1, 'kernel': 'poly',
                       'random_state': 1},
                 rBTree=False,
                 pBTree={'base_estimator__ccp_alpha': 0, 'base_estimator__criterion': 'entropy',
                         'base_estimator__max_depth': 7, 'n_estimators': 150, 'random_state': 1
                         }, )
    raise SystemExit(0)
