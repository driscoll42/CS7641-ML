import DecisionTree as DTree
import kNearestNeighbor as knn
import BoostedTree as BTree
import NeuralNetwork as NN
import SVM as SVM
import time
import util


def run_sl_algos(filename, result_col, full_param=False, debug=False):
    print(filename)
    X_train, X_test, y_train, y_test, X_train_n, X_test_n, y_train_n, y_test_n = util.data_load(filename, result_col,
                                                                                                debug)
    runTime, train_score, test_score = 0, 0, 0

    runTime, train_score, test_score = DTree.train_DTree(filename, X_train, X_test, y_train, y_test, full_param, debug)
    print('DTree: ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, test_score)

    runTime, train_score, test_score = knn.train_knn(filename, X_train, X_test, y_train, y_test, full_param, debug)
    print('knn:   ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, test_score)

    runTime, train_score, test_score = SVM.train_svm(filename, X_train_n, X_test_n, y_train_n, y_test_n, full_param, debug)
    print('SVM:   ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, test_score)

    runTime, train_score, test_score = NN.train_NN(filename, X_train, X_test, y_train, y_test, full_param, debug)
    print('NN:    ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, test_score)

    runTime, train_score, test_score = BTree.train_BTree(filename, X_train, X_test, y_train, y_test, full_param, debug)
    print('BTree: ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, test_score)


run_sl_algos('price_train.csv', 'price_range', False, False)
run_sl_algos('SteelPlates.csv', 'Outcome', False, False)
run_sl_algos('creditcard.csv', 'Class', False, False)

# TODO: Clean up Subprograms and maybe make more generic still?
# TODO: Test more Datasets
# TODO: Train vs test percentages
# TODO: Label features
# TODO: training set size
# TODO: Include Training Error, cross-validation error, test error
# TODO: Create Confusion Matrices
