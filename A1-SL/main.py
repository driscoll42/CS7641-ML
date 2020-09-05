import DecisionTree as DTree
import kNearestNeighbor as knn
import BoostedTree as BTree
import NeuralNetwork as NN
import SVM as SVM
import time
import util


def run_sl_algos(filename, result_col, full_param=False, test_all=False, debug=False, rDTree=True, rknn=True, rSVM=True, rNN=True,
                 rBTree=True, numFolds=10, njobs=-1, scalar=1):
    print(filename)
    start = time.time()
    X_train, X_test, y_train, y_test = util.data_load(filename, result_col,debug, scalar)
    print('data_load:', time.strftime("%H:%M:%S", time.gmtime(time.time() - start)))
    min_score = 1
    max_score = 0

    if rDTree:
        runTime, train_score, test_score = DTree.train_DTree(filename, X_train, X_test, y_train, y_test, full_param,
                                                             debug, numFolds, njobs, scalar)
        print('DTree: ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, test_score)
        min_score = min(min_score, test_score)
        max_score = max(max_score, test_score)

    if rNN:
        A_score, S_Score = 0, 0
        # In 10 out of 14 data sets adam was better than sgd, so testing it first
        runTime, train_score, S_Score = NN.train_NN(filename, X_train, X_test, y_train, y_test, 'adam', full_param,
                                                    debug, numFolds, njobs, scalar)
        print('NN-A:  ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, S_Score)

        if test_all or S_Score < 0.9:
            runTime, train_score, A_score = NN.train_NN(filename, X_train, X_test, y_train, y_test, 'sgd', full_param,
                                                        debug, numFolds, njobs, scalar)
            print('NN-S:  ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, A_score)


        overall_score = max(A_score, S_Score)
        min_score = min(min_score, overall_score)
        max_score = max(max_score, overall_score)



    if rknn:
        runTime, train_score, test_score = knn.train_knn(filename, X_train, X_test, y_train, y_test, full_param,
                                                         debug, numFolds, njobs, scalar)
        print('knn:   ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, test_score)
        min_score = min(min_score, test_score)
        max_score = max(max_score, test_score)


    if rSVM:
        L_score, R_Score, S_Score, P_Score = 0, 0, 0, 0
        # In 14 data sets, the most common scoring was rbf > poly > linear > sigmoid
        runTime, train_score, R_Score = SVM.train_svm(filename, X_train, X_test, y_train, y_test, 'rbf',
                                                      full_param, debug, numFolds, njobs, scalar)
        print('SVM-R: ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, R_Score)

        if test_all or R_Score < 0.9:
            runTime, train_score, P_Score = SVM.train_svm(filename, X_train, X_test, y_train, y_test, 'poly',
                                                          full_param, debug, numFolds, njobs, scalar)
            print('SVM-P: ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, P_Score)

        if test_all or max(R_Score, P_Score) < 0.9:
            runTime, train_score, L_score = SVM.train_svm(filename, X_train, X_test, y_train, y_test, 'linear',
                                                          full_param, debug, numFolds, njobs, scalar)
            print('SVM-L: ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, L_score)

        if test_all or max(R_Score, P_Score, L_score) < 0.9:
            runTime, train_score, S_Score = SVM.train_svm(filename, X_train, X_test, y_train, y_test, 'sigmoid',
                                                          full_param, debug, numFolds, njobs, scalar)
            print('SVM-S: ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, S_Score)

        overall_score = max(L_score, R_Score, S_Score, P_Score)
        min_score = min(min_score, overall_score)
        max_score = max(max_score, overall_score)

    if rBTree:
        runTime, train_score, test_score = BTree.train_BTree(filename, X_train, X_test, y_train, y_test, full_param,
                                                             debug, numFolds, njobs, scalar)
        print('BTree: ', time.strftime("%H:%M:%S", time.gmtime(runTime)), train_score, test_score)

        min_score = min(min_score, test_score)
        max_score = max(max_score, test_score)

    print('Overall Variance: ', round(max_score - min_score, 4), '\n')

# run_sl_algos('car.csv', 'class', debug = False)

# Finalists
#run_sl_algos('price_train.csv', 'price_range',   scalar=1, njobs=7, full_param=True, test_all=True, debug=True, rDTree=True, rknn=False, rSVM=False, rNN=False, rBTree=False)
#run_sl_algos('price_train.csv', 'price_range',   scalar=0, njobs=7, full_param=True, test_all=True, debug=True, rDTree=True, rknn=False, rSVM=False, rNN=False, rBTree=False)
#run_sl_algos('price_train.csv', 'price_range',   scalar=2, njobs=7, full_param=True, test_all=True, debug=True, rDTree=True, rknn=False, rSVM=False, rNN=False, rBTree=False)

# run_sl_algos('eyeDetection.csv', 'eyeDetection', scalar=1, njobs=7, full_param=True, test_all=True, debug=True, rDTree=False, rknn=False, rSVM=False, rNN=False, rBTree=False) # rSVM=True, rNN=True
# run_sl_algos('eyeDetection.csv', 'eyeDetection', scalar=0, njobs=7, full_param=True, test_all=True, debug=True, rDTree=True, rknn=False, rSVM=False, rNN=False, rBTree=False) # rSVM=True, rNN=True
# run_sl_algos('eyeDetection.csv', 'eyeDetection', scalar=2, njobs=7, full_param=True, test_all=True, debug=True, rDTree=True, rknn=False, rSVM=False, rNN=False, rBTree=False) # rSVM=True, rNN=True

run_sl_algos('ElectricalGrid.csv', 'stabf',      scalar=1, njobs=5, full_param=True, test_all=True, debug=True, rDTree=False, rknn=True, rSVM=False, rNN=False, rBTree=False)


run_sl_algos('Magic.csv', 'class',               scalar=1, njobs=4, full_param=True, test_all=True, debug=True, rDTree=True, rknn=True, rSVM=False, rNN=False, rBTree=True) # rSVM=True, rNN=True
run_sl_algos('Magic.csv', 'class',               scalar=0, njobs=4, full_param=True, test_all=True, debug=True, rDTree=True, rknn=True, rSVM=False, rNN=False, rBTree=True) # rSVM=True, rNN=True
run_sl_algos('Magic.csv', 'class',               scalar=2, njobs=4, full_param=True, test_all=True, debug=True, rDTree=True, rknn=True, rSVM=False, rNN=False, rBTree=True) # rSVM=True, rNN=True

run_sl_algos('ElectricalGrid.csv', 'stabf',      scalar=1, njobs=7, full_param=True, test_all=True, debug=True, rDTree=False, rknn=True, rSVM=False, rNN=False, rBTree=False)
run_sl_algos('ElectricalGrid.csv', 'stabf',      scalar=0, njobs=7, full_param=True, test_all=True, debug=True, rDTree=False, rknn=True, rSVM=False, rNN=False, rBTree=False)
run_sl_algos('ElectricalGrid.csv', 'stabf',      scalar=2, njobs=7, full_param=True, test_all=True, debug=True, rDTree=False, rknn=True, rSVM=False, rNN=False, rBTree=False)


run_sl_algos('SteelPlates.csv', 'Outcome',       scalar=1, njobs=7, full_param=True, test_all=True, debug=True, rDTree=True, rknn=True, rSVM=False, rNN=False, rBTree=True) # rSVM=True, rNN=True
run_sl_algos('SteelPlates.csv', 'Outcome',       scalar=0, njobs=7, full_param=True, test_all=True, debug=True, rDTree=True, rknn=True, rSVM=False, rNN=False, rBTree=True) # rSVM=True, rNN=True
run_sl_algos('SteelPlates.csv', 'Outcome',       scalar=2, njobs=7, full_param=True, test_all=True, debug=True, rDTree=True, rknn=True, rSVM=False, rNN=False, rBTree=True) # rSVM=True, rNN=True

run_sl_algos('Letter.csv', 'Letter',             scalar=1, njobs=7, full_param=True, test_all=True, debug=True, rDTree=False, rknn=False, rSVM=False, rNN=False, rBTree=False)
run_sl_algos('Letter.csv', 'Letter',             scalar=0, njobs=7, full_param=True, test_all=True, debug=True, rDTree=False, rknn=False, rSVM=False, rNN=False, rBTree=False)
run_sl_algos('Letter.csv', 'Letter',             scalar=2, njobs=7, full_param=True, test_all=True, debug=True, rDTree=False, rknn=False, rSVM=False, rNN=False, rBTree=False)

# Need to run
run_sl_algos('bank-full.csv', 'outcome', full_param=False, test_all=True, debug=True,  njobs=4, rDTree=False)
run_sl_algos('UCI_Credit_Card.csv', 'default.payment.next.month', False, False)
run_sl_algos('fordTrain.csv', 'IsAlert', False, False)
run_sl_algos('hepatitisC.csv', 'Decision', False, False, rDTree=False, rknn=False, rSVM=False, rNN=False)
run_sl_algos('Cover.csv', 'CoverType', False, False, rDTree=False)
run_sl_algos('creditcard_old.csv', 'Class', False, False)



# run_sl_algos('car.csv', 'class', full_param=True, test_all=True, debug=True,  njobs=7, rDTree=False)
# run_sl_algos('krkopt.csv', 'OpDepth', False, False)
# run_sl_algos('contra.csv', 'Contra', False, False)
# run_sl_algos('krkopt_num.csv', 'OpDepth', False, False)
# run_sl_algos('wine_cat.csv', 'quality', debug=True,  njobs=7, rDTree=False)
# run_sl_algos('wine_bin.csv', 'quality', debug=True,  njobs=7)
# run_sl_algos('bank_marketing.csv', 'outcome', True, True, rDTree=False, rknn=False, njobs=4)
# run_sl_algos('krvskp.csv', 'outcome', False, False)
# run_sl_algos('SkyData.csv', 'class', True, True, rknn=True, rDTree=False, rSVM=False, rNN=False, rBTree=False)
# run_sl_algos('price_train.csv', 'price_range', True, True, rknn=False, rDTree=False, rSVM=False, rNN=True, rBTree=False)

# run_sl_algos('churn.csv', 'Exited', False, debug=10, rknn=True, rDTree=False, rSVM=False, rNN=False, rBTree=True, njobs=4)
# run_sl_algos('adult.csv', 'income', False, False)
# run_sl_algos('SteelPlates.csv', 'Outcome', False, True, rDTree=False, rknn=True, rNN=False)
# run_sl_algos('spam.csv', 'spam', False, True, rDTree=False, rknn=True, rNN=False)
# run_sl_algos('Letter.csv', 'Letter', True, True, rDTree=True, rknn=False, rSVM=False, rNN=False, rBTree=False)

# Interesting
# run_sl_algos('Letter.csv', 'Letter', False, False)
# run_sl_algos('SkyData.csv', 'class', False, False)
# run_sl_algos('eyeDetection.csv', 'eyeDetection', False, False)
# run_sl_algos('spam.csv', 'spam', False, False)
# run_sl_algos('ElectricalGrid.csv', 'stabf', False, False)
# run_sl_algos('price_train.csv', 'price_range', False, False)
# run_sl_algos('SteelPlates.csv', 'Outcome', False, False)
# run_sl_algos('car_oh.csv', 'class', False, False)
# run_sl_algos('car.csv', 'class', False, False)
# run_sl_algos('car_oh.csv', 'class', False, False)

# Uninteresting
# run_sl_algos('wine.csv', 'quality', False, False, numFolds=4)
# run_sl_algos('phishing.csv', 'class', False, False) # Removing ID made uninteresting
# run_sl_algos('breast_cancer.csv', 'diagnosis', False, False) # Removing ID made uninteresting
# run_sl_algos('heart.csv', 'target', False, False) # Too small
# run_sl_algos('TaxInfo.csv', 'PoliticalParty', False, False)
# run_sl_algos('blocks.csv', 'block', False, False)
# run_sl_algos('Website_Phishing.csv', 'Result', False, False)
# run_sl_algos('diabetes.csv', 'Outcome', False, False)
# run_sl_algos('pendigits.csv', 'ClassCode', False, False)
# run_sl_algos('Magic.csv', 'class', False, False)
# run_sl_algos('HTRU_2.csv', 'Outcome', False, False)
# run_sl_algos('pulsar_stars.csv', 'target_class', False, False)
# run_sl_algos('optdata.csv', 'Class', False, True, rDTree=False, rknn=False, rSVM=False, rNN=False)
# run_sl_algos('nursery.csv', 'rec', False, False)
# run_sl_algos('mushroom.csv', 'class', False, False)


# Would need to clean data
# run_sl_algos('abalone_original.csv', 'rings', False, False) The least populated class in y has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.
# run_sl_algos('Avila.csv', 'Class', False, False) UserWarning: The least populated class in y has only 4 members, which is less than n_splits=10.
# run_sl_algos('audit_data.csv', 'Risk', False, False) Missing data


# TODO: training set size
# TODO: Include Training Error, cross-validation error, test error
# TODO: Create Confusion Matrices
# TODO: Skorch for NN
