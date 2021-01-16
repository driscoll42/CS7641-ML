from time import time

import pandas as pd

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import numpy as np
import util
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
import time


def train_kmeansNN(filename, X_train, X_test, y_train, y_test, debug=False, numFolds=10, njobs=-1, scalar=1,
                   make_graphs=False, pNN={}, nolegend=False, random_seed=1, num_clusts=4):
    np.random.seed(random_seed)
    algo = 'Neural Network'

    start = time.time()
    if num_clusts != 1:
        KClusters = KMeans(init='k-means++', n_clusters=num_clusts, n_init=100, random_state=random_seed,
                           max_iter=100).fit(X_train)
        X_train.insert(0, 'Cluster', KClusters.predict(X_train))
        X_train['Cluster'] = X_train['Cluster'].apply(str)
        X_test.insert(0, 'Cluster', KClusters.predict(X_test))
        X_test['Cluster'] = X_test['Cluster'].apply(str)

    X_train = pd.get_dummies(X_train, prefix='Cluster')
    X_test = pd.get_dummies(X_test, prefix='Cluster')

    param_grid = [{'hidden_layer_sizes': [(512, 512, 512, 512)],
                   'activation'        : ['relu'],  # 'identity',
                   'solver'            : ['adam'],
                   'alpha'             : [0.01], #[0.0001, 0.001, 0.01, 0.1],
                   'batch_size'        : ['auto'],
                   'learning_rate_init': [0.01], #[0.001, 0.01],
                   'max_iter'          : [10000],
                   'warm_start'        : [True],
                   'early_stopping'    : [True],
                   'random_state'      : [1]
                   }]

    nn_classifier = MLPClassifier()

    grid_search = GridSearchCV(nn_classifier, param_grid, cv=numFolds,
                               scoring='roc_auc_ovr_weighted',
                               return_train_score=True, n_jobs=njobs, verbose=debug)
    grid_search.fit(X_train, y_train)
    cvres = grid_search.cv_results_

    util.save_gridsearch_to_csv(cvres, algo, filename[:-4] + '-' + str(num_clusts), scalar, '-kmeans')


    start = time.time()
    nn_classifier.fit(X_train, y_train)
    print('NN Fit Time: ', time.time() - start)

    start = time.time()
    y_prob = nn_classifier.predict_proba(X_train)
    train_score = roc_auc_score(y_train, y_prob, multi_class="ovr", average="weighted")
    print('NN Train Score Time: ', train_score, time.time() - start)


    start = time.time()
    y_prob = nn_classifier.predict_proba(X_test)
    test_score = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    print('NN Test Score Time: ', test_score, time.time() - start)

    test_class = MLPClassifier()
    test_class.set_params(**pNN)

    if make_graphs:
        # computer Model Complexity/Validation curves
        util.plot_learning_curve(nn_classifier, 'K-Means', filename[:-4], X_train, y_train, ylim=(0.0, 1.05), cv=10,
                                 n_jobs=njobs, debug=debug)

    return time.time() - start, round(train_score, 4), round(test_score, 4)


def train_EM_NN(filename, X_train, X_test, y_train, y_test, debug=False, numFolds=10, njobs=-1, scalar=1,
                make_graphs=False, pNN={}, nolegend=False, random_seed=1, num_clusts=4, cov_type='spherical'):
    np.random.seed(random_seed)
    algo = cov_type + '-' + str(num_clusts)

    start = time.time()
    if num_clusts != 1:
        KClusters = GaussianMixture(n_components=num_clusts, covariance_type=cov_type, n_init=10).fit(X_train)

        X_train.insert(0, 'Cluster', KClusters.predict(X_train))
        X_test.insert(0, 'Cluster', KClusters.predict(X_test))


    param_grid = [{'hidden_layer_sizes': [(512, 512, 512, 512)],
                   'activation'        : ['relu'],  # 'identity',
                   'solver'            : ['adam'],
                   'alpha'             : [0.0001], #[0.0001, 0.001, 0.01, 0.1],
                   'batch_size'        : ['auto'],
                   'learning_rate_init': [0.01], #[0.001, 0.01],
                   'max_iter'          : [10000],
                   'warm_start'        : [True],
                   'early_stopping'    : [True],
                   'random_state'      : [1]
                   }]

    nn_classifier = MLPClassifier()

    grid_search = GridSearchCV(nn_classifier, param_grid, cv=numFolds,
                               scoring='roc_auc_ovr_weighted',
                               return_train_score=True, n_jobs=njobs, verbose=debug)
    grid_search.fit(X_train, y_train)

    start = time.time()
    nn_classifier.fit(X_train, y_train)
    print('NN Fit Time: ', time.time() - start)
    start = time.time()

    y_prob = nn_classifier.predict_proba(X_train)
    train_score = roc_auc_score(y_train, y_prob, multi_class="ovr", average="weighted")
    print('NN Train Score Time: ', train_score, time.time() - start)
    cvres = grid_search.cv_results_

    util.save_gridsearch_to_csv(cvres, algo, filename[:-4] + '-diag-' + str(num_clusts), scalar, cov_type)

    start = time.time()

    y_prob = nn_classifier.predict_proba(X_test)
    test_score = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    print('NN Test Score Time: ', test_score, time.time() - start)

    test_class = MLPClassifier()
    test_class.set_params(**pNN)

    if make_graphs:
        # computer Model Complexity/Validation curves
        util.plot_learning_curve(nn_classifier, 'EM', filename[:-4], X_train, y_train, ylim=(0.0, 1.05), cv=10,
                                 n_jobs=njobs, debug=debug)

    return time.time() - start, round(train_score, 4), round(test_score, 4)
