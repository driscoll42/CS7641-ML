from time import time

from sklearn.manifold import LocallyLinearEmbedding

import util
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
import time
import numpy as np

from sklearn.decomposition import FastICA, PCA
from sklearn.random_projection import GaussianRandomProjection as GRP

def train_NN_PCA(filename, X_train, X_test, y_train, y_test, debug=False, numFolds=10, njobs=-1, scalar=1,
                   make_graphs=False, pNN={}, nolegend=False, random_seed=1, num_dim=4):
    np.random.seed(random_seed)
    algo = 'PCA' + str(num_dim)

    start = time.time()
    pca = PCA(n_components=num_dim, random_state=random_seed)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    param_grid = [{'hidden_layer_sizes': [(512, 512, 512, 512)],
                   'activation'        : ['relu'],  # 'identity',
                   'solver'            : ['adam'],
                   'alpha'             : [0.0001, 0.001, 0.01, 0.1],
                   'batch_size'        : ['auto'],
                   'learning_rate_init': [0.001, 0.01],
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

    util.save_gridsearch_to_csv(cvres, algo, filename[:-4] + '-' + str(num_dim), scalar, '')


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
        util.plot_learning_curve(nn_classifier, algo, filename[:-4], X_train, y_train, ylim=(0.0, 1.05), cv=10,
                                 n_jobs=njobs, debug=debug)


    return time.time() - start, round(train_score, 4), round(test_score, 4)

def train_NN_ICA(filename, X_train, X_test, y_train, y_test, debug=False, numFolds=10, njobs=-1, scalar=1,
                   make_graphs=False, pNN={}, nolegend=False, random_seed=1, num_dim=4):
    np.random.seed(random_seed)
    algo = 'ICA-' + str(num_dim)

    start = time.time()
    ica = FastICA(n_components=num_dim, random_state=random_seed)
    ica.fit(X_train)
    X_train = ica.transform(X_train)
    X_test = ica.transform(X_test)

    param_grid = [{'hidden_layer_sizes': [(512, 512, 512, 512)],
                   'activation'        : ['relu'],  # 'identity',
                   'solver'            : ['adam'],
                   'alpha'             : [0.0001, 0.001, 0.01, 0.1],
                   'batch_size'        : ['auto'],
                   'learning_rate_init': [0.001, 0.01],
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

    util.save_gridsearch_to_csv(cvres, algo, filename[:-4] + '-' + str(num_dim), scalar, '-kmeans')


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
        util.plot_learning_curve(nn_classifier, algo, filename[:-4], X_train, y_train, ylim=(0.0, 1.05), cv=10,
                                 n_jobs=njobs, debug=debug)

        # util.compute_vc(algo, 'alpha',
        #               [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500,
        #                1000, 5000, 10000, 100000, 1000000], X_train, y_train, X_test, y_test, nn_classifier,
        #               filename[:-4], test_class, pNN, log=True, njobs=njobs, debug=debug)

    return time.time() - start, round(train_score, 4), round(test_score, 4)


def train_NN_RP(filename, X_train, X_test, y_train, y_test, debug=False, numFolds=10, njobs=-1, scalar=1,
                   make_graphs=False, pNN={}, nolegend=False, random_seed=1, num_dim=4):
    np.random.seed(random_seed)
    algo = 'RP' + str(num_dim)

    start = time.time()
    rp = GRP(n_components=num_dim, random_state=random_seed)
    rp.fit(X_train)
    X_train = rp.transform(X_train)
    X_test = rp.transform(X_test)

    param_grid = [{'hidden_layer_sizes': [(512, 512, 512, 512)],
                   'activation'        : ['relu'],  # 'identity',
                   'solver'            : ['adam'],
                   'alpha'             : [0.0001, 0.001, 0.01, 0.1],
                   'batch_size'        : ['auto'],
                   'learning_rate_init': [0.001, 0.01],
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

    util.save_gridsearch_to_csv(cvres, algo, filename[:-4] + '-' + str(num_dim), scalar, '')


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
        util.plot_learning_curve(nn_classifier, algo, filename[:-4], X_train, y_train, ylim=(0.0, 1.05), cv=10,
                                 n_jobs=njobs, debug=debug)

        # util.compute_vc(algo, 'alpha',
        #               [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500,
        #                1000, 5000, 10000, 100000, 1000000], X_train, y_train, X_test, y_test, nn_classifier,
        #               filename[:-4], test_class, pNN, log=True, njobs=njobs, debug=debug)

    return time.time() - start, round(train_score, 4), round(test_score, 4)


def train_NN_LLE(filename, X_train, X_test, y_train, y_test, debug=False, numFolds=10, njobs=-1, scalar=1,
                   make_graphs=False, pNN={}, nolegend=False, random_seed=1, num_dim=4):
    np.random.seed(random_seed)
    algo = 'LLE' + str(num_dim)

    start = time.time()
    lle = LocallyLinearEmbedding(n_neighbors=10, n_components=num_dim, random_state=random_seed, n_jobs=-1)
    lle.fit(X_train)
    X_train = lle.transform(X_train)
    X_test = lle.transform(X_test)

    param_grid = [{'hidden_layer_sizes': [(512, 512, 512, 512)],
                   'activation'        : ['relu'],  # 'identity',
                   'solver'            : ['adam'],
                   'alpha'             : [0.0001, 0.001, 0.01, 0.1],
                   'batch_size'        : ['auto'],
                   'learning_rate_init': [0.001, 0.01],
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

    util.save_gridsearch_to_csv(cvres, algo, filename[:-4] + '-' + str(num_dim), scalar, '')


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
        util.plot_learning_curve(nn_classifier, algo, filename[:-4], X_train, y_train, ylim=(0.0, 1.05), cv=10,
                                 n_jobs=njobs, debug=debug)


    return time.time() - start, round(train_score, 4), round(test_score, 4)