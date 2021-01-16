import numpy as np
import random
import util
import clustering
import dimred
import cluster_NN
import DR_NN
import dr_cluster
import copy
import vis

def run_ul_algos(filename, result_col, debug=False, numFolds=10, njobs=-1, scalar=1, make_graphs=False, nolegend=False,
                 verbose=False, random_seed=1, rNN=False, pNN={}):
    np.random.seed(random_seed)
    random.seed(random_seed)

    X_train, X_test, y_train, y_test = util.data_load(filename, result_col, debug, scalar, make_graphs, random_seed)


    vis.gen_vis(X_train, X_test, random_seed, filename[:-4])


    clustering.ul_Kmeans(X_train, y_train, random_seed, filename[:-4], result_col, verbose)

    clustering.ul_EM(X_train, y_train, random_seed, filename[:-4], result_col, verbose)

    dimred.ulPCA(X_train, y_train, random_seed, filename[:-4], verbose)

    dimred.ulICA(X_train, y_train, random_seed, filename[:-4], verbose)

    dimred.randProj(X_train, y_train, random_seed, filename[:-4], verbose)

    dimred.ul_LLE(X_train, y_train, random_seed, filename[:-4], verbose)
    new_Xtrain = copy.deepcopy(X_train)
    new_ytrain = copy.deepcopy(y_train)
    dr_cluster.pca_clust(new_Xtrain, new_ytrain, random_seed, filename[:-4], result_col, verbose)

    new_Xtrain = copy.deepcopy(X_train)
    new_ytrain = copy.deepcopy(y_train)
    dr_cluster.ica_clust(new_Xtrain, new_ytrain, random_seed, filename[:-4], result_col, verbose)

    new_Xtrain = copy.deepcopy(X_train)
    new_ytrain = copy.deepcopy(y_train)
    dr_cluster.rp_clust(new_Xtrain, new_ytrain, random_seed, filename[:-4], result_col, verbose)

    new_Xtrain = copy.deepcopy(X_train)
    new_ytrain = copy.deepcopy(y_train)
    dr_cluster.lle_clust(new_Xtrain, new_ytrain, random_seed, filename[:-4], result_col, verbose)

    # Run NNs for Dim Reduction
    if rNN:
        for n in range(1, 21):
            print('PCA', n)
            new_Xtrain = copy.deepcopy(X_train)
            new_Xtest = copy.deepcopy(X_test)
            new_ytrain = copy.deepcopy(y_train)
            new_ytest = copy.deepcopy(y_test)
            DR_NN.train_NN_PCA(filename[:-4], new_Xtrain, new_Xtest, new_ytrain, new_ytest,
                               random_seed=random_seed, scalar=scalar,
                               njobs=njobs, numFolds=numFolds, make_graphs=make_graphs, nolegend=nolegend,
                               pNN=pNN, num_dim=n)
            print('ICA', n)
            new_Xtrain = copy.deepcopy(X_train)
            new_Xtest = copy.deepcopy(X_test)
            new_ytrain = copy.deepcopy(y_train)
            new_ytest = copy.deepcopy(y_test)
            DR_NN.train_NN_ICA(filename[:-4], new_Xtrain, new_Xtest, new_ytrain, new_ytest,
                               random_seed=random_seed, scalar=scalar,
                               njobs=njobs, numFolds=numFolds, make_graphs=make_graphs, nolegend=nolegend,
                               pNN=pNN, num_dim=n)
            print('RP', n)
            new_Xtrain = copy.deepcopy(X_train)
            new_Xtest = copy.deepcopy(X_test)
            new_ytrain = copy.deepcopy(y_train)
            new_ytest = copy.deepcopy(y_test)
            DR_NN.train_NN_RP(filename[:-4], new_Xtrain, new_Xtest, new_ytrain, new_ytest,
                              random_seed=random_seed, scalar=scalar,
                              njobs=njobs, numFolds=numFolds, make_graphs=make_graphs, nolegend=nolegend,
                              pNN=pNN, num_dim=n)
        for n in range(1, 21):
            print('LLE', n)
            new_Xtrain = copy.deepcopy(X_train)
            new_Xtest = copy.deepcopy(X_test)
            new_ytrain = copy.deepcopy(y_train)
            new_ytest = copy.deepcopy(y_test)
            DR_NN.train_NN_LLE(filename[:-4], new_Xtrain, new_Xtest, new_ytrain, new_ytest,
                               random_seed=random_seed, scalar=scalar,
                               njobs=njobs, numFolds=numFolds, make_graphs=make_graphs, nolegend=nolegend,
                               pNN=pNN, num_dim=n)

    # Run NN for clustering
    if rNN:
        for n in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
            print('kmeans', n)
            new_Xtrain = copy.deepcopy(X_train)
            new_Xtest = copy.deepcopy(X_test)
            new_ytrain = copy.deepcopy(y_train)
            new_ytest = copy.deepcopy(y_test)
            cluster_NN.train_kmeansNN(filename[:-4], new_Xtrain, new_Xtest, new_ytrain, new_ytest,
                                      random_seed=random_seed, scalar=scalar, debug=verbose,
                                      njobs=njobs, numFolds=numFolds, make_graphs=make_graphs, nolegend=nolegend,
                                      pNN=pNN, num_clusts=n)

        print('diag')
        # cov_types = ['diag', 'tied', 'full', 'spherical']
        # for cov in cov_types:
        for n in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
            print('diag', n)
            new_Xtrain = copy.deepcopy(X_train)
            new_Xtest = copy.deepcopy(X_test)
            new_ytrain = copy.deepcopy(y_train)
            new_ytest = copy.deepcopy(y_test)
            cluster_NN.train_EM_NN(filename[:-4], new_Xtrain, new_Xtest, new_ytrain, new_ytest,
                                   random_seed=random_seed, scalar=scalar,
                                   njobs=njobs, numFolds=numFolds, make_graphs=make_graphs, nolegend=nolegend,
                                   pNN=pNN, num_clusts=n, cov_type='diag')


def main():
    if 1 == 1:
        run_ul_algos(filename='Mobile_Prices.csv',
                     result_col='price_range',
                     scalar=0,
                     random_seed=1,
                     make_graphs=False,
                     verbose=True,
                     rNN=True,
                     pNN={'hidden_layer_sizes': (512, 512, 512, 512),
                          'activation'        : 'relu',
                          'solver'            : 'adam',
                          'alpha'             : 0.1,
                          'learning_rate_init': 0.01,
                          'max_iter'          : 10000,
                          'warm_start'        : True,
                          'early_stopping'    : True,
                          'random_state'      : 1},
                     )
    if 1 == 1:
        run_ul_algos(filename='Chess.csv',
                     result_col='OpDepth',
                     scalar=2,
                     random_seed=1,
                     make_graphs=False,
                     verbose=True)


if __name__ == "__main__":
    main()
