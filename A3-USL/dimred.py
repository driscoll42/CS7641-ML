import numpy as np
from sklearn import metrics

import matplotlib.pyplot as plt


import pandas as pd
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection as GRP
from itertools import product
from collections import defaultdict
from sklearn.manifold import LocallyLinearEmbedding


def ulPCA(X, y, random_seed, filename, verbose=False):
    n_cols = len(X.columns)

    pca = PCA(svd_solver='full', random_state=random_seed)
    pca.fit(X)
    if verbose:
        print(pca.explained_variance_ratio_)
        print(pca.explained_variance_)
        print(pca.singular_values_)

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', -1)
        print(pd.DataFrame(pca.components_, columns=X.columns))

    cum_sum = np.cumsum(pca.explained_variance_ratio_)
    '''plt.figure()

    plt.plot(range(1, n_cols + 1), pca.explained_variance_ratio_, label='Variance Ratio', marker='o')
    plt.plot(range(1, n_cols + 1), cum_sum, label='Cumulative Variance Ratio', marker='o')

    plt.title()
    plt.xlabel('n_components', fontsize=16)
    plt.ylabel('Explained Variance Ratio', fontsize=16)
    plt.ylim(bottom=0)
    plt.xticks(range(1, n_cols + 1), fontsize=16)
    plt.yticks(np.linspace(0, 1, 11), fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.savefig("Images\\" + filename + " PCA Explained Variance")
    plt.show()

    plt.figure()
    plt.plot(pca.explained_variance_, label='Variance Ratio', marker='o')

    plt.title(filename + " PCA Eigenvalues", fontsize=16)
    plt.xlabel('n_components', fontsize=16)
    plt.ylabel('Eigenvalues', fontsize=16)
    plt.xticks(range(1, n_cols + 1), fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    # plt.legend()
    plt.savefig("Images\\" + filename + " PCA Eigenvalues")
    plt.show()'''

    fig, ax1 = plt.subplots()
    plt.title(filename + " PCA Explained Variance Ratio & Eigenvalues", fontsize=16)
    ax1.set_xlabel('# of Components', fontsize=16)
    ax1.set_ylabel('Explained Variance Ratio', color='blue', fontsize=16)
    ax1.plot(range(1, n_cols + 1), pca.explained_variance_ratio_, label='Variance Ratio', marker='o', color='blue')
    ax1.plot(range(1, n_cols + 1), cum_sum, label='Cumulative Variance Ratio', marker='o', color='cyan')
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=16)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.legend(prop={'size': 16}, loc='upper right')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Eigenvalue', color=color, fontsize=16)  # we already handled the x-label with ax1
    ax2.plot(range(1, n_cols + 1), pca.explained_variance_, label='Eigenvalues', marker='o', color=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=16)
    ax2.tick_params(axis='x', labelsize=16)

    ax2.legend(prop={'size': 16}, loc='right')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.xticks(n_clusters)
    plt.savefig("Images\\" + filename + " PCA Combined Plot")
    fig.show()

    n_cols = len(X.columns)
    n_com = range(1, n_cols + 1)

    re = defaultdict(dict)

    for i, n in product(range(50), n_com):
        random_projection = PCA(random_state=random_seed, n_components=n)
        X_Reduced = random_projection.fit_transform(X)

        p_inverse = np.linalg.pinv(random_projection.components_.T)
        Recon_X = X_Reduced.dot(p_inverse)

        MSE_RE = metrics.mean_squared_error(X, Recon_X)
        re[n][i] = MSE_RE

    rec = pd.DataFrame(re).T
    re_mean = rec.mean(axis=1).tolist()
    re_std = rec.std(axis=1).tolist()
    lower_axis = []
    upper_axis = []

    zip_object = zip(re_mean, re_std)
    for list1_i, list2_i in zip_object:
        lower_axis.append(list1_i - list2_i)
        upper_axis.append(list1_i + list2_i)

    if verbose:
        print('PCA RE')
        print(re_mean)
        print(re_std)
    fig, ax1 = plt.subplots()
    ax1.plot(n_com, re_mean, 'b-')
    ax1.fill_between(n_com, lower_axis, upper_axis, alpha=0.2)
    ax1.set_xlabel('# of Components', fontsize=16)
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Mean Reconstruction Error', color='b', fontsize=16)
    ax1.tick_params('y', colors='b', labelsize=16)
    ax1.tick_params('x', labelsize=16)
    plt.grid(False)
    plt.title(filename + " PCA Mean Reconstruction Error", fontsize=16)
    fig.tight_layout()
    plt.show()


def ulICA(X, y, random_seed, filename, verbose=False):
    n_cols = len(X.columns)

    n_com = range(1, n_cols + 1)
    ica = FastICA(random_state=random_seed)

    kurt_scores = []

    for n in n_com:
        ica.set_params(n_components=n)
        icaX = ica.fit_transform(X)
        icaX = pd.DataFrame(icaX)
        icaX = icaX.kurt(axis=0)
        kurt_scores.append(icaX.abs().mean())

    if verbose:
        print(kurt_scores)
    plt.figure(0)
    plt.xlabel("# of Components", fontsize=16)
    plt.ylabel("Average Kurtosis", fontsize=16)
    plt.title(filename + ' ICA', fontsize=16)
    plt.plot(n_com, kurt_scores, 'b-')
    plt.xticks(range(1, n_cols + 1), fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(linestyle='-', linewidth=1, axis="x")
    plt.savefig("Images\\" + filename + " ICA Kurtosis")
    plt.show()
    plt.close()

    n_cols = len(X.columns)
    n_com = range(1, n_cols + 1)

    re = defaultdict(dict)

    for i, n in product(range(50), n_com):
        random_projection = PCA(random_state=random_seed, n_components=n)
        X_Reduced = random_projection.fit_transform(X)

        p_inverse = np.linalg.pinv(random_projection.components_.T)
        Recon_X = X_Reduced.dot(p_inverse)

        MSE_RE = metrics.mean_squared_error(X, Recon_X)
        re[n][i] = MSE_RE

    rec = pd.DataFrame(re).T
    re_mean = rec.mean(axis=1).tolist()
    re_std = rec.std(axis=1).tolist()
    lower_axis = []
    upper_axis = []

    zip_object = zip(re_mean, re_std)
    for list1_i, list2_i in zip_object:
        lower_axis.append(list1_i - list2_i)
        upper_axis.append(list1_i + list2_i)

    if verbose:
        print('ICA RE')
        print(re_mean)
        print(re_std)
    fig, ax1 = plt.subplots()
    ax1.plot(n_com, re_mean, 'b-')
    ax1.fill_between(n_com, lower_axis, upper_axis, alpha=0.2)
    ax1.set_xlabel('# of Components', fontsize=16)
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Mean Reconstruction Error', color='b', fontsize=16)
    ax1.tick_params('y', colors='b', labelsize=16)
    ax1.tick_params('x', labelsize=16)
    plt.grid(False)
    plt.title(filename + " ICA Mean Reconstruction Error", fontsize=16)
    fig.tight_layout()
    plt.show()


def randProj(X, y, random_seed, filename, verbose=False):
    n_cols = len(X.columns)
    n_com = range(1, n_cols + 1)

    re = defaultdict(dict)

    for i, n in product(range(50), n_com):
        random_projection = GRP(random_state=i, n_components=n)
        X_Reduced = random_projection.fit_transform(X)

        p_inverse = np.linalg.pinv(random_projection.components_.T)
        Recon_X = X_Reduced.dot(p_inverse)

        MSE_RE = metrics.mean_squared_error(X, Recon_X)
        re[n][i] = MSE_RE

    rec = pd.DataFrame(re).T
    re_mean = rec.mean(axis=1).tolist()
    re_std = rec.std(axis=1).tolist()
    lower_axis = []
    upper_axis = []

    zip_object = zip(re_mean, re_std)
    for list1_i, list2_i in zip_object:
        lower_axis.append(list1_i - list2_i)
        upper_axis.append(list1_i + list2_i)

    if verbose:
        print('RP RE')
        print(re_mean)
        print(re_std)
    fig, ax1 = plt.subplots()
    ax1.plot(n_com, re_mean, 'b-')
    ax1.fill_between(n_com, lower_axis, upper_axis, alpha=0.2)
    ax1.set_xlabel('# of Components', fontsize=16)
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Mean Reconstruction Error', color='b', fontsize=16)
    ax1.tick_params('y', colors='b', labelsize=16)
    ax1.tick_params('x', labelsize=16)
    plt.grid(False)
    plt.title(filename + " RP Mean Reconstruction Error", fontsize=16)
    fig.tight_layout()
    plt.show()


def ul_LLE(X, y, random_seed, filename, verbose=False):
    n_cols = len(X.columns)
    re_list = []
    for i in range(n_cols):
        lle = LocallyLinearEmbedding(n_neighbors=10, n_components=i, random_state=random_seed, n_jobs=-1)
        lle.fit(X, y)
        re_list.append(lle.reconstruction_error_)
        if verbose:
            print(lle.reconstruction_error_)

    fig, ax1 = plt.subplots()
    ax1.plot(range(1, n_cols + 1), re_list, 'b-')
    ax1.set_xlabel('# of Components', fontsize=16)
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Mean Reconstruction Error', color='b', fontsize=16)
    ax1.tick_params('y', colors='b', labelsize=16)
    ax1.tick_params('x', labelsize=16)
    plt.grid(False)
    plt.title(filename + " LLE Mean Reconstruction Error", fontsize=16)
    fig.tight_layout()
    plt.show()
