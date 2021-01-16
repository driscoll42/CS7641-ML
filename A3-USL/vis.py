from time import time

import matplotlib.cm as cm

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.mixture import GaussianMixture
import copy
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
from sklearn import metrics

import matplotlib.pyplot as plt

import pandas as pd
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection as GRP

from sklearn.manifold import LocallyLinearEmbedding
import numpy as np
import random
import util
import clustering
import dimred
import cluster_NN
import DR_NN
import dr_cluster
import copy


def gen_vis(X, y, random_state, filename):
    if 1 == 1:
        if filename == "Chess":
            col1, col2 = 0, 5
            dotsize = 1000
        else:
            col1, col2 = 13, 11
            dotsize = 500
        # K-Means MP
        n_clusters = 2
        clusterer = KMeans(init='k-means++', n_clusters=n_clusters, n_init=100, random_state=random_state,
                           max_iter=100)

        cluster_labels = clusterer.fit_predict(X)

        plt.figure()

        cmap = cm.get_cmap("tab10")
        colors = cmap(cluster_labels.astype(float) / n_clusters)

        plt.scatter(X.iloc[:, col1], X.iloc[:, col2], marker='.', s=dotsize, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        plt.scatter(centers[:, col1], centers[:, col2], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            plt.scatter(c[col1], c[col2], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title("Visualization of the clustered data for %d KMeans clusters." % n_clusters, fontsize=18)
        plt.xlabel("Feature space for the 1st feature", fontsize=18)
        plt.ylabel("Feature space for the 2nd feature", fontsize=18)
        plt.savefig("Images\\" + filename + "Visualization " + str(n_clusters) + " clusters")

        plt.show()

        # EM MP
        if filename == 'Chess':
            n_clusters = 3
        else:
            n_clusters = 2
        clusterer = GaussianMixture(n_components=n_clusters, covariance_type='diag', n_init=100)

        cluster_labels = clusterer.fit_predict(X)

        plt.figure()

        cmap = cm.get_cmap("tab10")
        colors = cmap(cluster_labels.astype(float) / n_clusters)

        plt.scatter(X.iloc[:, col1], X.iloc[:, col2], marker='.', s=dotsize, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title("Visualization of the clustered data for %d KMeans clusters." % n_clusters, fontsize=18)
        plt.xlabel("Feature space for the 1st feature", fontsize=18)
        plt.ylabel("Feature space for the 2nd feature", fontsize=18)
        plt.savefig("Images\\" + filename + "Visualization " + str(n_clusters) + " clusters")

        plt.show()

# K-Means Chess
# EM Chess
# PCA Chess
# ICA Chess
# RP Chess
# LLE Chess


# The 16
