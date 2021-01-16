import numpy as np
from time import time

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pandas as pd

from sklearn import metrics
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.mixture import GaussianMixture
import copy
from yellowbrick.cluster import KElbowVisualizer


# Citation: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
def bench_k_means(estimator, labels, name, data, sample_size, n_clusters, random_state, filename, verbose=False):
    t0 = time()
    estimator.fit(data)
    fit_time = time() - t0

    homo = metrics.homogeneity_score(labels, estimator.predict(data))
    comp = metrics.completeness_score(labels, estimator.predict(data))
    v_meas = metrics.v_measure_score(labels, estimator.predict(data))
    ari = metrics.adjusted_rand_score(labels, estimator.predict(data))
    ami = metrics.adjusted_mutual_info_score(labels, estimator.predict(data))
    fks = metrics.fowlkes_mallows_score(labels, estimator.predict(data))
    silo = metrics.silhouette_score(data, estimator.predict(data), metric='euclidean', sample_size=sample_size,
                                    random_state=random_state)
    dbs = metrics.davies_bouldin_score(data, estimator.predict(data))
    chs = metrics.calinski_harabasz_score(data, estimator.predict(data))
    iner = estimator.inertia_
    if verbose:
        print('%-9s\t%d\t%.2fs\t%.2f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%i\t%i'
              % (name, n_clusters, (time() - t0), homo, comp, v_meas, ari, ami, fks, silo, dbs, chs, iner)
              )

    return fit_time, homo, comp, v_meas, ari, ami, fks, silo, dbs, chs, iner


# Citation: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
def silo_analysis(X, y, clusterer, n_clusters, random_state, filename):
    if filename == "Chess":
        col1, col2 = 0, 5
    else:
        col1, col2 = 13, 11

    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels, metric='euclidean', sample_size=400, random_state=random_state)

    plt.figure()

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    plt.xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    plt.ylim([0, len(X) + (n_clusters + 1) * 10])

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        cmap = cm.get_cmap("tab10")
        color = cmap(float(i) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    plt.title("The silhouette plot for the various %d KMeans clusters." % n_clusters)

    plt.xlabel("The silhouette coefficient values")
    plt.ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")

    plt.yticks([])  # Clear the yaxis labels / ticks
    plt.xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.savefig("Images\\" + filename + "Silhouette Plot for " + str(n_clusters) + " clusters")
    plt.show()

    # 2nd Plot showing the actual clusters formed
    plt.figure()

    cmap = cm.get_cmap("tab10")
    colors = cmap(cluster_labels.astype(float) / n_clusters)

    plt.scatter(X.iloc[:, col1], X.iloc[:, col2], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    plt.scatter(centers[:, col1], centers[:, col2], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        plt.scatter(c[col1], c[col2], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    plt.title("Visualization of the clustered data for %d KMeans clusters." % n_clusters)
    plt.xlabel("Feature space for the 1st feature")
    plt.ylabel("Feature space for the 2nd feature")
    plt.savefig("Images\\" + filename + "Visualization " + str(n_clusters) + " clusters")

    plt.show()


# Citation: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_stability_low_dim_dense.html#sphx-glr-auto-examples-cluster-plot-kmeans-stability-low-dim-dense-py
def init_analysis(X, y, n_clusters, filename):
    # Number of run (with randomly generated dataset) for each strategy so as
    # to be able to compute an estimate of the standard deviation
    n_runs = 5

    # k-means models can do several random inits so as to be able to trade
    # CPU time for convergence robustness
    n_init_range = np.array([1, 5, 10, 15, 20, 50, 75, 100])

    # Part 1: Quantitative evaluation of various init methods

    plt.figure()
    plots = []
    legends = []

    cases = [
        (KMeans, 'k-means++', {}),
        (KMeans, 'random', {}),
        (MiniBatchKMeans, 'k-means++', {'max_no_improvement': 3}),
        (MiniBatchKMeans, 'random', {'max_no_improvement': 3, 'init_size': 500}),
    ]

    for factory, init, params in cases:
        print("Evaluation of %s with %s init" % (factory.__name__, init))
        inertia = np.empty((len(n_init_range), n_runs))

        for run_id in range(n_runs):
            for i, n_init in enumerate(n_init_range):
                km = factory(n_clusters=n_clusters, init=init, random_state=run_id,
                             n_init=n_init, **params).fit(X)
                inertia[i, run_id] = km.inertia_
        p = plt.errorbar(n_init_range, inertia.mean(axis=1), inertia.std(axis=1))
        plots.append(p[0])
        legends.append("%s with %s init" % (factory.__name__, init))

    plt.xlabel('n_init')
    plt.ylabel('inertia')
    plt.legend(plots, legends)
    plt.title("Mean inertia for various k-means init across %d runs" % n_runs)
    plt.savefig(
            "Images\\" + filename + "Mean Interia across " + str(n_runs) + " runs and " + str(n_clusters) + " clusters")

    plt.show()


# Citation: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
def bench_EM(estimator, labels, name, data, sample_size, n_clusters, random_state, filename, verbose=False):
    t0 = time()
    estimator.fit(data)
    fit_time = time() - t0

    homo = 0  # metrics.homogeneity_score(labels, estimator.predict(data))
    comp = 0  # metrics.completeness_score(labels, estimator.predict(data))
    v_meas = metrics.v_measure_score(labels, estimator.predict(data))
    ari = metrics.adjusted_rand_score(labels, estimator.predict(data))
    ami = metrics.adjusted_mutual_info_score(labels, estimator.predict(data))
    fks = 0  # metrics.fowlkes_mallows_score(labels, estimator.predict(data))
    silo = metrics.silhouette_score(data, estimator.predict(data), metric='euclidean', sample_size=sample_size,
                                    random_state=random_state)
    dbs = metrics.davies_bouldin_score(data, estimator.predict(data))
    chs = metrics.calinski_harabasz_score(data, estimator.predict(data))
    aics = 0  # estimator.aic(data)  # Akaike information criterion for the current model on the input X. Lower is better
    bics = estimator.bic(data)  # Bayesian information criterion for the current model on the input X. Lower is Better
    scor = estimator.score(data)
    if verbose:
        print('%-9s\t%d\t%.2fs\t%.2f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%i\t%i\t%i\t%.3f'
              % (name, n_clusters, fit_time, homo, comp, v_meas, ari, ami, fks, silo, dbs, chs, aics, bics, scor)
              )
    return fit_time, homo, comp, v_meas, ari, ami, fks, silo, dbs, chs, aics, bics, scor


def ul_Kmeans(X, y, random_seed, filename, classifier_col, verbose=False):
    n_samples, n_features = X.shape

    print(X)
    n_y = len(np.unique(y))

    sample_size = 400
    n_init = 10

    total_n = 51

    n_clusters = [i for i in range(2, total_n)]

    if verbose:
        print("n_classes: %d, \t n_samples %d, \t n_features %d"
              % (n_y, n_samples, n_features))
        print('init\t\tn\ttime\thomo\tcompl\tv-meas\tARI \tAMI \tFMS \tsilo\tDBI \tCHI\tInertia')

    fit_time_arr, homo_arr, comp_arr, v_meas_arr, ari_arr, ami_arr, fks_arr, silo_arr, dbs_arr, chs_arr, iner_arr = [], [], [], [], [], [], [], [], [], [], []

    for n in n_clusters:
        clusterer = KMeans(init='k-means++', n_clusters=n, n_init=n_init, random_state=random_seed, max_iter=100)
        fit_time, homo, comp, v_meas, ari, ami, fks, silo, dbs, chs, iner = bench_k_means(clusterer, name="k-means++",
                                                                                          data=X, labels=y,
                                                                                          sample_size=sample_size,
                                                                                          n_clusters=n,
                                                                                          random_state=random_seed,
                                                                                          filename=filename,
                                                                                          verbose=verbose)

        fit_time_arr.append(fit_time)
        homo_arr.append(homo)
        comp_arr.append(comp)
        v_meas_arr.append(v_meas)
        ari_arr.append(ari)
        ami_arr.append(ami)
        fks_arr.append(fks)
        silo_arr.append(silo)
        dbs_arr.append(dbs)
        chs_arr.append(chs)
        iner_arr.append(iner)

        # TODO: Work on this later
        if 1 == 0:
            new_y = copy.deepcopy(y)
            clusters = pd.DataFrame(copy.deepcopy(clusterer.labels_))
            clusters.columns = ['clusters']

            combined = pd.concat([new_y, clusters], axis=1, sort=False)

            fig, axs = plt.subplots(1, n, sharey=True, tight_layout=True)

            for i in range(n):
                cluster_cnt = combined[combined['clusters'] == i]
                cluster_cnt = cluster_cnt[classifier_col]
                axs[i].hist(cluster_cnt, bins=[0, 1, 2, 3, 4, 5])
                axs[i].set_xlim(0, 4)
                axs[i].set_xticks([0, 1, 2, 3, 4])

            # plt.show()

        # silo_analysis(X, y, clusterer, n, random_state=random_seed, filename=filename)
        # init_analysis(X, y, n, filename)

    # Plot scores where ground truth is NOT known
    fig, ax1 = plt.subplots()
    plt.title("No Knowledge of Ground Truth Scores", fontsize=16)
    plt.xticks(fontsize=16)
    ax1.set_xlabel('Number of Clusters', fontsize=16)
    ax1.set_ylabel('score', color='blue', fontsize=16)
    ax1.plot(n_clusters, dbs_arr, label='DBS', color='blue')
    ax1.plot(n_clusters, silo_arr, label='Silo', color='cyan')
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=16)
    ax1.legend(prop={'size': 16}, loc='right')

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Calinski-Harabasz Index', color=color, fontsize=16)
    ax2.plot(n_clusters, chs_arr, label='Calinski-Harabasz', color=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=16)
    ax2.legend(prop={'size': 16})
    fig.tight_layout()
    plt.savefig("Images\\" + filename + " No Truth Scores")
    fig.show()

    # Plot scores where ground truth IS known
    plt.figure()

    # plt.plot(n_clusters, homo_arr, label='Homogeneity')
    # plt.plot(n_clusters, comp_arr, label='Completeness')
    plt.plot(n_clusters, v_meas_arr, label='V-measure')
    plt.plot(n_clusters, ari_arr, label='ARI')
    plt.plot(n_clusters, ami_arr, label='AMI')
    # plt.plot(n_clusters, fks_arr, label='FKS')

    plt.title("Ground Truth Scores", fontsize=16)
    plt.xlabel('Number of Clusters', fontsize=16)
    plt.ylabel('Score', fontsize=16)
    plt.grid(True)
    # plt.xticks(n_clusters)
    plt.legend(prop={'size': 16})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig("Images\\" + filename + " Ground Truth Scores")
    plt.show()

    # Plot Inertia
    plt.figure()
    plt.plot(n_clusters, iner_arr, label='Inertia')
    plt.title("Inertia", fontsize=16)
    plt.xlabel('Number of Clusters', fontsize=16)
    plt.ylabel('Score', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True)
    # plt.xticks(n_clusters)
    plt.savefig("Images\\" + filename + " Inertia")
    plt.show()

    # Plot scores where ground truth is NOT known
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_ylabel('Inertia', color=color)  # we already handled the x-label with ax1
    ax1.plot(n_clusters, chs_arr, label='Inertia', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(prop={'size': 16})

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Time to Fit (s)', color=color)  # we already handled the x-label with ax1
    ax2.plot(n_clusters, fit_time_arr, label='Fit Time', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='right')
    # plt.xticks(n_clusters)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig("Images\\" + filename + " Inertia + Fit Time")
    fig.show()

    if 1 == 0:
        model = KMeans()
        visualizer = KElbowVisualizer(
                model, k=(2, total_n), metric='distortion', timings=False
        )

        visualizer.fit(X)  # Fit the data to the visualizer
        visualizer.show()  # Finalize and render the figure


def ul_EM(X, y, random_seed, filename, classifier_col, verbose=False):
    # Generate random sample, two components
    np.random.seed(random_seed)
    n_samples, n_features = X.shape
    n_y = len(np.unique(y))

    sample_size = 400

    # If seeding of the centers is deterministic, only need to run kmeans algorithm once with n_init=1
    n_init = 100

    n_clusters = [i for i in range(2, 51)]
    print("n_classes: %d, \t n_samples %d, \t n_features %d"
          % (n_y, n_samples, n_features))
    print('init\t\tn\ttime\thomo\tcompl\tv-meas\tARI \tAMI \tFMS \tsilo\tDBI \tCHI\tAIC\tBIC\tScore')

    t_fit_time_arr, t_homo_arr, t_comp_arr, t_v_meas_arr, t_ari_arr, t_ami_arr, t_fks_arr, t_silo_arr, t_dbs_arr, t_chs_arr, t_aics_arr, t_bics_arr, t_scor_arr = [], [], [], [], [], [], [], [], [], [], [], [], []

    for n in n_clusters:
        clusterer = GaussianMixture(n_components=n, covariance_type='tied', n_init=n_init)
        fit_time, homo, comp, v_meas, ari, ami, fks, silo, dbs, chs, aics, bics, scor = bench_EM(clusterer, name="tied",
                                                                                                 data=X, labels=y,
                                                                                                 sample_size=sample_size,
                                                                                                 n_clusters=n,
                                                                                                 verbose=verbose,
                                                                                                 random_state=random_seed,
                                                                                                 filename=filename)
        t_fit_time_arr.append(fit_time)
        t_homo_arr.append(homo)
        t_comp_arr.append(comp)
        t_v_meas_arr.append(v_meas)
        t_ari_arr.append(ari)
        t_ami_arr.append(ami)
        t_fks_arr.append(fks)
        t_silo_arr.append(silo)
        t_dbs_arr.append(dbs)
        t_chs_arr.append(chs)
        t_aics_arr.append(aics)
        t_bics_arr.append(bics)
        t_scor_arr.append(scor)

    f_fit_time_arr, f_homo_arr, f_comp_arr, f_v_meas_arr, f_ari_arr, f_ami_arr, f_fks_arr, f_silo_arr, f_dbs_arr, f_chs_arr, f_aics_arr, f_bics_arr, f_scor_arr = [], [], [], [], [], [], [], [], [], [], [], [], []

    for n in n_clusters:
        clusterer = GaussianMixture(n_components=n, covariance_type='full', n_init=n_init)
        fit_time, homo, comp, v_meas, ari, ami, fks, silo, dbs, chs, aics, bics, scor = bench_EM(clusterer, name="full",
                                                                                                 data=X, labels=y,
                                                                                                 sample_size=sample_size,
                                                                                                 n_clusters=n,
                                                                                                 verbose=verbose,
                                                                                                 random_state=random_seed,
                                                                                                 filename=filename)
        f_fit_time_arr.append(fit_time)
        f_homo_arr.append(homo)
        f_comp_arr.append(comp)
        f_v_meas_arr.append(v_meas)
        f_ari_arr.append(ari)
        f_ami_arr.append(ami)
        f_fks_arr.append(fks)
        f_silo_arr.append(silo)
        f_dbs_arr.append(dbs)
        f_chs_arr.append(chs)
        f_aics_arr.append(aics)
        f_bics_arr.append(bics)
        f_scor_arr.append(scor)

    d_fit_time_arr, d_homo_arr, d_comp_arr, d_v_meas_arr, d_ari_arr, d_ami_arr, d_fks_arr, d_silo_arr, d_dbs_arr, d_chs_arr, d_aics_arr, d_bics_arr, d_scor_arr = [], [], [], [], [], [], [], [], [], [], [], [], []

    for n in n_clusters:
        clusterer = GaussianMixture(n_components=n, covariance_type='diag', n_init=n_init)
        fit_time, homo, comp, v_meas, ari, ami, fks, silo, dbs, chs, aics, bics, scor = bench_EM(clusterer, name="diag",
                                                                                                 data=X, labels=y,
                                                                                                 sample_size=sample_size,
                                                                                                 n_clusters=n,
                                                                                                 verbose=verbose,
                                                                                                 random_state=random_seed,
                                                                                                 filename=filename)
        d_fit_time_arr.append(fit_time)
        d_homo_arr.append(homo)
        d_comp_arr.append(comp)
        d_v_meas_arr.append(v_meas)
        d_ari_arr.append(ari)
        d_ami_arr.append(ami)
        d_fks_arr.append(fks)
        d_silo_arr.append(silo)
        d_dbs_arr.append(dbs)
        d_chs_arr.append(chs)
        d_aics_arr.append(aics)
        d_bics_arr.append(bics)
        d_scor_arr.append(scor)

    s_fit_time_arr, s_homo_arr, s_comp_arr, s_v_meas_arr, s_ari_arr, s_ami_arr, s_fks_arr, s_silo_arr, s_dbs_arr, s_chs_arr, s_aics_arr, s_bics_arr, s_scor_arr = [], [], [], [], [], [], [], [], [], [], [], [], []

    for n in n_clusters:
        clusterer = GaussianMixture(n_components=n, covariance_type='spherical', n_init=n_init)
        fit_time, homo, comp, v_meas, ari, ami, fks, silo, dbs, chs, aics, bics, scor = bench_EM(clusterer,
                                                                                                 name="spherical",
                                                                                                 data=X, labels=y,
                                                                                                 sample_size=sample_size,
                                                                                                 n_clusters=n,
                                                                                                 verbose=verbose,
                                                                                                 random_state=random_seed,
                                                                                                 filename=filename)
        s_fit_time_arr.append(fit_time)
        s_homo_arr.append(homo)
        s_comp_arr.append(comp)
        s_v_meas_arr.append(v_meas)
        s_ari_arr.append(ari)
        s_ami_arr.append(ami)
        s_fks_arr.append(fks)
        s_silo_arr.append(silo)
        s_dbs_arr.append(dbs)
        s_chs_arr.append(chs)
        s_aics_arr.append(aics)
        s_bics_arr.append(bics)
        s_scor_arr.append(scor)

    # BIC Plotting
    plt.figure()

    plt.plot(n_clusters, t_bics_arr, label='tied')
    plt.plot(n_clusters, f_bics_arr, label='full')
    plt.plot(n_clusters, s_bics_arr, label='spherical')
    plt.plot(n_clusters, d_bics_arr, label='diag')

    plt.title("BIC Scores by Covariance", fontsize=16)
    plt.xlabel('Number of Clusters', fontsize=16)
    plt.ylabel('BIC Score', fontsize=16)
    plt.grid(True)
    # plt.xticks(n_clusters)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(prop={'size': 16})
    plt.savefig("Images\\" + filename + " BIC Scores")
    plt.show()

    # AIC Plotting
    plt.figure()

    plt.plot(n_clusters, t_aics_arr, label='tied')
    plt.plot(n_clusters, f_aics_arr, label='full')
    plt.plot(n_clusters, s_aics_arr, label='spherical')
    plt.plot(n_clusters, d_aics_arr, label='diag')

    plt.title("AIC Scores by Covariance", fontsize=16)
    plt.xlabel('Number of Clusters', fontsize=16)
    plt.ylabel('AIC Score', fontsize=16)
    plt.grid(True)
    # plt.xticks(n_clusters)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(prop={'size': 16})
    plt.savefig("Images\\" + filename + " AIC Scores")
    plt.show()

    # Plot Score
    plt.figure()
    plt.plot(n_clusters, t_scor_arr, label='tied')
    plt.plot(n_clusters, f_scor_arr, label='full')
    plt.plot(n_clusters, s_scor_arr, label='spherical')
    plt.plot(n_clusters, d_scor_arr, label='diag')
    plt.title("Weighted Log Probability Score", fontsize=16)
    plt.xlabel('Number of Clusters', fontsize=16)
    plt.ylabel('Score', fontsize=16)
    plt.grid(True)
    plt.legend(prop={'size': 16})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.xticks(n_clusters)
    plt.savefig("Images\\" + filename + " EM Score")
    plt.show()

    # Plot scores where ground truth is NOT known
    fig, ax1 = plt.subplots()
    plt.title("No Knowledge of Ground Truth Scores - tied", fontsize=16)
    ax1.set_xlabel('Number of Clusters', fontsize=16)
    ax1.set_ylabel('score', color='blue', fontsize=16)
    ax1.plot(n_clusters, t_dbs_arr, label='DBS', color='blue')
    ax1.plot(n_clusters, t_silo_arr, label='Silo', color='cyan')
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=16)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.legend(prop={'size': 16}, loc='right')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Calinski-Harabasz Index', color=color, fontsize=16)  # we already handled the x-label with ax1
    ax2.plot(n_clusters, t_chs_arr, label='Calinski-Harabasz Index', color=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=16)
    ax2.tick_params(axis='x', labelsize=16)

    ax2.legend(prop={'size': 16}, loc='upper right')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.xticks(n_clusters)
    plt.savefig("Images\\" + filename + " EM tied No Truth Scores")
    fig.show()

    fig, ax1 = plt.subplots()
    plt.title("No Knowledge of Ground Truth Scores - diag", fontsize=16)
    ax1.set_xlabel('Number of Clusters', fontsize=16)
    ax1.set_ylabel('score', color='blue', fontsize=16)
    ax1.plot(n_clusters, d_dbs_arr, label='DBS', color='blue')
    ax1.plot(n_clusters, d_silo_arr, label='Silo', color='cyan')
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=16)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.legend(prop={'size': 16})

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Calinski-Harabasz Index', color=color, fontsize=16)  # we already handled the x-label with ax1
    ax2.plot(n_clusters, d_chs_arr, label='Calinski-Harabasz Index', color=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=16)
    ax2.tick_params(axis='x', labelsize=16)
    ax2.legend(prop={'size': 16}, loc='upper right')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.xticks(n_clusters)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig("Images\\" + filename + " EM diag No Truth Scores")
    fig.show()

    fig, ax1 = plt.subplots()
    plt.title("No Knowledge of Ground Truth Scores - full", fontsize=16)
    ax1.set_xlabel('Number of Clusters', fontsize=16)
    ax1.set_ylabel('score', color='blue', fontsize=16)
    ax1.plot(n_clusters, f_dbs_arr, label='DBS', color='blue')
    ax1.plot(n_clusters, f_silo_arr, label='Silo', color='cyan')
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=16)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.legend(prop={'size': 16})

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Calinski-Harabasz Index', color=color, fontsize=16)  # we already handled the x-label with ax1
    ax2.plot(n_clusters, f_chs_arr, label='Calinski-Harabasz Index', color=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=16)
    ax2.tick_params(axis='x', labelsize=16)
    ax2.legend(prop={'size': 16}, loc='upper right')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.xticks(n_clusters)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig("Images\\" + filename + " EM full No Truth Scores")
    fig.show()

    fig, ax1 = plt.subplots()
    plt.title("No Knowledge of Ground Truth Scores - spherical", fontsize=16)
    ax1.set_xlabel('Number of Clusters', fontsize=16)
    ax1.set_ylabel('score', color='blue', fontsize=16)
    ax1.plot(n_clusters, s_dbs_arr, label='DBS', color='blue')
    ax1.plot(n_clusters, s_silo_arr, label='Silo', color='cyan')
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=16)
    ax1.tick_params(axis='x', labelsize=16)

    ax1.legend(prop={'size': 16})

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Calinski-Harabasz Index', color=color, fontsize=16)  # we already handled the x-label with ax1
    ax2.plot(n_clusters, s_chs_arr, label='Calinski-Harabasz Index', color=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=16)
    ax2.tick_params(axis='x', labelsize=16)

    ax2.legend(prop={'size': 16}, loc='upper right')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.xticks(n_clusters)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig("Images\\" + filename + " EM spherical No Truth Scores")
    fig.show()

    # Plot scores where ground truth IS known
    plt.figure()

    plt.plot(n_clusters, t_v_meas_arr, label='V-measure')
    plt.plot(n_clusters, t_ari_arr, label='ARI')
    plt.plot(n_clusters, t_ami_arr, label='AMI')
    # plt.plot(n_clusters, fks_arr, label='FKS')

    plt.title("Ground Truth Scores - tied", fontsize=16)
    plt.xlabel('Number of Clusters', fontsize=16)
    plt.ylabel('Score', fontsize=16)
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.xticks(n_clusters)
    plt.legend(prop={'size': 16})
    plt.savefig("Images\\" + filename + " EM tied Ground Truth Scores")
    plt.show()

    plt.figure()

    plt.plot(n_clusters, d_v_meas_arr, label='V-measure')
    plt.plot(n_clusters, d_ari_arr, label='ARI')
    plt.plot(n_clusters, d_ami_arr, label='AMI')
    # plt.plot(n_clusters, fks_arr, label='FKS')

    plt.title("Ground Truth Scores - diag", fontsize=16)
    plt.xlabel('Number of Clusters', fontsize=16)
    plt.ylabel('Score', fontsize=16)
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.xticks(n_clusters)
    plt.legend(prop={'size': 16})
    plt.savefig("Images\\" + filename + " EM diag Ground Truth Scores")
    plt.show()

    plt.figure()

    plt.plot(n_clusters, f_v_meas_arr, label='V-measure')
    plt.plot(n_clusters, f_ari_arr, label='ARI')
    plt.plot(n_clusters, f_ami_arr, label='AMI')
    # plt.plot(n_clusters, fks_arr, label='FKS')

    plt.title("Ground Truth Scores - full", fontsize=16)
    plt.xlabel('Number of Clusters', fontsize=16)
    plt.ylabel('Score', fontsize=16)
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.xticks(n_clusters)
    plt.legend(prop={'size': 16})
    plt.savefig("Images\\" + filename + " EM full Ground Truth Scores")
    plt.show()

    plt.figure()

    plt.plot(n_clusters, s_v_meas_arr, label='V-measure')
    plt.plot(n_clusters, s_ari_arr, label='ARI')
    plt.plot(n_clusters, s_ami_arr, label='AMI')
    # plt.plot(n_clusters, fks_arr, label='FKS')

    plt.title("Ground Truth Scores - spherical", fontsize=16)
    plt.xlabel('Number of Clusters', fontsize=16)
    plt.ylabel('Score', fontsize=16)
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.xticks(n_clusters)
    plt.legend(prop={'size': 16})
    plt.savefig("Images\\" + filename + " EM spherical Ground Truth Scores")
    plt.show()
