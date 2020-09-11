import pandas as pd
import numpy as np
import csv

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import learning_curve, validation_curve, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def plot_class_distribution(df, class_col, file_name, X):
    # TODO: Only works with numerical classes

    X.hist(sharey=True, xlabelsize=10, ylabelsize=10)
    grp = df[class_col].value_counts()
    total_rows = len(df.index)
    fig, ax = plt.subplots()

    min_key = min(list(grp.keys()))
    max_key = max(list(grp.keys())) + 1
    labels = []
    data = []
    class_counts = {key: grp[key] for key in list(grp.keys())}
    colors = ['red', 'green', 'blue', 'yellow', 'black', 'cyan']

    for i in range(min_key, max_key):
        labels.append(i)
        data.append(float(grp[i]) / total_rows * 100)

    plt.xticks(range(len(data)), labels, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Class', fontsize=16)
    plt.ylabel('Occurances (% of total)', fontsize=16)
    plt.title("Class Distribution of {0} Dataset".format(file_name), fontsize=16)
    rects1 = plt.bar(range(len(data)), data, color=colors, edgecolor='black')
    for rect in rects1:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.0 * height,
                '{0}'.format(round(float(height), 1)), ha='center', va='bottom', fontsize=11)
    plt.show()
    return {val[0]: round(val[1] / df.shape[0], 4) for val in class_counts.items()}


def data_load(file_name, classifier_col, debug=False, scalar=1, make_graphs=False):
    df = pd.read_csv('Data/' + str(file_name), delimiter=',')
    df.head()
    row_count = df.shape[0]  # gives number of row count
    col_count = df.shape[1]  # gives number of col count

    if debug:
        print("Rows:", row_count)
        print("Columns:", col_count, )

    y = df[classifier_col]
    X = df.drop([classifier_col], axis=1)
    # y = label_binarize(y, classes=[0, 1, 2, 3])

    if make_graphs:
        plot_class_distribution(df, classifier_col, file_name[:-4], X)

    X_obj = X.loc[:, X.dtypes == np.object]
    X_nobj = X.loc[:, X.dtypes != np.object]

    if not X_nobj.empty:
        if scalar == 0:  # No scaling
            X_s = X_nobj
        if scalar == 1:  # Min_Max_Scaler
            min_max_scalar = MinMaxScaler()
            x_scaled = min_max_scalar.fit_transform(X_nobj)
            X_s = pd.DataFrame(x_scaled)
        elif scalar == 2:
            standard_scalar = StandardScaler()
            x_scaled = standard_scalar.fit_transform(X_nobj)
            X_s = pd.DataFrame(x_scaled)

    if not X_obj.empty:
        X_obj_OH = pd.get_dummies(X_obj)
        X = pd.concat([X_obj_OH, X_nobj], axis=1, sort=False)
        if not X_nobj.empty:
            X_s = pd.concat([X_obj_OH, X_s], axis=1, sort=False)
        else:
            X_s = X

    # TODO: If missing data, use nearest neighbor
    # TODO: What if only one or two samples
    # TODO: Return Random Choice and most common
    # TODO: Validation set

    # Split dataset into training set (70%) and test set (30%)
    X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.3, random_state=1, stratify=y, shuffle=True)

    return X_train, X_test, y_train, y_test


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
    plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                alpha=1.0, linewidth=1, marker='o',
                s=55, label='test set')


def compute_roc(algo, parameter, p_range, X_train, y_train, X_test, y_test, classifier, filename):
    # TODO: Make this be able to handle multilabel
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#plot-roc-curves-for-the-multilabel-problem
    plt.figure(figsize=(8, 6))
    for i, p in enumerate(p_range):
        classifier.set_params(**{parameter: p})
        classifier = classifier.fit(X_train, y_train)
        fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        print(roc_auc)
        plt.plot(fpr, tpr, label=parameter + ' = {0}, area = {1:.3f}'.format(str(p_range[i]), roc_auc))

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('{0} Receiver Operating Characteristics of {1}'.format(algo, filename), fontsize=16)
    plt.legend(loc='lower right')
    plt.show()


def compute_vc(algo, parameter, p_range, X_train, y_train, X_test, y_test, classifier, other_class, filename, test_class, log=False, njobs=-1):
    train_scores, vc_scores = validation_curve(
            classifier, X_train, y_train, param_name=parameter, param_range=p_range,
            scoring="roc_auc_ovr_weighted", n_jobs=njobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    vc_scores_mean = np.mean(vc_scores, axis=1)
    vc_scores_std = np.std(vc_scores, axis=1)

    train_scores, vc_scores = validation_curve(
            other_class, X_train, y_train, param_name=parameter, param_range=p_range,
            scoring="roc_auc_ovr_weighted", n_jobs=njobs)
    train_scores_mean_o = np.mean(train_scores, axis=1)
    vc_scores_mean_o = np.mean(vc_scores, axis=1)

    test_scores = []

    for i in p_range:
        test_class.set_params(**{parameter: i})
        test_class.fit(X_train, y_train)
        y_prob = test_class.predict_proba(X_test)
        weighted_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
        test_scores.append(weighted_roc_auc_ovr)
    print(test_scores)
    plt.title("{0} Validation Curve of {1}".format(algo, filename), fontsize=16)
    plt.xlabel(parameter)
    plt.ylabel("Score")
    plt.ylim(0.8, 1.05)
    lw = 2

    if p_range[0] is tuple:
        p_range = [str(ele) for ele in p_range]
    if log:
        plt.semilogx(p_range, train_scores_mean, label="Training score",
                     color="darkorange", lw=lw)
        plt.fill_between(p_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        plt.semilogx(p_range, vc_scores_mean, label="Cross-validation score",
                     color="navy", lw=lw)
        plt.fill_between(p_range, vc_scores_mean - vc_scores_std,
                         vc_scores_mean + vc_scores_std, alpha=0.2,
                         color="navy", lw=lw)
        plt.semilogx(p_range, test_scores, label="Test score", color="darkred", lw=lw)
    else:
        plt.plot(p_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
        plt.fill_between(p_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        plt.plot(p_range, vc_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
        plt.fill_between(p_range, vc_scores_mean - vc_scores_std,
                         vc_scores_mean + vc_scores_std, alpha=0.2,
                         color="navy", lw=lw)
        plt.plot(p_range, test_scores, label="Test score", color="darkred", lw=lw)

        plt.plot(p_range, train_scores_mean_o, label="Ent score", color="green", lw=lw)
        plt.plot(p_range, vc_scores_mean_o, label="Ent VC score", color="pink", lw=lw)

    plt.legend(loc="best")
    plt.show()


def plot_learning_curve(estimator, algo, filename, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title("{0} Learning Curves of {1}".format(algo, filename))

    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    plt.show()


def save_gridsearch_to_csv(cvres, algo, filename, scalar, solver=''):
    gs_df = pd.DataFrame(cvres)
    mtrs = gs_df["mean_train_score"]
    mtes = gs_df["mean_test_score"]
    msrs = gs_df["std_train_score"]
    mses = gs_df["std_test_score"]
    gs_dfp = pd.DataFrame(cvres["params"])

    out_df = pd.concat([mtrs, mtes, msrs, mses, gs_dfp], axis=1, sort=False)
    if solver:
        out_df.to_csv("ParamTests\\" + algo + "-" + solver + "-" + filename + "-" + str(scalar) + ".csv")
    else:
        out_df.to_csv("ParamTests\\" + algo + "-" + filename + "-" + str(scalar) + ".csv")
