import pandas as pd
from sklearn.model_selection import train_test_split  # Import train_test_split function
from matplotlib.colors import ListedColormap
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn import preprocessing


def get_class_counts(df, class_col):
    grp = df.groupby([class_col])['id'].nunique()
    print(grp)
    return {key: grp[key] for key in list(grp.keys())}


def get_class_proportions(df, class_col):
    class_counts = get_class_counts(df, class_col)
    return {val[0]: round(val[1] / df.shape[0], 4) for val in class_counts.items()}


def data_load(file_name, classifier_col, debug=False):
    df = pd.read_csv('Data/' + str(file_name), delimiter=',')
    df.head()
    row_count = df.shape[0]  # gives number of row count
    col_count = df.shape[1]  # gives number of col count

    if debug:
        print("Rows:", row_count)
        print("Columns:", col_count, )

    y = df[classifier_col]
    X = df.drop([classifier_col], axis=1)

    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(X)
    X_n = pd.DataFrame(x_scaled)

    # Split dataset into training set (70%) and test set (30%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_n, y, test_size=0.3, random_state=1, stratify=y)

    # print(get_class_proportions(y_train, classifier_col))
    # print(get_class_proportions(y_test, classifier_col))
    return X_train, X_test, y_train, y_test, X_train_n, X_test_n, y_train_n, y_test_n


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
