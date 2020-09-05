import pandas as pd
from sklearn.model_selection import train_test_split  # Import train_test_split function
from matplotlib.colors import ListedColormap
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import csv

def get_class_counts(df, class_col):
    grp = df.groupby([class_col])['id'].nunique()
    print(grp)
    return {key: grp[key] for key in list(grp.keys())}


def get_class_proportions(df, class_col):
    class_counts = get_class_counts(df, class_col)
    return {val[0]: round(val[1] / df.shape[0], 4) for val in class_counts.items()}


def data_load(file_name, classifier_col, debug=False, scalar=1):
    df = pd.read_csv('Data/' + str(file_name), delimiter=',')
    df.head()
    row_count = df.shape[0]  # gives number of row count
    col_count = df.shape[1]  # gives number of col count

    if debug:
        print("Rows:", row_count)
        print("Columns:", col_count, )

    y = df[classifier_col]
    X = df.drop([classifier_col], axis=1)

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
    X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.3, random_state=1, stratify=y)

    # print(get_class_proportions(y_train, classifier_col))
    # print(get_class_proportions(y_test, classifier_col))
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

# https://gist.github.com/willwest/fcb61b110b9f7f59db40
def print_GridCV_scores(gs_clf, export_file):
    '''Exports a CSV of the GridCV scores.
    gs_clf: A GridSearchCV object which has been fitted
    export_file: A file path
    Example output (file content):
    mean,std,feat__words__ngram_range,feat__words__stop_words,clf__alpha
    0.56805074971164937,0.0082019735998974941,"(1, 2)",english,0.0001
    0.57189542483660127,0.0066384723824170488,"(1, 2)",None,0.0001
    0.56839677047289505,0.0082404203511470264,"(1, 3)",english,0.0001
    0.57306164295783668,0.0095988722286300399,"(1, 3)",None,0.0001
    0.53524285531205951,0.0026015635012174854,"(1, 2)",english,0.001
    0.53742150454953219,0.0031141868512110649,"(1, 2)",None,0.001
    0.53510829168268614,0.0032487504805843725,"(1, 3)",english,0.001
    0.53717160066641034,0.0034025374855825019,"(1, 3)",None,0.001
    '''
    with open(export_file, 'w') as outfile:
        csvwriter = csv.writer(outfile, delimiter=',')

        # Create the header using the parameter names
        header = ["mean", "std"]
        param_names = [param for param in gs_clf.param_grid]
        header.extend(param_names)

        csvwriter.writerow(header)

        for config in gs_clf.cv_results_:
            # Get the list of parameter settings and add to row
            print(config)
            print(config[0])
            print(config[1])
            params = [str(p) for p in config[0].values()]
            csvwriter.writerow(params)