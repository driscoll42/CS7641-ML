import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def plot_class_distribution(df, class_col, file_name, X):
    X_hist = X.hist(sharey=True, xlabelsize=10, ylabelsize=10)
    # fig  = X_hist.get_figure()
    # fig.savefig('Images\\' + file_name + '-Hist.png')

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
    plt.savefig('Images\\' + file_name + '-ClassDist.png')
    plt.show()
    return {val[0]: round(val[1] / df.shape[0], 4) for val in class_counts.items()}


def data_load_no_split(file_name, classifier_col, debug=False, scalar=1, make_graphs=False):
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
        sns.pairplot(df, hue=classifier_col, corner=True)
        sns.pairplot(df, hue=classifier_col, diag_kind="hist", corner=True)
        sns.pairplot(df, hue=classifier_col, kind="kde", corner=True)
        sns.pairplot(df, kind="kde", corner=True)
        sns.pairplot(df, hue=classifier_col, kind="hist", corner=True)
        sns.pairplot(df, kind="hist", corner=True)
        sns.pairplot(df, hue=classifier_col, kind="hist", corner=True)
        sns.pairplot(df, kind="hist", corner=True)
        sns.pairplot(df, hue=classifier_col, kind="reg", corner=True)
        sns.pairplot(df, kind="reg", corner=True)
        sns.pairplot(df, hue=classifier_col, kind="scatter", corner=True)
        sns.pairplot(df, kind="scatter", corner=True)
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
    return X_s, y


def data_load(file_name, classifier_col, debug=False, scalar=1, make_graphs=False, random_seed=1):
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
        sns.pairplot(df, hue=classifier_col, corner=True)
        sns.pairplot(df, hue=classifier_col, diag_kind="hist", corner=True)
        sns.pairplot(df, hue=classifier_col, kind="kde", corner=True)
        sns.pairplot(df, kind="kde", corner=True)
        sns.pairplot(df, hue=classifier_col, kind="hist", corner=True)
        sns.pairplot(df, kind="hist", corner=True)
        sns.pairplot(df, hue=classifier_col, kind="hist", corner=True)
        sns.pairplot(df, kind="hist", corner=True)
        sns.pairplot(df, hue=classifier_col, kind="reg", corner=True)
        sns.pairplot(df, kind="reg", corner=True)
        sns.pairplot(df, hue=classifier_col, kind="scatter", corner=True)
        sns.pairplot(df, kind="scatter", corner=True)
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
    X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.3, random_state=random_seed, stratify=y,
                                                        shuffle=True)

    return X_train, X_test, y_train, y_test


def save_gridsearch_to_csv(cvres, algo, filename, scalar, solver=''):
    gs_df = pd.DataFrame(cvres)
    # mtrs = gs_df["mean_train_score"]
    mtes = gs_df["mean_test_score"]
    # msrs = gs_df["std_train_score"]
    mses = gs_df["std_test_score"]
    gs_dfp = pd.DataFrame(cvres["params"])

    # out_df = pd.concat([mtrs, mtes, msrs, mses, gs_dfp], axis=1, sort=False)
    out_df = pd.concat([mtes, mses, gs_dfp], axis=1, sort=False)

    if solver:
        out_df.to_csv("ParamTests\\" + algo + "-" + solver + "-" + filename + "-" + str(scalar) + ".csv")
    else:
        out_df.to_csv("ParamTests\\" + algo + "-" + filename + "-" + str(scalar) + ".csv")


# Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, algo, filename, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5), debug=False, rotatex=False):
    # plt.title("{0} Learning Curves of {1}".format(algo, filename))

    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples", size=15)
    plt.ylabel("Score", size=15)

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True, verbose=debug)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    plt.grid()
    plt.title(algo + ' Learning Curve', fontsize=20)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.xticks(rotation=45, size=20)
    plt.yticks(size=20)
    plt.legend(loc="best", fontsize=18)
    plt.tight_layout()
    plt.savefig('Images\\' + algo + '-' + filename + '-LearningCurveLC.png')
    plt.show()

    # Plot n_samples vs fit_times
    plt.grid()
    plt.plot(train_sizes, fit_times_mean, 'o-')
    plt.fill_between(train_sizes, fit_times_mean - fit_times_std,
                     fit_times_mean + fit_times_std, alpha=0.1)
    plt.xlabel("Training examples", size=15)
    plt.ylabel("fit_times", size=15)
    # plt.title("Scalability of the model")
    plt.xticks(rotation=45, size=20)
    plt.yticks(size=20)
    plt.tight_layout()
    plt.savefig('Images\\' + algo + '-' + filename + '-LearningCurveScaleofModel.png')
    plt.show()

    # Plot fit_time vs score
    plt.grid()
    plt.plot(fit_times_mean, test_scores_mean, 'o-')
    plt.fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1)
    plt.xlabel("fit_times", size=15)
    plt.ylabel("Score", size=15)
    # plt.title("Performance of the model")
    plt.xticks(rotation=45, size=20)
    plt.yticks(size=20)
    plt.tight_layout()
    plt.savefig('Images\\' + algo + '-' + filename + '-LearningCurvePerfofModel.png')

    plt.show()


def save_gridsearch_to_csv(cvres, algo, filename, scalar, solver=''):
    gs_df = pd.DataFrame(cvres)
    mtrs = gs_df["mean_train_score"]
    mtes = gs_df["mean_test_score"]
    msrs = gs_df["std_train_score"]
    mses = gs_df["std_test_score"]
    gs_dfp = pd.DataFrame(cvres["params"])

    out_df = pd.concat([mtrs, mtes, msrs, mses, gs_dfp], axis=1, sort=False)
    # out_df = pd.concat([mtes, mses, gs_dfp], axis=1, sort=False)

    if solver:
        out_df.to_csv("ParamTests\\" + algo + "-" + solver + "-" + filename + "-" + str(scalar) + ".csv")
    else:
        out_df.to_csv("ParamTests\\" + algo + "-" + filename + "-" + str(scalar) + ".csv")
