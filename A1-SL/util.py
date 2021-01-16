from itertools import cycle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from scipy import interp
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import learning_curve, validation_curve, GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# Utility function to move the midpoint of a colormap to be around
# the values of interest.

# Source: https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


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


# Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_validation_curve.html
def compute_vc(algo, parameter, p_range, X_train, y_train, X_test, y_test, classifier, filename, test_class, params,
               log=False, njobs=-1, debug=False, fString=False, extraText='', rotatex=False, smalllegend=False, nolegend=False):
    train_scores, vc_scores = validation_curve(
            classifier, X_train, y_train, param_name=parameter, param_range=p_range,
            scoring="roc_auc_ovr_weighted", n_jobs=njobs, verbose=debug)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    vc_scores_mean = np.mean(vc_scores, axis=1)
    vc_scores_std = np.std(vc_scores, axis=1)
    test_class.set_params(**params)
    '''train_scores, vc_scores = validation_curve(
            other_class, X_train, y_train, param_name=parameter, param_range=p_range,
            scoring="roc_auc_ovr_weighted", n_jobs=njobs)
    train_scores_mean_o = np.mean(train_scores, axis=1)
    vc_scores_mean_o = np.mean(vc_scores, axis=1)'''

    test_scores = []

    for i in p_range:
        test_class.set_params(**{parameter: i})
        test_class.fit(X_train, y_train)
        y_prob = test_class.predict_proba(X_test)
        weighted_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
        test_scores.append(weighted_roc_auc_ovr)

    # plt.title("{0}{1} {2} Validation Curve of {3}".format(algo, extraText, parameter, filename), fontsize=10)
    plt.xlabel(parameter, size=20)
    plt.ylabel("Score", size=20)
    train_min, vc_min, test_min = min(train_scores_mean), min(vc_scores_mean), min(test_scores)
    train_max, vc_max, test_max = max(train_scores_mean), max(vc_scores_mean), max(test_scores)

    overall_min = max(min(train_min, vc_min, test_min) - 0.025, 0)
    overall_max = min(max(train_max, vc_max, test_max) + 0.05, 1.025)

    plt.ylim(overall_min, overall_max)
    lw = 2

    if p_range[-1] is tuple or fString:
        p_range = [str(ele) for ele in p_range]

    if log:
        plt.semilogx(p_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
        plt.fill_between(p_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        plt.semilogx(p_range, vc_scores_mean, label="Cross-validation score", color="navy", lw=lw)
        plt.fill_between(p_range, vc_scores_mean - vc_scores_std, vc_scores_mean + vc_scores_std, alpha=0.2,
                         color="navy", lw=lw)
        plt.semilogx(p_range, test_scores, label="Test score", color="darkred", lw=lw)
    else:
        plt.plot(p_range, train_scores_mean, label="Training score", color="darkorange", lw=lw)
        plt.fill_between(p_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
        plt.plot(p_range, vc_scores_mean, label="Cross-validation score", color="navy", lw=lw)
        plt.fill_between(p_range, vc_scores_mean - vc_scores_std, vc_scores_mean + vc_scores_std, alpha=0.2,
                         color="navy", lw=lw)
        plt.plot(p_range, test_scores, label="Test score", color="darkred", lw=lw)
    if rotatex:
        plt.xticks(rotation=45, size=20)
        plt.yticks(size=20)
    else:
        plt.xticks(size=20)
        plt.yticks(size=20)
        # plt.plot(p_range, train_scores_mean_o, label="Ent score", color="green", lw=lw)
        # plt.plot(p_range, vc_scores_mean_o, label="Ent VC score", color="pink", lw=lw)
    if not nolegend:
        if smalllegend:
            plt.legend(loc="best", fontsize=14)
        else:
            plt.legend(loc="best", fontsize=18)
    plt.tight_layout()

    plt.savefig('Images\\' + algo + '-' + extraText + parameter + '-' + filename + '.png')

    plt.show()

#Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
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

# Source: https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
def svm_rbf_C_Gamma_viz(X, y, pSVM, njobs, filename, midscore):
    C_range = np.logspace(-5, 3, 9)
    gamma_range = np.logspace(-9, 0, 10)
    # print(gamma_range)
    # print(C_range)
    param_grid = dict(gamma=gamma_range, C=C_range)
    param_grid['kernel'] = [pSVM['kernel']]
    param_grid['random_state'] = [pSVM['random_state']]
    param_grid['probability'] = [pSVM['probability']]
    param_grid['break_ties'] = [pSVM['break_ties']]

    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=10, n_jobs=njobs, verbose=1, scoring='roc_auc_ovr_weighted')
    grid.fit(X, y)

    scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                         len(gamma_range))
    # #############################################################################
    # Visualization
    #
    # draw visualization of parameter effects

    # Draw heatmap of the validation accuracy as a function of gamma and C
    #
    # The score are encoded as colors with the hot colormap which varies from dark
    # red to bright yellow. As the most interesting scores are all located in the
    # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
    # as to make it easier to visualize the small variations of score values in the
    # interesting range while not brutally collapsing all the low score values to
    # the same color.
    midscore -= 0.1
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=midscore))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.savefig('Images\\' + filename + '-SVMHeatMap.png')
    plt.show()

# Source: https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
def boost_lr_vs_nest(X, y, pBTree, njobs, filename, midscore):
    LR_range = np.logspace(-7, 0, 8)
    nest_range = [1, 10, 100,  500, 1000, 5000, 10000]
    # print(gamma_range)
    # print(C_range)
    param_grid = dict(n_estimators=nest_range, learning_rate=LR_range)
    param_grid['base_estimator__ccp_alpha'] = [pBTree['base_estimator__ccp_alpha']]
    param_grid['base_estimator__criterion'] = [pBTree['base_estimator__criterion']]
    param_grid['base_estimator__max_depth'] = [pBTree['base_estimator__max_depth']]
    param_grid['base_estimator__max_leaf_nodes'] = [pBTree['base_estimator__max_leaf_nodes']]
    param_grid['base_estimator__splitter'] = [pBTree['base_estimator__splitter']]
    param_grid['random_state'] = [pBTree['random_state']]

    DTC = DecisionTreeClassifier()

    grid = GridSearchCV(AdaBoostClassifier(base_estimator=DTC), param_grid=param_grid, cv=10, n_jobs=njobs, verbose=1,
                        scoring='roc_auc_ovr_weighted')
    grid.fit(X, y)

    scores = grid.cv_results_['mean_test_score'].reshape(len(LR_range),
                                                         len(nest_range))

    # #############################################################################
    # Visualization
    #
    # draw visualization of parameter effects

    # Draw heatmap of the validation accuracy as a function of gamma and C
    #
    # The score are encoded as colors with the hot colormap which varies from dark
    # red to bright yellow. As the most interesting scores are all located in the
    # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
    # as to make it easier to visualize the small variations of score values in the
    # interesting range while not brutally collapsing all the low score values to
    # the same color.
    midscore -= 0.1
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=midscore, vmax=1.0))
    plt.xlabel('n Estimators')
    plt.ylabel('Learning Rate')
    plt.colorbar()
    plt.xticks(np.arange(len(nest_range)), nest_range, rotation=45)
    plt.yticks(np.arange(len(LR_range)), LR_range)
    plt.title('Validation accuracy')
    plt.savefig('Images\\' + filename + '-BoostHeatMap.png')
    plt.show()

    cvres = grid.cv_results_
    print(cvres)

    save_gridsearch_to_csv(cvres, 'BTreeHeatMap', filename, '', solver='')

