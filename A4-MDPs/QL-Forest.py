import mdptoolbox.example
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import colors
from hiive.mdptoolbox import mdp
import matplotlib
import matplotlib.patches
import random
import time

# Citation: https://learning.oreilly.com/library/view/reinforcement-learning-algorithms/9781789131116/7c6dfed0-1180-49fe-84a0-ea62131b5947.xhtml

# Citation: https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html
def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")
    plt.title('epsilon')
    plt.ylabel('gamma')

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=["black", "white"], threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def run_episodes(policy, S, R, p, num_episodes, num_resets):
    rew_arr = []
    for y in range(num_resets):
        forest_state = 0
        tot_rew = 0
        for x in range(num_episodes):
            forest_state = min(forest_state, S - 1)
            if np.random.rand(1) <= p:
                forest_state = -1
            else:
                tot_rew += R[forest_state][policy[forest_state]]
                if policy[forest_state] == 1:
                    forest_state = -1
            forest_state += 1
        rew_arr.append(tot_rew)
    return np.mean(rew_arr)


def run_forest(size):
    seed_val = 42
    np.random.seed(seed_val)
    random.seed(seed_val)

    S = size
    r1 = 10  # The reward when the forest is in its oldest state and action ‘Wait’ is performed
    r2 = 50  # The reward when the forest is in its oldest state and action ‘Cut’ is performed
    p = 0.1

    P, R = mdptoolbox.example.forest(S=S, r1=r1, r2=r2, p=p)  # Defaults left the same

    epsilons = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    epsilons = [0.00001, 0.000001]
    gammas = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999]

    learning_rates = [0.001, 0.01, 0.00001, 0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0]
    lr_decays = [1.0, 0.99, 0.9999, 0.999, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    lr_mins = [0.00001, 0.0001, 0.001, 0.01, 0]
    epsilons = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    epsilon_decays = [0.99, 0.9999, 0.99999, 0.999999, 0.999, 0.9, 0.8, 0.7]
    epsilon_mins = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0]

    best_lr, best_e, best_g, best_ed, best_em, best_rew = 0, 0, 0, 0, 0, -1

    '''for em in epsilon_mins:
        for am in lr_mins:
            for ad in lr_decays:
                for e in epsilons:
                    for g in gammas:
                        for a in learning_rates:
                            for ed in epsilon_decays:
                                pi = mdp.QLearning(P, R, gamma=g, epsilon=e, epsilon_decay=ed, epsilon_min=em, n_iter=10000,
                                                   alpha=a, alpha_min=am, alpha_decay=ad)
                                pi.run()
                                rew = run_episodes(pi.policy, S, R, p, 1000, 100)
                                print(rew, '-', e, ed, em, a, ad, am, g)'''

    # g	    e       	ed	    em  	a  	ad	am	    rew'''
    tests = [[0.1, 0.000001, 0.99, 0.0001, 0.6, 0.5, 0.001]]
    # g	    e       	ed	    em  	a  	ad	am	    rew
    # 0.1	1.00E-06	0.99	0.0001	0.6	0.5	0.001	4032

    # 0.1	1.00E-06	0.99	1.00E-05	0.001	0.5	0.01	429.2

    if size < 100:
        tests = [[0.1, 1.0, 0.7, 0.00001, 0.0001, 1.0, 0.00001]]
    else:
        tests = [[0.6, 1.0, 0.999999, 0.00001, 0.8, 1.0, 0.01]]

    if 1 == 1:
        # print(e, ed, em, a, ad, am, g, rew, )
        best_pol_arr = []

        print(size)
        for t in tests:
            for e in epsilons:
                Q_qlearning = mdp.QLearning(P, R, gamma=t[0], epsilon=e, epsilon_decay=t[2], epsilon_min=t[3], n_iter=10000,
                                            alpha=t[4], alpha_min=t[5], alpha_decay=t[6])
                Q_qlearning.run()
                best_pol_arr.append(list(Q_qlearning.policy))
                #print(run_episodes(Q_qlearning.policy, S, R, p, 100000, 100))


        # Plot out optimal policy
        # Citation: https://stackoverflow.com/questions/52566969/python-mapping-a-2d-array-to-a-grid-with-pyplot
        print(epsilons)
        cmap = colors.ListedColormap(['blue', 'red'])
        fig, ax = plt.subplots(figsize=(12, 3.5))
        plt.title("Forest Q-Learning Policy - Red = Cut, Blue = Wait")
        epsilons.reverse()

        plt.xticks(fontsize=15)
        plt.xlabel('State', fontsize=15)
        plt.ylabel('Epsilon', fontsize=15)
        plt.pcolor(best_pol_arr[::-1], cmap=cmap, edgecolors='k', linewidths=0)
        ax.set_yticklabels(epsilons, fontsize=15)
        ax.tick_params(left=False)  # remove the ticks
        plt.savefig('Images\\QL-Forest-Policy-' + str(size) + '.png')

        plt.show()

        mean_val = [i["Mean V"] for i in Q_qlearning.run_stats]
        error = [i["Error"] for i in Q_qlearning.run_stats]
        reward = [i["Reward"] for i in Q_qlearning.run_stats]

        # Plot Delta vs iterations
        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_ylabel('Reward/Error', color=color)
        ax1.semilogy(error, color=color, label='Error')
        ax1.semilogy(reward, color='darkblue', label='Reward')
        ax1.legend()
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:red'
        ax2.set_xlabel('Iterations')
        ax2.set_ylabel('Mean V', color=color)
        ax2.semilogy(mean_val, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        ax2.tick_params(axis='y', labelcolor=color)

        plt.title('V/Reward/Error vs. Iterations')
        plt.savefig('Images\\QL-Forest-RunStats' + str(size) + '.png')
        plt.show()



        best_rew = 0
        for em in epsilon_mins:
            for am in lr_mins:
                for ad in lr_decays:
                    for e in epsilons:
                        for g in gammas:
                            for a in learning_rates:
                                for ed in epsilon_decays:
                                    Q_qlearning = mdp.QLearning(P, R, gamma=g, epsilon=e, epsilon_decay=ed,
                                                                epsilon_min=em, n_iter=10000,
                                                                alpha=a, alpha_min=am, alpha_decay=ad)
                                    Q_qlearning.run()
                                    rew = run_episodes(Q_qlearning.policy, S, R, p, 1000, 200)
                                    if rew > best_rew:
                                        best_rew = rew
                                        print(e, ed, em, a, ad, am, g, rew, )

    for t in tests:

        num_seeds = 10

        for g in gammas:
            tot_rew = 0
            cnt = 0
            for x in range(num_seeds):
                cnt += 1
                seed_val = x
                np.random.seed(seed_val)
                random.seed(seed_val)
                Q_qlearning = mdp.QLearning(P, R, gamma=g, epsilon=t[1], epsilon_decay=t[2], epsilon_min=t[3],
                                            n_iter=10000,
                                            alpha=t[4], alpha_min=t[5], alpha_decay=t[6])
                Q_qlearning.run()
                rew = run_episodes(Q_qlearning.policy, S, R, p, 1000, 100)
                tot_rew += rew
                print('g', x, g, rew, Q_qlearning.run_stats[-1]['Mean V'])



        for em in epsilon_mins:
            tot_rew = 0
            cnt = 0
            for x in range(num_seeds):
                cnt += 1
                seed_val = x
                np.random.seed(seed_val)
                random.seed(seed_val)
                Q_qlearning = mdp.QLearning(P, R, gamma=t[0], epsilon=t[1], epsilon_decay=t[2], epsilon_min=em,
                                            n_iter=10000,
                                            alpha=t[4], alpha_min=t[5], alpha_decay=t[6])
                Q_qlearning.run()
                rew = run_episodes(Q_qlearning.policy, S, R, p, 1000, 100)
                tot_rew += rew
                print('em', x, em, rew, Q_qlearning.run_stats[-1]['Mean V'])


        for e in epsilons:
            tot_rew = 0
            cnt = 0
            for x in range(num_seeds):
                cnt += 1
                seed_val = x
                np.random.seed(seed_val)
                random.seed(seed_val)
                Q_qlearning = mdp.QLearning(P, R, gamma=t[0], epsilon=e, epsilon_decay=t[2], epsilon_min=t[3],
                                            n_iter=10000, alpha=t[4], alpha_min=t[5], alpha_decay=t[6])
                Q_qlearning.run()
                #print(Q_qlearning.policy)
                rew = run_episodes(Q_qlearning.policy, S, R, p, 1000, 100)
                tot_rew += rew
                print('e', x, e, rew, Q_qlearning.run_stats[-1]['Mean V'])


        for lr in learning_rates:
            tot_rew = 0
            cnt = 0
            for x in range(num_seeds):
                cnt += 1
                seed_val = x
                np.random.seed(seed_val)
                random.seed(seed_val)
                Q_qlearning = mdp.QLearning(P, R, gamma=t[0], epsilon=t[1], epsilon_decay=t[2], epsilon_min=t[3],
                                            n_iter=10000,
                                            alpha=lr, alpha_min=t[5], alpha_decay=t[6])
                Q_qlearning.run()
                rew = run_episodes(Q_qlearning.policy, S, R, p, 1000, 100)
                tot_rew += rew
                print('lr', x, lr, rew, Q_qlearning.run_stats[-1]['Mean V'])

        for ld in lr_decays:
            tot_rew = 0
            cnt = 0
            for x in range(num_seeds):
                cnt += 1
                seed_val = x
                np.random.seed(seed_val)
                random.seed(seed_val)
                Q_qlearning = mdp.QLearning(P, R, gamma=t[0], epsilon=t[1], epsilon_decay=t[2], epsilon_min=t[3],
                                            n_iter=10000,
                                            alpha=t[4], alpha_min=t[5], alpha_decay=ld)
                Q_qlearning.run()
                rew = run_episodes(Q_qlearning.policy, S, R, p, 1000, 100)
                tot_rew += rew
                print('ld', x, ld, rew, Q_qlearning.run_stats[-1]['Mean V'])

        for lm in lr_mins:
            tot_rew = 0
            cnt = 0
            for x in range(num_seeds):
                cnt += 1
                seed_val = x
                np.random.seed(seed_val)
                random.seed(seed_val)
                Q_qlearning = mdp.QLearning(P, R, gamma=t[0], epsilon=t[1], epsilon_decay=t[2], epsilon_min=t[3],
                                            n_iter=10000,
                                            alpha=t[4], alpha_min=lm, alpha_decay=t[6])
                Q_qlearning.run()
                rew = run_episodes(Q_qlearning.policy, S, R, p, 1000, 100)
                tot_rew += rew
                print('lm', x, lm, rew, Q_qlearning.run_stats[-1]['Mean V'])

        for e in epsilons:
            tot_rew = 0
            cnt = 0
            for x in range(num_seeds):
                cnt += 1
                seed_val = x
                np.random.seed(seed_val)
                random.seed(seed_val)
                Q_qlearning = mdp.QLearning(P, R, gamma=t[0], epsilon=e, epsilon_decay=t[2], epsilon_min=t[3],
                                            n_iter=10000, alpha=t[4], alpha_min=t[5], alpha_decay=t[6])
                Q_qlearning.run()
                rew = run_episodes(Q_qlearning.policy, S, R, p, 1000, 100)
                tot_rew += rew
                print('e', x, e, rew, Q_qlearning.run_stats[-1]['Mean V'])

        for ed in epsilon_decays:
            tot_rew = 0
            cnt = 0
            for x in range(num_seeds):
                cnt += 1
                seed_val = x
                np.random.seed(seed_val)
                random.seed(seed_val)
                Q_qlearning = mdp.QLearning(P, R, gamma=t[0], epsilon=t[1], epsilon_decay=ed, epsilon_min=t[3],
                                            n_iter=10000,
                                            alpha=t[4], alpha_min=t[5], alpha_decay=t[6])
                Q_qlearning.run()
                rew = run_episodes(Q_qlearning.policy, S, R, p, 1000, 100)
                tot_rew += rew
                print('ed', x, ed, rew, Q_qlearning.run_stats[-1]['Mean V'])

        for em in epsilon_mins:
            tot_rew = 0
            cnt = 0
            for x in range(num_seeds):
                cnt += 1
                seed_val = x
                np.random.seed(seed_val)
                random.seed(seed_val)
                Q_qlearning = mdp.QLearning(P, R, gamma=t[0], epsilon=t[1], epsilon_decay=t[2], epsilon_min=em,
                                            n_iter=10000,
                                            alpha=t[4], alpha_min=t[5], alpha_decay=t[6])
                Q_qlearning.run()
                rew = run_episodes(Q_qlearning.policy, S, R, p, 1000, 100)
                tot_rew += rew
                print('em', x, em, rew, Q_qlearning.run_stats[-1]['Mean V'])




run_forest(10)
# 10000 iters
# g	    e       	ed	    em  	a  	ad	am	    rew
# 0.1	1.00E-06	0.99	0.0001	0.6	0.5	0.001	4032
# 0.1	1.00E-06	0.99	1.00E-05	1	0.3	0.0001	4028
# 0.1	1.00E-06	0.99	1.00E-05	0.6	0.4	0.001	4022
# 0.1	1.00E-06	0.99	0.0001	1.00E-05	0.3	1.00E-05	4014
# 0.1	1.00E-06	0.99	1.00E-05	0.05	0.2	0	4000
# 0.1	1.00E-06	0.99	0.0001	0.7	0.7	0.0001	3984
# 0.1	1.00E-06	0.99	0.0001	0.3	0.5	0	3982

run_forest(1000)
# 10000 iters
# g	e	ed	em	a	ad	am	rew
# 0.1	1.00E-06	0.99	1.00E-05	0.01	0.1	0.001	431.4
# 0.1	1.00E-06	0.99	1.00E-05	0.001	0.5	0.01	429.2
# 0.1	1.00E-06	0.99	1.00E-05	0.01	0.1	1.00E-05	426.4
# 0.1	1.00E-06	0.99	1.00E-05	0.01	0.99	0.0001	425.6
# 0.1	1.00E-06	0.99	1.00E-05	0.001	0.1	0.01	424.4
# 0.1	1.00E-06	0.99	1.00E-05	0.01	0.4	0.001	423
# 0.1	1.00E-06	0.99	1.00E-05	0.01	0.8	0.0001	420
