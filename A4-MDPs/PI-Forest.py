import mdptoolbox.example
import numpy as np
from hiive.mdptoolbox import mdp
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib
import matplotlib.patches
import random


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
    gammas = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]
    gammas = [0.999]
    epsilons = [0.0001]

    per_won_hm = np.zeros((len(gammas), len(epsilons)))
    iters_hm = np.zeros((len(gammas), len(epsilons)))
    time_hm = np.zeros((len(gammas), len(epsilons)))

    best_rew = -1
    best_pol_arr = []
    g_cnt = 0
    e_cnt = 0
    for g in gammas:
        e_cnt = 0
        print(g)
        best_pol = []
        best_rew = -1

        for e in epsilons:
            pi = mdp.PolicyIteration(P, R, gamma=g)
            pi.run()
            rew = run_episodes(pi.policy, S, R, p, 1000, 100)
            if rew > best_rew:
                best_rew = rew
                best_pol = pi.policy
            per_won_hm[g_cnt][e_cnt] = rew
            iters_hm[g_cnt][e_cnt] = pi.iter
            time_hm[g_cnt][e_cnt] = pi.time * 1000
            e_cnt += 1
        best_pol_arr.append(list(best_pol))
        g_cnt += 1

    mean_val = [i["Mean V"] for i in pi.run_stats]
    error = [i["Error"] for i in pi.run_stats]
    reward = [i["Reward"] for i in pi.run_stats]

    fig, ax = plt.subplots()
    ax.plot(mean_val, label='Mean V')
    ax.plot(error, label='Error')
    ax.plot(reward, label='Reward')
    ax.legend()
    plt.xlabel('Iterations', fontsize=15)
    plt.ylabel('V/Error/Reward', fontsize=15)
    plt.title("Mean V/Error/ Reward vs iterations")
    plt.show()

    op_list = [list(best_pol)]
    print(best_pol_arr)

    # Plot Percent Games Won Heatmap
    fig, ax = plt.subplots()

    im, cbar = heatmap(per_won_hm, gammas, epsilons, ax=ax,
                       cmap="YlGn", cbarlabel="Average Reward")
    texts = annotate_heatmap(im, valfmt="{x:.0f}")

    fig.tight_layout()
    plt.savefig('Images\\PI-Forest-Per_Heatmap-' + str(size) + '.png')
    plt.show()

    # Plot Iterations Heatmap
    fig, ax = plt.subplots()

    im, cbar = heatmap(iters_hm, gammas, epsilons, ax=ax,
                       cmap="YlGn", cbarlabel="# of Iterations to Convergence")
    texts = annotate_heatmap(im, valfmt="{x:.0f}")

    fig.tight_layout()
    plt.savefig('Images\\PI-Forest-Iter_Heatmap-' + str(size) + '.png')
    plt.show()

    # Plot Run time Heatmap
    fig, ax = plt.subplots()

    im, cbar = heatmap(time_hm, gammas, epsilons, ax=ax,
                       cmap="YlGn", cbarlabel="Runtime (ms)")
    texts = annotate_heatmap(im, valfmt="{x:.0f}")

    fig.tight_layout()
    plt.savefig('Images\\PI-Forest-Time_Heatmap-' + str(size) + '.png')
    plt.show()

    # Plot out optimal policy
    # Citation: https://stackoverflow.com/questions/52566969/python-mapping-a-2d-array-to-a-grid-with-pyplot
    cmap = colors.ListedColormap(['blue', 'red'])
    fig, ax = plt.subplots(figsize=(12, 4))
    plt.title("Forest PI Policy - Red = Cut, Blue = Wait")
    gammas.reverse()
    ax.set_yticklabels(gammas, fontsize=15)
    plt.xticks(fontsize=15)
    ax.tick_params(left=False)  # remove the ticks
    plt.xlabel('State', fontsize=15)
    plt.ylabel('Gamma', fontsize=15)
    plt.pcolor(best_pol_arr[::-1], cmap=cmap, edgecolors='k', linewidths=0)
    plt.savefig('Images\\PI-Forest-Heatmap-' + str(size) + '.png')
    plt.show()


run_forest(10)
run_forest(1000)
