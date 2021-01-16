import mlrose_hiive as mlrose
from sklearn.metrics import roc_auc_score
import time
import util
from matplotlib import pyplot as plt

import pandas as pd


def NN_SA(file_name, classifier_col):
    X_train, X_test, y_train, y_test = util.data_load(file_name, classifier_col)

    # SA HPs
    nodes = [128, 128, 128, 128]

    act = 'relu'
    seed = 1
    sa_algo = 'simulated_annealing'
    sa_lr = 10
    sa_iter = 10000
    sa_temp = 10000
    sa_decay = 0.92
    sa_ma = 50
    sa_clip = 10

    temperature = [0.1, 1, 10, 100, 1000, 10000]
    plt.figure()
    for t in temperature:
        print('temperature', t)
        sa_nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act, random_state=seed, bias=True,
                                           is_classifier=True, early_stopping=True, curve=True, algorithm=sa_algo,
                                           max_iters=sa_iter, learning_rate=sa_lr, clip_max=sa_clip, max_attempts=sa_ma,
                                           schedule=mlrose.GeomDecay(init_temp=t, decay=sa_decay))
        sa_nn_model.fit(X_train, y_train)
        plt.plot(sa_nn_model.fitness_curve, label='temp =' + str(t))

    plt.title("NN SA - Temperature")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig("Images\\NN - SA - Temperature")
    plt.xscale('log')
    plt.savefig("Images\\NN - SA - Temperature - log")
    plt.show()

    decay_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.92]
    plt.figure()
    for dr in decay_rates:
        print('decay', dr)
        sa_nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act, random_state=seed, bias=True,
                                           is_classifier=True, early_stopping=True, curve=True, algorithm=sa_algo,
                                           max_iters=sa_iter, learning_rate=sa_lr, clip_max=sa_clip, max_attempts=sa_ma,
                                           schedule=mlrose.GeomDecay(init_temp=sa_temp, decay=dr))
        sa_nn_model.fit(X_train, y_train)
        plt.plot(sa_nn_model.fitness_curve, label='decay rate =' + str(dr))

    plt.title("NN SA - Decay Rate")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig("Images\\NN - SA - Decay Rate")
    plt.xscale('log')
    plt.savefig("Images\\NN - SA - Decay Rate - log")
    plt.show()


if __name__ == "__main__":
    NN_SA(file_name='Mobile_Prices_orig.csv',
          classifier_col='price_range')
