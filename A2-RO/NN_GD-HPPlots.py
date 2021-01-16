import mlrose_hiive as mlrose
from sklearn.metrics import roc_auc_score
import time
import util
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def NN_GD(file_name, classifier_col):
    X_train, X_test, y_train, y_test = util.data_load(file_name, classifier_col)

    # GD HPs
    nodes = [128, 128, 128, 128]

    act = 'relu'
    seed = 1
    gd_algo = 'gradient_descent'
    gd_lr = 0.00000009
    gd_iter = 10000
    gd_ma = 50
    gd_clip = 5

    learning_rates = [0.00000009, 0.00000001, 0.000000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    plt.figure()
    for lr in learning_rates:
        print('lr', lr)
        gd_nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act, random_state=seed, bias=True,
                                           is_classifier=True, early_stopping=True, curve=True, algorithm=gd_algo,
                                           max_iters=gd_iter, learning_rate=lr, clip_max=gd_clip, max_attempts=gd_ma)
        gd_nn_model.fit(X_train, y_train)
        gd_curve = gd_nn_model.fitness_curve

        inverted_gd_curve = np.array(gd_curve) * -1
        plt.plot(inverted_gd_curve, label='lr =' + str(lr))

    plt.title("NN GD - Learning Rates")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig("Images\\NN - GD - Learning Rates")
    plt.xscale('log')
    plt.savefig("Images\\NN - GD - Learning Rates - log")
    plt.show()



if __name__ == "__main__":
    NN_GD(file_name='Mobile_Prices_orig.csv',
          classifier_col='price_range')
