
import mlrose_hiive as mlrose
from sklearn.metrics import roc_auc_score
import time
import util
from matplotlib import pyplot as plt

import pandas as pd


def NN_GA(file_name, classifier_col):
    X_train, X_test, y_train, y_test = util.data_load(file_name, classifier_col)
    activation = ['relu']
    learning_rate = [5, 0.01, 0.1, 1, 2, 3, 4, 7, 10]
    algorithim = 'genetic_alg'
    iters = [1000, 10000, 50000, 100000]
    nodes = [128, 128, 128, 128]
    population = [2000, 2100,2200,2300]
    mutation = [ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1]
    outcomes = []
    max_attempts = [10, 50, 100, 200, 500, 1000]
    clips = [5, 10, 100, 1000, 10000, 100000]

    # GA HPs
    nodes = [128, 128, 128, 128]
    act = 'relu'
    seed = 1
    ga_algo = 'genetic_alg'
    ga_lr = 5
    ga_iter = 100
    ga_pop = 1500
    ga_mut = 0.1
    ga_ma = 100
    ga_clip = 5

    Population = [100, 200, 300, 400, 500, 750, 1000, 1240, 1500]
    plt.figure()
    for p in Population:
        print('Population', p)
        ga_nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act, max_iters=ga_iter,
                                        algorithm=ga_algo, pop_size=p, mutation_prob=ga_mut,
                                        bias=True, is_classifier=True, learning_rate=ga_lr,
                                        early_stopping=True, clip_max=ga_clip, max_attempts=ga_ma,
                                        random_state=seed, curve=True)
        ga_nn_model.fit(X_train, y_train)
        plt.plot(ga_nn_model.fitness_curve, label='pop =' + str(p))

    plt.title("NN GA - Population")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig("Images\\NN - GA - Population")
    plt.xscale('log')
    plt.savefig("Images\\NN - GA - Population - log")
    plt.show()





if __name__ == "__main__":
    NN_GA(file_name='Mobile_Prices_orig.csv',
          classifier_col='price_range')
