import mlrose_hiive as mlrose
from sklearn.metrics import roc_auc_score
import time
import util
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def NN_GA(file_name, classifier_col):
    # HPs shared by all
    nodes = [128, 128, 128, 128]

    act = 'relu'
    seed = 1

    # GD HPs
    gd_algo = 'gradient_descent'
    gd_lr = 0.00000009
    gd_iter = 10000
    gd_ma = 50
    gd_clip = 5

    # RHC HPs
    rhc_algo = 'random_hill_climb'
    rhc_lr = 8
    rhc_iter = 10000
    rhc_restarts = 10
    rhc_ma = 100
    rhc_clip = 100

    # SA HPs
    sa_algo = 'simulated_annealing'
    sa_lr = 10
    sa_iter = 10000
    sa_temp = 10000
    sa_decay = 0.92
    sa_ma = 50
    sa_clip = 10

    # GA HPs
    ga_algo = 'genetic_alg'
    ga_lr = 5
    ga_iter = 100
    ga_pop = 1500
    ga_mut = 0.1
    ga_ma = 100
    ga_clip = 5

    X_train, X_test, y_train, y_test = util.data_load(file_name, classifier_col, random_seed=seed)

    # Best GD Algorithm
    print('Training GD NN')
    gd_start = time.time()
    gd_nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act, random_state=seed, bias=True,
                                       is_classifier=True, early_stopping=True, curve=True, algorithm=gd_algo,
                                       max_iters=gd_iter, learning_rate=gd_lr, clip_max=gd_clip, max_attempts=gd_ma)
    gd_nn_model.fit(X_train, y_train)
    print('gd loss:', gd_nn_model.loss)

    gd_train_time = time.time() - gd_start
    print('gd_train_time:', gd_train_time)

    start = time.time()
    gd_y_train_pred = gd_nn_model.predict(X_train)
    gd_y_train_roc = roc_auc_score(y_train, gd_y_train_pred, multi_class="ovr", average="weighted")
    gd_y_train_query_time = time.time() - start
    print('gd_y_train_roc', gd_y_train_roc, 'gd_y_train_roc: ', gd_y_train_query_time)

    start = time.time()
    gd_y_test_pred = gd_nn_model.predict(X_test)
    gd_y_test_roc = roc_auc_score(y_test, gd_y_test_pred, multi_class="ovr", average="weighted")
    gd_y_test_query_time = time.time() - start
    print('gd_y_test_roc', gd_y_test_roc, 'gd_y_test_query_time: ', gd_y_test_query_time)

    # Best RHC Algorithm
    print('Training RHC NN')
    rhc_start = time.time()
    rhc_nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act, random_state=seed, bias=True,
                                        is_classifier=True, early_stopping=True, curve=True, algorithm=rhc_algo,
                                        max_iters=rhc_iter, learning_rate=rhc_lr, clip_max=rhc_clip,
                                        max_attempts=rhc_ma, restarts=rhc_restarts)
    rhc_nn_model.fit(X_train, y_train)
    print('rhc loss:', rhc_nn_model.loss)

    rhc_train_time = time.time() - rhc_start
    print('rhc_train_time:', rhc_train_time)

    start = time.time()
    rhc_y_train_pred = rhc_nn_model.predict(X_train)
    rhc_y_train_roc = roc_auc_score(y_train, rhc_y_train_pred, multi_class="ovr", average="weighted")
    rhc_y_train_query_time = time.time() - start
    print('rhc_y_train_roc', rhc_y_train_roc, 'rhc_y_train_roc: ', rhc_y_train_query_time)

    start = time.time()
    rhc_y_test_pred = rhc_nn_model.predict(X_test)
    rhc_y_test_roc = roc_auc_score(y_test, rhc_y_test_pred, multi_class="ovr", average="weighted")
    rhc_y_test_query_time = time.time() - start
    print('rhc_y_test_roc', rhc_y_test_roc, 'rhc_y_test_query_time: ', rhc_y_test_query_time)

    # Best SA Algorithm
    print('Training SA NN')
    sa_start = time.time()
    sa_nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act, random_state=seed, bias=True,
                                       is_classifier=True, early_stopping=True, curve=True, algorithm=sa_algo,
                                       max_iters=sa_iter, learning_rate=sa_lr, clip_max=sa_clip, max_attempts=sa_ma,
                                       schedule=mlrose.GeomDecay(init_temp=sa_temp, decay=sa_decay))
    sa_nn_model.fit(X_train, y_train)
    print('sa loss:', sa_nn_model.loss)

    sa_train_time = time.time() - sa_start
    print('sa_train_time:', sa_train_time)

    start = time.time()
    sa_y_train_pred = sa_nn_model.predict(X_train)
    sa_y_train_roc = roc_auc_score(y_train, sa_y_train_pred, multi_class="ovr", average="weighted")
    sa_y_train_query_time = time.time() - start
    print('sa_y_train_roc', sa_y_train_roc, 'sa_y_train_roc: ', sa_y_train_query_time)

    start = time.time()
    sa_y_test_pred = sa_nn_model.predict(X_test)
    sa_y_test_roc = roc_auc_score(y_test, sa_y_test_pred, multi_class="ovr", average="weighted")
    sa_y_test_query_time = time.time() - start
    print('sa_y_test_roc', sa_y_test_roc, 'sa_y_test_query_time: ', sa_y_test_query_time)

    # Best Genetic Algorithm
    print('Training GA NN')
    ga_start = time.time()
    ga_nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act, random_state=seed, bias=True,
                                       is_classifier=True, early_stopping=True, curve=True, algorithm=ga_algo,
                                       max_iters=ga_iter, learning_rate=ga_lr, clip_max=ga_clip, max_attempts=ga_ma,
                                       pop_size=ga_pop, mutation_prob=ga_mut)
    ga_nn_model.fit(X_train, y_train)
    print('ga loss:', ga_nn_model.loss)

    ga_train_time = time.time() - ga_start
    print('ga_train_time:', ga_train_time)

    start = time.time()
    ga_y_train_pred = ga_nn_model.predict(X_train)
    ga_y_train_roc = roc_auc_score(y_train, ga_y_train_pred, multi_class="ovr", average="weighted")
    ga_y_train_query_time = time.time() - start
    print('ga_y_train_roc', ga_y_train_roc, 'ga_y_train_roc: ', ga_y_train_query_time)

    start = time.time()
    ga_y_test_pred = ga_nn_model.predict(X_test)
    ga_y_test_roc = roc_auc_score(y_test, ga_y_test_pred, multi_class="ovr", average="weighted")
    ga_y_test_query_time = time.time() - start
    print('ga_y_test_roc', ga_y_test_roc, 'ga_y_test_query_time: ', ga_y_test_query_time)

    # Plot Loss Curves
    plt.figure()

    plt.plot(ga_nn_model.fitness_curve, label='GA Loss Curve')
    plt.plot(sa_nn_model.fitness_curve, label='SA Loss Curve')
    plt.plot(gd_nn_model.fitness_curve, label='GD Loss Curve')
    plt.plot(rhc_nn_model.fitness_curve, label='RHC Loss Curve')

    plt.title("Neural Network Loss Curves")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')

    plt.savefig("Images\\Neural Network Loss Curves")
    plt.show()

    # Plot Loss Curves (GD Inverted)
    plt.figure()
    gd_curve = gd_nn_model.fitness_curve
    inverted_gd_curve = np.array(gd_curve) * -1

    plt.plot(ga_nn_model.fitness_curve, label='GA Loss Curve')
    plt.plot(sa_nn_model.fitness_curve, label='SA Loss Curve')
    plt.plot(inverted_gd_curve, label='GD Loss Curve')
    plt.plot(rhc_nn_model.fitness_curve, label='RHC Loss Curve')

    plt.title("Neural Network Loss Curves")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')

    plt.savefig("Images\\Neural Network Loss Curves-inverted GD")
    plt.show()

    # Plot Loss Curves - No GD
    plt.figure()

    plt.plot(ga_nn_model.fitness_curve, label='GA Loss Curve')
    plt.plot(sa_nn_model.fitness_curve, label='SA Loss Curve')
    plt.plot(rhc_nn_model.fitness_curve, label='RHC Loss Curve')

    plt.title("Neural Network Loss Curves")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.legend()
    plt.xscale('log')

    plt.savefig("Images\\Neural Network Loss Curves - No GD")
    plt.show()


if __name__ == "__main__":
    NN_GA(file_name='Mobile_Prices_orig.csv',
          classifier_col='price_range')
