import mlrose_hiive as mlrose
from sklearn.metrics import roc_auc_score
import time
import util

import pandas as pd


def NN_GA(file_name, classifier_col):
    X_train, X_test, y_train, y_test = util.data_load(file_name, classifier_col)
    activation = ['relu']
    learning_rate = [5, 0.01, 0.1, 1, 2, 3, 4, 7, 10]
    algorithim = 'genetic_alg'
    iters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    nodes = [128, 128, 128, 128]
    population = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    mutation = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1]
    outcomes = []
    max_attempts = [10, 50, 100, 200, 500, 1000]
    clips = [5, 10, 100, 1000, 10000, 100000]

    act = 'relu'
    lr = 5
    itera = 100
    pop = 1500
    mut = 0.1
    ma = 100
    clip = 5
    seed = 1

    while 1 == 1:

        iters_outs = {}

        for iter_test in iters:
            start = time.time()

            print(algorithim, act, lr, iter_test, ' ', pop, mut, ma, clip)
            nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act, max_iters=iter_test,
                                            algorithm=algorithim, pop_size=pop, mutation_prob=mut,
                                            bias=True, is_classifier=True, learning_rate=lr,
                                            early_stopping=True, clip_max=clip, max_attempts=ma,
                                            random_state=seed, curve=True)
            nn_model.fit(X_train, y_train)
            train_time = time.time() - start
            print('Train time', train_time)

            start = time.time()
            y_train_pred = nn_model.predict(X_train)
            y_train_roc = roc_auc_score(y_train, y_train_pred, multi_class="ovr", average="weighted")
            print('y_train_roc', y_train_roc)

            y_train_query_time = time.time() - start
            print('y_train_query_time', y_train_query_time)

            start = time.time()
            y_test_pred = nn_model.predict(X_test)
            y_test_roc = roc_auc_score(y_test, y_test_pred, multi_class="ovr", average="weighted")
            print('y_test_roc', y_test_roc)

            y_test_query_time = time.time() - start
            print('y_test_query_time', y_test_query_time)
            nn_loss = nn_model.loss
            print('loss', nn_loss)
            outcome = {}
            outcome['schedule'] = ' '
            outcome['activation'] = act
            outcome['learning_rate'] = lr
            outcome['max_iters'] = iter_test
            outcome['population'] = pop
            outcome['mutation'] = mut
            outcome['max_attempts'] = ma
            outcome['clip'] = clip
            outcome['y_train_roc'] = y_train_roc
            outcome['y_test_roc'] = y_test_roc
            outcome['runtime'] = train_time + y_train_query_time + y_test_query_time
            outcome['Train time'] = train_time
            outcome['y_train_query_time'] = y_train_query_time
            outcome['y_test_query_time'] = y_test_query_time
            outcome['loss'] = nn_loss
            outcomes.append(outcome)
            pd.DataFrame(outcomes).to_csv('NN-GA-iteretsets.csv', mode='a', header=False)
            iters_outs[iter_test] = y_test_roc

        old_val = itera
        itera = max(iters_outs, key=iters_outs.get)
        print('best iter', itera, 'old', old_val)
        raise SystemExit(0)

        mut_outs = {}

        for mut_test in mutation:
            start = time.time()

            print(algorithim, act, lr, itera, ' ', pop, mut_test, ma, clip)
            nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act, max_iters=itera,
                                            algorithm=algorithim, pop_size=pop, mutation_prob=mut_test,
                                            bias=True, is_classifier=True, learning_rate=lr,
                                            early_stopping=True, clip_max=clip, max_attempts=ma,
                                            random_state=seed, curve=True)
            nn_model.fit(X_train, y_train)
            y_train_pred = nn_model.predict(X_train)
            y_train_roc = roc_auc_score(y_train, y_train_pred, multi_class="ovr", average="weighted")
            print('y_train_roc', y_train_roc)

            y_test_pred = nn_model.predict(X_test)
            y_test_roc = roc_auc_score(y_test, y_test_pred, multi_class="ovr", average="weighted")
            print('y_test_roc', y_test_roc)

            runtime = time.time() - start
            print('curr run time', time.time() - start)

            outcome = {}
            outcome['schedule'] = ' '
            outcome['activation'] = act
            outcome['learning_rate'] = lr
            outcome['max_iters'] = itera
            outcome['population'] = pop
            outcome['mutation'] = mut_test
            outcome['max_attempts'] = ma
            outcome['clip'] = clip
            outcome['y_train_roc'] = y_train_roc
            outcome['y_test_roc'] = y_test_roc
            outcome['runtime'] = runtime
            outcomes.append(outcome)
            pd.DataFrame(outcomes).to_csv('NN-GA.csv', mode='a', header=False)
            mut_outs[mut_test] = y_test_roc

        old_val = mut
        mut = max(mut_outs, key=mut_outs.get)
        print('best mut', mut, 'old', old_val)

        clips_outs = {}

        for clip_test in clips:
            start = time.time()

            print(algorithim, act, lr, itera, ' ', pop, mut, ma, clip_test)
            nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act, max_iters=itera,
                                            algorithm=algorithim, pop_size=pop, mutation_prob=mut,
                                            bias=True, is_classifier=True, learning_rate=lr,
                                            early_stopping=True, clip_max=clip_test, max_attempts=ma,
                                            random_state=seed, curve=True)
            nn_model.fit(X_train, y_train)
            y_train_pred = nn_model.predict(X_train)
            y_train_roc = roc_auc_score(y_train, y_train_pred, multi_class="ovr", average="weighted")
            print('y_train_roc', y_train_roc)

            y_test_pred = nn_model.predict(X_test)
            y_test_roc = roc_auc_score(y_test, y_test_pred, multi_class="ovr", average="weighted")
            print('y_test_roc', y_test_roc)

            runtime = time.time() - start
            print('curr run time', time.time() - start)

            outcome = {}
            outcome['schedule'] = ' '
            outcome['activation'] = act
            outcome['learning_rate'] = lr
            outcome['max_iters'] = itera
            outcome['population'] = pop
            outcome['mutation'] = mut
            outcome['max_attempts'] = ma
            outcome['clip'] = clip_test
            outcome['y_train_roc'] = y_train_roc
            outcome['y_test_roc'] = y_test_roc
            outcome['runtime'] = runtime
            outcomes.append(outcome)
            pd.DataFrame(outcomes).to_csv('NN-GA.csv', mode='a', header=False)
            clips_outs[clip_test] = y_test_roc

        old_val = clip
        clip = max(clips_outs, key=clips_outs.get)
        print('best clip', clip, 'old', old_val)

        maxa_outs = {}

        for maxa_test in max_attempts:
            start = time.time()

            print(algorithim, act, lr, itera, ' ', pop, mut, maxa_test, clip)
            nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act, max_iters=itera,
                                            algorithm=algorithim, pop_size=pop, mutation_prob=mut,
                                            bias=True, is_classifier=True, learning_rate=lr,
                                            early_stopping=True, clip_max=clip, max_attempts=maxa_test,
                                            random_state=seed, curve=True)
            nn_model.fit(X_train, y_train)
            y_train_pred = nn_model.predict(X_train)
            y_train_roc = roc_auc_score(y_train, y_train_pred, multi_class="ovr", average="weighted")
            print('y_train_roc', y_train_roc)

            y_test_pred = nn_model.predict(X_test)
            y_test_roc = roc_auc_score(y_test, y_test_pred, multi_class="ovr", average="weighted")
            print('y_test_roc', y_test_roc)

            runtime = time.time() - start
            print('curr run time', time.time() - start)

            outcome = {}
            outcome['schedule'] = ' '
            outcome['activation'] = act
            outcome['learning_rate'] = lr
            outcome['max_iters'] = itera
            outcome['population'] = pop
            outcome['mutation'] = mut
            outcome['max_attempts'] = maxa_test
            outcome['clip'] = clip
            outcome['y_train_roc'] = y_train_roc
            outcome['y_test_roc'] = y_test_roc
            outcome['runtime'] = runtime
            outcomes.append(outcome)
            pd.DataFrame(outcomes).to_csv('NN-GA.csv', mode='a', header=False)
            maxa_outs[maxa_test] = y_test_roc

        old_val = ma
        ma = max(maxa_outs, key=maxa_outs.get)
        print('best ma', ma, 'old', old_val)
        lr_outs = {}

        for lr_test in learning_rate:
            start = time.time()

            print(algorithim, act, lr_test, itera, ' ', pop, mut, ma, clip)
            nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act, max_iters=itera,
                                            algorithm=algorithim, pop_size=pop, mutation_prob=mut,
                                            bias=True, is_classifier=True, learning_rate=lr_test,
                                            early_stopping=True, clip_max=clip, max_attempts=ma,
                                            random_state=seed, curve=True)
            nn_model.fit(X_train, y_train)
            y_train_pred = nn_model.predict(X_train)
            y_train_roc = roc_auc_score(y_train, y_train_pred, multi_class="ovr", average="weighted")
            print('y_train_roc', y_train_roc)

            y_test_pred = nn_model.predict(X_test)
            y_test_roc = roc_auc_score(y_test, y_test_pred, multi_class="ovr", average="weighted")
            print('y_test_roc', y_test_roc)

            runtime = time.time() - start
            print('curr run time', time.time() - start)

            outcome = {}
            outcome['schedule'] = ' '
            outcome['activation'] = act
            outcome['learning_rate'] = lr_test
            outcome['max_iters'] = itera
            outcome['population'] = pop
            outcome['mutation'] = mut
            outcome['max_attempts'] = ma
            outcome['clip'] = clip
            outcome['y_train_roc'] = y_train_roc
            outcome['y_test_roc'] = y_test_roc
            outcome['runtime'] = runtime
            outcomes.append(outcome)
            pd.DataFrame(outcomes).to_csv('NN-GA.csv', mode='a', header=False)
            lr_outs[lr_test] = y_test_roc

        old_lr = lr
        lr = max(lr_outs, key=lr_outs.get)
        print('best lr', lr, 'old', old_lr)

        pop_outs = {}

        for pop_test in population:
            start = time.time()

            print(algorithim, act, lr, itera, ' ', pop_test, mut, ma, clip)
            nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act, max_iters=itera,
                                            algorithm=algorithim, pop_size=pop_test, mutation_prob=mut,
                                            bias=True, is_classifier=True, learning_rate=lr,
                                            early_stopping=True, clip_max=clip, max_attempts=ma,
                                            random_state=seed, curve=True)
            nn_model.fit(X_train, y_train)
            y_train_pred = nn_model.predict(X_train)
            y_train_roc = roc_auc_score(y_train, y_train_pred, multi_class="ovr", average="weighted")
            print('y_train_roc', y_train_roc)

            y_test_pred = nn_model.predict(X_test)
            y_test_roc = roc_auc_score(y_test, y_test_pred, multi_class="ovr", average="weighted")
            print('y_test_roc', y_test_roc)

            runtime = time.time() - start
            print('curr run time', time.time() - start)

            outcome = {}
            outcome['schedule'] = ' '
            outcome['activation'] = act
            outcome['learning_rate'] = lr
            outcome['max_iters'] = itera
            outcome['population'] = pop_test
            outcome['mutation'] = mut
            outcome['max_attempts'] = ma
            outcome['clip'] = clip
            outcome['y_train_roc'] = y_train_roc
            outcome['y_test_roc'] = y_test_roc
            outcome['runtime'] = runtime
            outcomes.append(outcome)
            pd.DataFrame(outcomes).to_csv('NN-GA.csv', mode='a', header=False)
            pop_outs[pop_test] = y_test_roc

        old_val = pop
        pop = max(pop_outs, key=pop_outs.get)
        print('best pop', pop, 'old', old_val)


if __name__ == "__main__":
    NN_GA(file_name='Mobile_Prices_orig.csv',
          classifier_col='price_range')
