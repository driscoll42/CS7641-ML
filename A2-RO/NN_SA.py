import mlrose_hiive as mlrose
from sklearn.metrics import roc_auc_score
import time
import util

import pandas as pd


def NN_SA(file_name, classifier_col):
    X_train, X_test, y_train, y_test = util.data_load(file_name, classifier_col)

    activation = 'relu'
    learning_rate = [0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    algorithim = 'simulated_annealing'
    iters = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    nodes = [128, 128, 128, 128]
    temperatures = [0.001, 0.01]
    decay_rates = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    outcomes = []
    max_attempts = [10, 50, 100, 200, 500, 1000]
    clips = [5, 10, 100, 1000, 10000, 100000]

    csv_file = 'NN_SA-itertests.csv'
    act = 'relu'
    lr = 10
    itera = 10000
    temp = 10000
    dec = 0.92
    ma = 50
    clip = 10

    while 1 == 1:

        iters_outs = {}

        for iter_test in iters:
            start = time.time()

            print(algorithim, act, lr, iter_test, 'GeomDecay', temp, dec, ma, clip)
            nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act, max_iters=iter_test,
                                            algorithm=algorithim,
                                            schedule=mlrose.GeomDecay(init_temp=temp, decay=dec),
                                            bias=True, is_classifier=True, learning_rate=lr,
                                            early_stopping=True, clip_max=clip, max_attempts=ma,
                                            random_state=1, curve=True)
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
            outcome['schedule'] = 'GeomDecay'
            outcome['activation'] = act
            outcome['learning_rate'] = lr
            outcome['max_iters'] = iter_test
            outcome['temperatures'] = temp
            outcome['decay_rates'] = dec
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
            pd.DataFrame(outcomes).to_csv(csv_file)

            iters_outs[iter_test] = y_test_roc

        old_val = itera
        itera = max(iters_outs, key=iters_outs.get)
        print('best iter', itera, 'old', old_val)


        raise SystemExit(0)



        temp_outs = {}

        for temp_test in temperatures:
            start = time.time()

            print(algorithim, act, lr, itera, 'GeomDecay', temp_test, dec, ma, clip)
            nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act, max_iters=itera,
                                            algorithm=algorithim,
                                            schedule=mlrose.GeomDecay(init_temp=temp_test, decay=dec),
                                            bias=True, is_classifier=True, learning_rate=lr,
                                            early_stopping=True, clip_max=clip, max_attempts=ma,
                                            random_state=1, curve=True)
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
            outcome['schedule'] = 'GeomDecay'
            outcome['activation'] = act
            outcome['learning_rate'] = lr
            outcome['max_iters'] = itera
            outcome['temperatures'] = temp_test
            outcome['decay_rates'] = dec
            outcome['max_attempts'] = ma
            outcome['clip'] = clip
            outcome['y_train_roc'] = y_train_roc
            outcome['y_test_roc'] = y_test_roc
            outcome['runtime'] = runtime
            outcomes.append(outcome)
            pd.DataFrame(outcomes).to_csv(csv_file)

            temp_outs[temp_test] = y_test_roc

        old_temp = temp
        temp = max(temp_outs, key=temp_outs.get)
        print('best temp', temp, 'old', old_temp)

        decay_outs = {}

        for decay_test in decay_rates:
            start = time.time()

            print(algorithim, act, lr, itera, 'GeomDecay', temp, decay_test, ma, clip)
            nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act, max_iters=itera,
                                            algorithm=algorithim,
                                            schedule=mlrose.GeomDecay(init_temp=temp, decay=decay_test),
                                            bias=True, is_classifier=True, learning_rate=lr,
                                            early_stopping=True, clip_max=clip, max_attempts=ma,
                                            random_state=1, curve=True)
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
            outcome['schedule'] = 'GeomDecay'
            outcome['activation'] = act
            outcome['learning_rate'] = lr
            outcome['max_iters'] = itera
            outcome['temperatures'] = temp
            outcome['decay_rates'] = decay_test
            outcome['max_attempts'] = ma
            outcome['clip'] = clip
            outcome['y_train_roc'] = y_train_roc
            outcome['y_test_roc'] = y_test_roc
            outcome['runtime'] = runtime
            outcomes.append(outcome)
            pd.DataFrame(outcomes).to_csv(csv_file)

            decay_outs[decay_test] = y_test_roc

        old_val = dec
        dec = max(decay_outs, key=decay_outs.get)
        print('best decay', dec, 'old', old_val)

        clips_outs = {}
        for clip_test in clips:
            start = time.time()

            print(algorithim, act, lr, itera, 'GeomDecay', temp, dec, ma, clip_test)
            nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act, max_iters=itera,
                                            algorithm=algorithim,
                                            schedule=mlrose.GeomDecay(init_temp=temp, decay=dec),
                                            bias=True, is_classifier=True, learning_rate=lr,
                                            early_stopping=True, clip_max=clip_test, max_attempts=ma,
                                            random_state=1, curve=True)
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
            outcome['schedule'] = 'GeomDecay'
            outcome['activation'] = act
            outcome['learning_rate'] = lr
            outcome['max_iters'] = itera
            outcome['temperatures'] = temp
            outcome['decay_rates'] = dec
            outcome['max_attempts'] = ma
            outcome['clip'] = clip_test
            outcome['y_train_roc'] = y_train_roc
            outcome['y_test_roc'] = y_test_roc
            outcome['runtime'] = runtime
            outcomes.append(outcome)
            pd.DataFrame(outcomes).to_csv(csv_file)
            clips_outs[clip_test] = y_test_roc

        old_val = clip
        clip = max(clips_outs, key=clips_outs.get)
        print('best clip', clip, 'old', old_val)

        maxa_outs = {}

        for maxa_test in max_attempts:
            start = time.time()

            print(algorithim, act, lr, itera, 'GeomDecay', temp, dec, maxa_test, clip)
            nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act, max_iters=itera,
                                            algorithm=algorithim,
                                            schedule=mlrose.GeomDecay(init_temp=temp, decay=dec),
                                            bias=True, is_classifier=True, learning_rate=lr,
                                            early_stopping=True, clip_max=clip, max_attempts=maxa_test,
                                            random_state=1, curve=True)
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
            outcome['schedule'] = 'GeomDecay'
            outcome['activation'] = act
            outcome['learning_rate'] = lr
            outcome['max_iters'] = itera
            outcome['temperatures'] = temp
            outcome['decay_rates'] = dec
            outcome['max_attempts'] = maxa_test
            outcome['clip'] = clip
            outcome['y_train_roc'] = y_train_roc
            outcome['y_test_roc'] = y_test_roc
            outcome['runtime'] = runtime
            outcomes.append(outcome)
            pd.DataFrame(outcomes).to_csv(csv_file)

            maxa_outs[maxa_test] = y_test_roc

        old_val = ma
        ma = max(maxa_outs, key=maxa_outs.get)
        print('best ma', ma, 'old', old_val)

        lr_outs = {}

        for lr_test in learning_rate:
            start = time.time()

            print(algorithim, act, lr_test, itera, 'GeomDecay', temp, dec, ma, clip)
            nn_model = mlrose.NeuralNetwork(hidden_nodes=nodes, activation=act, max_iters=itera,
                                            algorithm=algorithim,
                                            schedule=mlrose.GeomDecay(init_temp=temp, decay=dec),
                                            bias=True, is_classifier=True, learning_rate=lr_test,
                                            early_stopping=True, clip_max=clip, max_attempts=ma,
                                            random_state=1, curve=True)
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
            outcome['schedule'] = 'GeomDecay'
            outcome['activation'] = act
            outcome['learning_rate'] = lr_test
            outcome['max_iters'] = itera
            outcome['temperatures'] = temp
            outcome['decay_rates'] = dec
            outcome['max_attempts'] = ma
            outcome['clip'] = clip
            outcome['y_train_roc'] = y_train_roc
            outcome['y_test_roc'] = y_test_roc
            outcome['runtime'] = runtime
            outcomes.append(outcome)
            pd.DataFrame(outcomes).to_csv(csv_file)

            lr_outs[lr_test] = y_test_roc

        old_lr = lr
        lr = max(lr_outs, key=lr_outs.get)
        print('best lr', lr, 'old', old_lr)


if __name__ == "__main__":
    NN_SA(file_name='Mobile_Prices_orig.csv',
          classifier_col='price_range')
