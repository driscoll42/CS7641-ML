import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, label_binarize, OneHotEncoder
import mlrose_hiive as mlrose
from matplotlib import pyplot as plt
import random
import time
import copy

start = 0.
times = []
evals = []
iter_total = 0
algo_in = 'NOT'


def callback_func(iteration, attempt=None, done=None, state=None, fitness=None, curve=None, user_data=None):
    global start, times, evals, iter_total, rhc_curve, eval_count, algo_in

    if iteration == 0 and algo_in == 'NOT_RHC':
        evals = []
        start = time.time()
        times = []
    else:
        evals.append(eval_count)
        iter_total += 1
        times.append(time.time() - start)

    return True


def fit_eval_count(fit_fun, *args, **kwargs):
    def wrap(state):
        global eval_count
        fitness = fit_fun(*args, **kwargs)
        eval_count += 1
        return fitness.evaluate(state)

    return wrap


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
    # One hot encode target values
    one_hot = OneHotEncoder()

    y_train = one_hot.fit_transform(y_train.values.reshape(-1, 1)).todense()
    y_test = one_hot.transform(y_test.values.reshape(-1, 1)).todense()

    return X_train, X_test, y_train, y_test


def RHC_tests(problem_fit, ma, maxIter=1000, seed=1, input_size=-1, restarts=0):
    global eval_count, iter_total, algo_in, start, times, evals

    start = time.time()
    eval_count, iter_total, times, evals = 0, 0, [], []
    algo_in = 'RHC'
    best_state, best_fitness, rhcfitness_curve = mlrose.random_hill_climb(problem_fit, curve=True, max_attempts=ma,
                                                                          state_fitness_callback=callback_func,
                                                                          callback_user_info=[], restarts=restarts,
                                                                          random_state=seed, max_iters=maxIter)
    rhc_times = copy.deepcopy(times)
    rhc_evals = copy.deepcopy(evals)
    print('RHC', input_size,
          restarts,
          seed,
          best_fitness,
          np.where(rhcfitness_curve == best_fitness)[0][0],  # iter of best fitness
          iter_total,  # total iters
          np.where(rhcfitness_curve == best_fitness)[0][0],  # num evals for best fitness
          eval_count,  # total evals
          # round(rhc_times[np.where(rhcfitness_curve == best_fitness)[0][0]], 4),  # time of best fitness
          round(rhc_times[-1], 4)  # total time
          )


def GenerateFitnessCurves(problem_fit, ma, sa_sched=None, gapop=200, gamut=0.5, mimpop=200, mimpct=0.2, maxIter=1000,
                          seed=1, min=False, input_size=-1, restarts=0):
    global eval_count, iter_total, algo_in, start, times, evals

    start = time.time()
    eval_count, iter_total, times, evals = 0, 0, [], []
    algo_in = 'RHC'
    print('RHC')
    best_state, best_fitness, rhcfitness_curve = mlrose.random_hill_climb(problem_fit, curve=True, max_attempts=ma,
                                                                          state_fitness_callback=callback_func,
                                                                          callback_user_info=[], restarts=restarts,
                                                                          random_state=seed, max_iters=maxIter)
    rhc_times = copy.deepcopy(times)
    rhc_evals = copy.deepcopy(evals)
    print('RHC',
          input_size,
          seed,
          best_fitness,
          np.where(rhcfitness_curve == best_fitness)[0][0],  # iter of best fitness
          iter_total,  # total iters
          np.where(rhcfitness_curve == best_fitness)[0][0],  # num evals for best fitness
          eval_count,  # total evals
          # round(rhc_times[np.where(rhcfitness_curve == best_fitness)[0][0]], 4),  # time of best fitness
          round(rhc_times[-1], 4)  # total time
          )
    eval_count, iter_total, times, evals = 0, 0, [], []
    algo_in = 'NOT_RHC'
    print('SA')

    if sa_sched:
        best_state, best_fitness, safitness_curve = mlrose.simulated_annealing(problem_fit, schedule=sa_sched,
                                                                               state_fitness_callback=callback_func,
                                                                               callback_user_info=[],
                                                                               curve=True, max_attempts=ma,
                                                                               random_state=seed, max_iters=maxIter)
    else:
        best_state, best_fitness, safitness_curve = mlrose.simulated_annealing(problem_fit,  # schedule=SA_schedule,
                                                                               curve=True, max_attempts=ma,
                                                                               state_fitness_callback=callback_func,
                                                                               callback_user_info=[],
                                                                               random_state=seed, max_iters=maxIter)
    sa_times = copy.deepcopy(times)
    sa_evals = copy.deepcopy(evals)

    print('SA',
          input_size,
          seed,
          best_fitness,
          np.where(safitness_curve == best_fitness)[0][0],  # iter of best fitness
          len(safitness_curve),  # total iters
          evals[np.where(safitness_curve == best_fitness)[0][0]],  # num evals for best fitness
          eval_count,  # total evals
          round(sa_times[np.where(safitness_curve == best_fitness)[0][0]], 4),  # time of best fitness
          round(sa_times[-1], 4)  # total time
          )
    eval_count, iter_total, times, evals = 0, 0, [], []

    print('GA')

    best_state, best_fitness, gafitness_curve = mlrose.genetic_alg(problem_fit, curve=True, pop_size=gapop,
                                                                   mutation_prob=gamut, max_attempts=ma,
                                                                   state_fitness_callback=callback_func,
                                                                   callback_user_info=[],

                                                                   random_state=seed, max_iters=maxIter)
    ga_times = copy.deepcopy(times)
    ga_evals = copy.deepcopy(evals)

    print('GA', input_size,
          seed,
          best_fitness,
          np.where(gafitness_curve == best_fitness)[0][0],  # iter of best fitness
          len(gafitness_curve),  # total iters
          evals[np.where(gafitness_curve == best_fitness)[0][0]],  # num evals for best fitness
          eval_count,  # total evals
          round(ga_times[np.where(gafitness_curve == best_fitness)[0][0]], 4),  # time of best fitness
          round(ga_times[-1], 4)  # total time
          )

    eval_count, iter_total, times, evals = 0, 0, [], []

    print('MIMIC')

    best_state, best_fitness, mimicfitness_curve = mlrose.mimic(problem_fit, curve=True, max_attempts=ma,
                                                                pop_size=mimpop, keep_pct=mimpct,
                                                                state_fitness_callback=callback_func,
                                                                callback_user_info=[],
                                                                random_state=seed, max_iters=maxIter)
    mim_times = copy.deepcopy(times)
    mim_evals = copy.deepcopy(evals)
    print('MIMIC', input_size,
          seed,
          best_fitness,
          np.where(mimicfitness_curve == best_fitness)[0][0],  # iter of best fitness
          len(mimicfitness_curve),  # total iters
          evals[np.where(mimicfitness_curve == best_fitness)[0][0]],  # num evals for best fitness
          eval_count,  # total evals
          round(mim_times[np.where(mimicfitness_curve == best_fitness)[0][0]], 4),  # time of best fitness
          round(mim_times[-1], 4)  # total time
          )

    if min:
        # To Maximize TSP, need to make everything negative
        gafitness_curve = np.array(gafitness_curve) * -1
        rhcfitness_curve = np.array(rhcfitness_curve) * -1
        safitness_curve = np.array(safitness_curve) * -1
        mimicfitness_curve = np.array(mimicfitness_curve) * -1
        return gafitness_curve, rhcfitness_curve, safitness_curve, mimicfitness_curve, ga_times, rhc_times, sa_times, mim_times, rhc_evals, sa_evals, ga_evals, mim_evals
    else:
        return gafitness_curve, rhcfitness_curve, safitness_curve, mimicfitness_curve, ga_times, rhc_times, sa_times, mim_times, rhc_evals, sa_evals, ga_evals, mim_evals


def plot_ItervsTime(ga_curve, rhc_curve, sa_curve, mim_curve, problem):
    plt.title(problem + " Iterations vs. Time (s)")
    plt.plot(ga_curve, label='GA', color='r')
    plt.plot(rhc_curve, label='RHC', color='b')
    plt.plot(sa_curve, label='SA', color='orange')
    plt.plot(mim_curve, label='MIMIC', color='g')
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('time (s)')
    plt.grid(True)
    plt.legend()
    plt.savefig("Images\\" + problem + " Iter vs Time")
    plt.show()

    plt.title(problem + " Iterations vs. Time (s)")
    plt.plot(ga_curve, label='GA', color='r')
    plt.plot(rhc_curve, label='RHC', color='b')
    plt.plot(sa_curve, label='SA', color='orange')
    plt.plot(mim_curve, label='MIMIC', color='g')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('iterations')
    plt.ylabel('time (s)')
    plt.grid(True)
    plt.legend()
    plt.savefig("Images\\" + problem + " Iter vs Time - log")
    plt.show()


def plot_FitvsTime(ga_curve, rhc_curve, sa_curve, mim_curve, ga_time, rhc_time, sa_time, mim_time, problem):
    new_rhc_time = np.linspace(0, rhc_time[-1], num=len(rhc_curve))

    plt.title(problem + " Fitness vs. Time (s)")
    plt.plot(ga_time, ga_curve, label='GA', color='r')
    plt.plot(new_rhc_time, rhc_curve, label='RHC', color='b')
    plt.plot(sa_time, sa_curve, label='SA', color='orange')
    plt.plot(mim_time, mim_curve, label='MIMIC', color='g')
    plt.xlabel('time (s)')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("Images\\" + problem + " Fit vs Time")
    plt.show()

    plt.title(problem + " Fitness vs. Time (s)")
    plt.plot(ga_time, ga_curve, label='GA', color='r')
    plt.plot(new_rhc_time, rhc_curve, label='RHC', color='b')
    plt.plot(sa_time, sa_curve, label='SA', color='orange')
    plt.plot(mim_time, mim_curve, label='MIMIC', color='g')
    plt.xscale('log')
    plt.xlabel('time (s)')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("Images\\" + problem + " Fit vs Time - log")
    plt.show()


def plot_vsIterations(ga_curve, rhc_curve, sa_curve, mim_curve, problem):
    plt.title(problem + " Fitness vs. Iterations")
    plt.plot(ga_curve, label='GA', color='r')
    plt.plot(rhc_curve, label='RHC', color='b')
    plt.plot(sa_curve, label='SA', color='orange')
    plt.plot(mim_curve, label='MIMIC', color='g')
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("Images\\" + problem + " Fitness Curve")
    plt.show()


def plot_vsEvaluations(ga_curve, rhc_curve, sa_curve, mim_curve, ga_evals, rhc_evals, sa_evals, mim_evals, problem):
    new_rhc_evals = np.linspace(0, rhc_evals[-1], num=len(rhc_curve))

    plt.title(problem + " Fitness vs. Fitness Evaluation Calls")
    plt.plot(ga_evals, ga_curve, label='GA', color='r')
    plt.plot(new_rhc_evals, rhc_curve, label='RHC', color='b')
    plt.plot(sa_evals, sa_curve, label='SA', color='orange')
    plt.plot(mim_evals, mim_curve, label='MIMIC', color='g')
    plt.xscale('log')
    plt.xlabel('Fitness Evaluations Calls')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("Images\\" + problem + " Fitness vs Evaluation Curve")
    plt.show()


def plot_GAmut(problem, problem_fit, max_attempts, gapop, maxIter, seed, min=False):
    Mutate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    plt.figure()
    for m in Mutate:
        best_state, best_fitness, gafitness_curve = mlrose.genetic_alg(problem_fit, mutation_prob=m, curve=True,
                                                                       max_iters=maxIter, max_attempts=max_attempts,
                                                                       pop_size=gapop, random_state=seed)
        if min:
            gafitness_curve = np.array(gafitness_curve) * -1
        plt.plot(gafitness_curve, label='mut prob =' + str(m))
        print(m, best_fitness)

    plt.title(problem + " - GA - Mutation Probabilities")
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("Images\\" + problem + " - GA - Mutation Probabilities")
    plt.show()


def plot_GAmutpop(problem, problem_fit, max_attempts, gamut, maxIter, seed, min=False):
    Mutatepop = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
    plt.figure()
    for m in Mutatepop:
        best_state, best_fitness, gafitness_curve = mlrose.genetic_alg(problem_fit, pop_size=m, mutation_prob=gamut,
                                                                       curve=True, max_attempts=max_attempts,
                                                                       max_iters=maxIter, random_state=seed)
        if min:
            gafitness_curve = np.array(gafitness_curve) * -1
        plt.plot(gafitness_curve, label='mut pop = ' + str(m))
        print(m, best_fitness)

    plt.title(problem + " - GA - Mutation Populations")
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("Images\\" + problem + " - GA - Mutation Populations")
    plt.show()


def plot_RHCRestarts(problem, problem_fit, max_attempts, maxIter, seed, min=False):
    restarts = [0, 1, 5, 10, 50, 100]
    plt.figure()
    global start, times, evals, iter_total, rhc_curve, eval_count, algo_in

    eval_count, iter_total, times, evals = 0, 0, [], []

    for r in restarts:
        print('restarts', r)
        best_state, best_fitness, rhcfitness_curve = mlrose.random_hill_climb(problem_fit, curve=True,
                                                                              max_attempts=max_attempts,
                                                                              state_fitness_callback=callback_func,
                                                                              callback_user_info=[], restarts=r,
                                                                              random_state=seed, max_iters=maxIter)
        if min:
            rhcfitness_curve = np.array(rhcfitness_curve) * -1
        plt.plot(rhcfitness_curve, label='restarts = ' + str(r))

    plt.title(problem + " - RHC - Restarts")
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("Images\\" + problem + " - RHC - Restarts")
    plt.show()


def plot_SAdecay(problem, problem_fit, max_attempts, init_temp, maxIter, seed, min=False):
    decay_r = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    plt.figure()
    for d in decay_r:
        SAschedule = mlrose.GeomDecay(init_temp=init_temp, decay=d, min_temp=0.01)
        best_state, best_fitness, safitness_curve = mlrose.simulated_annealing(problem_fit, schedule=SAschedule,
                                                                               curve=True, max_attempts=max_attempts,
                                                                               random_state=seed, max_iters=maxIter)
        if min:
            safitness_curve = np.array(safitness_curve) * -1
        plt.plot(safitness_curve, label='decay rate = ' + str(d))

    plt.title(problem + " - SA - Decay Rates")
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("Images\\" + problem + " - SA - Decay Rates")
    plt.show()


def plot_SATemps(problem, problem_fit, max_attempts, decay, maxIter, seed, min=False):
    temps = [10000000, 1000000, 100000, 10000, 1000, 100, 10, 1, 0.1]
    plt.figure()
    for t in temps:
        SAschedule = mlrose.GeomDecay(init_temp=t, decay=decay, min_temp=0.01)
        best_state, best_fitness, safitness_curve = mlrose.simulated_annealing(problem_fit, schedule=SAschedule,
                                                                               curve=True, max_attempts=max_attempts,
                                                                               random_state=seed, max_iters=maxIter)
        if min:
            safitness_curve = np.array(safitness_curve) * -1
        plt.plot(safitness_curve, label='Temperature = ' + str(t))

    plt.title(problem + " - SA - Initial Temperature")
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("Images\\" + problem + " - SA - Initial Temperature")
    plt.show()


def plot_MIMpop(problem, problem_fit, max_attempts, mimpct, maxIter, seed, min=False):
    plt.figure()
    pop = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for p in pop:
        print(p)
        best_state, best_fitness, mimicfitness_curve = mlrose.mimic(problem_fit,
                                                                    pop_size=p,
                                                                    keep_pct=mimpct, curve=True,
                                                                    max_attempts=max_attempts,
                                                                    max_iters=maxIter,
                                                                    random_state=seed)
        print(p, best_fitness)
        if min:
            mimicfitness_curve = np.array(mimicfitness_curve) * -1
        plt.plot(mimicfitness_curve, label='pop size = ' + str(p))

    plt.title(problem + " - MIMIC - Population Size")
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("Images\\" + problem + " - MIMIC - Population Size")
    plt.show()


def plot_MIMICpct(problem, problem_fit, max_attempts, mimpop, maxIter, seed, min=False):
    PCT = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    plt.figure()
    for p in PCT:
        print(p)
        best_state, best_fitness, mimicfitness_curve = mlrose.mimic(problem_fit, keep_pct=p, curve=True,
                                                                    pop_size=mimpop,
                                                                    max_attempts=max_attempts,
                                                                    max_iters=maxIter,
                                                                    random_state=seed)
        print(p, best_fitness)
        if min:
            mimicfitness_curve = np.array(mimicfitness_curve) * -1
        plt.plot(mimicfitness_curve, label='pct = ' + str(p))

    plt.title(problem + " - MIMIC - Keep Percent")
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig("Images\\" + problem + " - MIMIC - Keep Percent")
    plt.show()
