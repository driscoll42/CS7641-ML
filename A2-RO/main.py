import mlrose_hiive as mlrose

import util
import random
import itertools


def run_ro_algos(algo, problem_fit, gapop=200, gamut=0.1, mimpop=200, mimpct=0.2, sa_sched=None, init_temp=10,
                 decay=0.55, max_attempts=250, maxIter=5000, seed=1, input_size=-1, restarts=0, min=False):
    gafitness_curve, rhcfitness_curve, safitness_curve, mimicfitness_curve, ga_times, rhc_times, sa_times, mim_times, rhc_evals, sa_evals, ga_evals, mim_evals = util.GenerateFitnessCurves(
            problem_fit, max_attempts, gapop=gapop, gamut=gamut, mimpop=mimpop, mimpct=mimpct, sa_sched=sa_sched,
            maxIter=maxIter, seed=seed, input_size=input_size, restarts=restarts, min=min)

    util.plot_vsIterations(gafitness_curve, rhcfitness_curve, safitness_curve, mimicfitness_curve, algo)
    util.plot_vsEvaluations(gafitness_curve, rhcfitness_curve, safitness_curve, mimicfitness_curve, ga_evals, rhc_evals,
                            sa_evals, mim_evals, algo)
    util.plot_FitvsTime(gafitness_curve, rhcfitness_curve, safitness_curve, mimicfitness_curve, ga_times, rhc_times,
                        sa_times, mim_times, algo)
    if 1 == 1:
        # RHC - Restarts
        util.plot_RHCRestarts(algo, problem_fit, max_attempts, maxIter, seed, min)

        # GA - Mutation Probability
        util.plot_GAmut(algo, problem_fit, max_attempts, gapop, maxIter, seed, min)

        # GA - Mutation Population
        util.plot_GAmutpop(algo, problem_fit, max_attempts, gamut, maxIter, seed, min)

        # SA - Decay Rate
        util.plot_SAdecay(algo, problem_fit, max_attempts, init_temp, maxIter, seed, min)

        # SA - Initial Temperature
        util.plot_SATemps(algo, problem_fit, max_attempts, decay, maxIter, seed, min)

        # MIMIC - Population
        util.plot_MIMpop(algo, problem_fit, max_attempts, mimpct, maxIter, seed, min)

        # MIMIC - Keep Percent
        util.plot_MIMICpct(algo, problem_fit, max_attempts, mimpop, maxIter, seed)


if __name__ == "__main__":
    print('Continuous Peaks - SA best')
    seed = 1
    random.seed(seed)
    bit_length = 100
    SAschedule = mlrose.GeomDecay(init_temp=10000, decay=0.95, min_temp=0.01)
    # fitness = mlrose.ContinuousPeaks(t_pct=0.02)
    fitness = mlrose.CustomFitness(
            util.fit_eval_count(mlrose.ContinuousPeaks, t_pct=0.02))
    problem_fit = mlrose.DiscreteOpt(length=bit_length, fitness_fn=fitness, maximize=True, max_val=2)

    run_ro_algos('CP-testingran', problem_fit, sa_sched=SAschedule, init_temp=1, gapop=900, gamut=0.7, mimpop=800,
                 mimpct=0.3, seed=seed, restarts=100, input_size=bit_length)

    # TSP - GA's best
    print('TSP - GA best')
    random.seed(seed)
    num_cities = 50
    random_list = list(itertools.product(range(0, 99), range(0, 99)))
    coords_list = random.sample(random_list, num_cities)

    # Initialize fitness function object using coords_list
    # fitness_coords = mlrose.TravellingSales(coords=coords_list)
    fitness = mlrose.CustomFitness(
            util.fit_eval_count(mlrose.TravellingSales, coords=coords_list))
    fitness.problem_type = 'tsp'

    SAschedule = mlrose.GeomDecay(init_temp=0.1, decay=0.5, min_temp=0.01)

    # Define optimization problem object
    problem_fit = mlrose.TSPOpt(length=num_cities, fitness_fn=fitness, maximize=True)

    run_ro_algos('TSP', problem_fit, gapop=600, gamut=0.6, mimpop=700, mimpct=0.3, seed=seed, min=True, sa_sched=SAschedule)

    # Knapsack - MIMIC's best
    print('Knapsack - MIMIC best')
    random.seed(seed)
    num_items = 100
    weights = []
    values = []
    for z in range(num_items):  # number of items
        values.append((random.random() + 0.1) * 500)
        weights.append((random.random() + 0.1) * 30)

    max_weight_pct = 0.8
    SAschedule = mlrose.GeomDecay(init_temp=10000000, decay=0.5, min_temp=0.01)

    # fitness = mlrose.Knapsack(weights, values, max_weight_pct)
    fitness = mlrose.CustomFitness(
            util.fit_eval_count(mlrose.Knapsack, weights=weights, values=values, max_weight_pct=max_weight_pct))
    problem_fit = mlrose.DiscreteOpt(length=num_items, fitness_fn=fitness, maximize=True)

    run_ro_algos('Knapsack', problem_fit, gapop=200, gamut=0.1, mimpop=900, mimpct=0.3, seed=seed, sa_sched=SAschedule)
