## CS 7641 Assignment 2: Randomized Optimization
The purpose of this project is to explore some techniques in Randomized Optimization
 
 * Randomized Hill Climbing
 * Simulated Annealing
 * Genetic Algorithms
 * MIMIC
    
Each algorithm will be run on three optimization problem domains, that is trying to maximize a fitness function. Once done RHC, SA, and GA will be used to compare against SGD on a Neural Network used in Assignment 1, in this case I used the Neural Network I made for the Mobile Prices dataset.

Dataset:  Mobile Prices - https://www.kaggle.com/iabhishekofficial/mobile-price-classification


## Getting Started & Prerequisites
For testing on your own machine, you need only to install python 3.7 and the following packages:
- pandas, numpy, scikit-learn, matplotlib, itertools, timeit, scipy, mlrose-hiive

## Run Instructions

1. First run the code in main.py. This will generate Validation Curves for three optimization problems, listed below, for the main hyperparameters for each algorithm as well as plotting vs iterations, evaluations and fitness vs time. 
   * Continuous Peaks
   * Traveling Sales Person
   * Knapsack

2. Run NN_RHC.py, NN_SA.py, NN_GA.py, NN_GD.py to pseudo-gridsearch (optimizes one parameter at a time, holding the rest constant and loops), through the HPs of each algorithim until the best HPs are determined, saving results to a csv.
3. Find the best HPs from step 2 and input them into the variables in NN_Graph to generate loss curves for all four Neural Networks and generate the ROC vs iteration curves for all four Neural Networks
4. The NN_<algo>-HPPlots.py files will generate validation curves for each Neural Network and RO algo combo