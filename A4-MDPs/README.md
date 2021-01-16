## CS 7641 Assignment 3: Reinforcement Learning - Markov Decision Processes
The purpose of this project is to explore some techniques in reinforcement learning, specifically this assignment explores three RL algorithms on two RL environments, when the environment is both small and large to compare and contrast how the algorithms work.

* Value-Iteration
* Policy Iteration
* Q-Learning


## Getting Started & Prerequisites
For testing on your own machine, you need only to install python 3.7 and the following packages:
- pandas, numpy, matplotlib, itertools, timeit, gym, mdptoolbox-hiive, mdptoolbox

## Run Instructions

There are six files to run, grouped by RL environment and then algorithm, with the following graphs generated per algorithm:

* Value/Policy Iteration: 
  * Epsilon vs Gamma heatmaps relative to: Average Reward, # Iterations to Convergence, Runtime 
  * Mean V/Error/Rewards vs iterations
  * The policy graphed
* Q-Learning:
  * Prints out the parameters and average rewards to be taken into excel to plot
  * V/Error/Rewards vs iterations
  * The policy graphed

To run both sets of environments, run: 

* Forest - https://pymdptoolbox.readthedocs.io/en/latest/api/example.html#mdptoolbox.example.forest
  * _By Default will run for forests of size 10 and 1000_
  * VI-Forest.py
  * PI-Forest.py
  * QL-Forest.py

* Frozen Lake - https://gym.openai.com/envs/FrozenLake-v0/
  * _By Default will run for FrozenLake of size 4 and 16_
  * VI-FrozenLake.py
  * PI-FrozenLake.py
  * QL-FrozenLake.py

Run each file the output images will be sored in the Images subdirectory
