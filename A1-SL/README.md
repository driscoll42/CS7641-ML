## CS 7641 Assignment 1: Supervised Learning Classification
The purpose of this project is to explore some techniques in supervised learning.
 
 * Decision Trees
     * For the decision tree, you should implement or steal a decision tree algorithm. Be sure to use some form of pruning. You are not required to use information gain (for example, there is something called the GINI index that is sometimes used) to split attributes, but you should describe whatever it is that you do use.
 * k-Nearest Neighbors (k-NN)
     * You should implement kNN. Use different values of k.  
 * Boosted Trees
     * Implement a boosted version of your decision trees. As before, you will want to use some form of pruning, but presumably because you're using boosting you can afford to be much more aggressive about your pruning.
 * Support Vector Machines (SVM)
     * You should implement (for sufficiently loose definitions of implement including "download") SVMs. This should be done in such a way that you can swap out kernel functions. I'd like to see at least two.
 * Neural Networks
     * For the neural network you should implement or steal your favorite kind of network and training algorithm. You may use networks of nodes with as many layers as you like and any activation function you see fit. 
    
Each algorithm will be run for two binary classification datasets so that we can compare and contrast them for two different problems (one for a balanced target variable and the other for an unbalanced target variable).

Dataset 1:  Mobile Prices - https://www.kaggle.com/iabhishekofficial/mobile-price-classification

Dataset 2: Chess - https://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King%29


## Getting Started & Prerequisites
For testing on your own machine, you need only to install python 3.7 and the following packages:
- pandas, numpy, scikit-learn, matplotlib, itertools, timeit, scipy

## Run Instructions

1. Place a dataset in to the Data subfolder.
   * Data must be to solve a classification problem and have a specific column containing the class
   * There should be no missing data
   * If there is any non-numeric data, the code will treat it as a categorical and one hot encode those columns
2. In the main.py under main.py call a run_sl_algos function, see description below. This code will run the dataset on all five algorithms, performing a GridSearchCV for each algorithm to determine the optimal hyperparameters and once determined for each HP plots validation curves, also the learning curve for each algorithm. You can specify a specific set of parameters for each algorithm to run for those parameters instead of performing a Grid Search.    
    * run_sl_algos(
      * filename - the file name,
      * result_col - The column storing the class to be determined
      * scalar - 0, 1, 2 - 0 applies no scaling to the data, 1 applies minMaxScalar, 2 applies StandardScalar
      * njobs - For parallelzable jobs how many jobs to run, -1 runs on all cores
      * full_param - Boolean - Run an exhaustive GridSearch or a shorter GridSearch (the shorter one worked the majority of the time to find "good enough" results in testing)
      * make_graphs - Boolean, Whether to generate graphs when running
      * numFolds - int - Number of folds for cross validation, default is 10
      * nolegend - Boolean - For Neural Networks if True suppresses the legend on Validation Curves   
      * test_all - Boolean, Test all the solvers for SVM and NN or break off based on score returned
      * debug - Boolean - print debug statements or not
      * rDTree - Boolean - Test the dataset on Decision Trees
      * pDTree - Specify HPs to run the algo on rather than GridSearch, see main.py for example
      * rknn - Boolean - Test the dataset on KNNs
      * pknn - Specify HPs to run the algo on rather than GridSearch, see main.py for example
      * rSVM - Boolean - Test the on SVMs 
      * pSVM - Specify HPs to run the algo on rather than GridSearch, see main.py for example
      * rNN - Boolean - Test the dataset on Neural Networks
      * pNN - Specify HPs to run the algo on rather than GridSearch, see main.py for example
      * rBTree - Boolean - Test the dataset on Boosted Trees
      * pBTree - Specify HPs to run the algo on rather than GridSearch, see main.py for example
3. Graphs will be sored in Images, and results of the GridSearch will be sored in ParamTests 