## CS 7641 Assignment 3: Unsupervised Learning and Dimensionality Reduction 
The purpose of this project is to explore some techniques in unsupervised learning.
 
#### Clustering

* k-means clustering
* Expectation Maximization

Each clustering algorithm will be run on two datasets and compared and contrasted, using measures which have no knowledge of the ground truth (i.e. the labels) the BIC Scores, Inertia Curve, Davies-Bouldin Index, Calinski-Harabasz Index, and Silohuette scores, but then also contrasting to ground truth scores of V-Measure, Adjusted Rand Index, and Adjusted Mutual Information Index.

#### Dimensionality Reduction

Each DR algorithm will be run on the same datasets as clustering and by reviewing a specific metric per algorithm it will be possible to determine the optimal number of clusters.

* PCA
   * Explained Variance Ratio
* ICA
  * Average Kurtosis
* Randomized Projections
  * Mean Reconstruction Error
* Locally Linear Embedding
  * Mean Reconstruction Error
   
After running the clustering and DR algorithms independently, the same clustering metrics and algorithms will be run on the datasets after applying the optimal number of dimensionality reduction.

Finally, one dataset is chosen to run the neural network from A1, first replacing the data with the dimension reduced data and finding the Train/Test ROC, then determining the ideal clusters and appending them to the data and running the neural network classification.

Each algorithm will be run for two binary classification datasets so that we can compare and contrast them for two different problems (one for a balanced target variable and the other for an unbalanced target variable).

Dataset 1:  Mobile Prices - https://www.kaggle.com/iabhishekofficial/mobile-price-classification

Dataset 2: Chess - https://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King%29


## Getting Started & Prerequisites
For testing on your own machine, you need only to install python 3.7 and the following packages:
- pandas, numpy, scikit-learn, matplotlib, itertools, timeit, scipy, yellowbrick

## Run Instructions

1. Place a dataset in to the Data subfolder.
   * Data must be to solve a classification problem and have a specific column containing the class
   * There should be no missing data
   * If there is any non-numeric data, the code will treat it as a categorical and one hot encode those columns
2. In the main.py under main.py call a run_ul_algos function, see description below. This code will run the dataset to generate a series of plots and scores for the algorithms as described above.    
    * run_ul_algos(
      * filename - the file name,
      * result_col - The column storing the class to be determined
      * scalar - 0, 1, 2 - 0 applies no scaling to the data, 1 applies minMaxScalar, 2 applies StandardScalar
      * random_seed - int - random seed to use for the various algorithms for reproducibility
      * make_graphs - Boolean - Whether to generate graphs when running
      * verbose - Boolean - Whether to print a number of de bug statements
      * rNN - Boolean - Whether to generate Neural Networks
      * pNN - Specify HPs to run the Neural Network for, if blank and rNN = True a GridSearch will be run to find the best HPs
3. Graphs will be sored in Images, and results of the GridSearch will be sored in ParamTests 