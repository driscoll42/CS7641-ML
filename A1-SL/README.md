## CS 7641 Assignment 1: Supervised Learning Classification
The purpose of this project is to explore some techniques in supervised learning.
 
 * Neural Networks
     * For the neural network you should implement or steal your favorite kind of network and training algorithm. You may use networks of nodes with as many layers as you like and any activation function you see fit. 
 * Support Vector Machines (SVM)
     * You should implement (for sufficiently loose definitions of implement including "download") SVMs. This should be done in such a way that you can swap out kernel functions. I'd like to see at least two. 
 * k-Nearest Neighbors (k-NN)
     * You should implement kNN. Use different values of k. 
 * Decision Trees
     * For the decision tree, you should implement or steal a decision tree algorithm. Be sure to use some form of pruning. You are not required to use information gain (for example, there is something called the GINI index that is sometimes used) to split attributes, but you should describe whatever it is that you do use.
 * Boosted Trees
     * Implement a boosted version of your decision trees. As before, you will want to use some form of pruning, but presumably because you're using boosting you can afford to be much more aggressive about your pruning. 
 
Each algorithm will be run for two binary classification datasets so that we can compare and contrast them for two different problems (one for a balanced target variable and the other for an unbalanced target variable).

Dataset 1:  TBD

Dataset 2: TBD


## Getting Started & Prerequisites
For testing on your own machine, you need only to install python 3.7 and the following packages:
- pandas, numpy, scikit-learn, matplotlib, itertools, timeit

Make sure this is up to date


## Running the Classifiers

TBD

Linear Algebra

    Linear Algebra and Eigenproblems

ML is the ROX

    Mitchell Ch 1

Decision Trees

    Mitchell Ch 3

Regression and Classification
Neural Networks

    Mitchell Ch 4

Instance-Based Learning

    Mitchell Ch 8

Ensemble Learning

    Schapire's Introduction
    Jiri Matas and Jan Sochman's Slides

Kernel Methods and SVMs

    An introduction to SVMs for data mining
    Christopher Burges tutorial on SVMs for pattern recognition
    Scholkopf's NIPS tutorial slides on SVMs and kernel methods
    
Note: Below you should do feature extraction AFTER test/train data split - https://redd.it/ib8d90
![Supervised Learning Workflow](https://i.redd.it/7zfxkyey2ih51.png)