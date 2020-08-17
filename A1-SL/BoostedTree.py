
# https://scikit-learn.org/stable/modules/ensemble.html
# No a boosted tree comes with 2 APIs. the API for the booster and one for the base learner (decision tree)

from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
