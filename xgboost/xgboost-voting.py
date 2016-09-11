# Imports and Global Vars => add any which are necessary
# Ensure that you have the data in the folder './data/undersampled_datasets?*'

# Standard imports
import numpy as np
import pandas as pd
import scipy as sc
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import xgboost as xgb
#%matplotlib inline

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Preprocessing modules (Shouldn't be needed by now)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split

# Gridsearching and Parameter Optimization
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

# For efficient saving of models
from sklearn.externals import joblib

# The data
X_train_file = "./data/undersampled_datasets/X_train.npz"
X_valid_file = "./data/undersampled_datasets/X_valid.npy"
y_train_file = "./data/undersampled_datasets/y_train.npz"
y_valid_file = "./data/undersampled_datasets/y_valid.npy"
X_test_file = "./data/undersampled_datasets/X_test.npy"

X_train_clusters = np.load(X_train_file)
X_valid = np.load(X_valid_file)
y_train_clusters = np.load(y_train_file)
y_valid = np.load(y_valid_file)
X_test = np.load(X_test_file)

# Function for setting the parameters
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = {'clf__penalty': ['l1', 'l2'],
              'clf__C': param_range,
              "clf__fit_intercept": [True, False],
              "clf__kernel": ["rbf", "sigmoid", "poly", "linear"],
              "clf__gamma": param_range}

def get_params(*args):
    """
    Returns a list of a dictionary of parameter options

    Usage:
        get_params('penalty', 'C', 'kernel')

    Returns:
        [{
            'clf__penalty': ...,
            'clf__C': ...,
            'clf__kernel': ...,
        }]
    """
    to_return = [{}]
    for arg in args:
        to_return[0]["clf__" + arg] = param_grid["clf__" + arg]

    return to_return

# Parameters for the gridsearch
cv = 5 # Cross validation
n_jobs = -1
scoring = "log_loss"

# Set up the pipeline of your classifier
name = "XGBoost"
pipe = Pipeline([("clf", xgb.XGBClassifier())])

# Get the params
params = get_params("penalty", "C")

# The GridSearch
gs_pipe = GridSearchCV(pipe, params, scoring=scoring, cv=cv, verbose=1, n_jobs=n_jobs)

# What we actually want to do is generate a gs_pipe for each undersampled set
num_examples = len(X_train_clusters.keys())

# Group the clusters
print 'Starting Clusters'
X_trains = [X_train_clusters["arr_{0}".format(i)] for i in range(num_examples)]
y_trains = [y_train_clusters["arr_{0}".format(i)] for i in range(num_examples)]

# Create a gridsearch for each dataset
gs_pipes = [gs_pipe for i in range(num_examples)]
pipes = [pipe for i in range(num_examples)]

# Now the fun part - hyperparameter search for each model for each undersample

param_search = False
X_y_pairs = zip(X_trains, y_trains)
data_gs_pipe_pairs = zip(gs_pipes, X_y_pairs)
data_pipe_pairs = zip(pipes, X_y_pairs)

def fit_custom(pair):
    gs_pipe, X_y_pair = pair
    X_train, y_train = X_y_pair
    gs_pipe.fit(X_train, y_train)
    return gs_pipe.best_estimator_

def fit_no_search(pair):
    pipe, X_y_pair = pair
    X_train, y_train = X_y_pair
    pipe.fit(X_train, y_train)
    return pipe

# Let's do this in parallel
from multiprocessing import Pool

print 'Starting Parallelization'
p = Pool(5)

if param_search:
    best_classifiers = p.map(fit_custom, data_gs_pipe_pairs)
else:
    best_classifiers = p.map(fit_no_search, data_pipe_pairs)

# Save each sklearn classifier in a folder called clfs
print 'Saving model'
filenames = ["clfs/{name}_{num}.pkl".format(name=name, num=i) for i in range(num_examples)]

for fname, best_clf in zip(filenames, best_classifiers):
    joblib.dump(best_clf, fname)

# Inference
num_tests = X_test.shape[0]

def vote():
    predictions = np.zeros((num_tests,1))
    predictions_weighted = np.zeros((num_tests,1))
    predictions_inverse_weighted = np.zeros((num_tests,1))
    scores = [c.score(X_valid, y_valid) for c in best_classifiers]
    for idx, clf in enumerate(best_classifiers):
        probs = clf.predict_proba(X_test)
        # Take the right column
        probs = probs[:, 1]
        probs = np.reshape(probs, (num_tests, 1))
        probs_weighted = probs * scores[idx]
        probs_inverse_weighted = probs * (1 / scores[idx])
        predictions = np.hstack((predictions, probs))
        predictions_weighted = np.hstack((predictions_weighted, probs_weighted))
        predictions_inverse_weighted = np.hstack((predictions_inverse_weighted, probs_inverse_weighted))
    means = np.mean(predictions, axis=1)
    means_weighted = np.mean(predictions_weighted, axis=1)
    means_inverse_weighted = np.mean(predictions_inverse_weighted, axis=1)
    return (means, means_weighted, means_inverse_weighted)

res, res_weight, res_inverse_weight = vote()

np.savetxt("{name}_votes.csv".format(name=name), res)
np.savetxt("{name}_votes_weighted.csv".format(name=name), res_weight)
np.savetxt("{name}_votes_inverse_weighted.csv".format(name=name), res_inverse_weight)
