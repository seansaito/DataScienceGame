
# coding: utf-8

# # Notebook for optimizing different classifiers and ensembling them
# 
# Here is how the algorithm works for training:
# - Transform the dataset by deleting and adding features (should be done by this point)
# - We create N (adjustable) sets of undersampled data (should be done by this point)
# - For each undersampled set, there will be a blend of classifiers:
#     - Tensorflow
#     - XGBoost
#     - RandomForest
#     - SVM
#     - ...
# - For each classifier for each undersampled set, we optimize the hyperparameters
# - After optimization, we do a small ensemble learning for each undersampled data
# - Every model is saved
# 
# For testing:
# - Transform the dataset (should be done by this point)
# - Ensemble classification on the whole dataset (no undersampling)
# 
# Generating and ensembling predictions
# - Each model generates the likelihood that the user will convert
# - Ensembling works by taking a weighted mean of all the votes, the weights being the accuracy of the model.
# 
# 
# ## The dataset

# In[1]:

# Imports and Global Vars => add any which are necessary

# Standard imports
import numpy as np
import pandas as pd
import scipy as sc
import sklearn
import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

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


# In[ ]:

# The data
X_train_file = "data/X_train.npz"
X_valid_file = "data/X_valid.npy"
y_train_file = "data/y_train.npz"
y_valid_file = "data/y_valid.npy"
X_test_file = "data/X_test.npy"

X_train_clusters = np.load(X_train_file)
X_valid = np.load(X_valid_file)
y_train_clusters = np.load(y_train_file)
y_valid = np.load(y_valid_file)
X_test = np.load(X_test_file)


# In[ ]:

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


# In[ ]:

# Parameters for the gridsearch
cv = 5 # Cross validation
n_jobs = -1
scoring = "log_loss"

# Set up the pipeline of your classifier
name = "RandomForestClassifier"
pipe = Pipeline([("clf", RandomForestClassifier())])

# Get the params
params = get_params("penalty", "C")

# The GridSearch
gs_pipe = GridSearchCV(pipe, params, scoring=scoring, cv=cv, verbose=1, n_jobs=n_jobs)


# In[ ]:

X_train_clusters.keys()


# In[ ]:

# What we actually want to do is generate a gs_pipe for each undersampled set
num_examples = len(X_train_clusters.keys())

# Group the clusters
X_trains = [X_train_clusters["arr_{0}".format(i)] for i in range(num_examples)]
y_trains = [y_train_clusters["arr_{0}".format(i)] for i in range(num_examples)]

# Create a gridsearch for each dataset
gs_pipes = [gs_pipe for i in range(num_examples)]
pipes = [pipe for i in range(num_examples)]


# In[ ]:

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

p = Pool(5)

if param_search:
    best_classifiers = p.map(fit_custom, data_gs_pipe_pairs)
else:
    best_classifiers = p.map(fit_no_search, data_pipe_pairs)


# In[ ]:

# Save each sklearn classifier in a folder called clfs
filenames = ["clfs/{name}_{num}.pkl".format(name=name, num=i) for i in range(num_examples)]

for fname, best_clf in zip(filenames, best_classifiers):
    joblib.dump(best_clf, fname)


# In[ ]:

# Inference
num_tests = X_test.shape[0]

def vote():
    predictions = np.zeros((num_tests,1))
    for clf in best_classifiers:
        probs = clf.predict_proba(X_test)
        probs = np.reshape(probs, (num_tests, 1))
        predictions = np.hstack(predictions, probs)
    means = np.mean(predictions, axis=1)
    return means


# In[ ]:

res = vote()

