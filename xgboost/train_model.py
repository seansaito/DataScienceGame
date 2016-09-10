######################## IMPORT LIBRARIES ############################################

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score, log_loss
from sklearn.externals import joblib
import pandas as pd
import csv
rng = np.random.RandomState(31337)

####################### EXTRACT DATASET #############################################

our_file = np.load('forty_column_dataset.npz')
your_file = np.load('juice_test.npz')
print our_file.files
X_original = our_file['arr_0']
Y_original = our_file['arr_1']
X_evaluate = your_file['arr_0']

X = X_original
Y = Y_original

print X_train.shape # As a check to ensure the right dataset is loaded
################################## RUN XGBOOST ############################################
from sklearn.grid_search import GridSearchCV
import xgboost as xgb

kf = KFold(Y.shape[0], n_folds=2, shuffle=True, random_state=rng)
for train_index, test_index in kf:
    xgb_model = xgb.XGBClassifier().fit(X[train_index],Y[train_index])
    predictions = xgb_model.predict(X[test_index])
    actuals = Y[test_index]
    print predictions

################################## GET SCORES & LOG LOSS ##################################

print(confusion_matrix(actuals, predictions))
print(accuracy_score(actuals, predictions))
predictions_probs = xgb_model.predict_proba(X[test_index])
loss = log_loss(actuals, predictions_probs)
print loss

##################### SAVE MODEL ##############################################
saved_model = 'xgboost_model.sav'
joblib.dump(xgb_model, saved_model)

###################### RUN MODEL #################################################
print "Starting to predict model..."
pred = xgb_model.predict_proba(X_evaluate)
print pred.shape # To check and verify shape of the CSV
pred = pd.DataFrame(pred)
pred = pd.DataFrame(pred[1])

######################## GENERATE CSV ############################################
pred.to_csv("Y_priv.predict", headers=False, index=False)
a_predictions = pd.read_csv("Y_priv.predict")
np.savetxt("Y_priv.predict", a_predictions)
print a_predictions.shape
