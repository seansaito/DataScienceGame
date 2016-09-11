
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
import joblib


# In[ ]:

train = pd.read_csv('forty_column_dataset.csv')
juiceboy = pd.read_csv('juice_test.csv')
target = 'labels'

# In[ ]:

xgbl = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

supplied_arr = juiceboy
predictors = [x for x in train_3.columns if x not in [target,'Unnamed: 0', u'CustomerMD5Key']]
dtrain = train_3
alg = xgbl
useTrainCV=True
cv_folds=5
early_stopping_rounds=50
predictors = [x for x in train_3.columns if x not in [target,'Unnamed: 0', u'CustomerMD5Key']]
    
if useTrainCV:
    xgb_param = alg.get_xgb_params()
    xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
        metrics='auc', early_stopping_rounds=early_stopping_rounds)
    alg.set_params(n_estimators=cvresult.shape[0])

    
#Fit the algorithm on the data
alg.fit(dtrain[predictors], dtrain['labels'],eval_metric='auc')

joblib.dump(clf, 'woru_full_fit.pkl') 

#Predict training set:
dtrain_predictions = alg.predict(dtrain[predictors])
dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]

#these are the labels to be written to file
dtrain_predlabels = alg.predict_proba(supplied_arr)
np.savetxt("hyper_shiokalpha_woru.csv", [x[1] for x in dtrain_predprob] , delimiter="\n")

print "\nModel Report"
print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['labels'].values, dtrain_predictions)
print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['labels'], dtrain_predprob)

