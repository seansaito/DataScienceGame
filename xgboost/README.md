# **Training the XGBoost model for AXA Dataset**

For the sake of brevity, you should already have the necessary scikit libraries and just need to install XGBoost Libraris
'''
$   cd /usr/local/src && mkdir xgboost && cd xgboost && \
    git clone --depth 1 --recursive https://github.com/dmlc/xgboost.git && cd xgboost && \
    make && cd python-package && python setup.py install && \
'''
Then you can run the train model script. Ensure you have downloaded the .npz files for the Training Set and the Test Set from Rohan
