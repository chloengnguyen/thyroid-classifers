# imports
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
from statsmodels.tools import add_constant
from imblearn.over_sampling import RandomOverSampler
​
from collections import Counter
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import seaborn as sns
# %matplotlib inline
plt.style.use('fivethirtyeight')
​
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, randint
​
# load in data
df = pd.read_json('data/data.json')
​
# create a Fraud column based on acct_type
df['fraud'] = 1
df.loc[df['acct_type'] == 'fraudster_event', 'fraud'] = 0
df.loc[df['acct_type'] == 'fraudster', 'fraud'] = 0
df.loc[df['acct_type'] == 'fraudster_att', 'fraud'] = 0
​
# replace NaN values with 0s
cols = ['approx_payout_date', 'body_length', 'channels', 'delivery_method', 'event_start', 'event_end', 
    'event_published', 'event_start', 'fb_published', 'gts', 'has_analytics', 'has_header', 
    'has_logo', 'name_length', 'object_id', 'org_facebook', 'org_twitter', 'sale_duration', 
    'sale_duration2', 'show_map', 'user_created', 'user_type', 'fb_published', 'num_order', 
    'num_payouts', 'user_age']
​
cols_fr = ['approx_payout_date', 'body_length', 'channels', 'delivery_method', 'event_start', 'event_end', 
    'event_published', 'event_start', 'fb_published', 'gts', 'has_analytics', 'has_header', 
    'has_logo', 'name_length', 'object_id', 'org_facebook', 'org_twitter', 'sale_duration', 
    'sale_duration2', 'show_map', 'user_created', 'user_type', 'fb_published', 'num_order', 
    'num_payouts', 'user_age', 'fraud']
​
df[cols] = df[cols].fillna(value = 0)
​
# set aside 100 data points to predict on after model is created and save them to a csv
# test_script_examples2 = df[cols].iloc[:100, :]
# test_script_examples2.to_csv('test_script_examples3.csv', index=False)
​
# set up X (feature dataframe) and y (target values) for model creation
X = df[['approx_payout_date', 'body_length', 'channels', 'delivery_method', 'event_start', 
        'event_end', 'event_published', 'event_start', 'fb_published', 'gts', 'has_analytics', 
        'has_header', 'has_logo', 'name_length', 'object_id', 'org_facebook', 'org_twitter', 
        'sale_duration', 'sale_duration2', 'show_map', 'user_created', 'user_type', 
        'fb_published', 'num_order', 'num_payouts', 'user_age']].iloc[100:, :].values
y = df['fraud'][100:]
​
# set up a train-test split on X and y, using stratify to preserve class ratios in target column
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=df['fraud'][100:])
​
# take an oversample of data to account for the 10%/90% fraud/not fraud ratio
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train, y_train)
​
# create an XGBoost classifer model
# xgb_model = xgb.XGBClassifier()
​
# set up randomized parameters
# params = {
#     "colsample_bytree": uniform(0.9, 0.5),
#     "gamma": uniform(0, 0.5),
#     "learning_rate": uniform(0.02, 0.2), # default = 0.1 
#     "max_depth": randint(2, 5), # default = 3
#     "n_estimators": randint(100, 300), # default = 100
#     "subsample": uniform(0.9, 0.5),
#     "reg_lambda": randint(0,6),
#     "min_child_weight" : uniform(1,7)
#     }
​
# conduct a randomized GridSearchCV
# search= RandomizedSearchCV(xgb_model, param_distributions=params,scoring="neg_log_loss", random_state=42, n_iter=1500, cv=5, verbose=1, n_jobs=-1, return_train_score=True)
​
# fit to oversampled training data
# search.fit(X_res, y_res)
​
# predict on test data
# xgb_pred= search.best_estimator_.predict(X_test)
​
# print confusion matrix metrics
# print(accuracy_score(y_test, xgb_pred))
# print(precision_score(y_test, xgb_pred))
# print(recall_score(y_test, xgb_pred))
# print(confusion_matrix(y_test, xgb_pred))
# print(search.best_estimator_)
​
# save model and hyperparameters
​
bst6 = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.9621486117427724,
              gamma=0.36543373760182213, learning_rate=0.20766809136420758,
              max_delta_step=0, max_depth=4,
              min_child_weight=1.4654738715674425, missing=None,
              n_estimators=139, n_jobs=1, nthread=None,
              objective='binary:logistic', random_state=0, reg_alpha=0,
              reg_lambda=5, scale_pos_weight=1, seed=None, silent=None,
              subsample=0.9240473219820439, verbosity=1)
​
# fit model to oversampled training data
bst6.fit(X_res, y_res)
​
# predict on test data
xgb_pred = bst6.predict(X_test)
​
# save predicted probabilities of target values on test data
xgb_pred_proba = bst6.predict_proba(X_test)
​
# print metric results of model
print('Test Set Scores:')
print(f'Accuracy: {accuracy_score(y_test, xgb_pred):.4f}')
print(f'Precision: {precision_score(y_test, xgb_pred):.4f}')
print(f'Recall: {recall_score(y_test, xgb_pred):.4f}')
TN = confusion_matrix(y_test, xgb_pred)[0][0]
FP = confusion_matrix(y_test, xgb_pred)[0][1]
FN = confusion_matrix(y_test, xgb_pred)[1][0]
TP = confusion_matrix(y_test, xgb_pred)[1][1]
print(f'TN: {TN} FP: {FP} FN: {FN} TP: {TP}')