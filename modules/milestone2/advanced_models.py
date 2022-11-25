from comet_ml import Experiment
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.utils.class_weight import compute_class_weight
from argparse import ArgumentParser
import os
import configparser
np.random.seed(42)
from sklearn.metrics import confusion_matrix
from matplotlib.gridspec import GridSpec

import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN 
from graphs import *

####### comet setup, always check command arg and model naming

'''
model code 0: with distance and angle
model code 1: all features
model code 2: with feature selection
''' 
import sys
parser = ArgumentParser()
parser.add_argument("-m", "--model_code", help="model type to choose from the 3", type = int, default=0)
args = parser.parse_args()
model_code = args.model_code

### TODO:keep experiment and filename same
exp_type = ["xgboost_dist_angle","xgboost_all_features","xgboost_feature_selection"]
exp_name = exp_type[model_code]+'_class_weights'

config = configparser.ConfigParser()
config.read('../configfile.ini')
type_env = "comet_ml_prod" #comet_ml_prod
COMET_API_KEY = config[type_env]['api_key']
COMET_PROJECT_NAME = config[type_env]['project_name_advanced']
COMET_WORKSPACE = config[type_env]['workspace']

comet_exp_obj = Experiment(api_key=COMET_API_KEY,
                           project_name=COMET_PROJECT_NAME,
                           workspace=COMET_WORKSPACE,
                           log_code=True
                          )
comet_exp_obj.set_name(name=exp_name)
comet_exp_obj.log_code("advanced_models.py")
#######

###### load data and class imbalance handling
ada = ADASYN(random_state=42)
x_train = pd.read_pickle("../data/dataset/x_train.pkl").drop(columns=['is_goal','game_id','season'])
x_val = pd.read_pickle("../data/dataset/x_val.pkl").drop(columns=['is_goal','game_id','season'])
y_train = pd.read_pickle("../data/dataset/y_train.pkl")
y_val = pd.read_pickle("../data/dataset/y_val.pkl")

#x_train, y_train = ada.fit_resample(x_train,y_train) 
cw = (len(y_train)- y_train.sum())/y_train.sum()
#cw = 1
######

###### model gridsearch and graphs

### 0: XGB with distance and angle
if model_code == 0:
    params = {
                'objective':['binary:logistic'],
                'max_depth': [0],
                'reg_alpha': [0.1,1.0],
                'learning_rate': [0.1,0.3],
                'n_estimators':[4,6,8,10]
    }

    model = XGBClassifier(random_state=42, scale_pos_weight = cw)
    grid = GridSearchCV(estimator=model,param_grid = params, scoring='f1',cv=3,return_train_score=True)
    grid.fit(x_train[['angle','distance']].to_numpy(),y_train.to_numpy())
    print(grid.best_params_)
    y_pred = grid.best_estimator_.predict(x_train[['angle','distance']].to_numpy())
    print("training:\n",classification_report(y_train.to_numpy(),y_pred))
    y_pred = grid.best_estimator_.predict(x_val[['angle','distance']].to_numpy())
    print("validation:\n",classification_report(y_val.to_numpy(),y_pred))

    log_comet(comet_exp_obj,exp_name,grid.best_estimator_,x_val[['angle','distance']].to_numpy(),y_val.to_numpy())
    create_roc_auc_curve(comet_exp_obj,exp_name,grid.best_estimator_,x_val[['angle','distance']].to_numpy(),y_val.to_numpy())
    plot_cumulative_goal(comet_exp_obj,exp_name,grid.best_estimator_,x_val[['angle','distance']].to_numpy(),y_val.to_numpy())
    create_extimator_plot(comet_exp_obj,exp_name,grid.best_estimator_,x_val[['angle','distance']].to_numpy(),y_val.to_numpy())
    plot_goal_shot_rate(comet_exp_obj,exp_name,grid.best_estimator_,x_val[['angle','distance']].to_numpy(),y_val.to_numpy())

###### 1: XGB all features
elif model_code == 1:
    params = {
            'objective':['binary:logistic'],
            'max_depth': [0],
            'reg_alpha': [1.0],
            'reg_lambda':[1.0],
            'learning_rate': [0.3],
            'n_estimators':[25,45,70,100]
        }

    model = XGBClassifier(random_state=42,scale_pos_weight = cw)
    grid1 = GridSearchCV(estimator=model,param_grid = params, scoring='f1',cv=3,return_train_score=True)
    grid1.fit(x_train.to_numpy(),y_train.to_numpy())
    results = grid1.cv_results_
    grid1.fit(x_train.to_numpy(),y_train.to_numpy())
    print(grid1.best_params_)
    y_pred = grid1.best_estimator_.predict(x_train.to_numpy())
    print(classification_report(y_train.to_numpy(),y_pred))
    y_pred = grid1.best_estimator_.predict(x_val.to_numpy())
    print(classification_report(y_val.to_numpy(),y_pred))
    
    log_comet(comet_exp_obj,exp_name,grid1.best_estimator_,x_val.to_numpy(),y_val.to_numpy())
    create_roc_auc_curve(comet_exp_obj,exp_name,grid1.best_estimator_,x_val.to_numpy(),y_val.to_numpy())
    plot_cumulative_goal(comet_exp_obj,exp_name,grid1.best_estimator_,x_val.to_numpy(),y_val.to_numpy())
    create_extimator_plot(comet_exp_obj,exp_name,grid1.best_estimator_,x_val.to_numpy(),y_val.to_numpy())
    plot_goal_shot_rate(comet_exp_obj,exp_name,grid1.best_estimator_,x_val.to_numpy(),y_val.to_numpy())

###### 1: XGB Feature Selection
elif model_code == 2:

    fs = ['angle', 'distance_from_last_event', 'empty_net', 'shot_type_Wrap-around', 'y_coordinate', 'speed', 'distance', 'x_coordinate', 'game_period', 'shot_type_Tip-In', 'shot_type_Wrist Shot', 'game_seconds'] 
    params = {
            'objective':['binary:logistic'],
            'max_depth': [0],
            'reg_alpha': [1.0],
            'reg_lambda':[1.0],
            'learning_rate': [0.1,0.4],
            'n_estimators':[25,35,50,70,100]
        }

    model = XGBClassifier(random_state=42,scale_pos_weight = cw)
    grid3 = GridSearchCV(estimator=model,param_grid = params, scoring='f1',cv=3,return_train_score=True)
    grid3.fit(x_train[fs].to_numpy(),y_train.to_numpy())
    results = grid3.cv_results_

    y_pred = grid3.best_estimator_.predict(x_train[fs].to_numpy())
    print(classification_report(y_train.to_numpy(),y_pred))
    y_pred = grid3.best_estimator_.predict(x_val[fs].to_numpy())
    print(classification_report(y_val.to_numpy(),y_pred))

    log_comet(comet_exp_obj,exp_name,grid3.best_estimator_,x_val[fs].to_numpy(),y_val.to_numpy())
    create_roc_auc_curve(comet_exp_obj,exp_name,grid3.best_estimator_,x_val[fs].to_numpy(),y_val.to_numpy())
    plot_cumulative_goal(comet_exp_obj,exp_name,grid3.best_estimator_,x_val[fs].to_numpy(),y_val.to_numpy())
    create_extimator_plot(comet_exp_obj,exp_name,grid3.best_estimator_,x_val[fs].to_numpy(),y_val.to_numpy())
    plot_goal_shot_rate(comet_exp_obj,exp_name,grid3.best_estimator_,x_val[fs].to_numpy(),y_val.to_numpy())
