import matplotlib.pyplot as plt
from comet_ml import Experiment
import os
import pickle
import configparser
import pandas as pd
from sklearn.metrics import confusion_matrix
from matplotlib.gridspec import GridSpec
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score
from sklearn.calibration import CalibrationDisplay
import warnings
warnings.filterwarnings('ignore')


def log_comet(comet_exp_obj, name, model, x_val, y_val):
    filename = '../../model/' + name + "_Model.pkl"
    pickle.dump(model, open(filename, 'wb'))
    comet_exp_obj.log_model(name, file_or_folder=filename, overwrite=True, file_name=name + "_Model")

    y_pred = model.predict(x_val)
    y_prob = model.predict_proba(x_val)[:,1]
    metrics = {
        "Accuracy":accuracy_score(y_val,y_pred),
        "Roc_auc":roc_auc_score(y_val,y_prob),
        "f1_score":f1_score(y_val,y_pred),
    }

    comet_exp_obj.log_metrics(metrics)
    comet_exp_obj.log_confusion_matrix(y_true=y_val, y_predicted=y_pred, title="Confusion Matrix", 
                                           file_name="Confusion Matrix for "+name)

def plot_cumulative_goal(comet_exp_obj,name, model, x_val, y_val):
    
    testy = y_val
    ns_probs = [0 for _ in range(len(y_val))]
    
    predicted_probablities = model.predict_proba(x_val)

    goal_prob = predicted_probablities[:, 1] # taking only the goal probablities
    
    df = pd.DataFrame(y_val, columns=["is_goal"])
    df["probablity_of_goal"] = goal_prob
    
    df['percentile_of_goal'] = round(df["probablity_of_goal"].rank(pct = True)*100)
    goal_rate = df.groupby(by='percentile_of_goal').sum()
    goal_rate['percentile'] = goal_rate.index
    
    goal_rate['cum_sum'] = goal_rate.loc[::-1, 'is_goal'].cumsum()[::-1]
    goal_rate['cum_perc'] = 100*goal_rate['cum_sum']/ goal_rate['is_goal'].sum()


    graph, (plt1) = plt.subplots(1, 1)
    plt1.plot(goal_rate["percentile"], goal_rate["cum_perc"], label=name)
    plt1.set_ylim(0,100)
    plt1.invert_xaxis()
    plt1.set_xlabel('Shot probablity model percentile')
    plt1.set_ylabel('Proportion')
    plt.grid()
    plt.title("Cumulative % of goals")
    
    comet_exp_obj.log_figure(figure_name="Cumulative % of goals ", figure=plt,
                    overwrite=False, step=None)
    
    plt.show()


def create_extimator_plot(comet_exp_obj,name, model, x_val, y_val):
    testy = y_val
    ns_probs = [0 for _ in range(len(y_val))]
    
    predicted_probablities = model.predict_proba(x_val)
    goal_prob = predicted_probablities[:, 1] # taking only the goal probablities
    
    disp = CalibrationDisplay.from_predictions(y_val, goal_prob)
    disp = CalibrationDisplay.from_estimator(model, x_val, y_val)
    
    comet_exp_obj.log_figure(figure_name="Calibration Display ", figure=plt,
                 overwrite=False, step=None)
    
    plt.show()

def plot_goal_shot_rate(comet_exp_obj,name, model, x_val, y_val):
    testy = y_val
    ns_probs = [0 for _ in range(len(y_val))]
    
    predicted_probablities = model.predict_proba(x_val)

    goal_prob = predicted_probablities[:, 1] # taking only the goal probablities
    
    df = pd.DataFrame(y_val, columns=["is_goal"])
    df["probablity_of_goal"] = goal_prob
    
    df['percentile_of_goal'] = round(df["probablity_of_goal"].rank(pct = True)*100)
    goal_rate = round((df.groupby(by='percentile_of_goal').sum() / 
                       df.groupby(by='percentile_of_goal').count())*100)
    goal_rate['percentile'] = goal_rate.index

    fig, (plt1) = plt.subplots(1, 1)
    plt1.plot(goal_rate["percentile"], goal_rate["is_goal"], label=name)
    plt1.set_ylim(0,100)
    plt1.xaxis.set_major_formatter('{x:1.0f}%')
    plt1.yaxis.set_major_formatter('{x:1.0f}%')
    plt1.invert_xaxis()
    plt1.set_xlabel('Shot probablity model percentile')
    plt1.set_ylabel('Goals / (No Goals + Goals)')
    plt.grid()
    plt.title("Goal Rate")

    comet_exp_obj.log_figure(figure_name="Goal Rate", figure=plt,
                   overwrite=False, step=None)
    
    plt.show()


def create_roc_auc_curve(comet_exp_obj,name, model, x_val, y_val):    
    testy = y_val
    ns_probs = [0 for _ in range(len(testy))]

    predicted_probablities = model.predict_proba(x_val)
    goal_prob = predicted_probablities[:, 1] # taking only the goal probablities
    # https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
    # calculate scores
    ns_auc = roc_auc_score(testy, ns_probs)
    lr_auc = roc_auc_score(testy, goal_prob)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(testy, goal_prob)
    # plot the roc curve for the model
    plt.figure()
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Base Line')
    plt.plot(lr_fpr, lr_tpr, marker='.', label=name)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Receiver Operating Characteristic (ROC) curves and the AUC metric")
    # show the legend
    plt.legend()
    plt.text(0, 1, 'No Skill: ROC AUC=%.3f\nLogistic: ROC AUC=%.3f\n'%(ns_auc, lr_auc), fontsize=10, 
             transform=plt.gcf().transFigure)
    # show the plot    
    
    #metrics = {"roc" : lr_auc}
    #comet_exp_obj.log_metrics(metrics)
    #comet_exp_obj.log_dataset_hash(x_train)
    comet_exp_obj.log_figure(figure_name="Receiver Operating Characteristic (ROC) curves and the AUC metric", 
                             figure=plt, overwrite=False, step=None)
    
    plt.show()
    



    