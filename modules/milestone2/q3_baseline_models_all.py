#!/usr/bin/env python
# coding: utf-8

# In[1]:


import configparser
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from comet_ml import Experiment
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from constant import COMET_FILE
warnings.filterwarnings('ignore')
np.random.seed(42)
# In[2]:


config = configparser.ConfigParser()
# config.read('../configfile.ini')
config.read(COMET_FILE)
type_env = "comet_ml_dev"  # comet_ml_prod
COMET_API_KEY = config[type_env]['api_key']
COMET_PROJECT_NAME = config[type_env]['project_name_baseline']
COMET_WORKSPACE = config[type_env]['workspace']

comet_exp_obj = Experiment(api_key=COMET_API_KEY,
                           project_name=COMET_PROJECT_NAME,
                           workspace=COMET_WORKSPACE,
                           log_code=True
                           )
comet_exp_obj.set_name(name="Baseline Models")
comet_exp_obj.log_notebook("9_baseline_models_all.ipynb")


# In[3]:


def create_train_val_datafile():
    """
    This function takes the original dataset and split the dataset in a stratified form into train and valid
    dataset.
    :return: None
    """
    data = pd.read_pickle("../../data/trainvaldata/train_set.pkl")
    data = data.dropna()
    x_train, x_val, y_train, y_val = train_test_split(data, data["is_goal"], test_size=0.2, stratify=data["is_goal"],
                                                      random_state=42)
    train_df = x_train.copy()
    train_df["is_goal"] = y_train
    train_df["train-val"] = 0

    val_df = x_val.copy()
    val_df["is_goal"] = y_val
    val_df["train-val"] = 1

    df = train_df.append(val_df, ignore_index=True)
    df.to_pickle(
        "../../data/trainvaldata/train_val_df.pkl")  # Raphael use this and preprocess and share the x_train, y_train ....


# create_train_val_datafile()


# In[4]:


x_train = pd.read_pickle("../../data/trainvaldata/x_train.pkl")
x_val = pd.read_pickle("../../data/trainvaldata/x_val.pkl")
y_train = pd.read_pickle("../../data/trainvaldata/y_train.pkl")
y_val = pd.read_pickle("../../data/trainvaldata/y_val.pkl")

comet_exp_obj.log_dataframe_profile(x_train, "x_train")
comet_exp_obj.log_dataframe_profile(y_train, "y_train")
comet_exp_obj.log_dataframe_profile(x_val, "x_val")
comet_exp_obj.log_dataframe_profile(y_val, "y_val")

x_train_df_distance = x_train[['distance']]
x_train_df_angle = x_train[['angle']]
x_train_df_distance_angle = x_train[['distance', 'angle']]
y_train = y_train.values

x_val_df_distance = x_val[['distance']]
x_val_df_angle = x_val[['angle']]
x_val_df_distance_angle = x_val[['distance', 'angle']]
y_val = y_val.values

data_train_dict = {
    "LinearModel_Distance": (x_train_df_distance, y_train, 1),
    "LinearModel_Angle": (x_train_df_angle, y_train, 1),
    "LinearModel_Angle_Distance": (x_train_df_distance_angle, y_train, 2),
    "Random_Base_Line": ()
}

data_val_dict = {
    "LinearModel_Distance": (x_val_df_distance, y_val, 1),
    "LinearModel_Angle": (x_val_df_angle, y_val, 1),
    "LinearModel_Angle_Distance": (x_val_df_distance_angle, y_val, 2),
    "Random_Base_Line": ()
}


# In[5]:


def build_log_reg_model(model, x_val, y_val, no_of_features, k):
    """
    This functions tests the accuracy of the logistic model on the validation dataset and log the various
    data into comet ml
    :param model: Trained model in order to predict on the validation set.
    :param x_val: input data for the validation set
    :param y_val: taret values of the validation data set
    :param no_of_features: no of features does the validation set have.
    :param k:
    :return:  None
    """
    if no_of_features == 1:
        x_val = x_val.values.reshape(-1, 1)

    y_pred = model.predict(x_val)
    accuracy = np.mean(y_val == y_pred) * 100
    print("correctly predicted / total is ", accuracy)
    print(classification_report(y_val, y_pred))
    print(confusion_matrix(y_val, y_pred))
    metrics = {k + "_feature_accuracy": accuracy}
    classNames = np.unique(y_val)

    comet_exp_obj.log_metrics(metrics)
    comet_exp_obj.log_confusion_matrix(y_true=y_val, y_predicted=y_pred, title="Confusion Matrix for " + k,
                                       file_name="Confusion Matrix for " + k)


for k, v in data_train_dict.items():
    print("Classification Matrix for BaseLine Model for features - ", k)
    if k != "Random_Base_Line":
        x_train = v[0]
        y_train = v[1]
        x_val = data_val_dict[k][0]
        y_val = data_val_dict[k][1]
        no_of_features = v[2]
        if no_of_features == 1:
            x_train = x_train.values.reshape(-1, 1)
        model = LogisticRegression()
        model.fit(x_train, y_train)
        build_log_reg_model(model, x_val, y_val, no_of_features, k)

        filename = '../../model/' + k + "_Model.pkl"
        pickle.dump(model, open(filename, 'wb'))
        comet_exp_obj.log_model(k, file_or_folder=filename, overwrite=True, file_name=k + "_Model")

    else:
        y_pred = np.array([np.random.uniform(low=0.0, high=1.0) for i in range(y_val.shape[0])])
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        y_pred = y_pred.astype(int)

        accuracy = np.mean(y_val == y_pred) * 100
        metrics = {k + "_feature_accuracy": accuracy}
        comet_exp_obj.log_metrics(metrics)
        print("correctly predicted / total is ", accuracy)
        print(classification_report(y_val, y_pred))
        print(confusion_matrix(y_val, y_pred))
        comet_exp_obj.log_confusion_matrix(y_true=y_val, y_predicted=y_pred, title="Confusion Matrix for " + k,
                                           file_name="Confusion Matrix for " + k)


# ## Question 1
# 
# Evaluate the accuracy (i.e. correctly predicted / total) of your model on the validation set. What do you notice? Look
# at the predictions and discuss your findings. What could be a potential issue? Include these discussions in your blog
# post.
# 
# - Although the accuracy is showing up 90% of accuracy, but this number isn't right in my option. As the major class
# is 0s, and the ratio of training data of shot being goal and not-goal is highly imbalance and is in the order of
# 1:10.
# - From the classification report its clear that all the test data (validation data) has been predicted to 0s as the
# output.
# 

# ## Question 2
# Receiver Operating Characteristic (ROC) curves and the AUC metric of the ROC curve. Include a random classifier
# baseline, i.e. each shot has a 50% chance of being a goal.

# In[6]:


def create_roc_auc_curve(model, x_val, y_val):
    """
    This function is a helper function in order to plot the ROC AUC curve.
    :param model: Trained model which is used to predict on the validation dataset
    :param x_val: Input data of validation dataset
    :param y_val: target values of the valdiation dataset
    :return: values which are required for plotting the roc auc curve which inclues true positive rate, false postive
    rate etc.
    """
    testy = y_val
    ns_probs = [0 for _ in range(len(testy))]

    predicted_probablities = model.predict_proba(x_val)
    goal_prob = predicted_probablities[:, 1]  # taking only the goal probablities
    # https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
    # calculate scores
    ns_auc = roc_auc_score(testy, ns_probs)
    lr_auc = round(roc_auc_score(testy, goal_prob), 8)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(testy, goal_prob)

    return ns_fpr, ns_tpr, lr_fpr, lr_tpr, ns_auc, lr_auc


plt.figure()
table_values = []
markers_list = [",", ",", ",", ",", "P"]
for index, (k, v) in enumerate(data_train_dict.items()):
    lw = 5 - 4 * index / len(data_train_dict)
    ls = ['-', '--', '-.', ':'][index % 4]
    print("AUC ROC Scores for BaseLine Model for features - ", k)
    if k != "Random_Base_Line":
        x_train = v[0]
        y_train = v[1]
        x_val = data_val_dict[k][0]
        y_val = data_val_dict[k][1]
        no_of_features = v[2]
        if no_of_features == 1:
            x_train = x_train.values.reshape(-1, 1)
        model = LogisticRegression()
        model.fit(x_train, y_train)
        ns_fpr, ns_tpr, lr_fpr, lr_tpr, ns_auc, lr_auc = create_roc_auc_curve(model, x_val, y_val)
        table_values.append([ns_auc, lr_auc])
        # plot the roc curve for the model
        plt.plot(lr_fpr, lr_tpr, marker=markers_list[index], label=k, linewidth=lw, linestyle=ls)
    else:
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]
        y_pred = np.array([np.random.uniform(low=0.0, high=1.0) for i in range(y_val.shape[0])])
        goal_prob = y_pred

        ns_auc = roc_auc_score(testy, ns_probs)
        lr_auc = round(roc_auc_score(testy, goal_prob), 8)
        # summarize scores
        print('No Skill: ROC AUC=%.3f' % (ns_auc))
        print('Logistic: ROC AUC=%.3f' % (lr_auc))
        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(testy, goal_prob)
        table_values.append([ns_auc, lr_auc])
        plt.plot(lr_fpr, lr_tpr, marker=markers_list[index], label=k, linewidth=lw, linestyle=ls)

col_labels = ['No Skill: ROC AUC', 'Logistic: ROC AUC']
row_labels = [*data_train_dict]
table_vals = table_values
the_table = plt.table(cellText=table_vals,
                      colWidths=[0.15] * 3,
                      rowLabels=row_labels,
                      colLabels=col_labels,
                      loc='center right')
plt.text(0.8, 0.75, 'Table ROC AUC', size=8)

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Base Line')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("Receiver Operating Characteristic (ROC) curves and the AUC metric")
plt.legend()
comet_exp_obj.log_figure(figure_name="Receiver Operating Characteristic (ROC) curves and the AUC metric",
                         figure=plt, overwrite=True, step=None)
plt.show()


# ## Question 2
# The goal rate (#goals / (#no_goals + #goals)) as a function of the shot probability model percentile, i.e. if a
# value is the 70th percentile, it is above 70% of the data.

# In[7]:


def plot_goal_shot_rate(plt1, model, x_val, y_val, k):
    """
    This is a helper function which will plot the goal shot rate by percentile.
    :param plt1: subplot for which we will plot the goal shot rate
    :param model: trained model which is used to predict on the validation dataset.
    :param x_val: input features of the validation dataset
    :param y_val: target data of the validation dateset
    :param k: model name
    :return: None
    """
    predicted_probablities = model.predict_proba(x_val)
    goal_prob = predicted_probablities[:, 1]  # taking only the goal probablities

    df = pd.DataFrame(y_val, columns=["is_goal"])
    df["probablity_of_goal"] = goal_prob
    df['percentile_of_goal'] = round(df["probablity_of_goal"].rank(pct=True) * 100)
    goal_rate = round((df.groupby(by='percentile_of_goal').sum() /
                       df.groupby(by='percentile_of_goal').count()) * 100)
    goal_rate['percentile'] = goal_rate.index

    plt1.plot(goal_rate["percentile"], goal_rate["is_goal"], label=k)


fig, (plt1) = plt.subplots(1, 1)
for index, (k, v) in enumerate(data_train_dict.items()):
    print("Goal Shot Rate for BaseLine Model for features - ", k)
    if k != "Random_Base_Line":
        x_train = v[0]
        y_train = v[1]
        x_val = data_val_dict[k][0]
        y_val = data_val_dict[k][1]
        no_of_features = v[2]
        if no_of_features == 1:
            x_train = x_train.values.reshape(-1, 1)
        model = LogisticRegression()
        model.fit(x_train, y_train)
        plot_goal_shot_rate(plt1, model, x_val, y_val, k)

    else:
        goal_prob = np.array([np.random.uniform(low=0.0, high=1.0) for i in range(x_val.shape[0])])
        df = pd.DataFrame(y_val, columns=["is_goal"])
        df["probablity_of_goal"] = goal_prob

        df['percentile_of_goal'] = round(df["probablity_of_goal"].rank(pct=True) * 100)
        goal_rate = round((df.groupby(by='percentile_of_goal').sum() /
                           df.groupby(by='percentile_of_goal').count()) * 100)
        goal_rate['percentile'] = goal_rate.index
        plt1.plot(goal_rate["percentile"], goal_rate["is_goal"], label=k)

plt1.set_ylim(0, 100)
plt1.xaxis.set_major_formatter('{x:1.0f}%')
plt1.yaxis.set_major_formatter('{x:1.0f}%')
plt1.invert_xaxis()
plt1.set_xlabel('Shot probablity model percentile')
plt1.set_ylabel('Goals / (No Goals + Goals)')
plt.grid()
plt.title("Goal Rate")
plt.legend()
comet_exp_obj.log_figure(figure_name="Goal Rate", figure=fig,
                         overwrite=True, step=None)
plt.show()


# In[8]:


def plot_cumulative_goal(plt1, model, x_val, y_val, k, ls, lw):
    """
    This is a helper function which will plot the cumulative goal proportion by shot probablity model
    percentile.
    :param plt1: subplot for which we will plot the goal shot rate
    :param model: trained model which is used to predict on the validation dataset.
    :param x_val: input features of the validation dataset
    :param y_val: target data of the validation dateset
    :param k: model name
    :param ls: line space of the plot
    :param lw: linde width of the plot
    :return:
    """
    predicted_probablities = model.predict_proba(x_val)

    goal_prob = predicted_probablities[:, 1]  # taking only the goal probablities

    df = pd.DataFrame(y_val, columns=["is_goal"])
    df["probablity_of_goal"] = goal_prob

    df['percentile_of_goal'] = round(df["probablity_of_goal"].rank(pct=True) * 100)
    goal_rate = df.groupby(by='percentile_of_goal').sum()
    goal_rate['percentile'] = goal_rate.index

    goal_rate['cum_sum'] = goal_rate.loc[::-1, 'is_goal'].cumsum()[::-1]
    goal_rate['cum_perc'] = 100 * goal_rate['cum_sum'] / goal_rate['is_goal'].sum()

    plt1.plot(goal_rate["percentile"], goal_rate["cum_perc"], label=k, linewidth=lw, linestyle=ls)


graph, (plt1) = plt.subplots(1, 1)

for index, (k, v) in enumerate(data_train_dict.items()):
    print("Cumulative Goal for BaseLine Model for features - ", k)
    lw = 5 - 4 * index / len(data_train_dict)
    ls = ['-', '--', '-.', ':'][index % 4]
    if k != "Random_Base_Line":
        x_train = v[0]
        y_train = v[1]
        x_val = data_val_dict[k][0]
        y_val = data_val_dict[k][1]
        no_of_features = v[2]
        if no_of_features == 1:
            x_train = x_train.values.reshape(-1, 1)
        model = LogisticRegression()
        model.fit(x_train, y_train)
        plot_cumulative_goal(plt1, model, x_val, y_val, k, ls, lw)

    else:
        ggoal_prob = np.array([np.random.uniform(low=0.0, high=1.0) for i in range(x_val.shape[0])])
        df = pd.DataFrame(y_val, columns=["is_goal"])
        df["probablity_of_goal"] = goal_prob

        df['percentile_of_goal'] = round(df["probablity_of_goal"].rank(pct=True) * 100)
        goal_rate = df.groupby(by='percentile_of_goal').sum()
        goal_rate['percentile'] = goal_rate.index

        goal_rate['cum_sum'] = goal_rate.loc[::-1, 'is_goal'].cumsum()[::-1]
        goal_rate['cum_perc'] = 100 * goal_rate['cum_sum'] / goal_rate['is_goal'].sum()
        plt1.plot(goal_rate["percentile"], goal_rate["cum_perc"], label=k, linewidth=lw, linestyle=ls)

plt1.set_ylim(0, 100)
plt1.invert_xaxis()
plt1.set_xlabel('Shot probablity model percentile')
plt1.set_ylabel('Proportion')
plt.grid()
plt.title("Cumulative % of goals")
plt.legend()
comet_exp_obj.log_figure(figure_name="Cumulative % of goals", figure=plt,
                         overwrite=True, step=None)

plt.show()


# In[9]:


def create_estimator_plot(model, x_val, y_val, k):
    """
    This is a helper function which is used to create the estimator plot for the given input model.
    :param model: trained model which is used to predict on the validation dataset.
    :param x_val: input features of the validation dataset
    :param y_val: target data of the validation dateset
    :param k: model name
    :return:
    """

    predicted_probablities = model.predict_proba(x_val)
    goal_prob = predicted_probablities[:, 1]  # taking only the goal probablities
    #     return CalibrationDisplay.from_predictions(y_val, goal_prob, ax=ax_calibration_curve,name=k)
    return CalibrationDisplay.from_estimator(model, x_val, y_val, ax=ax_calibration_curve, name=k)


fig = plt.figure(figsize=(10, 10))
gs = GridSpec(4, 2)
colors = plt.cm.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}

for index, (k, v) in enumerate(data_train_dict.items()):
    print("Estimator Plot for BaseLine Model for features - ", k)
    if k != "Random_Base_Line":
        x_train = v[0]
        y_train = v[1]
        x_val = data_val_dict[k][0]
        y_val = data_val_dict[k][1]
        no_of_features = v[2]
        if no_of_features == 1:
            x_train = x_train.values.reshape(-1, 1)
        model = LogisticRegression()
        model.fit(x_train, y_train)
        calibration_displays[k] = create_estimator_plot(model, x_val, y_val, k)

    else:
        goal_prob = np.array([np.random.uniform(low=0.0, high=1.0) for i in range(x_val.shape[0])])
        calibration_displays[k] = CalibrationDisplay.from_predictions(y_val, goal_prob, ax=ax_calibration_curve,
                                                                      name=k)

ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots")
plt.tight_layout()
plt.legend(loc='upper left')

comet_exp_obj.log_figure(figure_name="Calibration Display", figure=plt,
                         overwrite=True, step=None)
plt.show()

comet_exp_obj.end()