#!/usr/bin/env python
# coding: utf-8

# In[1]:


import configparser
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
from comet_ml import Experiment
from comet_ml.api import API
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, auc
from torch import nn
from torch.utils.data import DataLoader

from constant import COMET_FILE

warnings.filterwarnings('ignore')
np.random.seed(42)
seed = 42

# In[2]:


config = configparser.ConfigParser()
# config.read('../configfile.ini')
config.read(COMET_FILE)
type_env = "comet_ml_prod"  # comet_ml_prod
COMET_API_KEY = config[type_env]['api_key']
COMET_PROJECT_NAME_REGULAR = config[type_env]['project_name_final_test_regular']
COMET_WORKSPACE = config[type_env]['workspace']
DOWNLOADED_MODEL_PATH = "../../downloaded_models/"


# In[3]:


class NeuralNets(nn.Module):
    """
    A custom Neural Network Class developed on PyTorch framework
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# In[4]:


def download_models():
    """
    This function will download the model locally from the comet ml repository which are published.
    :return:
    """
    api = API(api_key=COMET_API_KEY)
    list_of_models = ["neural-network-model", "linearmodel-angle", "linearmodel-distance",
                      "linearmodel-angle-distance", "xgboost-feature-selection-class-weights"]
    # Download a Registry Model:
    for model_name in list_of_models:
        api.download_registry_model(COMET_WORKSPACE, model_name, "1.0.0",
                                    output_path=DOWNLOADED_MODEL_PATH, expand=True)


def load_test_data():
    """
    This function will load the test data from the year 2019/2020 and filter the features based on the feature selection
    task which we had done.
    :return: input features, target labels
    """
    df = pd.read_pickle("../../data/testdata/final_evaluation_set.pkl")
    x_test = df
    x_test["game_type"] = np.where(x_test["game_id"].str[5] == "2", 'regular', 'playoffs')
    x_test = x_test[x_test["game_type"] == "regular"]
    y_test = x_test["is_goal"].values

    x_test = x_test[['angle', 'distance_from_last_event', 'empty_net', 'shot_type_Wrap-around',
                     'y_coordinate', 'speed', 'distance', 'x_coordinate', 'game_period', 'shot_type_Tip-In',
                     'shot_type_Wrist Shot', 'game_seconds']]

    x_test = (x_test - x_test.mean()) / x_test.std()

    return x_test, y_test


def transform_data_for_nn(x_test, y_test):
    """
    This functions transform the data which is required to ingest in neural network. The code is written in
    PyTorch hence it needs to convert into datasets and dataloaders in respective format.
    :param x_test: input features of training data
    :param y_test: targets of the training data
    :return: Dataloaders of training and validation sets of data.
    """
    test_data = data_utils.TensorDataset(torch.Tensor(x_test.values.astype(np.float32)),
                                         torch.LongTensor(y_test.astype(np.float32)))
    batch_size = 64
    # Create data loaders.
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    return test_dataloader


# In[5]:


download_models()

# In[6]:


comet_exp_obj = Experiment(api_key=COMET_API_KEY,
                           project_name=COMET_PROJECT_NAME_REGULAR,
                           workspace=COMET_WORKSPACE,
                           log_code=True
                           )
comet_exp_obj.set_name(name="Final Evaluation Testing")
# comet_exp_obj.log_notebook("q7_final_evaluation_playoffs.py")

# In[7]:


x_test, y_test = load_test_data()
comet_exp_obj.log_dataframe_profile(x_test, "x_test", dataframe_format='csv')
comet_exp_obj.log_dataframe_profile(pd.DataFrame(y_test, columns=["is_goal"]), "y_test", dataframe_format='csv')

# Data Prep for Neural Network
test_dataloader = transform_data_for_nn(x_test, y_test)

data_dict = {
    "Neural_Network": test_dataloader,
    # "kNN": (x_test, y_test),
    # "Random_Forest": (x_test, y_test),
    "xgboost_feature_selection_class_weights" : (x_test, y_test),
    "LinearModel_Angle": (x_test, y_test),
    "LinearModel_Angle_Distance": (x_test, y_test),
    "LinearModel_Distance": (x_test, y_test)
}

valdataloader = data_dict["Neural_Network"]
x_val = data_dict["LinearModel_Angle"][0]
y_val = data_dict["LinearModel_Angle"][1]

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# In[8]:


def get_probs_nn(model, dataloader):
    """
    This functions retrieve the probabilities of the Neural network of the model
    :param model: Neural Network trained model
    :param dataloader: validation dataset in the form of dataloader
    :return: pred_list, testy, ns_probs
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    pred_list = np.empty((0, 2), float)
    true_y = np.empty((0), int)
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred_list = np.append(pred_list, pred.numpy(), axis=0)
            true_y = np.append(true_y, y.numpy(), axis=0)

    y_val = true_y
    testy = y_val
    ns_probs = [0 for _ in range(len(testy))]
    return pred_list, testy, ns_probs


# In[9]:


def create_roc_auc_curve(dataloader, x_val, y_val, k):
    """
    This function is a helper function in order to plot the ROC AUC curve.
    :param dataloader: validation dataloader for neural network model as its written in pytorch
    :param x_val: Input data of validation dataset
    :param y_val: target values of the validation dataset
    :param k: model name
    :return: values which are required for plotting the roc auc curve which includes true positive rate, false postive
    rate etc.
    """
    if k == "Neural_Network":
        nn_model = NeuralNets()
        nn_model.load_state_dict(torch.load(DOWNLOADED_MODEL_PATH + k + "_Model.pth"))
        pred_list, testy, ns_probs = get_probs_nn(nn_model, dataloader)

    elif k == "kNN":
        knn_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = knn_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    elif k == "xgboost_feature_selection_class_weights":
        xgb_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = xgb_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    elif k == "Random_Forest":
        rf_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = rf_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    elif k == "Decision_Tree":
        dt_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = dt_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    elif k == "LinearModel_Angle":
        x_val = x_val[["angle"]]
        dt_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = dt_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    elif k == "LinearModel_Angle_Distance":
        x_val = x_val[["angle", "distance"]]
        dt_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = dt_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    elif k == "LinearModel_Distance":
        x_val = x_val[["distance"]]
        dt_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = dt_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    else:
        pass

    metrics = {}
    predicted_probablities = pred_list
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

    pred_list_class = np.argmax(pred_list, axis=1)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_val, pred_list_class)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print("roc_auc", roc_auc)
    metrics["accuracy"] = np.mean(y_val == pred_list_class) * 100
    metrics["roc_auc"] = lr_auc

    class_report = classification_report(y_val, pred_list_class, output_dict=True)
    for ke, v in class_report.items():
        if type(v) is dict:
            for vk, val in v.items():
                metrics[ke + "_" + vk] = round(val, 3)
    comet_exp_obj.log_metrics(metrics)
    comet_exp_obj.log_confusion_matrix(y_true=y_val, y_predicted=pred_list, title="Confusion Matrix",
                                       file_name="Confusion Matrix for " + k)

    return ns_fpr, ns_tpr, lr_fpr, lr_tpr, ns_auc, lr_auc


plt.figure(figsize=(10, 8))
table_values = []
markers_list = [",", ",", ",", ",", ",", ",", ","]

for index, (k, v) in enumerate(data_dict.items()):
    #     if k == "Random_Forest":
    lw = 5 - 4 * index / len(data_dict)
    ls = ['-', '--', '-.', ':'][index % 4]
    print("AUC ROC Scores for BaseLine Model for features - ", k)
    ns_fpr, ns_tpr, lr_fpr, lr_tpr, ns_auc, lr_auc = create_roc_auc_curve(valdataloader, x_val, y_val, k)
    table_values.append([ns_auc, lr_auc])
    # plot the roc curve for the model
    plt.plot(lr_fpr, lr_tpr, marker=markers_list[index], label=k, linewidth=lw, linestyle=ls)

col_labels = ['No Skill: ROC AUC', 'Logistic: ROC AUC']
row_labels = [*data_dict]
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
plt.title("Receiver Operating Characteristic (ROC) curves and the AUC metric for regular games")
plt.legend()
comet_exp_obj.log_figure(
    figure_name="Receiver Operating Characteristic (ROC) curves and the AUC metric for regular games",
    figure=plt, overwrite=True, step=None)
plt.show()


# ## Question 2
# The goal rate (#goals / (#no_goals + #goals)) as a function of the shot probability model percentile, i.e. if a value is the 70th percentile, it is above 70% of the data. 

# In[10]:


def plot_goal_shot_rate(dataloader, x_val, y_val, k):
    """
    This is a helper function which will plot the goal shot rate by percentile.
    :param dataloader: validation dataloader for neural network model as its written in pytorch
    :param x_val: input features of the validation dataset
    :param y_val: target data of the validation dateset
    :param k: model name
    :return: None
    """
    if k == "Neural_Network":
        nn_model = NeuralNets()
        nn_model.load_state_dict(torch.load(DOWNLOADED_MODEL_PATH + k + "_Model.pth"))
        pred_list, testy, ns_probs = get_probs_nn(nn_model, dataloader)

    elif k == "kNN":
        knn_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = knn_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    elif k == "xgboost_feature_selection_class_weights":
        xgb_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = xgb_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    elif k == "Random_Forest":
        rf_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = rf_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    elif k == "Decision_Tree":
        dt_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = dt_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    elif k == "LinearModel_Angle":
        x_val = x_val[["angle"]]
        dt_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = dt_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    elif k == "LinearModel_Angle_Distance":
        x_val = x_val[["angle", "distance"]]
        dt_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = dt_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    elif k == "LinearModel_Distance":
        x_val = x_val[["distance"]]
        dt_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = dt_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]
    else:
        pass

    predicted_probablities = pred_list
    goal_prob = predicted_probablities[:, 1]  # taking only the goal probablities

    df = pd.DataFrame(y_val, columns=["is_goal"])
    df["probablity_of_goal"] = goal_prob
    df['percentile_of_goal'] = round(df["probablity_of_goal"].rank(pct=True) * 100)
    goal_rate = round((df.groupby(by='percentile_of_goal').sum() /
                       df.groupby(by='percentile_of_goal').count()) * 100)
    goal_rate['percentile'] = goal_rate.index

    plt1.plot(goal_rate["percentile"], goal_rate["is_goal"], label=k)


fig, (plt1) = plt.subplots(1, 1)

for index, (k, v) in enumerate(data_dict.items()):
    print("Goal Shot Rate for BaseLine Model for features - ", k)
    plot_goal_shot_rate(valdataloader, x_val, y_val, k)

plt1.set_ylim(0, 100)
plt1.xaxis.set_major_formatter('{x:1.0f}%')
plt1.yaxis.set_major_formatter('{x:1.0f}%')
plt1.invert_xaxis()
plt1.set_xlabel('Shot probablity model percentile')
plt1.set_ylabel('Goals / (No Goals + Goals)')
plt.grid()
plt.title("Goal Rate for regular games")
plt.legend()
comet_exp_obj.log_figure(figure_name="Goal Rate for regular games", figure=fig,
                         overwrite=True, step=None)
plt.show()


# In[11]:


def plot_cumulative_goal(dataloader, x_val, y_val, k, ls, lw):
    """
    This is a helper function which will plot the cumulative goal proportion by shot probability model
    percentile.
    :param dataloader: validation dataloader for neural network model as its written in pytorch
    :param x_val: input features of the validation dataset
    :param y_val: target data of the validation dateset
    :param k: model name
    :param ls: line space of the plot
    :param lw: linde width of the plot
    :return: None
    """
    if k == "Neural_Network":
        nn_model = NeuralNets()
        nn_model.load_state_dict(torch.load(DOWNLOADED_MODEL_PATH + k + "_Model.pth"))
        pred_list, testy, ns_probs = get_probs_nn(nn_model, dataloader)

    elif k == "kNN":
        knn_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = knn_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    elif k == "xgboost_feature_selection_class_weights":
        xgb_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = xgb_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    elif k == "Random_Forest":
        rf_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = rf_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    elif k == "Decision_Tree":
        dt_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = dt_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    elif k == "LinearModel_Angle":
        x_val = x_val[["angle"]]
        dt_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = dt_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    elif k == "LinearModel_Angle_Distance":
        x_val = x_val[["angle", "distance"]]
        dt_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = dt_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    elif k == "LinearModel_Distance":
        x_val = x_val[["distance"]]
        dt_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = dt_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]
    else:
        pass

    predicted_probablities = pred_list

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

for index, (k, v) in enumerate(data_dict.items()):
    print("Cumulative Goal for BaseLine Model for features - ", k)
    lw = 5 - 4 * index / len(data_dict)
    ls = ['-', '--', '-.', ':'][index % 4]
    plot_cumulative_goal(valdataloader, x_val, y_val, k, ls, lw)

plt1.set_ylim(0, 100)
plt1.invert_xaxis()
plt1.set_xlabel('Shot probablity model percentile')
plt1.set_ylabel('Proportion')
plt.grid()
plt.title("Cumulative % of goals for regular games")
plt.legend()
comet_exp_obj.log_figure(figure_name="Cumulative % of goals for regular games", figure=plt,
                         overwrite=True, step=None)

plt.show()


# In[12]:


def create_estimator_plot(dataloader, x_val, y_val, k):
    """
    This is a helper function which is used to create the estimator plot for the given input model.
    :param dataloader: validation dataloader for neural network model as its written in pytorch
    :param x_val: input features of the validation dataset
    :param y_val: target data of the validation dateset
    :param k: model name
    :return: None
    """
    if k == "Neural_Network":
        nn_model = NeuralNets()
        nn_model.load_state_dict(torch.load(DOWNLOADED_MODEL_PATH + k + "_Model.pth"))
        pred_list, testy, ns_probs = get_probs_nn(nn_model, dataloader)

    elif k == "kNN":
        knn_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = knn_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    elif k == "xgboost_feature_selection_class_weights":
        xgb_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = xgb_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    elif k == "Random_Forest":
        rf_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = rf_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    elif k == "Decision_Tree":
        dt_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = dt_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    elif k == "LinearModel_Angle":
        x_val = x_val[["angle"]]
        dt_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = dt_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    elif k == "LinearModel_Angle_Distance":
        x_val = x_val[["angle", "distance"]]
        dt_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = dt_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]

    elif k == "LinearModel_Distance":
        x_val = x_val[["distance"]]
        dt_model = pickle.load(open(DOWNLOADED_MODEL_PATH + k + "_Model.pkl", 'rb'))
        pred_list = dt_model.predict_proba(x_val)
        testy = y_val
        ns_probs = [0 for _ in range(len(testy))]
    else:
        pass

    predicted_probablities = pred_list
    goal_prob = predicted_probablities[:, 1]  # taking only the goal probablities
    return CalibrationDisplay.from_predictions(y_val, goal_prob, ax=ax_calibration_curve, name=k)


#     return CalibrationDisplay.from_estimator(model, x_val, y_val, ax=ax_calibration_curve, name=k)

fig = plt.figure(figsize=(10, 8))
gs = GridSpec(4, 2)
colors = plt.cm.get_cmap("Dark2")

ax_calibration_curve = fig.add_subplot(gs[:2, :2])
calibration_displays = {}

for index, (k, v) in enumerate(data_dict.items()):
    print("Estimator Plot for BaseLine Model for features - ", k)
    calibration_displays[k] = create_estimator_plot(valdataloader, x_val, y_val, k)

ax_calibration_curve.grid()
ax_calibration_curve.set_title("Calibration plots for regular games")
plt.tight_layout()
plt.legend(loc='upper left')

comet_exp_obj.log_figure(figure_name="Calibration Display for regular games", figure=plt,
                         overwrite=True, step=None)
plt.show()

comet_exp_obj.end()

# In[ ]:
