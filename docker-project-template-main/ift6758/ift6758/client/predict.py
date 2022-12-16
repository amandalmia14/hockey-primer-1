import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
import sklearn

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_probs_nn(model, X):
    """
    This functions retrieve the probabilities of the Neural network of the model
    :param model: Neural Network trained model
    :param dataloader: validation dataset in the form of dataloader
    :return: pred_list, testy, ns_probs
    """
    
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        pred = model(X)
            

    return pred.numpy()

def predict(model,model_name, x):
    if model_name == "Neural_Network":
        pred = get_probs_nn(model, x)
    else:
        pred_list = model.predict_proba(x)
    