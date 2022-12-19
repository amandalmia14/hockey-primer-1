import numpy as np
import torch
import torch.utils.data as data_utils
from torch import nn
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


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


def get_probs_nn(model, df_feg):
    """
    This functions retrieve the probabilities of the Neural network of the model
    :param model: Neural Network trained model
    :param dataloader: validation dataset in the form of dataloader
    :return: pred_list, testy, ns_probs
    """
    with torch.no_grad():
        X = torch.Tensor(torch.Tensor(df_feg.values.astype(np.float32))).to(device)
        pred = model(X)
    return pred


# def transform_data_for_nn(X_test):
#     """
#     This functions transform the data which is required to ingest in neural network. The code is written in
#     PyTorch hence it needs to convert into datasets and dataloaders in respective format.
#     :param x_test: input features of training data
#     :param y_test: targets of the training data
#     :return: Dataloaders of training and validation sets of data.
#     """
#     test_data = data_utils.TensorDataset(torch.Tensor(X_test.values.astype(np.float32)))
#     batch_size = 64
#     # Create data loaders.
#     test_dataloader = DataLoader(test_data, batch_size=batch_size)
#     return test_dataloader
