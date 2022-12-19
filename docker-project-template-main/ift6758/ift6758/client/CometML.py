import os
import pickle

import torch
from comet_ml.api import API

from ift6758.ift6758.client.NeuralNet import NeuralNets
# from serving.constant import COMET_FILE
from serving.constant import DOWNLOADED_MODEL_PATH
from serving.constant import model_name_map

# config = configparser.ConfigParser()
# # config.read('../configfile.ini')
# config.read(COMET_FILE)
# print(COMET_FILE)
# type_env = "comet_ml_prod"  # comet_ml_prod
COMET_API_KEY = os.environ['COMET_API_KEY']  # config[type_env]['api_key']


class CometMLClient():
    def __init__(self, workspace, model_name, version):
        self.workspace = workspace
        self.model_name = model_name
        self.version = version
        self.downloaded_model_path = DOWNLOADED_MODEL_PATH
        self.api = API(api_key=COMET_API_KEY)

    def get_model(self):
        if model_name_map[self.model_name] == "Neural_Network":
            model = NeuralNets()
            model.load_state_dict(torch.load(self.downloaded_model_path + model_name_map[self.model_name] +
                                             "_Model.pth"))

        else:
            model = pickle.load(open(self.downloaded_model_path + model_name_map[self.model_name] + "_Model.pkl", 'rb'))

        return model

    def download_model(self):
        try:
            self.api.download_registry_model(self.workspace, self.model_name, self.version,
                                             output_path=self.downloaded_model_path, expand=True)
            download_success = False
            list_of_models = os.listdir(DOWNLOADED_MODEL_PATH)
            print(list_of_models)
            for i in list_of_models:
                if model_name_map[self.model_name] in i:
                    download_success = True
                    break
            if download_success:
                response = {"message": "Download Successfully"}
            else:
                response = {"message": "Some Error occurred while downloading, please check model name / version etc."}
            print("response", response)
            return response
        except Exception as e:
            print(e)
            return {"message": "Some Error occurred while downloading, please check model name / version etc."}
