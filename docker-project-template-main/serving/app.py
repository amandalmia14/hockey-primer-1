"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import sys

sys.path.append(".")

import os
import logging
import json
from flask import Flask, jsonify, request
import pandas as pd
from ift6758.ift6758.client.CometML import CometMLClient
from ift6758.ift6758.client.serving_client import ServingClient
from ift6758.ift6758.client.NeuralNet import get_probs_nn
from serving.constant import DOWNLOADED_MODEL_PATH
from constant import model_name_map, model_feature_map

serving_client_obj = ServingClient()
LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, filemode='w')
# COMET_WORKSPACE = config[type_env]['workspace']


app = Flask(__name__)


@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    # TODO: setup basic logging configuration
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

    # TODO: any other initialization before the first request (e.g. load default model)
    # default model
    # default_model = "linearmodel-distance"
    default_model = "neural-network-model"
    comet_ml_obj = CometMLClient(model_name=default_model, version="1.0.0", workspace="data-science-workspace")
    file_path = DOWNLOADED_MODEL_PATH + default_model + ".pkl"
    serving_client_obj.model_name = model_name_map[default_model]
    serving_client_obj.features = model_feature_map[serving_client_obj.model_name]
    if os.path.isfile(file_path):
        serving_client_obj.model = comet_ml_obj.get_model()
    else:
        app.logger.info(f"Downloading from COMET {default_model}")
        response = comet_ml_obj.download_model()
        serving_client_obj.model = comet_ml_obj.get_model()


@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""

    # TODO: read the log file specified and return the data

    with open(LOG_FILE, 'r') as log_data:
        response = log_data.readlines()
    return jsonify(response)  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    
    """
    # Download a Registry Model:

    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    download_exist = False
    # TODO: check to see if the model you are querying for is already downloaded
    list_of_models_downloaded = os.listdir(DOWNLOADED_MODEL_PATH)
    for i in list_of_models_downloaded:
        if json["model"] in model_name_map and json["version"] == "1.0.0":
            if model_name_map[json["model"]] in i:
                download_exist = True
                break
        else:
            response = {"message": "Invalid Model name / Version, kindly re-check model name / version. Default Model "
                                   "Loaded"}
            app.logger.info(response["message"])
            return jsonify(response)

    # TODO: if yes, load that model and write to the log about the model change.  
    # eg: app.logger.info(<LOG STRING>)

    response = None
    comet_ml_obj = CometMLClient(model_name=json["model"], version=json["version"], workspace=json["workspace"])
    serving_client_obj.model_name = model_name_map[json["model"]]

    if download_exist:
        app.logger.info("Loading the existing model")
        response = {"message": "Loading the existing model"}
        serving_client_obj.model = comet_ml_obj.get_model()

    # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the 
    # currently loaded model
    else:
        app.logger.info("Downloading the model from comel ml space")
        response = comet_ml_obj.download_model()
        serving_client_obj.model = comet_ml_obj.get_model()
    app.logger.info(response["message"])
    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here

    if model_name_map[json["model"]] in model_feature_map:
        serving_client_obj.features = model_feature_map[model_name_map[json["model"]]]

    # raise NotImplementedError("TODO: implement this endpoint")

    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    json_data = request.get_json()

    # TODO:
    str_json = json.dumps(json_data)
    X = pd.read_json(str_json, orient="records")
    # X = X.to_numpy()
    if serving_client_obj.model_name == "Neural_Network":
        # test_dataloader = transform_data_for_nn(X_test=X)
        Y_pred_proba_list = get_probs_nn(model=serving_client_obj.model, df_feg=X)
    else:
        X = X[serving_client_obj.features]
        Y_pred_proba_list = serving_client_obj.model.predict_proba(X)
    response = {
        # "prediction": np.argmax(Y_pred_proba_list, axis=1).tolist(),
        "goal_probabilities": [round(x, 4) for x in Y_pred_proba_list[:, 1].tolist()]
    }
    return jsonify(response)  # response must be json serializable!


@app.route("/fetch_data", methods=["POST"])
def fetch_data():
    from ift6758.ift6758.client.game_client import GameClient
    gc_obj = GameClient()
    data = gc_obj.get_live_data(game_id=2021020329)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False, port=8080)
