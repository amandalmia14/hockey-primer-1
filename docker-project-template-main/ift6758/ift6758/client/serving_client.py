import pandas as pd
import logging
import requests
from ift6758.ift6758.client.CometML import CometMLClient
logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features
        self.model = None
        self.model_name = None

        # any other potential initialization

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the inputs into an appropriate payload for a POST request, and queries the
        prediction service. Retrieves the response from the server, and processes it back into a
        dataframe that corresponds index-wise to the input dataframe.
        
        Args:
            X (Dataframe): Input dataframe to submit to the prediction service.
        """
        response = requests.post(self.base_url + "/predict", json=X.to_json(orient="table"))
        prediction = response.json()
        df = pd.DataFrame(prediction)
        return df
        # raise NotImplementedError("TODO: implement this function")

    def logs(self) -> dict:
        """Get server logs"""
        r = requests.get(f"{self.base_url}/logs")
        # print(r)
        logs = r.json()
        print(logs)
        raise NotImplementedError("TODO: implement this function")

    def download_registry_model(self, workspace: str, model: str, version: str):
        """
        Triggers a "model swap" in the service; the workspace, model, and model version are
        specified and the service looks for this model in the model registry and tries to
        download it. 

        See more here:

            https://www.comet.ml/docs/python-sdk/API/#apidownload_registry_model
        
        Args:
            workspace (str): The Comet ML workspace
            model (str): The model in the Comet ML registry to download
            version (str): The model version to download
        """

        comet_ml_obj = CometMLClient(model_name=model, version=version, workspace=workspace)
        response = comet_ml_obj.download_model()
        model = comet_ml_obj.get_model()
        return model, response

        # raise NotImplementedError("TODO: implement this function")


if __name__ == '__main__':
    sc_obj = ServingClient()
    from game_client import GameClient

    gc_obj = GameClient()
    data = gc_obj.get_live_data(game_id=2021020329)

    # print(data.columns)
    from feature_engineering import main_feature_engg

    df_feg = main_feature_engg(df=data)

    model, _ = sc_obj.download_registry_model(workspace="data-science-workspace", model="neural-network-model",
                                              version="1.0.0")

    from ift6758.ift6758.client.NeuralNet import get_probs_nn, transform_data_for_nn

    test_dataloader = transform_data_for_nn(X_test=df_feg)
    # Y_pred_proba_list, ns_probs = get_probs_nn(model=model, dataloader=test_dataloader)
    model.eval()
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    import numpy as np

    with torch.no_grad():
        X = torch.Tensor(torch.Tensor(df_feg.values.astype(np.float32))).to(device)
        pred = model(X)
    print(pred)