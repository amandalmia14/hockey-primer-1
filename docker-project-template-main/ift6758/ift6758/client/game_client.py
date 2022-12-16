import json
import requests
import pandas as pd
import logging
from data_processing import *
from data_retrival import *

logger = logging.getLogger(__name__)

class GameClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features
        self.data = None
        self.last_entry_idx = None
        self.last_event_data = None
        # any other potential initialization

    def get_live_data(self,game_id):
        # diff based on all event types, but our df only concerns about shot and goal events
        game_json = get_data_by_gameid(game_id)
        event_arr = game_json['liveData']['plays']['allPlays']
        if len(event_arr) > self.last_entry_idx:
            
            df = self.process_live_data()
            self.last_entry_idx = len(event_arr) - 1 
            self.last_event_data = last_data_parsing(event_arr[-1])

        return df

    def process_live_data(self):
        pass
