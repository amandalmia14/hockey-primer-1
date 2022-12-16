import json
import requests
import pandas as pd
import logging
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
        game_dict = get_data_by_gameid(game_id)
        event_arr = game_dict['liveData']['plays']['allPlays']
        if len(event_arr) > self.last_entry_idx:
            
            df = self.process_live_data(game_id,game_dict,self.last_entry_idx+1)
            self.last_entry_idx = len(event_arr) - 1 
            self.last_event_data = last_data_parsing(event_arr[-1])

        return df

    def process_live_data(self,game_id,game_dict,entry_idx):
        period_dict = get_side(game_meta=game_dict)
        teams_type = get_home_away_team(game_meta=game_dict)
        live_data = game_dict["liveData"]["plays"]["allPlays"]
        final_list = []
        for i in live_data[entry_idx:]:
            if i["result"]["event"] in TYPES_OF_SHOTS:
                try:
                    parsed_data = data_parsing(data=i, id=game_id, event_type=i["result"]["event"],
                                            period_dict=period_dict, team_detail_dict=teams_type)
                    final_list.append(parsed_data)
                except Exception as e:
                    print(e)
                    import traceback
                    print(traceback.print_exc())
                    break

        shots_goals_df = pd.DataFrame(final_list)
        return shots_goals_df
