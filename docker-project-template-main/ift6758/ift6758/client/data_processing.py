import json
import os
import pickle
import re
import traceback
from os.path import exists

import pandas as pd
import requests

game_id_url = "https://statsapi.web.nhl.com/api/v1/game/{}/feed/live/"

def get_data_by_gameid(game_id: str):
    """
    This function takes an input formatted game id and returns the data of that partical game
    @param game_id:Game id in the form of "2017020007"
    @return: Metadata for that particular game id
    """
    try:
        data = requests.get(game_id_url.format(game_id))
        return data.json()
    except Exception as e:
        print(e)
        print(traceback.print_exc())
        pass
