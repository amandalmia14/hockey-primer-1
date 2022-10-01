import os
from os.path import exists

import pandas as pd
import requests
from tqdm import tqdm

from constant import game_type_map, seasons_year_matches_map


def get_url(game_id: str):
    """
    This funtions formats the url for a given game_id
    @param game_id: input game id
    @return: API in order to get the metadata for the given game id
    """
    return "https://statsapi.web.nhl.com/api/v1/game/{}/feed/live/".format(game_id)


def get_data_by_gameid(game_id: str):
    """
    This function takes an input formatted game id and returns the data of that partical game
    @param game_id:Game id in the form of "2017020007"
    @return: Metadata for that particular game id
    """
    data = requests.get(get_url(game_id=game_id))
    return data.json()


def get_data_by_season(year: int, game_type: str, save_path: str):
    """
    This function fetches all the metadata for a given game type occurred in the given season
    @param year: Year for which we need metadata for
    @param game_type: type of game of that season, (regular or playoffs)
    @param save_path: file where the csv will be saved
    @return: the metadata for a particular game type for the given season
    """
    datafile_path = save_path + os.path.sep + str(year) + ".csv"
    if exists(datafile_path):
        return pd.read_csv(datafile_path)
    else:
        no_of_matches = seasons_year_matches_map[year]
        total_season_data = []
        for i in tqdm(range(1, no_of_matches + 1)):
            formatted_game_id = str(year) + game_type_map[game_type] + format(i, "04")
            match_data = get_data_by_gameid(game_id=formatted_game_id)
            total_season_data.append(match_data)

        df = pd.DataFrame.from_records(total_season_data)
        df.to_csv(".." + os.path.sep + save_path + os.path.sep + str(year) + "_ " + game_type + ".csv", index=False)
        return total_season_data


if __name__ == "__main__":
    results = get_data_by_season(year=2016, game_type="regular_season", save_path="data")
