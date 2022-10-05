import os
import re
import traceback
from os.path import exists

import pandas as pd
import requests
from tqdm import tqdm

from constant import APIList, CustomRegex, data_dir


def get_url(game_id: str):
    """
    This funtions formats the url for a given game_id
    @param game_id: input game id
    @return: API in order to get the metadata for the given game id
    """
    return APIList.GET_ALL_DATA_FOR_A_GIVEN_MATCH.format(game_id)


def get_data_by_gameid(game_id: str):
    """
    This function takes an input formatted game id and returns the data of that partical game
    @param game_id:Game id in the form of "2017020007"
    @return: Metadata for that particular game id
    """
    try:
        data = requests.get(get_url(game_id=game_id))
        return data.json()
    except Exception as e:
        print(e)
        print(traceback.print_exc())
        pass


def get_all_relevant_game_ids_by_season(season_year: int):
    """
    This functions fetches all the game ids having game type Regular Season and Playoffs
    @param season_year: Season Year
    @return: two list of game ids, one for regular and another for playoffs
    """
    try:
        response = requests.get(APIList.GET_ALL_MATCHES_FOR_A_GIVEN_SEASON + str(season_year) + str(season_year + 1))
        list_of_all_matches = response.json()["dates"]
        reg_season_gameid_list = []
        playoffs_gameid_list = []
        for i in list_of_all_matches:
            for j in i["games"]:
                if re.match(CustomRegex.REGULAR_GAME_ID, str(j["gamePk"])):
                    reg_season_gameid_list.append(j["gamePk"])
                elif re.match(CustomRegex.PLAYOFFS_ID, str(j["gamePk"])):
                    playoffs_gameid_list.append(j["gamePk"])
                else:
                    pass
        reg_season_gameid_list.sort()
        playoffs_gameid_list.sort()
        return reg_season_gameid_list, playoffs_gameid_list
    except Exception as e:
        print(e)
        print(traceback.print_exc())
        pass


def save_data_to_csv(year, game_type, data, save_path):
    """
    A function which will help the extracted data from the NHL API locally into a csv file
    @param year: Year / Season which we want to extract the data
    @param game_type: Type of game, regular season or playoffs
    @param data: fetched data
    @param save_path: csv path
    @return: None
    """
    df = pd.DataFrame.from_records(data)
    df.to_csv(data_dir + save_path + os.path.sep + str(year) + "_" + game_type + ".csv", index=False)
    return None


def get_all_data_by_season(year: int, out_path: str):
    """
    This function will fetch both the games which are regular seasons and playoffs for the entire season
    @param year: season year
    @param out_path: directory where the file will save
    @return:None
    """
    try:
        if 2015 < year < 2021:
            if exists(data_dir + out_path):
                regular_season_file_path = data_dir + out_path + os.path.sep + str(year) \
                                           + "_regular_season.csv"
                playoffs_file_path = data_dir + out_path + os.path.sep + str(year) + "_playoffs.csv"
                reg_season_game_data_list = pd.read_csv(regular_season_file_path)
                playoffs_game_data_list = pd.read_csv(playoffs_file_path)
                return reg_season_game_data_list, playoffs_game_data_list
            else:
                os.mkdir(data_dir + out_path)
                reg_season_game_data_list = []
                playoffs_game_data_list = []
                reg_season_gameid_list, playoffs_gameid_list = get_all_relevant_game_ids_by_season(season_year=year)
                for reg_season_game_id in tqdm(reg_season_gameid_list):
                    match_data = get_data_by_gameid(game_id=reg_season_game_id)
                    reg_season_game_data_list.append(match_data)
                for playoff_game_id in tqdm(playoffs_gameid_list):
                    match_data = get_data_by_gameid(game_id=playoff_game_id)
                    playoffs_game_data_list.append(match_data)

                save_data_to_csv(year=year, game_type="regular_season", season_data=reg_season_game_data_list,
                                 save_path=out_path)
                save_data_to_csv(year=year, game_type="playoffs", season_data=playoffs_game_data_list,
                                 save_path=out_path)

                return reg_season_game_data_list, playoffs_game_data_list, "Success"
        else:
            return [], [], "Invalid Year"
    except Exception as e:
        print(e)
        print(traceback.print_exc())
        pass


if __name__ == "__main__":
    results = get_all_data_by_season(year=2017, out_path="2017")
