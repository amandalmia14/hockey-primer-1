import json
import os
import pickle
import re
import traceback
from os.path import exists

import pandas as pd
import requests
from tqdm import tqdm

from constant import APIList, CustomRegex, Directory


def get_no_of_matches_team_map(season_year):
    """
    This function will fetch no of matches a team has played season wise.
    Args:
        season_year: season for which we need to find it out no of matches a team has played.
    Returns: None; creates a pickle file

    """
    response = requests.get(APIList.GET_ALL_MATCHES_FOR_A_GIVEN_SEASON + str(season_year) + str(season_year + 1))
    list_of_all_matches = response.json()["dates"]
    game_id = []
    home_team = []
    away_team = []
    for i in list_of_all_matches:
        for j in i["games"]:
            game_id.append(j["gamePk"])
            home_team.append(j["teams"]["home"]["team"]["name"])
            away_team.append(j["teams"]["away"]["team"]["name"])
    df = pd.DataFrame({'game_id': game_id, 'home_team': home_team, 'away_team': away_team})
    with open(Directory.DATA_DIR + str(season_year) + os.path.sep + 'no_of_matches_' + str(season_year) + '_.p',
              'wb') as fp:
        pickle.dump(df, fp, protocol=pickle.HIGHEST_PROTOCOL)


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


def save_data_to_json(year, game_type, data, save_path):
    """
    A function which will help the extracted data from the NHL API locally into a json file
    @param year: Year / Season which we want to extract the data
    @param game_type: Type of game, regular season or playoffs
    @param data: fetched data
    @param save_path: json path
    @return: None
    """
    path = Directory.DATA_DIR + save_path + os.path.sep + str(year) + "_" + game_type + ".json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    return None


def get_all_data_by_season(year: int, out_path: str):
    """
    This function will fetch both the games which are regular seasons and playoffs for the entire season
    @param year: season year
    @param out_path: directory where the file will save
    @return:None
    """
    try:
        if 2014 < year < 2021:
            if exists(Directory.DATA_DIR + out_path):
                print("Data exists locally, fetching the data from your system !!")
                regular_season_file_path = Directory.DATA_DIR + out_path + os.path.sep + str(year) \
                                           + "_regular_season.json"
                playoffs_file_path = Directory.DATA_DIR + out_path + os.path.sep + str(year) + "_playoffs.json"
                with open(regular_season_file_path, "r") as f:
                    reg_season_game_data_list = json.load(f)

                with open(playoffs_file_path, "r") as f:
                    playoffs_game_data_list = json.load(f)

                return reg_season_game_data_list, playoffs_game_data_list
            else:
                os.mkdir(Directory.DATA_DIR + out_path)
                get_no_of_matches_team_map(season_year=year)
                reg_season_game_data_dict = {}
                playoffs_game_data_dict = {}
                reg_season_gameid_list, playoffs_gameid_list = get_all_relevant_game_ids_by_season(season_year=year)
                print("Starting Data extraction for the season ", year)
                for reg_season_game_id in tqdm(reg_season_gameid_list):
                    match_data = get_data_by_gameid(game_id=reg_season_game_id)
                    reg_season_game_data_dict[reg_season_game_id] = match_data
                for playoff_game_id in tqdm(playoffs_gameid_list):
                    match_data = get_data_by_gameid(game_id=playoff_game_id)
                    playoffs_game_data_dict[playoff_game_id] = match_data
                print("Please wait !! Data is persisting into file ")
                save_data_to_json(year=year, game_type="regular_season", data=reg_season_game_data_dict,
                                  save_path=out_path)
                save_data_to_json(year=year, game_type="playoffs", data=playoffs_game_data_dict,
                                  save_path=out_path)
                print("Completed for the season ", year)
                return reg_season_game_data_dict, playoffs_game_data_dict, "Success"
        else:
            return {}, {}, "Invalid Year"
    except Exception as e:
        print(e)
        print(traceback.print_exc())


if __name__ == "__main__":
    year = [2015, 2016, 2017, 2018, 2019, 2020]
    for y in year:
        results = get_all_data_by_season(year=y, out_path=str(y))