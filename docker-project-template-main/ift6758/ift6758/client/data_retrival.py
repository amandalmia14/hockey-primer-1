import json
import os
import traceback
import numpy as np
import pandas as pd
import requests

TYPES_OF_SHOTS = ["Goal", "Shot"]

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


def flatten_player_data(player_list):
    """
    This function transform list of players into a flatten encoded string in the form of (Full Name)_(Player Type)|
    (Full Name)_(Player Type)|.....
    @param player_list: list of players data
    @return: flatten string
    """
    flatten_string = ""
    for player in player_list:
        # flatten_string += "(" + player["player"]["id"] + ")_" # Can be uncommented in future if required
        flatten_string += "(" + player["player"]["fullName"] + ")_"
        flatten_string += "(" + player["playerType"] + ")|"
    return flatten_string[:-1]


def get_shooter_goalie(player_list):
    """
    This function gets the name of the goalie and the shooter
    @param player_list: return the shooter and goalie player names
    @return:
    """
    shooter = ""
    goalie = ""
    for player in player_list:
        if player["playerType"] == "Shooter":
            shooter = player["player"]["fullName"]
        elif player["playerType"] == "Goalie":
            goalie = player["player"]["fullName"]
        else:
            pass
    return shooter, goalie


def get_home_away_team(game_meta):
    """
    This functions get the team data
    @param game_meta: game metadata
    @return: dictionary of the team information
    """
    teams_data = game_meta["gameData"]["teams"]
    return {"home": teams_data["home"]["name"], "home_abv": teams_data["home"]["abbreviation"],
            "away": teams_data["away"]["name"], "away_abv": teams_data["away"]["abbreviation"]}


def get_side(game_meta):
    """
    This fucntion gets the team on which  rink side they were there in each period.
    @param game_meta: game metadata
    @return: a dictionary for each period home and away team rink side
    """
    periods_data = game_meta["liveData"]["linescore"]["periods"]
    period_dict = {}
    if len(periods_data) > 0:
        for i, period in enumerate(periods_data):
            if "rinkSide" in period["home"]:
                period_dict[i + 1] = {"home": period["home"]["rinkSide"], "away": period["away"]["rinkSide"]}
            else:
                period_dict[i + 1] = {"home": "Side Not Available", "away": "Side Not Available"}
    return period_dict


def get_coordinates(coordinates_data):
    """
    This functions return the coordinates as a tuple, if either of the data isn't available, it returns None
    Args:
        coordinates_data: coordinate dict
    Returns: x, y as a tuple
    """
    if "x" not in coordinates_data or "y" not in coordinates_data:
        return None
    return coordinates_data["x"], coordinates_data["y"]


def last_data_parsing(data):
    """
    This function will take the data of the previous event and return a new dict which contains certain data which will
    require to append the data as a previous event details for the next event.
    :param data: previous data event
    :return: relevant details of the previous event
    """
    result_data = data["result"]
    about_data = data["about"]
    data_dict = {"event_code": result_data["eventCode"], "event_type_id": result_data["eventTypeId"],
                 "coordinates": get_coordinates(data["coordinates"]),
                 "about_period": about_data["period"], "game_time": about_data["periodTime"]
                 }

    return data_dict


def data_parsing(data, id, event_type, period_dict, team_detail_dict, last_event_data):
    """
    This functions transforms the json data into the relevant information for the usecase
    @param data: entire metadata and details of the given game id
    @param id: game id
    @param event_type: type of game Shot / Goal
    @return: json object
    """
    players_data = data["players"]
    result_data = data["result"]
    print("result_data")
    print(result_data)
    about_data = data["about"]
    coordinates_data = data["coordinates"]
    team_data = data["team"]
    shooter, goalie = get_shooter_goalie(players_data)
    data_dict = {"game_id": id, "event_code": result_data["eventCode"],
                 "player_info": flatten_player_data(players_data),
                 "shooter": shooter, "goalie": goalie, "event": result_data["event"],
                 "event_type_id": result_data["eventTypeId"], "event_description": result_data["description"],

                 "home_team": team_detail_dict["home"], "home_team_abv": team_detail_dict["home_abv"],
                 "away_team": team_detail_dict["away"], "away_team_abv": team_detail_dict["away_abv"],

                 "about_event_id": about_data["eventId"], "about_period": about_data["period"],
                 "about_period_type": about_data["periodType"], "game_time": about_data["periodTime"],
                 "about_time_remaining": about_data["periodTimeRemaining"], "about_date_time": about_data["dateTime"],
                 "about_goal_away": about_data["goals"]["away"], "about_goal_home": about_data["goals"]["home"],
                 "action_team_name": team_data["name"]}

    if last_event_data is not None:
        data_dict["last_event_code"] = last_event_data['event_code']
        data_dict["last_event_type_id"] = last_event_data['event_type_id']
        data_dict["last_event_coordinates"] = last_event_data["coordinates"]
        data_dict["last_event_time"] = last_event_data["game_time"]
        data_dict["last_event_period"] = last_event_data["about_period"]
    else:
        data_dict["last_event_id"] = np.nan
        data_dict["last_event_type"] = np.nan
        data_dict["last_event_coordinates"] = np.nan
        data_dict["last_event_time"] = np.nan
        data_dict["last_event_period"] = np.nan

    if "secondaryType" not in result_data:
        data_dict["event_secondary_type"] = "NA"
    else:
        data_dict["event_secondary_type"] = result_data["secondaryType"]

    data_dict["coordinates"] = get_coordinates(coordinates_data)

    if about_data["period"] not in period_dict:
        data_dict["home_team_side"] = "NA-Shootout"
        data_dict["away_team_side"] = "NA-Shootout"
    else:
        data_dict["home_team_side"] = period_dict[about_data["period"]]["home"]
        data_dict["away_team_side"] = period_dict[about_data["period"]]["away"]

    if event_type == "Goal":
        data_dict["event_strength_name"] = result_data["strength"]["name"]
        data_dict["event_strength_code"] = result_data["strength"]["code"]
        if "gameWinningGoal" not in result_data:
            data_dict["event_game_winning_goal"] = None
        else:
            data_dict["event_game_winning_goal"] = result_data["gameWinningGoal"]
        if "emptyNet" not in result_data:
            data_dict["event_empty_net"] = "Missing Data"
        else:
            data_dict["event_empty_net"] = result_data["emptyNet"]
    else:
        data_dict["event_strength_name"] = "NA"
        data_dict["event_strength_code"] = "NA"
        data_dict["event_game_winning_goal"] = "NA"
        data_dict["event_empty_net"] = "NA"
    return data_dict


