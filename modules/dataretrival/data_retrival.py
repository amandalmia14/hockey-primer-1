import json
import os

import pandas as pd

from constant import Directory, TYPES_OF_SHOTS


def get_json_path(game_id: int):
    """
    This function takes an input game id and return the location of the json file
    @param game_id: game id for which we need to get the data
    @return: local system path
    """
    year = str(game_id)[:4]
    season = str(game_id)[4:6]
    if season == "02":
        game_type = "regular_season"
    elif season == "03":
        game_type = "playoffs"
    else:
        game_type = ""
    return Directory.DATA_DIR + year + os.path.sep + str(year) + "_" + game_type + ".json"


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
    teams_data = game_meta["gameData"]["teams"]
    return {"home": teams_data["home"]["name"], "away": teams_data["away"]["name"]}


def get_side(game_meta):
    periods_data = game_meta["liveData"]["linescore"]["periods"]
    period_dict = {}
    if len(periods_data) > 0:
        for i, period in enumerate(periods_data):
            period_dict[i + 1] = {"home": period["home"]["rinkSide"], "away": period["away"]["rinkSide"]}
    return period_dict


def data_parsing(data, id, event_type, period_dict, team_detail_dict):
    """
    This functions transforms the json data into the relevant information for the usecase
    @param data: entire metadata and details of the given game id
    @param id: game id
    @param event_type: type of game Shot / Goal
    @return: json object
    """
    players_data = data["players"]
    result_data = data["result"]
    about_data = data["about"]
    coordinates_data = data["coordinates"]
    team_data = data["team"]

    shooter, goalie = get_shooter_goalie(players_data)

    data_dict = {"game_id": id, "event_code": result_data["eventCode"],
                 "player_info": flatten_player_data(players_data),
                 "shooter": shooter, "goalie": goalie, "event": result_data["event"],
                 "event_type_id": result_data["eventTypeId"], "event_description": result_data["description"],
                 "event_secondary_type": result_data["secondaryType"],

                 "home_team": team_detail_dict["home"], "away_team": team_detail_dict["away"],
                 "home_team_side": period_dict[about_data["period"]]["home"],
                 "away_team_size": period_dict[about_data["period"]]["away"],

                 "about_event_id": about_data["eventId"], "about_period": about_data["period"],
                 "about_period_type": about_data["periodType"], "game_time": about_data["periodTime"],
                 "about_time_remaining": about_data["periodTimeRemaining"], "about_date_time": about_data["dateTime"],
                 "about_goal_away": about_data["goals"]["away"], "about_goal_home": about_data["goals"]["home"],
                 "coordinates": (coordinates_data["x"], coordinates_data["y"]), "action_team_name": team_data["name"]}
    if event_type == "Goal":
        data_dict["event_strength_name"] = result_data["strength"]["name"]
        data_dict["event_strength_code"] = result_data["strength"]["code"]
        data_dict["event_game_winning_goal"] = result_data["gameWinningGoal"]
        data_dict["event_empty_net"] = result_data["emptyNet"]
    else:
        data_dict["event_strength_name"] = "NA"
        data_dict["event_strength_code"] = "NA"
        data_dict["event_game_winning_goal"] = "NA"
        data_dict["event_empty_net"] = "NA"
    return data_dict


def get_goal_shots_data_by_game_id(game_id: int):
    """
    This functions transforms the json data into a df by filtering the relevant live data of the matchs which is
    restricted to "Shots" and "Goals"
    @param game_id: game id for which the transformed data needs to be done
    @return: data frame which consists of shots and goals data
    """
    json_path = get_json_path(game_id=game_id)
    with open(json_path, "r") as f:
        playoffs_game_data_dict = json.load(f)
    game_data = playoffs_game_data_dict[str(game_id)]
    period_dict = get_side(game_meta=game_data)
    teams_type = get_home_away_team(game_meta=game_data)
    live_data = game_data["liveData"]["plays"]["allPlays"]
    final_list = []
    for i in live_data:
        if i["result"]["event"] in TYPES_OF_SHOTS:
            try:
                parsed_data = data_parsing(data=i, id=game_id, event_type=i["result"]["event"],
                                           period_dict=period_dict, team_detail_dict=teams_type)
                final_list.append(parsed_data)
            except Exception as e:
                print(e)

    shots_goals_df = pd.DataFrame(final_list)
    return shots_goals_df


if __name__ == '__main__':
    print(get_goal_shots_data_by_game_id(game_id=2017020001).head())
