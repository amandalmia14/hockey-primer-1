import os
import pickle

import pandas as pd

from constant import Directory, year_list


def get_all_season_teams_matches_played_map():
    """
    This function fetch the count of the matches a team has played for a given season
    Returns: a dictionary of map of the year wise
    {
        2016 : {
                Team A : 23,
                Team B : 24.....
        .....
    }
    """
    season_dict_map = {}
    for season_year in year_list:
        with open(Directory.DATA_DIR + str(season_year) + os.path.sep + 'no_of_matches_' + str(season_year) + '_.p',
                  'rb') as rp:
            df = pickle.load(rp)
            dict_count = {}
            home_list = df["home_team"].tolist()
            away_list = df["away_team"].tolist()
            merged_list = home_list + away_list
            for i in merged_list:
                if i not in dict_count:
                    dict_count[i] = 1
                else:
                    dict_count[i] += 1
            season_dict_map[season_year] = dict_count
    return season_dict_map


def get_offense_corr(row):
    """
    # TODO
    Args:
        row:

    Returns:

    """
    team_side = row.loc["side"]
    corr = row.loc['coordinates']
    x, y = 0, 0
    if corr is not None:
        if team_side == 'right':
            (x, y) = (corr[1], -corr[0])
        elif team_side == 'left':
            (x, y) = (-corr[1], corr[0])
        if y < 0:
            return None
        else:
            return x, y
    else:
        return None


def transform_coordinates(df):
    """
    # TODO
    Args:
        df:

    Returns:

    """
    new_df = df[['game_id', 'action_team_name', 'home_team', 'away_team', 'home_team_side', 'away_team_side',
                 'coordinates']]
    df_home = new_df[new_df["action_team_name"] == new_df["home_team"]][["action_team_name", 'game_id',
                                                                         "home_team_side", 'coordinates']]
    df_home = df_home.rename(columns={"home_team_side": "side"})

    df_away = new_df[new_df["action_team_name"] == new_df["away_team"]][["action_team_name", 'game_id',
                                                                         "away_team_side", 'coordinates']]
    df_away = df_away.rename(columns={"away_team_side": "side"})

    df = pd.concat([df_home, df_away], axis=0)

    final_df = df[df["side"] != "Side Not Available"]
    final_df = final_df[final_df["side"] != 'NA-Shootout']

    final_df = final_df[final_df["side"] != 'NA-Shootout']

    final_df["new_corr"] = final_df.apply(get_offense_corr, axis=1)
    final_df = final_df[final_df["new_corr"].notna()]

    return final_df


def get_count(final_df):
    """
    This method fetches the aggregate count of the shots which has been taken by a given team in a given year for all
    the coordinates.
    Args:
        final_df: input dataframe which contains all the shot details.
    Returns: group by dataframe by year, team, coordinates and count.
    """
    final_df["year"] = final_df["game_id"].apply(lambda x: int(x[:4]))
    count_df = final_df.groupby(["year", "action_team_name", "new_corr"])["count_nos"].agg('count').reset_index()
    return count_df


def get_default_dict_for_shots_frequency():
    """
    This function creates a dict of the coordinates having a coordinates as the key and a default 0 value. This
    can be used to update the count of frequency which has been taken for that particular point.
    Returns: default dict of the coordinates.
    """
    empty_dict = {}
    for i in range(-42, 43):
        for j in range(0, 91):
            empty_dict[(i / 1.0, j / 1.0)] = 0
    return empty_dict


def get_default_dict_for_teams(df):
    """
    This function creates an initialization of the coordinates dict for every teams.
    Args:
        df: input dataframe of the events
    Returns: initialization of the default dict.
    """
    team_list = df["action_team_name"].unique().tolist()
    empty_dict = {}
    for i in team_list:
        empty_dict[i] = get_default_dict_for_shots_frequency()
    return empty_dict


def get_all_season_shot_perhr_map(df):
    """
    This method returns the shots per hour data for all the coordinates. In order to calculate this we took all the
    seasons data which includes from [2016, 2017, 2018, 2019, 2020]
    Args:
        df: input dataframe of the events

    Returns: a dictionary of the coordinates with the average short per hour.
    """
    league_shot_per_hr_map = get_default_dict_for_shots_frequency()
    no_of_games = len(df["game_id"].unique().tolist())  # Equivalent to no of hrs
    df_all_season_map = df.groupby(["new_corr"])["count_nos"].agg('count').reset_index()

    for index, row in df_all_season_map.iterrows():
        league_shot_per_hr_map[row["new_corr"]] = row["count_nos"] / no_of_games
    return league_shot_per_hr_map


def create_data_for_adv_viz():
    """
    Entry point function in order to create the data file for advance visualisation, this file further is incorporated
    in the notebook - 6_advance_visualization
    Returns:

    """
    no_matches_per_season_map = get_all_season_teams_matches_played_map()
    final_df = pd.read_pickle(Directory.ALL_SEASON_DATA_PKL_FILE)
    final_df = transform_coordinates(final_df)
    final_df['count_nos'] = 0
    all_season_shot_perhr_map = get_all_season_shot_perhr_map(df=final_df)
    count_df = get_count(final_df)

    major_dict = {}
    for k, v in count_df.iterrows():
        try:
            no_of_games_in_season = no_matches_per_season_map[v["year"]][v["action_team_name"]]

            avg_shot_per_hr_season = all_season_shot_perhr_map[v["new_corr"]]
            avg_shot_per_hr_game = v["count_nos"] / no_of_games_in_season

            if v["year"] not in major_dict:
                major_dict[v["year"]] = get_default_dict_for_teams(count_df)
                major_dict[v["year"]][v["action_team_name"]][v["new_corr"]] = \
                    (avg_shot_per_hr_season - avg_shot_per_hr_game) / avg_shot_per_hr_season
            else:
                major_dict[v["year"]][v["action_team_name"]][v["new_corr"]] = \
                    (avg_shot_per_hr_season - avg_shot_per_hr_game) / avg_shot_per_hr_season
        except Exception as e:
            print(e)
            print(v)
            break

    with open(Directory.ADV_VIZ_PKL_FILE, 'wb') as fp:
        pickle.dump(major_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    create_data_for_adv_viz()
    print("Prepared Data for Advance Visualisation completed !!")
