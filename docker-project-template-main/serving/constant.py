import os

# For playoff games, the 2nd digit of the specific number gives the round of the playoffs,
# the 3rd digit specifies the matchup, and the 4th digit specifies the game (out of 7).
game_type_map = {"regular_season": "02",
                 "playoffs": "03"}
year_list = [2016, 2017, 2018, 2019, 2020]


class Directory:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "data" + os.path.sep
    FIGURE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "figures" + os.path.sep
    ADV_VIZ_PKL_FILE = DATA_DIR + 'major_dict_1px.p'
    ALL_SEASON_DATA_PKL_FILE = DATA_DIR + 'all_season.pkl'
    TIDY_DATA_PKL_FILENAME = 'tidy_data.pkl'


class APIList():
    GET_ALL_MATCHES_FOR_A_GIVEN_SEASON = "https://statsapi.web.nhl.com/api/v1/schedule?season="
    GET_ALL_DATA_FOR_A_GIVEN_MATCH = "https://statsapi.web.nhl.com/api/v1/game/{}/feed/live/"


class CustomRegex():
    REGULAR_GAME_ID = r"\d{0,4}02\d{0,4}"  # 02 for regular season
    PLAYOFFS_ID = r"\d{0,4}03\d{0,4}"  # 03 for playoffs


TYPES_OF_SHOTS = ["Goal", "Shot"]
COMET_FILE = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "configfile.ini"
DOWNLOADED_MODEL_PATH = "../model/"

model_name_map = {
    "neural-network-model": "Neural_Network",
    "xgboost-feature-selection-class-weights": "xgboost_feature_selection_class_weights",
    "linearmodel-angle": "LinearModel_Angle",
    "linearmodel-angle-distance": "LinearModel_Angle_Distance",
    "linearmodel-distance": "LinearModel_Distance",
}

all_imp_features = ['angle', 'distance_from_last_event', 'empty_net', 'shot_type_Wrap-around', 'y_coordinate', 'speed',
                    'distance', 'x_coordinate', 'game_period', 'shot_type_Tip-In', 'shot_type_Wrist Shot',
                    'game_seconds']
model_feature_map = {
    "Neural_Network": all_imp_features,
    "xgboost-feature-selection-class-weights": all_imp_features,
    "LinearModel_Angle": ["angle"],
    "LinearModel_Angle_Distance": ["angle", "distance"],
    "LinearModel_Distance": ["distance"],
}
