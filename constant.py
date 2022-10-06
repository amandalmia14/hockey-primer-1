import os

game_type_map = {"regular_season": "02",
                 "playoffs": "03"}


class Directory:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + "data" + os.path.sep


class APIList():
    GET_ALL_MATCHES_FOR_A_GIVEN_SEASON = "https://statsapi.web.nhl.com/api/v1/schedule?season="
    GET_ALL_DATA_FOR_A_GIVEN_MATCH = "https://statsapi.web.nhl.com/api/v1/game/{}/feed/live/"


class CustomRegex():
    REGULAR_GAME_ID = r"\d{0,4}02\d{0,4}"  # 02 for regular season
    PLAYOFFS_ID = r"\d{0,4}03\d{0,4}"  # 03 for playoffs

TYPES_OF_SHOTS = ["Goal", "Shot"]

# For playoff games, the 2nd digit of the specific number gives the round of the playoffs,
# the 3rd digit specifies the matchup, and the 4th digit specifies the game (out of 7).
