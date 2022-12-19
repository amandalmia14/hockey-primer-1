import json

import pandas as pd
import requests
import streamlit as st

from ift6758.ift6758.client.feature_engineering import main_feature_engg
from ift6758.ift6758.client.game_client import GameClient

gc_obj = GameClient()

drop_features_for_display = ['about_time_remaining', 'home_team', 'away_team', 'action_team_name',
                             'event_type_id', 'about_goal_away', 'about_goal_home']

st.title("Hockey Visualization App")
with st.sidebar:
    workspace = st.text_input("Workspace", value="")
    model = st.text_input("Model", value="")
    version = st.text_input("Version", value="")
    data = {"workspace": workspace, "model": model, "version": version}
    if st.button('Load Model'):
        # Load model from Comet ML here using the 3 inputs above
        response = requests.post(
            "http://127.0.0.1:5000/download_registry_model",
            json=data
        )
        st.write(response.json()["message"])

gameID = st.text_input("Game ID", value="", max_chars=10)
if gameID:
    if st.button('Ping game'):
        # call the game client here in order to get a new sample of events differnt from previous ones
        # and return the 9 parameters below
        data = gc_obj.get_live_data(game_id=gameID)
        df_feg = main_feature_engg(df=data)

        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json=json.loads(df_feg.to_json())
        )
        df_input = pd.DataFrame.from_dict(response.json())
        df_feg["Goal Probabilities"] = response.json()["goal_probabilities"]
        last_row_data = df_feg.iloc[-1]
        time_left = last_row_data["about_time_remaining"]

        sum_df = df_feg.groupby(['action_team_name'])['Goal Probabilities'].sum()

        json_data_total_prob = json.loads(sum_df.to_json())
        home_team_name = df_feg["home_team"].unique().tolist()[0]
        away_team_name = df_feg["away_team"].unique().tolist()[0]
        period = last_row_data["game_period"]
        home_team_current_score = last_row_data["about_goal_home"]
        away_team_current_score = last_row_data["about_goal_away"]

        home_team_sum_of_expected_goals = json_data_total_prob[home_team_name]
        away_team_sum_of_expected_goals = json_data_total_prob[away_team_name]

        # Display:
        st.subheader(str('Game #' + str(gameID) + "\:  " + str(home_team_name) + " vs\. " + str(away_team_name)))
        st.subheader(str('Period: ' + str(period) + "   -   " + str(time_left) + " minutes left"))
        col1, col2 = st.columns(2)
        col1.metric(label=str(str(home_team_name) + " xG (actual)"), value=str(
            str(round(home_team_sum_of_expected_goals, 1)) + " (" + str(home_team_current_score) + ")"),
                    delta=round(home_team_sum_of_expected_goals - home_team_current_score, 1))
        col2.metric(label=str(str(away_team_name) + " xG (actual)"), value=str(
            str(round(away_team_sum_of_expected_goals, 1)) + " (" + str(away_team_current_score) + ")"),
                    delta=round(away_team_sum_of_expected_goals - away_team_current_score, 1))
        st.header("Data and Predictions")

        df_feg = df_feg.drop(drop_features_for_display, axis=1)
        st.write(df_feg)

        # Bonus: display a graph - the input will be the dataframe "df_input"
        # *** I will continue working on this part ***
        st.write("caption for bonus graph")
