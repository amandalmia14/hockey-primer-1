import pandas as pd
import numpy as np
import math

def distance(x_coordinate: float, y_coordinate: float, shoot_side: str) -> float:
    """
    This functions computes the distance at which a shot was taken, i.e. the distance between the coordinates and the goal,
    taking the goal side into account
    @param x_coordinate: x coordinates on the ice
    @param y_coordinate: y coordinates on the ice
    @param shoot_side: side of the goal at which the shot is aimed at
    @return: distance
    """
    x = np.nan
    y = np.nan
    if shoot_side == 'right':
        x = 89 - x_coordinate
        y = y_coordinate
    elif shoot_side == 'left':
        x = -89 - x_coordinate
        y = y_coordinate
    else:
        pass
    distance = math.hypot(x,y)
    return distance

# %%
def angle(x_coordinate: float, y_coordinate: float, shoot_side: str) -> float:
    """
    This functions computes the angle at which a shot was taken, i.e. the distance is 0 if the shot is taken from
    in front of the goal, positive if from the right side, and negative if from the left side
    @param x_coordinate: x coordinates on the ice
    @param y_coordinate: y coordinates on the ice
    @param shoot_side: side of the goal at which the shot is aimed at
    @return: angle
    """
    x = np.nan
    y = np.nan
    if shoot_side == 'right':
        x = 89 - x_coordinate
        y = y_coordinate
    elif shoot_side == 'left':
        x = -(-89 - x_coordinate)
        y = y_coordinate
    else:
        pass
    angle = math.atan2(y,x)*180/math.pi
    return angle

# %%
def distance_between_events(x_coordinate: float, y_coordinate: float, last_event_x_coordinate: float, last_event_y_coordinate: float) -> float:
    """
    This functions computes the distance between two points on the ice
    @param x_coordinate: x coordinates on the ice
    @param y_coordinate: y coordinates on the ice
    @return: distance
    """
    x = abs(x_coordinate - last_event_x_coordinate)
    y = abs(y_coordinate - last_event_y_coordinate)
    return math.hypot(x,y)


def main_feature_engg(df):
    df = df.loc[df['coordinates'].notnull()]
    df = df.loc[df['last_event_coordinates'].notnull()]
    def test(x):
        return (len(x) == 2)


    df = df[np.vectorize(test)(df['coordinates'])]
    df = df[np.vectorize(test)(df['last_event_coordinates'])]

    # %%
    # Add distinct columns for x and y coordinates
    df['x_coordinate'] = df['coordinates'].apply(lambda x: x[0])
    df['y_coordinate'] = df['coordinates'].apply(lambda x: x[1])
    df['last_event_x_coordinate'] = df['last_event_coordinates'].apply(lambda x: x[0])
    df['last_event_y_coordinate'] = df['last_event_coordinates'].apply(lambda x: x[1])

    # %%
    # Replace NAs by np.nan and type as float
    df['x_coordinate'].replace({'NA': np.nan}, inplace=True)
    df['x_coordinate'] = df['x_coordinate'].astype('float')
    df['y_coordinate'].replace({'NA': np.nan}, inplace=True)
    df['y_coordinate'] = df['y_coordinate'].astype('float')

    # Add the side of the goal at which the shot is aimed - this is the opposite side of that of the team which is shooting
    df.loc[(df['action_team_name'] == df['away_team']), 'shoot_side'] = df.loc[
        (df['action_team_name'] == df['away_team']), 'home_team_side']
    df.loc[(df['action_team_name'] == df['home_team']), 'shoot_side'] = df.loc[
        (df['action_team_name'] == df['home_team']), 'away_team_side']

    # Compute the distance from the goal
    df['distance'] = np.vectorize(distance)(df['x_coordinate'], df['y_coordinate'], df['shoot_side'])

    # %%
    # Compute the angle
    df['angle'] = np.vectorize(angle)(df['x_coordinate'], df['y_coordinate'], df['shoot_side'])

    # %%
    # Create Goal variable
    df['is_goal'] = np.where(df['event'] == 'Goal', 1, 0)

    # %%
    # Create Empty Net variable
    df['empty_net'] = np.where(df['event_empty_net'] == True, 1, 0)

    # %%
    # # Add season
    # df['season'] = df['game_id'].str[:4]
    #
    # # %%
    # df['season'].unique()

    # %%
    # Period
    df['game_period'] = df['about_period']

    # %%
    # Shot type
    df['shot_type'] = df['event_secondary_type']

    # %%
    # Last event type
    df['last_event_type'] = df['last_event_type_id']

    # %%
    # Distance from last event
    df['distance_from_last_event'] = np.vectorize(distance_between_events)(df['x_coordinate'], df['y_coordinate'],
                                                                           df['last_event_x_coordinate'],
                                                                           df['last_event_y_coordinate'])

    # %%
    # Rebound
    df['rebound'] = np.where(df['last_event_type'] == 'SHOT', True, False)

    # %%
    # Change in shot angle
    df['last_shot_angle'] = df['angle'].shift(1)  # Get the last shot angle
    df['change_in_shot_angle'] = abs(df['angle'] - df['last_shot_angle'])  # Absolute difference of angle
    df.loc[df[
               'rebound'] != True, 'change_in_shot_angle'] = 0  # Only include this value if the shot is a rebund, otherwise zero

    # %%
    # Speed: this is given in feet by second
    df['game_seconds'] = df['game_time'].str[:2].astype(int) * 60 + df['game_time'].str[3:].astype(int)
    df['last_event_seconds'] = df['last_event_time'].str[:2].astype(int) * 60 + df['last_event_time'].str[3:].astype(
        int)
    df['speed'] = df['distance_from_last_event'] / (df['game_seconds'] - df['last_event_seconds'])

    # %%
    # df['about_period_type'].unique

    # %%

    # Keep only relevant columns
    df = df[['game_id', 'distance', 'angle', 'is_goal', 'empty_net', 'game_period', 'shot_type',
             'last_event_type', 'distance_from_last_event',
             'rebound', 'change_in_shot_angle', 'speed', 'x_coordinate', 'y_coordinate', 'game_seconds',
             'about_time_remaining', 'home_team', 'away_team', 'action_team_name', 'event_type_id', 'about_goal_away',
             'about_goal_home']]
    # %%
    # Clean final dataset
    df['speed'] = df['speed'].replace([np.inf], np.nan)
    df = df.dropna(axis=0)
    df = pd.get_dummies(df, columns=['shot_type', 'last_event_type'])
    if 'shot_type_Tip-In' not in df.columns.tolist():
        df['shot_type_Tip-In'] = np.zeros(df.shape[0])
    if 'shot_type_Wrist Shot' not in df.columns.tolist():
        df['shot_type_Wrist Shot'] = np.zeros(df.shape[0])
    if 'shot_type_Wrap-around' not in df.columns.tolist():
        df['shot_type_Wrap-around'] = np.zeros(df.shape[0])

    df = df[['angle', 'distance_from_last_event', 'empty_net', 'is_goal', 'shot_type_Wrap-around', 'y_coordinate', 'speed',
             'distance', 'x_coordinate', 'game_period', 'shot_type_Tip-In', 'shot_type_Wrist Shot', 'game_seconds',
             'about_time_remaining', 'home_team', 'away_team', 'action_team_name', 'event_type_id', 'about_goal_away',
             'about_goal_home']]

    # df = df[['angle', 'distance_from_last_event', 'empty_net', 'shot_type_Wrap-around', 'y_coordinate', 'speed',
    #          'distance', 'x_coordinate', 'game_period', 'shot_type_Tip-In', 'shot_type_Wrist Shot', 'game_seconds',
    #          'about_time_remaining', 'home_team', 'away_team', 'action_team_name', 'event_type_id', 'about_goal_away',
    #          'about_goal_home']]

    return df
