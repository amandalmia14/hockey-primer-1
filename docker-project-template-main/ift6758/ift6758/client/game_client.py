import logging

from ift6758.ift6758.client.data_retrival import *

logger = logging.getLogger(__name__)


class GameClient:
    def __init__(self, ip: str = "0.0.0.0", port: int = 5001, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance"]
        self.features = features
        self.data = None
        self.last_entry_idx = 0

        # any other potential initialization

    def get_live_data(self, game_id):
        # diff based on all event types, but our df only concerns about shot and goal events
        game_dict = get_data_by_gameid(game_id)
        message = "Success"
        empty_df = pd.DataFrame(columns=['game_id', 'event_code', 'player_info', 'shooter', 'goalie', 'event',
                'event_type_id', 'event_description', 'home_team', 'home_team_abv',
                'away_team', 'away_team_abv', 'about_event_id', 'about_period',
                'about_period_type', 'game_time', 'about_time_remaining',
                'about_date_time', 'about_goal_away', 'about_goal_home',
                'action_team_name', 'last_event_code', 'last_event_type_id',
                'last_event_coordinates', 'last_event_time', 'last_event_period',
                'event_secondary_type', 'coordinates', 'home_team_side',
                'away_team_side', 'event_strength_name', 'event_strength_code',
                'event_game_winning_goal', 'event_empty_net'])
        if "message" in game_dict:
            message = "Game data couldn't be found / Invalid Game Id"
            df = empty_df
        else:
            event_arr = game_dict['liveData']['plays']['allPlays']
            if len(event_arr) > self.last_entry_idx:
                df = self.process_live_data(game_id, game_dict, self.last_entry_idx)
                self.last_entry_idx = len(event_arr) - 1
            else:
                df = empty_df
            
        return df, message

    def process_live_data(self, game_id, game_dict, entry_idx):
        period_dict = get_side(game_meta=game_dict)
        teams_type = get_home_away_team(game_meta=game_dict)
        live_data = game_dict["liveData"]["plays"]["allPlays"]
        final_list = []
        last_event = live_data[entry_idx]
        for i in live_data[entry_idx + 1:]:
            if i["result"]["event"] in TYPES_OF_SHOTS:
                try:
                    last_event_data = last_data_parsing(last_event)
                    parsed_data = data_parsing(data=i, id=game_id, event_type=i["result"]["event"],
                                               period_dict=period_dict, team_detail_dict=teams_type,
                                               last_event_data=last_event_data)
                    final_list.append(parsed_data)
                except Exception as e:
                    print(game_id)
                    print(e)
                    import traceback
                    print(traceback.print_exc())
                    break
            last_event = i

        shots_goals_df = pd.DataFrame(final_list)
        return shots_goals_df
