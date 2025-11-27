from enum import Enum

class CommandType(Enum):
    CONNECT = 'connect'
    DISCONNECT = 'disconnect'
    LOGIN = 'login'
    SHUTDOWN = 'shutdown'
    START_WATCH = 'start_watch'
    STOP_WATCH = 'stop_watch'
    GET_CAMERA_LIST = 'get_camera_list'
    GET_DAILY_EMOTIONS_LIST = 'get_daily_emotions_list'
    GET_MONTHLY_EMOTIONS_LIST = 'get_monthly_emotions_list'
    GET_EMOTIONS_LIST = 'get_emotions_list'
    GET_EMOTIONS_IN_LAST_MINUTES = 'get_emotions_in_last_minutes'
    GET_CAMERA_EMOTIONS_IN_LAST_MINUTES = 'get_camera_emotions_in_last_minutes'
    GET_EMOTIONS_IN_LAST_HOURS = 'get_emotions_in_last_hours'
    GET_EMOTIONS_IN_LAST_DAYS = 'get_emotions_in_last_days'
    GET_EMOTIONS_CUSTOM = 'get_emotions_custom'
    GET_HEATMAP_DATA_IN_LAST_MINUTES = 'get_heatmap_data_in_last_minutes'
    GET_HEATMAP_DATA_IN_LAST_HOURS = 'get_heatmap_data_in_last_hours'
    GET_HEATMAP_DATA_IN_LAST_DAYS = 'get_heatmap_data_in_last_days'
    GET_HEATMAP_DATA_CUSTOM = 'get_heatmap_data_custom'

    

