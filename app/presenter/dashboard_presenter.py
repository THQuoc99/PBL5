from model.serializers.account_serializer import AccountSerializer
from typing import Protocol 

class AppController(Protocol) :
    def send_command(self,command) -> None :
        ...
    def login_success(self, account=None) -> None:
        ...

class DashboardPresenter:
    def __init__(self, view, app_controller : AppController):
        self.view = view
        self.name = "dashboard_presenter"
        self.app_controller = app_controller
        self.app_controller.append_presenter(self)
    
    def get_emotions_in_last_minutes(self, minutes):
        command = {
            'presenter' : self.name,
            'type' : 'get_emotions_in_last_minutes',
            'minutes' : minutes
        }
        self.app_controller.send_command(command)
    
    def get_emotions_in_last_hours(self,hours):
        command = {
            'presenter' : self.name,
            'type' : 'get_emotions_in_last_hours',
            'hours' : hours
        }
        self.app_controller.send_command(command)
    
    def get_emotions_in_last_days(self,days):
        command = {
            'presenter' : self.name,
            'type' : 'get_emotions_in_last_days',
            'days' : days
        }
        self.app_controller.send_command(command)
    
    def get_emotions_custom(self,time_unit, start_time, end_time):
        command = {
            'presenter' : self.name,
            'type' : 'get_emotions_custom',
            'time_unit' : time_unit,
            'start_time' : start_time,
            'end_time' : end_time,
        }
        self.app_controller.send_command(command)
    
    def get_heatmap_data_in_last_minutes(self, minutes):
        command = {
            'presenter' : self.name,
            'type' : 'get_heatmap_data_in_last_minutes',
            'minutes' : minutes
        }
        self.app_controller.send_command(command)
    
    def get_heatmap_data_in_last_hours(self,hours):
        command = {
            'presenter' : self.name,
            'type' : 'get_heatmap_data_in_last_hours',
            'hours' : hours
        }
        self.app_controller.send_command(command)
    
    def get_heatmap_data_in_last_days(self,days):
        command = {
            'presenter' : self.name,
            'type' : 'get_heatmap_data_in_last_days',
            'days' : days
        }
        self.app_controller.send_command(command)
    
    def get_heatmap_data_custom(self,time_unit, start_time, end_time):
        command = {
            'presenter' : self.name,
            'type' : 'get_heatmap_data_custom',
            'time_unit' : time_unit,
            'start_time' : start_time,
            'end_time' : end_time,
        }
        self.app_controller.send_command(command)
    
    def handle_response(self,response):
        if response['status'] == 'success':
            if response['type'] == 'total_emotion_ratios':
                # self.view.update_status_bar_signal.emit(response['emotion_ratios'])
                pass
            elif response['type'] == 'emotion_per_minute':
                self.view.append_emotion_per_minute_signal.emit(response['data'])
            elif response['type'] == 'emotion_per_hour':
                self.view.append_emotion_per_hour_signal.emit(response['data'])
            elif response['type'] == 'emotion_per_day':
                self.view.append_emotion_per_day_signal.emit(response['data'])
            elif response['type'] == 'emotions_in_last_minutes':
                self.view.update_emotions_per_minute_signal.emit(response['emotions_dict'])
            elif response['type'] == 'emotions_in_last_hours':
                self.view.update_emotions_per_hour_signal.emit(response['emotions_dict'])
            elif response['type'] == 'emotions_in_last_days':
                self.view.update_emotions_per_day_signal.emit(response['emotions_dict'])
            elif response['type'] == 'emotions_custom':
                self.view.update_emotions_custom_signal.emit(response['emotions_dict'])
            
            elif response['type'] == 'heatmap_data_per_minute':
                self.view.append_heatmap_data_per_minute_signal.emit(response['data'])
            elif response['type'] == 'heatmap_data_per_hour':
                self.view.append_heatmap_data_per_hour_signal.emit(response['data'])
            elif response['type'] == 'heatmap_data_per_day':
                self.view.append_heatmap_data_per_day_signal.emit(response['data'])
            elif response['type'] == 'heatmap_data_in_last_minutes':
                self.view.update_heatmap_data_per_minute_signal.emit(response['heatmap_data'])
            elif response['type'] == 'heatmap_data_in_last_hours':
                self.view.update_heatmap_data_per_hour_signal.emit(response['heatmap_data'])
            elif response['type'] == 'heatmap_data_in_last_days':
                self.view.update_heatmap_data_per_day_signal.emit(response['heatmap_data'])
            elif response['type'] == 'heatmap_data_custom':
                self.view.update_heatmap_data_custom_signal.emit(response['heatmap_data'])





            


