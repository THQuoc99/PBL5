from model.serializers.account_serializer import AccountSerializer
from typing import Protocol 

class AppController(Protocol) :
    def send_command(self,command) -> None :
        ...
    def login_success(self, account=None) -> None:
        ...

class ObservePresenter:
    def __init__(self, view, app_controller : AppController):
        self.view = view
        self.name = 'observe_presenter'
        self.app_controller = app_controller
        self.app_controller.append_presenter(self)

    def get_camera_list(self):
        command = {
            'presenter' : self.name,
            'type' : 'get_camera_list',
        }
        self.app_controller.send_command(command)
    
    def view_camera(self,camera_id):
        command = {
            'presenter' : self.name,
            'type' : 'start_watch',
            'target_camera' : camera_id,
            'host' : self.app_controller.receiver_host,
            'port' : self.app_controller.receiver_port
        }
        self.app_controller.send_command(command)
    
    def stop_view(self):
        command = {
            'presenter' : self.name,
            'type' : 'stop_watch',
            'host' : self.app_controller.receiver_host,
            'port' : self.app_controller.receiver_port
        }
        self.app_controller.send_command(command)
    
    def get_camera_emotions_in_last_minutes(self,camera_id, minutes):
        command = {
            'presenter' : self.name,
            'type' : 'get_camera_emotions_in_last_minutes',
            'camera_id' : camera_id,
            'minutes' : minutes
        }
        self.app_controller.send_command(command)

    def handle_response(self, response):
        if response['status'] == "success":
            response_type = response['type']
            if response_type == 'camera_list':
                self.view.update_camera_list_signal.emit(response['list'])
            if response_type == 'emotions_in_last_minutes':
                self.view.update_emotions_per_minute_signal.emit(response['emotions_dict'])
            if response_type == 'emotion_per_minute':
                self.view.append_emotion_per_minute_signal.emit(response['data'])
            



