from typing import Protocol 

class AppController(Protocol) :
    def send_command(self,command) -> None :
        ...
    def login_success(self, account=None) -> None:
        ...

class HomePresenter:
    def __init__(self, view, app_controller : AppController):
        self.view = view
        self.name = "home_presenter"
        self.app_controller = app_controller
        self.app_controller.append_presenter(self)
    
    def handle_response(self,response):
        if response['status'] == 'success':
            if response['type'] == 'total_emotion_ratios':
                self.view.update_status_bar_signal.emit(response['emotion_ratios'])