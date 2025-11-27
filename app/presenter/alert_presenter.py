from typing import Protocol 
from view.components.alert_dialog import AlertDialog
from PyQt6.QtCore import QObject, pyqtSignal
from view.layouts.notification_panel import NotificationPanel

class AppController(Protocol) :
    def send_command(self,command) -> None :
        ...
    def login_success(self, account=None) -> None:
        ...

class AlertPresenter(QObject):
    show_alert_signal = pyqtSignal(str)

    def __init__(self,view, app_controller):
        super().__init__()
        self.name = "alert_presenter"
        self.app_controller = app_controller
        self.app_controller.append_presenter(self)
        self.show_alert_signal.connect(self._show_alert_dialog)
        self.view = view

    def handle_response(self, response):
        alert_type = response.get('type')
        camera = response.get('camera')
        if alert_type in ['negative', 'neutral', 'unhappy']:
            message = self._build_message(alert_type, camera)
            self.view.add_notification_signal.emit(message)
            # Phát tín hiệu hiển thị cảnh báo
            #self.show_alert_signal.emit(message)

    def _show_alert_dialog(self, message):
        alert_dialog = AlertDialog(message, parent=self.app_controller.main_window)
        alert_dialog.show()

    def _build_message(self, alert_type, camera):
        alert_map = {
            'negative': '😠 Nhiều cảm xúc tiêu cực',
            'neutral': '😐 Cảm xúc trung lập kéo dài',
            'unhappy': '😟 Nhiều khách không hài lòng'
        }
        return f"📷 Camera {camera}: {alert_map.get(alert_type, 'Cảnh báo không xác định')}"