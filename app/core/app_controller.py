from view.dashboard_view import DashboardView
from view.observe_view import ObserveView
from view.home_view import HomeView
from presenter.alert_presenter import AlertPresenter
from view.layouts.menu_bar import MenuBar
from view.layouts.notification_panel import NotificationPanel


from PyQt6.QtWidgets import QMainWindow, QStackedWidget, QApplication, QWidget, QHBoxLayout, QGridLayout


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sentio")
        self.resize(1280, 720)

        self.center()
        self.setStyleSheet("background-color: #f8f2e2;")
        self.current_index = 0

        # ========== TỔNG LAYOUT ==========
        main_widget = QWidget()
        main_layout = QGridLayout()
        main_layout.setContentsMargins(16,16,16,16)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # ========== SIDEBAR ==========
        self.sidebar = MenuBar()
        self.sidebar.menu.currentRowChanged.connect(self.display_view)
        main_layout.addWidget(self.sidebar, 0, 0, 9, 2)

        # ========== STACKED VIEW ==========
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack, 0, 2, 9, 6)  # 👈 chiếm 6 cột thay vì 7

        # ========== SIDEBAR PHẢI ==========
        self.notification_panel = NotificationPanel()
        main_layout.addWidget(self.notification_panel, 0, 8, 9, 2)
        main_layout.setColumnStretch(0, 2)  # sidebar
        main_layout.setColumnStretch(2, 5)  # stacked widget
        main_layout.setColumnStretch(7, 3)  # notification panel

        self.sidebar.setMinimumWidth(0)
        self.stack.setMinimumWidth(0)
        self.notification_panel.setMinimumWidth(0)

        self.sidebar.menu.setCurrentRow(0)

    def closeEvent(self, event):
        event.accept()
    
    def display_view(self, index):
        # Nếu đang rời khỏi ObserveView thì stop camera stream
        old_widget = self.stack.widget(self.current_index)
        if isinstance(old_widget, ObserveView):
            old_widget.presenter.stop_view()
            print("[MainWindow] ⏹ Gửi lệnh stop_watch do rời khỏi ObserveView")
        elif isinstance(old_widget, DashboardView):
            old_widget.set_current_display(False)

        self.current_index = index
        self.stack.setCurrentIndex(index)
        if isinstance(self.stack.currentWidget(), DashboardView):
            print("[DEBUG] Hiển thị dashboard")
            self.stack.currentWidget().set_current_display(True)
    
    def center(self):
        screen = QApplication.primaryScreen()  # lấy màn hình chính
        screen_geometry = screen.availableGeometry()  # lấy kích thước màn hình khả dụng (không tính taskbar)

        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2

        self.move(x, y)

class AppController:
    def __init__(self, command_client, video_receiver):
        self.command_client = command_client
        self.command_client.set_response_callback(self.handle_response)
        self.video_receiver = video_receiver
        self.receiver_host = self.video_receiver.host
        self.receiver_port = self.video_receiver.port
        self.presenters = {}

        self.main_window = MainWindow()
        self.home_view = HomeView(self)
        self.dashboard_view = DashboardView(self)
        self.observe_view = ObserveView(self, self.video_receiver)

        self.alert_presenter = AlertPresenter(self.main_window.notification_panel ,self)
        

        # Thêm các view vào QStackedWidget
        self.main_window.stack.addWidget(self.home_view)
        self.main_window.stack.addWidget(self.dashboard_view)
        self.main_window.stack.addWidget(self.observe_view)

        self.main_window.show()

    # === COMMAND CLIENT COMMUNICATION ===

    def send_command(self, command):
        self.command_client.send_command(command)
    
    def handle_response(self, response):
        command_type = response.get("presenter")
        presenter = self.presenters.get(command_type)

        if presenter:
            presenter.handle_response(response)
        else:
            print(f"[AppController] ⚠️ Không tìm thấy presenter cho loại lệnh: {command_type}")

    # === PRESENTER COMMUNICATION ===

    def append_presenter(self, presenter):
        self.presenters[presenter.name] = presenter

    def get_command_client_ip(self):
        return self.command_client.receiver_host

    


