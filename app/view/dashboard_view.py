from PyQt6.QtWidgets import QWidget, QHBoxLayout, QGridLayout, QGraphicsDropShadowEffect, QVBoxLayout, QLabel, QStackedWidget, QListWidget, QListView
from PyQt6.QtCore import pyqtSignal, Qt, QDateTime
from PyQt6.QtGui import QColor, QFont
from typing import Protocol
import os
from .components.time_selector import TimeSelectorWidget
from .components.chart_panel import ChartPanel
from .components.heatmap_panel import HeatmapPanel
from presenter.dashboard_presenter import DashboardPresenter
from view.utils.fonts import set_font


shadow = QGraphicsDropShadowEffect()
shadow.setBlurRadius(10)
shadow.setXOffset(0)
shadow.setYOffset(4)
shadow.setColor(QColor(0, 0, 0, 50))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class AppController(Protocol) :
    def change_view(self, selected_menu) -> None:
        ...

class DashboardView(QWidget):
    update_emotions_per_minute_signal = pyqtSignal(dict)
    update_emotions_per_hour_signal = pyqtSignal(dict)
    update_emotions_per_day_signal = pyqtSignal(dict)
    update_emotions_custom_signal = pyqtSignal(dict)
    append_emotion_per_minute_signal = pyqtSignal(dict)
    append_emotion_per_hour_signal = pyqtSignal(dict)
    append_emotion_per_day_signal = pyqtSignal(dict)

    update_heatmap_data_per_minute_signal = pyqtSignal(dict)
    update_heatmap_data_per_hour_signal = pyqtSignal(dict)
    update_heatmap_data_per_day_signal = pyqtSignal(dict)
    update_heatmap_data_custom_signal = pyqtSignal(dict)
    append_heatmap_data_per_minute_signal = pyqtSignal(dict)
    append_heatmap_data_per_hour_signal = pyqtSignal(dict)
    append_heatmap_data_per_day_signal = pyqtSignal(dict)

    def __init__(self, controller : AppController):
        super().__init__()
        self.controller = controller
        self.presenter = DashboardPresenter(self,self.controller)
        self.setContentsMargins(0, 0, 0, 0)

        # Biểu đồ
        self.update_emotions_per_minute_signal.connect(self.update_emotions_per_minute)
        self.update_emotions_per_hour_signal.connect(self.update_emotions_per_hour)
        self.update_emotions_per_day_signal.connect(self.update_emotions_per_day)
        self.update_emotions_custom_signal.connect(self.update_emotions_custom)
        self.append_emotion_per_minute_signal.connect(self.append_emotion_per_minute)
        self.append_emotion_per_hour_signal.connect(self.append_emotion_per_hour)
        self.append_emotion_per_day_signal.connect(self.append_emotion_per_day)

        # Heatmap
        self.update_heatmap_data_per_minute_signal.connect(self.update_heatmap_data_per_minute)
        self.update_heatmap_data_per_hour_signal.connect(self.update_heatmap_data_per_hour)
        self.update_heatmap_data_per_day_signal.connect(self.update_heatmap_data_per_day)
        self.update_heatmap_data_custom_signal.connect(self.update_heatmap_data_custom)
        self.append_heatmap_data_per_minute_signal.connect(self.append_heatmap_data_per_minute)
        self.append_heatmap_data_per_hour_signal.connect(self.append_heatmap_data_per_hour)
        self.append_heatmap_data_per_day_signal.connect(self.append_heatmap_data_per_day)

        
        # Create central widget and main layout
        layout = QGridLayout()
        layout.setContentsMargins(24, 0, 24, 0)  # 24px padding
        layout.setSpacing(16)

        # Fonts
        OpenSans = set_font("assets/fonts/OpenSans-VariableFont_wdth,wght.ttf")
        EduNSWACTCursive = set_font("assets/fonts/EduNSWACTCursive-VariableFont_wght.ttf")

        # Create Title
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)

        title = QLabel("Thống kê")
        title.setFont(QFont(OpenSans,16, weight = QFont.Weight.Bold))
        title.setStyleSheet("color: #121212;")
        title_layout.addWidget(title)

        layout.addWidget(title_container, 0, 0, 1, 4)


        # Line Chart and Bar Chart

        self.main_stack = QStackedWidget()
        self.chart_panel = ChartPanel()
        self.heatmap_panel = HeatmapPanel()  # bạn sẽ tạo giống ChartPanel
        self.main_stack.addWidget(self.chart_panel)
        self.main_stack.addWidget(self.heatmap_panel)

        self.presenter.get_emotions_in_last_minutes(self.chart_panel.limit_minutes)
        self.presenter.get_emotions_in_last_hours(self.chart_panel.limit_hours)
        self.presenter.get_emotions_in_last_days(self.chart_panel.limit_days)
        self.presenter.get_heatmap_data_in_last_minutes(self.heatmap_panel.limit_minutes)
        self.presenter.get_heatmap_data_in_last_hours(self.heatmap_panel.limit_hours)
        self.presenter.get_heatmap_data_in_last_days(self.heatmap_panel.limit_days)

        layout.addWidget(self.main_stack, 2, 0, 4, 4) 

        # Display Type
        self.display_mode_selector = QListWidget()
        self.display_mode_selector.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.display_mode_selector.addItem('Biểu đồ')
        self.display_mode_selector.addItem('Heatmap')
        self.display_mode_selector.setStyleSheet("""
            QListWidget {
                background-color: #ede7d9;
                border: none;
                color: #f8f2e2;
                border-radius: 12px;
            }
            QListWidget::item {
                padding: 8px;
                background-color: #121212;
                border-radius: 12px;
            }
            QListWidget::item:selected {
                color: #f5b8da;
                border: none;
                margin-left: 4px;
            }
            QListWidget::item:hover {
                color: #f5b8da;
            }
        """)
        self.display_mode_selector.setSpacing(12)
        self.display_mode_selector.setFont(QFont(OpenSans, 11, weight=QFont.Weight.Bold))
        self.display_mode_selector.setCursor(Qt.CursorShape.PointingHandCursor)
        self.display_mode_selector.setMaximumWidth(144)
        self.display_mode_selector.setMaximumHeight(128)
        layout.addWidget(self.display_mode_selector, 1, 0, 1, 1)

        # Time
        self.time_selector = TimeSelectorWidget()
        layout.addWidget(self.time_selector, 1, 1, 1, 3)

        self.display_mode_selector.currentRowChanged.connect(self.switch_display_mode)
        self.time_selector.time_range_selector.currentRowChanged.connect(self.switch_time_view)
        self.time_selector.time_selected.connect(self.handle_time_selected)

        # Cho layout biết hàng 2 (main_stack) là phần nên chiếm chiều cao chính
        layout.setRowStretch(0, 0)  # Tiêu đề
        layout.setRowStretch(1, 0)  # Bộ lọc
        layout.setRowStretch(2, 1)  # main_stack: nên "giãn" theo chiều cao
        layout.setRowStretch(3, 1)  # nếu bạn dùng 2 hàng cho main_stack
        layout.setRowStretch(4, 1)

        self.setLayout(layout)

        self.display_mode_selector.setCurrentRow(0)
        self.time_selector.time_range_selector.setCurrentRow(0)
    
    def set_current_display(self, is_current):
        self.chart_panel.is_current_display = is_current
        self.heatmap_panel.is_current_display = is_current
        if is_current:
            self.chart_panel.update_chart()
            self.heatmap_panel.update_map()
    
    def switch_display_mode(self, index):
        self.main_stack.setCurrentIndex(index)

    def switch_time_view(self, time_index):
        self.chart_panel.update_time_type(time_index)
        self.heatmap_panel.update_time_type(time_index)
    
    def handle_time_selected(self, start: QDateTime, end: QDateTime):
        start_str = start.toString("yyyy-MM-dd HH:mm:ss")
        end_str = end.toString("yyyy-MM-dd HH:mm:ss")
        self.presenter.get_emotions_custom(self.chart_panel.time_type, start_str, end_str)
        self.presenter.get_heatmap_data_custom(self.heatmap_panel.time_type, start_str, end_str)
        self.chart_panel.update_time_type(3)
        self.heatmap_panel.update_time_type(3)


    
    def update_emotions_per_minute(self, emotions):
        self.chart_panel.update_emotions_per_minute(emotions)
    
    def update_emotions_per_hour(self, emotions):
        self.chart_panel.update_emotions_per_hour(emotions)
    
    def update_emotions_per_day(self, emotions):
        self.chart_panel.update_emotions_per_day(emotions)

    def update_emotions_custom(self, emotions):
        self.chart_panel.update_emotions_custom(emotions)
    
    def append_emotion_per_minute(self,emotion):
        self.chart_panel.append_emotion_per_minute(emotion)
    
    def append_emotion_per_hour(self,emotion):
        self.chart_panel.append_emotion_per_hour(emotion)

    def append_emotion_per_day(self, emotion):
        self.chart_panel.append_emotion_per_day(emotion)


    

    def update_heatmap_data_per_minute(self, heatmap_data):
        self.heatmap_panel.update_heatmap_data_per_minute(heatmap_data)
    
    def update_heatmap_data_per_hour(self, heatmap_data):
        self.heatmap_panel.update_heatmap_data_per_hour(heatmap_data)
    
    def update_heatmap_data_per_day(self, heatmap_data):
        self.heatmap_panel.update_heatmap_data_per_day(heatmap_data)

    def update_heatmap_data_custom(self, heatmap_data):
        self.heatmap_panel.update_heatmap_data_custom(heatmap_data)
    
    def append_heatmap_data_per_minute(self,heatmap_data):
        self.heatmap_panel.append_heatmap_data_per_minute(heatmap_data)
    
    def append_heatmap_data_per_hour(self,heatmap_data):
        self.heatmap_panel.append_heatmap_data_per_hour(heatmap_data)

    def append_heatmap_data_per_day(self, heatmap_data):
        self.heatmap_panel.append_heatmap_data_per_day(heatmap_data)

    







