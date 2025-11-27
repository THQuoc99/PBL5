from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QStackedWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import os
import numpy as np
from view.utils.charts import create_chart, update_line_chart, update_bar_chart

def add_with_limit(d, key, value, limit):
    d[key] = value
    if len(d) > limit:
        oldest_key = sorted(d.keys())[0]
        del d[oldest_key]

def sort_emotion_dict_by_time(emotion_dict):
    for emotion in emotion_dict:
        sorted_items = sorted(emotion_dict[emotion].items())  # sort by string key
        emotion_dict[emotion] = dict(sorted_items)

def has_data(emotion_dict):
    return any(emotion_dict[emotion] for emotion in emotion_dict)

class ChartPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color:#f5b8da; border-radius:12px;")
        self.setContentsMargins(12,12,12,12)
        self.chart, self.canvas = create_chart("Biểu đồ tỉ lệ các cảm xúc theo phút", "#f5b8da")
        self.time_type = 'minute'
        self.is_custom = False
        self.is_current_display = False

        layout = QVBoxLayout()

        layout.addWidget(self.chart)

        self.limit_minutes = 30
        self.limit_hours = 30
        self.limit_days = 7

        self.emotion_per_minute = {
            "happy" : {},
            "neutral" : {},
            "negative" : {},
            "surprise" : {}
        }

        self.emotion_per_hour = {
            "happy" : {},
            "neutral" : {},
            "negative" : {},
            "surprise" : {} 
        }

        self.emotion_per_day = {
            "happy" : {},
            "neutral" : {},
            "negative" : {},
            "surprise" : {} 
        }

        self.emotion_custom = {
            "happy" : {},
            "neutral" : {},
            "negative" : {},
            "surprise" : {} 
        }

        self.setLayout(layout)
    
    def update_chart(self):
        data_dict = None
        if not self.is_current_display:
            return

        if self.is_custom and self.emotion_custom:
            data_dict = self.emotion_custom
        else:
            if self.time_type == 'minute':
                data_dict = self.emotion_per_minute
            elif self.time_type == 'hour':
                data_dict = self.emotion_per_hour
            elif self.time_type == 'day':
                data_dict = self.emotion_per_day

        if not data_dict or not has_data(data_dict):
            print("[DEBUG] Không có dữ liệu cảm xúc để vẽ.")
            return  # Không vẽ nếu không có dữ liệu

        if self.time_type == 'minute':
            update_line_chart(self.canvas, data_dict, "Biểu đồ tỉ lệ các cảm xúc theo phút", "#f5b8da", "%H:%M")
        elif self.time_type == 'hour':
            update_line_chart(self.canvas, data_dict, "Biểu đồ tỉ lệ các cảm xúc theo giờ", "#f5b8da", "%H:00")
        elif self.time_type == 'day':
            update_bar_chart(self.canvas, data_dict, "Biểu đồ tỉ lệ các cảm xúc theo ngày", "#f5b8da", "%d/%m")

    def update_time_type(self, index):
        if index == 0:
            self.time_type = 'minute'
            self.is_custom = False
        elif index == 1:
            self.time_type = 'hour'
            self.is_custom = False
        elif index == 2:
            self.time_type = 'day'
            self.is_custom = False
        else:
            self.is_custom = True
        self.update_chart()


    def update_emotions_custom(self, emotions):
        self.emotion_custom = emotions
        sort_emotion_dict_by_time(self.emotion_custom)
        self.update_chart()
    

    def update_emotions_per_minute(self, emotions):
        self.emotion_per_minute = emotions
        sort_emotion_dict_by_time(self.emotion_per_minute)
        self.update_chart()

    def append_emotion_per_minute(self,emotion):
        time_key = emotion["timestamp"].strftime("%Y-%m-%d %H:%M:%S")

        add_with_limit(self.emotion_per_minute['happy'], time_key, emotion['happy'], self.limit_minutes)
        add_with_limit(self.emotion_per_minute['negative'], time_key, emotion['negative'], self.limit_minutes)
        add_with_limit(self.emotion_per_minute['surprise'], time_key, emotion['surprise'], self.limit_minutes)
        add_with_limit(self.emotion_per_minute['neutral'], time_key, emotion['neutral'], self.limit_minutes)

        if self.time_type == 'minute' and not self.is_custom : 
            self.update_chart()
    
    def update_emotions_per_hour(self, emotions):
        self.emotion_per_hour = emotions
        sort_emotion_dict_by_time(self.emotion_per_hour)
        self.update_chart()

    def append_emotion_per_hour(self, emotion):
        time_key = emotion["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        latest_key = sorted(self.emotion_per_hour['happy'].keys(), reverse=True)[0] if self.emotion_per_hour['happy'] else None

        if latest_key and latest_key == time_key:
            # Thay thế giá trị nếu time_key trùng với phần tử gần nhất
            self.emotion_per_hour['happy'][time_key] = emotion['happy']
            self.emotion_per_hour['neutral'][time_key] = emotion['neutral']
            self.emotion_per_hour['surprise'][time_key] = emotion['surprise']
            self.emotion_per_hour['negative'][time_key] = emotion['negative']
        else:
            # Thêm mới nếu không trùng
            add_with_limit(self.emotion_per_hour['happy'], time_key, emotion['happy'], self.limit_hours)
            add_with_limit(self.emotion_per_hour['neutral'], time_key, emotion['neutral'], self.limit_hours)
            add_with_limit(self.emotion_per_hour['surprise'], time_key, emotion['surprise'], self.limit_hours)
            add_with_limit(self.emotion_per_hour['negative'], time_key, emotion['negative'], self.limit_hours)

        if self.time_type == 'hour' and not self.is_custom : 
            self.update_chart()

    def update_emotions_per_day(self, emotions):
        self.emotion_per_day = emotions
        sort_emotion_dict_by_time(self.emotion_per_day)
        self.update_chart()

    def append_emotion_per_day(self, emotion):
        time_key = emotion["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        latest_key = sorted(self.emotion_per_day['happy'].keys(), reverse=True)[0] if self.emotion_per_day['happy'] else None

        if latest_key and latest_key == time_key:
            # Thay thế giá trị nếu time_key trùng với phần tử gần nhất
            self.emotion_per_day['happy'][time_key] = emotion['happy']
            self.emotion_per_day['neutral'][time_key] = emotion['neutral']
            self.emotion_per_day['surprise'][time_key] = emotion['surprise']
            self.emotion_per_day['negative'][time_key] = emotion['negative']
        else:
            # Thêm mới nếu không trùng
            add_with_limit(self.emotion_per_day['happy'], time_key, emotion['happy'], self.limit_days)
            add_with_limit(self.emotion_per_day['neutral'], time_key, emotion['neutral'], self.limit_days)
            add_with_limit(self.emotion_per_day['surprise'], time_key, emotion['surprise'], self.limit_days)
            add_with_limit(self.emotion_per_day['negative'], time_key, emotion['negative'], self.limit_days)

        if self.time_type == 'day' and not self.is_custom : 
            self.update_chart()

