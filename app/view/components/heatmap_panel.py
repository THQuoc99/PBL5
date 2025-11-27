from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QStackedWidget, QListWidget, QSizePolicy
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from datetime import datetime
import os
import numpy as np
from view.utils.fonts import set_font

def add_with_limit(d, key, value, limit):
    d[key] = value
    if len(d) > limit:
        oldest_key = sorted(d.keys())[0]
        del d[oldest_key]

class HeatmapPanel(QWidget):  # Thay đổi từ QStackedWidget thành QWidget để quản lý layout
    def __init__(self):
        super().__init__()

        # Fonts
        OpenSans = set_font("assets/fonts/OpenSans-VariableFont_wdth,wght.ttf")
        
        # Layout chính (ngang) để chứa heatmap và sidebar
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Khởi tạo các stack widgets
        self.chart_container = QWidget()
        self.chart_layout = QVBoxLayout(self.chart_container)
        self.chart_container.setStyleSheet("background-color: #b6caeb; border-radius: 12px;")

        # Khởi tạo các heatmap
        self.time_type = 'minute'
        self.is_custom = False
        self.emotion_type = 'happy'
        self.is_current_display = False

        self.limit_minutes = 30
        self.limit_hours = 30
        self.limit_days = 7

        self.heatmap_data_per_minute = {
            "happy": {},
            "neutral": {},
            "negative": {},
            "surprise": {}
        }

        self.heatmap_data_per_hour = {
            "happy": {},
            "neutral": {},
            "negative": {},
            "surprise": {}
        }

        self.heatmap_data_per_day = {
            "happy": {},
            "neutral": {},
            "negative": {},
            "surprise": {}
        }

        self.heatmap_data_custom = {
            "happy": {},
            "neutral": {},
            "negative": {},
            "surprise": {}
        }

        self.heatmap, self.canvas = self.create_heatmap("Heatmap cảm xúc theo phút", "happy")

        # Tạo sidebar
        self.sidebar = QListWidget()
        self.sidebar.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sidebar.setFont(QFont(OpenSans, 10, weight = QFont.Weight.Bold))
        self.sidebar.addItems(["Happy", "Neutral", "Negative", "Surprise"])
        self.sidebar.setStyleSheet("""
            QListWidget {
                background-color: #9aab64;
                border: none;
                color: #f8f2e2;
                border-radius: 12px;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 12px;
            }
            QListWidget::item:selected {
                color: #121212;
                border: none;
                background-color: #808d53;
            }
            QListWidget::item:hover {
                color: #121212;
            }
        """)
        self.sidebar.setSpacing(8)
        self.sidebar.setCursor(Qt.CursorShape.PointingHandCursor)
        self.sidebar.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.sidebar.setFixedWidth(128)  # Độ rộng cố định cho sidebar

        # Kết nối sự kiện chọn item trong sidebar
        self.sidebar.currentRowChanged.connect(self.on_emotion_selected)

        # Thêm center_widget và sidebar vào main_layout
        self.chart_layout.addWidget(self.heatmap)
        layout.addWidget(self.chart_container, stretch=5)
        layout.addWidget(self.sidebar, stretch=1)

        self.setLayout(layout)

        # Khởi tạo với heatmap happy
        self.sidebar.setCurrentRow(0)

        

    def create_heatmap(self, title, emotion_type):
        canvas = FigureCanvas(Figure(figsize=(8, 4)))  # Tăng kích thước để hiển thị rõ hơn
        ax = canvas.figure.add_subplot(111)
        canvas.figure.set_facecolor("#b6caeb")
        ax.set_title(f"{title} - {emotion_type}")
        ax.set_xlabel("Thời gian")
        ax.set_ylabel("Camera")

        widget = QWidget()
        vbox = QVBoxLayout(widget)
        vbox.addWidget(canvas)
        return widget, canvas
    
    def update_heatmap(self, canvas, emotion_data, cameras, time_labels, title, emotion_type, time_format):
        if len(cameras) < 1 or len(time_labels) < 1:
            return
        # Xóa và tạo lại axes
        canvas.figure.clear()
        ax = canvas.figure.add_subplot(111)

        # Chuẩn bị dữ liệu cho heatmap
        heatmap_data = np.zeros((len(cameras), len(time_labels)))
        
        for i, camera in enumerate(cameras):
            for j, time in enumerate(time_labels):
                value = emotion_data[emotion_type].get(time, {}).get(camera, 0)  # Giả sử dữ liệu có camera
                heatmap_data[i, j] = value * 100

        time_label_dts = sorted([
            datetime.strptime(t, "%Y-%m-%d %H:%M:%S") if isinstance(t, str) else t
            for t in time_labels
        ])
        # Vẽ heatmap
        im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=100)
        ax.set_xticks(np.arange(len(time_labels)))
        ax.set_xticklabels([t.strftime(time_format) for t in time_label_dts])
        ax.set_yticks(np.arange(len(cameras)))
        ax.set_yticklabels(cameras)
        ax.set_title(title)
        ax.set_xlabel("Thời gian")
        ax.set_ylabel("Camera")
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            
        canvas.figure.colorbar(im, ax=ax, label=emotion_type)
        canvas.figure.tight_layout()
        canvas.draw()
    
    def update_map(self):
        time_format = None
        if not self.is_current_display:
            return
        elif self.is_custom:
            heatmap_data = self.heatmap_data_custom
            cameras = self.extract_cameras(heatmap_data)
            time_labels = sorted(self.extract_time_labels(heatmap_data))
            if self.time_type == 'minute' :
                title = "Biểu đồ tỉ lệ cảm xúc theo phút"
                time_format = "%H:%M"
            elif self.time_type == 'hour' :
                title = "Biểu đồ tỉ lệ cảm xúc theo giờ"
                time_format = "%H:00"
            elif self.time_type == 'day' :
                title = "Biểu đồ tỉ lệ cảm xúc theo ngày"
                time_format = "%d/%m"
        else:
            if self.time_type == 'minute' :
                heatmap_data = self.heatmap_data_per_minute
                cameras = self.extract_cameras(heatmap_data)
                time_labels = sorted(self.extract_time_labels(heatmap_data))
                title = "Biểu đồ tỉ lệ cảm xúc theo phút"
                time_format = "%H:%M"
            elif self.time_type == 'hour' :
                heatmap_data = self.heatmap_data_per_hour
                cameras = self.extract_cameras(heatmap_data)
                time_labels = sorted(self.extract_time_labels(heatmap_data))
                title = "Biểu đồ tỉ lệ cảm xúc theo giờ"
                time_format = "%H:00"
            elif self.time_type == 'day' :
                heatmap_data = self.heatmap_data_per_day
                cameras = self.extract_cameras(heatmap_data)
                time_labels = sorted(self.extract_time_labels(heatmap_data))
                title = "Biểu đồ tỉ lệ cảm xúc theo ngày"
                time_format = "%d/%m"
        
        if time_format is not None:
            self.update_heatmap(self.canvas, heatmap_data, cameras, time_labels, 
                                title, self.emotion_type, time_format)
    
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
        self.update_map()

    def on_emotion_selected(self, index):
        EMOTION_TYPES = ["happy", "neutral", "negative", "surprise"]
        # Cập nhật index của tất cả các stack dựa trên lựa chọn trong sidebar
        self.emotion_type = EMOTION_TYPES[index]
        self.update_map()

    # Các phương thức khác (append, update,...) có thể giữ nguyên hoặc điều chỉnh tương tự như trước

    def update_heatmap_data_per_minute(self, heatmap_data):
        self.heatmap_data_per_minute = heatmap_data
        self.update_map()

    def update_heatmap_data_per_hour(self, heatmap_data):
        self.heatmap_data_per_hour = heatmap_data
        self.update_map()

    def update_heatmap_data_per_day(self, heatmap_data):
        self.heatmap_data_per_day = heatmap_data
        self.update_map()

    def update_heatmap_data_custom(self, heatmap_data):
        self.heatmap_data_custom = heatmap_data
        self.update_map()
    
    def append_heatmap_data_per_minute(self, heatmap_data):
        time_key = heatmap_data['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        data = heatmap_data['heatmap_data']

        emotion_heatmap_data = {
            'happy' : {},
            'neutral' : {},
            'negative' : {},
            'surprise' : {}
        }

        for camera, emotion_ratios in data.items():
            emotion_heatmap_data["happy"][camera] = emotion_ratios["Happy"] 
            emotion_heatmap_data["neutral"][camera] = emotion_ratios["Neutral"] 
            emotion_heatmap_data["negative"][camera] = emotion_ratios["Negative"] 
            emotion_heatmap_data["surprise"][camera] = emotion_ratios["Surprise"] 

        add_with_limit(self.heatmap_data_per_minute['happy'], time_key, emotion_heatmap_data['happy'], self.limit_minutes)
        add_with_limit(self.heatmap_data_per_minute['negative'], time_key, emotion_heatmap_data['negative'], self.limit_minutes)
        add_with_limit(self.heatmap_data_per_minute['surprise'], time_key, emotion_heatmap_data['surprise'], self.limit_minutes)
        add_with_limit(self.heatmap_data_per_minute['neutral'], time_key, emotion_heatmap_data['neutral'], self.limit_minutes)

        if self.time_type == 'minute' and not self.is_custom :
            self.update_map()
    
    def append_heatmap_data_per_hour(self, heatmap_data):
        time_key = heatmap_data['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        data = heatmap_data['heatmap_data']

        emotion_heatmap_data = {
            'happy': {},
            'neutral': {},
            'negative': {},
            'surprise': {}
        }

        for camera, emotion_ratios in data.items():
            emotion_heatmap_data["happy"][camera] = emotion_ratios["Happy"] 
            emotion_heatmap_data["neutral"][camera] = emotion_ratios["Neutral"] 
            emotion_heatmap_data["negative"][camera] = emotion_ratios["Negative"] 
            emotion_heatmap_data["surprise"][camera] = emotion_ratios["Surprise"] 

        for emotion in emotion_heatmap_data:
            add_with_limit(self.heatmap_data_per_hour[emotion], time_key, emotion_heatmap_data[emotion], self.limit_hours)

        if self.time_type == 'hour' and not self.is_custom :
            self.update_map()
    
    def append_heatmap_data_per_day(self, heatmap_data):
        time_key = heatmap_data['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        data = heatmap_data['heatmap_data']

        emotion_heatmap_data = {
            'happy': {},
            'neutral': {},
            'negative': {},
            'surprise': {}
        }

        for camera, emotion_ratios in data.items():
            emotion_heatmap_data["happy"][camera] = emotion_ratios["Happy"] 
            emotion_heatmap_data["neutral"][camera] = emotion_ratios["Neutral"] 
            emotion_heatmap_data["negative"][camera] = emotion_ratios["Negative"] 
            emotion_heatmap_data["surprise"][camera] = emotion_ratios["Surprise"] 

        for emotion in emotion_heatmap_data:
            add_with_limit(self.heatmap_data_per_day[emotion], time_key, emotion_heatmap_data[emotion], self.limit_days)

        if self.time_type == 'day' and not self.is_custom :
            self.update_map()
        
    def extract_time_labels(self, data):
        # Trả về tập hợp tất cả mốc thời gian có trong dữ liệu
        time_labels = set()
        for emotion_dict in data.values():
            time_labels.update(emotion_dict.keys())
        return list(time_labels)

    def extract_cameras(self, data):
        # Trả về tập hợp tất cả các camera xuất hiện
        cameras = set()
        for emotion_dict in data.values():
            for time_point in emotion_dict.values():
                cameras.update(time_point.keys())
        return list(cameras)