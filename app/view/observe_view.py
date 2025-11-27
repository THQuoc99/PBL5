from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox, QGridLayout, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QFont
import pickle
from presenter.observe_presenter import ObservePresenter
from typing import Protocol
from view.utils.fonts import set_font
from view.utils.charts import create_chart, update_line_chart

def add_with_limit(d, key, value, limit):
    d[key] = value
    if len(d) > limit:
        oldest_key = sorted(d.keys())[0]
        del d[oldest_key]


class VideoThread(Protocol) :
    change_pixmap_signal = pyqtSignal(QImage)


    
class ObserveView(QWidget):
    update_camera_list_signal = pyqtSignal(list)
    update_emotions_per_minute_signal = pyqtSignal(dict)
    append_emotion_per_minute_signal = pyqtSignal(dict)

    def __init__(self, controller , udp_video_client : VideoThread):
        super().__init__()
        self.controller = controller
        self.presenter = ObservePresenter(self, self.controller)
        self.update_camera_list_signal.connect(self.set_camera_list)
        self.update_emotions_per_minute_signal.connect(self.update_emotions_per_minute)
        self.append_emotion_per_minute_signal.connect(self.append_emotion_per_minute)

        self.limit_minutes = 15
        self.emotion_per_minute = {
            "happy" : {},
            "neutral" : {},
            "negative" : {},
            "surprise" : {}
        }
        
        # Create central widget and layout
        layout = QGridLayout()
        layout.setContentsMargins(24, 0, 24, 0)  # 24px padding
        layout.setSpacing(16)

        # Fonts
        OpenSans = set_font("assets/fonts/OpenSans-VariableFont_wdth,wght.ttf")
        EduNSWACTCursive = set_font("assets/fonts/EduNSWACTCursive-VariableFont_wght.ttf")

        # Create Title
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)

        title = QLabel("Giám sát")
        title.setFont(QFont(OpenSans,16, weight = QFont.Weight.Bold))
        title.setStyleSheet("color: #121212;")
        title_layout.addWidget(title)

        layout.addWidget(title_container, 0, 0, 1, 4)


        # === Create Content Area ===
        content_area_container = QWidget()
        content_area = QVBoxLayout(content_area_container)
        content_area_container.setStyleSheet("background-color: #f5b8da; border-radius: 12px;")

        # Create video label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")  # Tuỳ chọn, để dễ thấy
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        ) 
        content_area.addWidget(self.video_label)

        layout.addWidget(content_area_container, 1, 0, 3, 3)


        # === Create Line Chart ===
        chart_container = QWidget()
        chart_container.setStyleSheet("background-color: #b6caeb; border-radius: 12px;")

        chart_layout = QVBoxLayout(chart_container)
        chart_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.line_chart, self.chart_canvas = create_chart("Biểu đồ tỉ lệ các cảm xúc theo phút", "#b6caeb")
        self.line_chart.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.chart_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        chart_layout.addWidget(self.line_chart)
        layout.addWidget(chart_container, 4, 0, 2, 3)


        # Create Camera List
        camera_list_container = QWidget()
        self.camera_list = QVBoxLayout(camera_list_container)
        self.camera_list.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.camera_list.setContentsMargins(16,16,16,16)
        camera_list_container.setStyleSheet("background-color: #f3da65; border-radius: 12px;")

        # Create Camera List Title
        camera_list_title = QLabel("Danh sách camera")
        camera_list_title.setStyleSheet("font-size: 16px; font-weight: 800; color: #000000;")
        self.camera_list.addWidget(camera_list_title)

        # Create Camera List Content
        self.presenter.get_camera_list()
        
        layout.addWidget(camera_list_container, 1, 3, 5, 2)

        # Cho layout biết hàng 2 (main_stack) là phần nên chiếm chiều cao chính
        layout.setRowStretch(0, 0)  # Title
        layout.setRowStretch(1, 3)  # Video
        layout.setRowStretch(2, 0)  # Không dùng, nên bỏ hoặc dùng layout.addWidget widget nào đó
        layout.setRowStretch(3, 0)  # Không dùng
        layout.setRowStretch(4, 2)  # Chart
        layout.setRowStretch(5, 0)

        self.setLayout(layout)

        # Start video thread
        self.video_thread = udp_video_client
        self.video_thread.change_pixmap_signal.connect(self.update_image)
    
    def closeEvent(self, event):
        print("Closing ObserveView, stopping video thread...")
        self.video_thread.stop()
        event.accept()
        
    def update_image(self, qt_image):
        # Chuyển thành QPixmap từ QImage
        if qt_image.isNull():
            print("[ERROR] QImage is null! Ảnh bị lỗi.")
            return
        
        if not self.video_label:
            print("[ERROR] video_label is not initialized.")
            return
        
        try:
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(
                self.video_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
        except Exception as e:
            print("[EXCEPTION] Khi hiển thị ảnh:", e)
    
    def set_camera_list(self, camera_list_content):
        if camera_list_content:
            for content in camera_list_content:
                camera_list_item = QPushButton(content['name'])
                camera_list_item.setStyleSheet("font-size: 16px; font-weight: 600; color: #000000; border-radius: 16px; background-color: #fff; padding: 8px 16px;")
                camera_list_item.clicked.connect(lambda x, cam_id=content['camera_id']: self.set_target_camera(cam_id))
                self.camera_list.addWidget(camera_list_item)
    
    def set_target_camera(self,camera_id):
        self.presenter.view_camera(camera_id)
        self.presenter.get_camera_emotions_in_last_minutes(camera_id, self.limit_minutes)
    
    def update_emotions_per_minute(self, emotions):
        self.emotion_per_minute = emotions
        update_line_chart(self.chart_canvas, self.emotion_per_minute, "Biểu đồ tỉ lệ các cảm xúc theo phút", "#b6caeb", "%H:%M")

    def append_emotion_per_minute(self,emotion):
        time_key = emotion["timestamp"].strftime("%Y-%m-%d %H:%M:%S")

        add_with_limit(self.emotion_per_minute['happy'], time_key, emotion['happy'], self.limit_minutes)
        add_with_limit(self.emotion_per_minute['negative'], time_key, emotion['negative'], self.limit_minutes)
        add_with_limit(self.emotion_per_minute['surprise'], time_key, emotion['surprise'], self.limit_minutes)
        add_with_limit(self.emotion_per_minute['neutral'], time_key, emotion['neutral'], self.limit_minutes)

        update_line_chart(self.chart_canvas, self.emotion_per_minute, "Biểu đồ tỉ lệ các cảm xúc theo phút", "#b6caeb", "%H:%M")


