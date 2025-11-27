from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QGridLayout, QTabWidget, QTabBar, QGraphicsDropShadowEffect
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor, QPixmap
from PyQt6.QtSvgWidgets import QSvgWidget
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class EmotionNode(QWidget):
    def __init__(self,emotion):
        super().__init__()
        self.emotion = emotion["name"]
        self.emotion_name = emotion["label"]
        self.emotion_bg = emotion["bg"]
        self.emotion_color = emotion["color"]
        self.emotion_rate = emotion["rate"]
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        emotion_node_container = QWidget()
        emotion_node = QGridLayout(emotion_node_container)
        emotion_node.setContentsMargins(24, 8, 24, 8)
        emotion_node_container.setStyleSheet(f"background-color: {self.emotion_bg}; border-radius: 12px;")

        emotion_image = QLabel()
        icon_path = os.path.join(BASE_DIR, "assets", "icons", f"{self.emotion}.svg")
        emotion_image = QSvgWidget(icon_path)
        emotion_image.setFixedSize(64, 64)
        emotion_node.addWidget(emotion_image, 0, 0, 2, 1)

        emotion_label = QLabel(self.emotion_name)
        emotion_label.setStyleSheet(f"font-size: 16px; font-weight: 800; color: {self.emotion_color}; margin-left : 24px;")

        self.emotion_rate_label = QLabel(f"{self.emotion_rate*100:.1f}%")
        self.emotion_rate_label.setStyleSheet(f"font-size: 24px; font-weight: 600; color: {self.emotion_color}; margin-left : 24px;")

        emotion_node.addWidget(emotion_image, 0, 0, 2, 1)
        emotion_node.addWidget(emotion_label, 0, 1)
        emotion_node.addWidget(self.emotion_rate_label, 1, 1)

        layout.addWidget(emotion_node_container)

    def updateRate(self, rate):
        self.emotion_rate = rate
        self.emotion_rate_label.setText(f"{rate * 100:.1f}%")  # Cập nhật text
        self.emotion_rate_label.update()
        self.update()