from PyQt6.QtWidgets import QDialog, QLabel, QVBoxLayout
from PyQt6.QtCore import QTimer

class AlertDialog(QDialog):
    def __init__(self, message: str, parent=None, timeout_ms=10000):
        super().__init__(parent)
        self.setWindowTitle("🚨 Cảnh Báo")
        self.setFixedSize(480, 64)
        self.setModal(False)  # Non-blocking

        layout = QVBoxLayout()
        self.setStyleSheet("background-color: #121212; color: #ffffff;")
        layout.addWidget(QLabel(message))
        self.setLayout(layout)

        QTimer.singleShot(timeout_ms, self.close)

        # Di chuyển về góc dưới phải nếu có parent
        if parent:
            self.move_to_bottom_right(parent)

    def move_to_bottom_right(self, parent_window):
        parent_geom = parent_window.geometry()
        x = parent_geom.x() + parent_geom.width() - self.width() - 20
        y = parent_geom.y() + parent_geom.height() - self.height() - 40
        self.move(x, y)