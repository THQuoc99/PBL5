from PyQt6.QtWidgets import (
    QWidget, QListWidget, QDateTimeEdit, QComboBox, QPushButton,
    QVBoxLayout, QHBoxLayout, QListView
)
from PyQt6.QtCore import Qt, QDateTime, QDate, QTime, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from view.utils.fonts import set_font

class TimeSelectorWidget(QWidget):
    time_selected = pyqtSignal(QDateTime, QDateTime)
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout(self)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background-color:#f3da65; border-radius:12px;")
        self.setContentsMargins(16,16,16,16)
        self.setMinimumHeight(0)
        self.setMaximumHeight(128)

        # Fonts
        OpenSans = set_font("assets/fonts/OpenSans-VariableFont_wdth,wght.ttf")

        # === Hàng 1: chọn mốc thời gian ===
        self.time_range_selector = QListWidget()
        self.time_range_selector.setFlow(QListView.Flow.LeftToRight)
        self.time_range_selector.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.time_range_selector.setWrapping(False)
        self.time_range_selector.addItems(['Theo phút', 'Theo giờ', 'Theo ngày'])
        self.time_range_selector.setFixedHeight(48)
        self.time_range_selector.setSpacing(10)
        self.time_range_selector.setStyleSheet("""
            QListWidget {
                background-color: transparent;
                border: none;
                color: #f8f2e2;
            }           
            QListWidget::item {
                padding: 4px 12px;
            }
            QListWidget::item:selected {
                color: #121212;
                background-color: #e8c84d;
                border: none;
                border-radius:12px;
            }
            QListWidget::item:hover {
                color: #121212;
            }                        
        """)
        
        self.time_range_selector.setFont(QFont(OpenSans, 10, weight=QFont.Weight.Bold))

        self.layout.addWidget(self.time_range_selector)

        # === Hàng 2: chỉnh thời gian ===
        self.time_controls = QWidget()
        time_controls_layout = QHBoxLayout(self.time_controls)
        self.time_controls.setStyleSheet("background-color: #e8c84d;")

        self.start_time = QDateTimeEdit(QDateTime.currentDateTime())
        self.start_time.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.start_time.setCalendarPopup(True)

        self.end_time = QDateTimeEdit(QDateTime.currentDateTime())
        self.end_time.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.end_time.setCalendarPopup(True)

        self.view_button = QPushButton("Xem")
        self.view_button.setFont(QFont(OpenSans, 10, weight=QFont.Weight.Bold))
        self.view_button.setStyleSheet("""
            QPushButton:hover {
                color: #121212;
            }
        """)

        self.view_button.clicked.connect(self.emit_selected_time)

        # Add widgets vào layout hàng 2
        time_controls_layout.addWidget(self.start_time)
        time_controls_layout.addWidget(self.end_time)
        time_controls_layout.addWidget(self.view_button)

        self.layout.addWidget(self.time_controls)

        # === Sự kiện ===
        self.time_range_selector.currentRowChanged.connect(self.adjust_time_range)

        # Gọi lần đầu để setup mặc định
        QTimer.singleShot(100, lambda: self.time_range_selector.setCurrentRow(0))

    def adjust_time_range(self, index):
        end = QDateTime.currentDateTime()

        if index == 0:  # Theo phút
            start = end.addSecs(-30 * 60)
        elif index == 1:  # Theo giờ
            start = end.addSecs(-30 * 3600)
        elif index == 2:  # Theo ngày
            start = end.addDays(-7)
        else:
            start = end.addSecs(-1800)

        self.start_time.setDateTime(start)
        self.end_time.setDateTime(end)
    
    def emit_selected_time(self):
        start = self.start_time.dateTime()
        end = self.end_time.dateTime()
        self.time_selected.emit(start, end)
