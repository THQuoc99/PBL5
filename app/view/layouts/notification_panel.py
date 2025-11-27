from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QListWidgetItem, QMenu
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from view.utils.fonts import set_font

class NotificationPanel(QWidget):
    add_notification_signal = pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("""
            background-color: #ede7d9;
            border-radius: 12px;
        """)

        self.add_notification_signal.connect(self.add_notification)
        self.OpenSans = set_font("assets/fonts/OpenSans-VariableFont_wdth,wght.ttf")

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setLayout(layout)

        title = QLabel("Thông báo")
        title.setFont(QFont(self.OpenSans, 16, weight= QFont.Weight.Bold))
        title.setStyleSheet("margin: 10px;")
        layout.addWidget(title)

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        # Cho phép xóa item với phím Delete
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list_widget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self.show_context_menu)

    def add_notification(self, message: str):
        if self.list_widget.count() >= 15:
            self.list_widget.takeItem(0)

        item = QListWidgetItem()
        label = QLabel(message)
        label.setFont(QFont(self.OpenSans, 10, weight= QFont.Weight.Medium))
        label.setWordWrap(True)  # Cho phép xuống dòng
        label.setFixedWidth(240)
        label.setStyleSheet("border-radius: 0; padding: 5px; color: #121212;")  # Tùy chỉnh thêm nếu muốn

        item.setSizeHint(label.sizeHint())  # Đảm bảo item đủ cao để hiển thị toàn bộ nội dung
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, label)

    def show_context_menu(self, pos):
        menu = QMenu()
        delete_action = menu.addAction("Xóa thông báo")
        action = menu.exec(self.list_widget.mapToGlobal(pos))
        if action == delete_action:
            selected_items = self.list_widget.selectedItems()
            if selected_items:
                for item in selected_items:
                    self.list_widget.takeItem(self.list_widget.row(item))
