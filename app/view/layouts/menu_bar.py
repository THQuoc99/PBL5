from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QListWidgetItem
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QPixmap, QIcon, QFont, QPainter
from PyQt6.QtSvg import QSvgRenderer
import os
from view.utils.fonts import set_font


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def svg_to_pixmap(svg_path, size=QSize(24, 24)):
    renderer = QSvgRenderer(svg_path)
    pixmap = QPixmap(size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()
    return pixmap

class MenuBar(QWidget):
    menu_selected = pyqtSignal(str)
    def __init__(self, selectedMenu = "dashboard"):
        super().__init__()
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self.setStyleSheet("background-color: #121212; color: #333; border-radius : 12px;")

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 16, 8, 16)
        layout.setSpacing(10)

        # Tiêu đề
        title_font = set_font("assets/fonts/EduNSWACTCursive-VariableFont_wght.ttf")
        title = QLabel("Sentio")
        title.setFont(QFont(title_font, 24, weight=QFont.Weight.Bold))
        title.setStyleSheet("color: #f8f2e2; margin-bottom: 24px;")
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Menu chính
        self.menu = QListWidget()
        self.menu.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.add_menu_item('assets/icons/home.svg', "Home")
        self.add_menu_item('assets/icons/dashboard.svg', "Dashboard")
        self.add_menu_item('assets/icons/observer.svg', "Observer")
        self.menu.setStyleSheet("""
            QListWidget {
                background-color: transparent;
                border: none;
                color: #f8f2e2;
            }
            QListWidget::item {
                padding: 8px;
            }
            QListWidget::item:selected {
                color: #f5b8da;
                border: none;
                margin-left: 4px;
                border-left: 2px solid #f5b8da;
            }
            QListWidget::item:hover {
                color: #f5b8da;
            }
        """)
        self.menu.setSpacing(8)
        self.menu.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(self.menu)

        layout.addStretch()  # 🔥 Đẩy nút xuống dưới

        self.setLayout(layout)
    
    def add_menu_item(self, icon_path, text):
        icon = QIcon(svg_to_pixmap(icon_path))
        item = QListWidgetItem(icon, f" {text}")
        # Tạo font đậm
        font = QFont("Roboto", 11)
        font.setWeight(QFont.Weight.Medium)  # hoặc Bold nếu muốn
        item.setFont(font)
        self.menu.addItem(item)

        
        
       