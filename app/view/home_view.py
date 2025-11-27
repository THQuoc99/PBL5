from PyQt6.QtWidgets import QWidget, QHBoxLayout, QGridLayout, QGraphicsDropShadowEffect, QVBoxLayout, QLabel
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor, QFont
from typing import Protocol
import os
from .components.emotion_node import EmotionNode
from presenter.home_presenter import HomePresenter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from view.utils.fonts import set_font


shadow = QGraphicsDropShadowEffect()
shadow.setBlurRadius(10)
shadow.setXOffset(0)
shadow.setYOffset(4)
shadow.setColor(QColor(0, 0, 0, 50))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

class AppController(Protocol) :
    def change_view(self, selected_menu) -> None:
        ...

class HomeView(QWidget):
    update_status_bar_signal = pyqtSignal(dict)

    def __init__(self, controller : AppController):
        super().__init__()
        self.controller = controller
        self.presenter = HomePresenter(self,self.controller)
        self.setContentsMargins(0, 0, 0, 0)
        self.update_status_bar_signal.connect(self.updateRate)

        
        # Create and main layout
        layout = QGridLayout()
        layout.setContentsMargins(24, 0, 24, 0)  # 24px padding
        layout.setSpacing(16)

        # Fonts
        OpenSans = set_font("assets/fonts/OpenSans-VariableFont_wdth,wght.ttf")
        EduNSWACTCursive = set_font("assets/fonts/EduNSWACTCursive-VariableFont_wght.ttf")

        # Create Title
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)

        title = QLabel("Tổng quan")
        title.setFont(QFont(OpenSans,16, weight = QFont.Weight.Bold))
        title.setStyleSheet("color: #121212;")

        description = QLabel("Xin chào, Dưới đây là tổng quan tình hình hiện tại!")
        description.setFont(QFont(OpenSans, 12, weight=QFont.Weight.Bold))
        description.setStyleSheet("color: #94948f;")

        title_layout.setSpacing(4)
        title_layout.addWidget(title)
        title_layout.addWidget(description)


        layout.addWidget(title_container, 0, 0, 1, 4)



        # Create Status Bar
        emotion_rates = [0.0, 0.0, 0.0, 0.0]

        emotions = [
            {"name" : "happy" ,"rate" : emotion_rates[0] , "color" : "#121212", "bg" : "#9aab64" , "label" : "Tích cực"},
            {"name" : "neutral" ,"rate" : emotion_rates[1] , "color" : "#121212", "bg" : "#b6caeb" , "label" : "Trung tính"},
            {"name" : "surprise" ,"rate" : emotion_rates[2] , "color" : "#121212", "bg" : "#f3da65" , "label" : "Ngạc nhiên"},
            {"name" : "negative" ,"rate" : emotion_rates[3] , "color" : "#121212", "bg" : "#ff6b6b" , "label" : "Tiêu cực"}
        ]

        # Create Emotion Nodes
        self.emotion_nodes = []
        for emotion in emotions:
            emotion_node = EmotionNode(emotion)
            self.emotion_nodes.append(emotion_node)
        
        layout.addWidget(self.emotion_nodes[0], 1, 0, 1, 2)
        layout.addWidget(self.emotion_nodes[1], 1, 2, 1, 2)
        layout.addWidget(self.emotion_nodes[2], 2, 0, 1, 2)
        layout.addWidget(self.emotion_nodes[3], 2, 2, 1, 2)

        
        overall_chart_container = QWidget()
        overall_chart = QVBoxLayout(overall_chart_container)
        overall_chart.setContentsMargins(16,16,16,16)
        overall_chart_container.setStyleSheet("background-color: #f5b8da; border-radius: 12px;")

        overall_chart_title =  QLabel("Biểu đồ tổng quan")
        overall_chart_title.setFont(QFont(OpenSans, 16, weight=QFont.Weight.Bold))
        overall_chart_title.setStyleSheet("color: #121212;")
        overall_chart.addWidget(overall_chart_title)

        # Tạo canvas pie chart
        self.chart_canvas = MplCanvas(self, width=3, height=3, dpi=100)

        # Dữ liệu ví dụ - thay bằng dữ liệu thực tế bạn muốn hiển thị
        sizes = [25, 35, 20, 20]
        labels = ['Tích cực', 'Trung tính', 'Ngạc nhiên', 'Tiêu cực']
        colors = ['#9aab64', '#b6caeb', '#f3da65', '#ff6b6b']

        # Vẽ Pie Chart
        self.chart_canvas.axes.clear()
        self.chart_canvas.figure.set_facecolor('#f5b8da')
        self.chart_canvas.axes.pie(
            sizes, 
            labels=None, 
            colors=colors, 
            autopct='%1.1f%%', 
            startangle=140,
            labeldistance=0, 
        )
        self.chart_canvas.axes.axis('equal')  # Đảm bảo pie chart là hình tròn

        overall_chart.addWidget(self.chart_canvas)

        layout.addWidget(overall_chart_container, 3, 0, 3, 2)


        # Create Overall Status
        overall_container = QWidget()
        overall = QVBoxLayout(overall_container)
        overall.setContentsMargins(16,16,16,16)
        overall_container.setStyleSheet("background-color: #fff; border-radius: 12px;")

        overall_title = QLabel("Tình trạng")
        overall_title.setStyleSheet("""
                                        color : #121212; font-size : 16px; font-weight : 800;
                                        
                                    """)
        self.overall_status = QLabel()
        self.overall_status.setStyleSheet(f"""
                                        color : #333; font-size : 16px; font-weight : 600;
                                    """)
        overall.addWidget(overall_title)
        overall.addWidget(self.overall_status)
        layout.addWidget(overall_container, 3, 2, 1, 2)

        self.setLayout(layout)

    
    def update_status_bar(self, emotion_rates):
        pass
    
    def updateRate(self, rates):
        index = 0
        for _ , rate in rates.items():
            self.emotion_nodes[index].updateRate(rate)
            index+=1
        
        status, status_color = self.updateStatus(rates)
        self.overall_status.setText(status)
        self.overall_status.setStyleSheet(f"""
                                        color : {status_color}; font-size : 16px; font-weight : 600;
                                    """)
        
        self.updateOverallChart(rates)
        
        self.update()
    
    def updateStatus(self, rates : dict) :
        if rates["Happy"] >= 0.6 :
            return "Hài lòng", "#2E8B57"
        if rates["Negative"] >= 0.5 :
            return "Không hài lòng" , "#B22222"
        if rates["Surprise"] >= 0.4 :
            if rates["Happy"] >= rates["Negative"] :
                return "Hứng thú" , "#2E8B57"
            else :
                return "Bất mãn" , "#B22222"
        if rates["Neutral"] >= 0.4 :
            if rates["Happy"] >= rates["Negative"] :
                return "Ổn định" , "#1E90FF"
            else :
                return "Hơi khó chịu" , "#DAA520"
        return "Bình thường", "#1E90FF"

    def updateOverallChart(self, emotion_rates):
        self.chart_canvas.axes.clear()
        self.chart_canvas.figure.set_facecolor('#f5b8da')
        
        # Trích xuất giá trị từ emotion_rates
        emotions = ["Happy", "Neutral", "Surprise", "Negative"]
        values = [float(emotion_rates.get(emotion, 0)) for emotion in emotions]

        if not values or sum(values) == 0 or any(v is None or not isinstance(v, (int, float)) or v != v for v in values):
            self.chart_canvas.axes.clear()
            self.chart_canvas.axes.text(0.5, 0.5, "Không có dữ liệu", ha='center', va='center', fontsize=14)
            self.chart_canvas.draw()
            return
        
        labels = ['Tích cực', 'Trung tính', 'Ngạc nhiên', 'Tiêu cực']
        colors = ['#9aab64', '#b6caeb', '#f3da65', '#ff6b6b']
        
        # Vẽ Pie Chart
        wedges, texts, autotexts = self.chart_canvas.axes.pie(
            values,
            labels=None,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            labeldistance=0
        )
        self.chart_canvas.axes.axis('equal')  # Đảm bảo hình tròn
        
        # Tùy chỉnh văn bản phần trăm
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
        
        self.chart_canvas.axes.set_title("Tỷ lệ cảm xúc")
        self.chart_canvas.figure.tight_layout()
        self.chart_canvas.draw()

        