from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QGridLayout
from .emotion_node import EmotionNode
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class StatusBar(QWidget):
    def __init__(self,emotion_rates):
        super().__init__()
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        emotions = [
            {"name" : "happy" ,"rate" : emotion_rates[0] , "color" : "#121212", "bg" : "#9aab64" , "label" : "Tích cực"},
            {"name" : "neutral" ,"rate" : emotion_rates[1] , "color" : "#121212", "bg" : "#b6caeb" , "label" : "Trung tính"},
            {"name" : "surprise" ,"rate" : emotion_rates[2] , "color" : "#121212", "bg" : "#f3da65" , "label" : "Ngạc nhiên"},
            {"name" : "negative" ,"rate" : emotion_rates[3] , "color" : "#121212", "bg" : "#ff6b6b" , "label" : "Tiêu cực"}
        ]

        # Create Emotion Nodes
        self.emotion_nodes = []
        row = 0
        col = 0
        for emotion in emotions:
            emotion_node = EmotionNode(emotion)
            self.emotion_nodes.append(emotion_node)
            layout.addWidget(emotion_node, row, col, 1, 2)
            col += 2
        
        # Create Overall Status
        overall_container = QWidget()
        overall = QVBoxLayout(overall_container)
        overall.setContentsMargins(16,16,16,16)
        overall_container.setStyleSheet("background-color: #fff; border-radius: 16px;")

        overall_title = QLabel("Tình trạng")
        overall_title.setStyleSheet("""
                                        color : #333; font-size : 16px; font-weight : 800;
                                        
                                    """)
        self.overall_status = QLabel()
        self.overall_status.setStyleSheet(f"""
                                        color : #333; font-size : 16px; font-weight : 600;
                                    """)
        overall.addWidget(overall_title)
        overall.addWidget(self.overall_status)
        layout.addWidget(overall_container, 0, col, 1, 2)


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
        
        