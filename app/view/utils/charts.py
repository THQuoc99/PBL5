from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QStackedWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from datetime import datetime
import os
import numpy as np

def create_chart(title, facecolor):
    canvas = FigureCanvas(Figure(figsize=(5, 3)))
    ax = canvas.figure.add_subplot(111)
    canvas.figure.set_facecolor(facecolor)
    ax.set_title(title)
    ax.legend()

    widget = QWidget()
    vbox = QVBoxLayout(widget)
    vbox.addWidget(canvas)
    return widget, canvas

def update_line_chart(canvas, data_dict, title, facecolor, time_format):
    ax = canvas.figure.axes[0]
    ax.clear()
    canvas.figure.set_facecolor(facecolor)

    # Chuyển time_labels về datetime nếu đang là chuỗi
    raw_keys = next(iter(data_dict.values())).keys()
    time_labels = sorted([
        datetime.strptime(t, "%Y-%m-%d %H:%M:%S") if isinstance(t, str) else t
        for t in raw_keys
    ])

    for emotion, values in data_dict.items():
        y = [values.get(t.strftime("%Y-%m-%d %H:%M:%S"), 0) * 100 for t in time_labels]
        x_full = [t.strftime("%Y-%m-%d %H:%M:%S") for t in time_labels]  # Dùng để vẽ
        x_label = [t.strftime(time_format) for t in time_labels]       # Dùng để hiển thị nhãn
        ax.plot(x_full, y, label=emotion)

    # Đặt nhãn trục x là x_label
    ax.set_xticks(x_full)  # Dùng x_full để xác định vị trí
    ax.set_xticklabels(x_label, rotation=45, ha='right', fontsize=8)

    ax.set_title(title)
    ax.tick_params(axis='x', labelsize=8)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_ylabel("Tỷ lệ cảm xúc (%)")
    ax.set_xlabel("Thời gian")
    ax.legend()
    canvas.figure.tight_layout()
    ax.figure.canvas.draw()

def update_bar_chart(canvas, data_dict, title, facecolor, time_format):
    ax = canvas.figure.axes[0]
    ax.clear()
    canvas.figure.set_facecolor(facecolor)

    days = sorted(set(k for v in data_dict.values() for k in v.keys()))
    x = np.arange(len(days))
    width = 0.2

    emotions = list(data_dict.keys())
    colors = ['#4CAF50', '#2196F3', '#F44336', '#FFC107']

    for i, emotion in enumerate(emotions):
        y = [data_dict[emotion].get(day, 0)*100 for day in days]
        ax.bar(x + (i - 1.5) * width, y, width, label=emotion, color=colors[i])

    day_dt = [datetime.strptime(t, "%Y-%m-%d %H:%M:%S") for t in days]
    x_labels = [day.strftime(time_format) for day in day_dt]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_title(title)
    ax.set_ylabel("Tỷ lệ cảm xúc (%)")
    ax.set_xlabel("Thời gian")
    ax.tick_params(axis='x', labelsize=8)  # cỡ chữ trục X nhỏ hơn
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    ax.tick_params(axis='y', labelsize=8)  # cỡ chữ trục Y nhỏ hơn
    ax.legend()
    ax.figure.canvas.draw()    