import torch
import cv2
import numpy as np
import argparse
from models.common import YOLOV5m  # Import model YOLOv5m đã định nghĩa
from utils.metrics import non_max_suppression, cells_to_bboxes  # Hàm NMS để lọc bbox trùng lặp

def draw_boxes(img, detections, class_names):
    for (x1, y1, x2, y2) in detections:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img

def load_image(image_path=None, image_array=None):
    if image_path is not None:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
    elif image_array is not None:
        img = image_array  # Nếu có frame từ webcam
    else:
        raise ValueError("Cần truyền vào image_path hoặc image_array")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Chuyển từ BGR sang RGB
    img_resized = cv2.resize(img, (640, 640))  # Resize về kích thước phù hợp với model
    
    return img_resized

bboxes = np.array([
    [0.1748046875, 0.16976127320954906, 0.076171875, 0.259946949602122], 
    [0.4755859375, 0.34748010610079577, 0.095703125, 0.16976127320954906], 
    [0.65625, 0.28514588859416445, 0.09765625, 0.1883289124668435], 
    [0.830078125, 0.20159151193633953, 0.1015625, 0.23342175066312998]
])

source = "1_Handshaking_Handshaking_1_71.jpg"


if __name__ == "__main__":

    boxs = bboxes * 640  # Chuyển sang pixel
    boxs[:, 0] -= boxs[:, 2] / 2  # x_min = x - w/2
    boxs[:, 1] -= boxs[:, 3] / 2  # y_min = y - h/2
    boxs[:, 2] += boxs[:, 0]  # x_max = x_min + w
    boxs[:, 3] += boxs[:, 1]  # y_max = y_min + h

    img = load_image(source)

    img = draw_boxes(img, boxs, class_names=["face"])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Hiển thị ảnh kết quả
    cv2.imshow("YOLOv5 Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


