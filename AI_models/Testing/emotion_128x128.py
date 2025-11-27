import torch
import cv2
import numpy as np
from datetime import datetime
import sys
import os
from pathlib import Path

# Lấy đường dẫn gốc của project
ROOT_DIR = Path(__file__).parent.parent

# Thêm đường dẫn của các thư mục vào sys.path
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / 'yolo'))
sys.path.append(str(ROOT_DIR / 'resnet'))

# Import các module
from yolo.video_processor import VideoProcessor as YOLOVideoProcessor
from yolo.models.common import YOLOV5m
from resnet.models.resnet50_128x128 import ResNet50Plus
from resnet.utils.preprocess import preprocess_face

# Thiết lập device
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Gộp nhãn
def map_emotion(emotion_idx):
    if emotion_idx in [0, 1, 2, 4]:  # Angry, Disgust, Fear, Sad
        return "Negative"
    elif emotion_idx == 3:  # Happy
        return "Happy"
    elif emotion_idx == 5:  # Surprise
        return "Surprise"
    elif emotion_idx == 6:  # Neutral
        return "Neutral"

# Khởi tạo model YOLO
def init_yolo():
    num_classes = 1  # Số class của dataset
    anchors = [
        [[0.88, 1.13],  [1.38, 2.0],  [2.5, 2.75]],
        [[1.44, 2.31],  [2.56, 3.44],  [3.63, 4.88]],
        [[2.34, 3.44],  [3.75, 5.63],  [5.94, 9.06]]
    ]
    
    model = YOLOV5m(first_out=64, nc=num_classes, anchors=anchors, ch=[256, 512, 1024]).to(device)
    checkpoint_path = ROOT_DIR / "weights" / "yolo" / "yolov5m_epoch_44_loss_0.2633.pt"
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

# Khởi tạo model ResNet
def init_resnet():
    model = ResNet50Plus().to(device)
    checkpoint_path = ROOT_DIR / "weights" / "resnet" / "resnet-50-128x128.pth"
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    # Nếu checkpoint là 1 dict với 'state_dict' bên trong
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']

    # Xóa prefix "module." nếu có
    new_state_dict = {}
    for k, v in checkpoint.items():
        new_key = k.replace("module.", "")  # xoá prefix "module."
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model

# Các nhãn cảm xúc
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class EmotionDetector:
    def __init__(self):
        self.yolo_model = init_yolo()
        self.resnet_model = init_resnet()
        self.yolo_processor = YOLOVideoProcessor(self.yolo_model, device)
        self.yolo_processor.start()
        
    def process_frame(self, frame):
        # Xử lý frame với YOLO
        img, img_tensor = self.yolo_processor.load_image(frame)
        with torch.no_grad():
            predictions = self.yolo_model(img_tensor)
        detections = self.yolo_processor.process_predictions(predictions)

        # Scale hệ số từ ảnh 640x640 về ảnh gốc
        orig_h, orig_w = img.shape[:2]
        scale_x = orig_w / 640
        scale_y = orig_h / 640
        
        # Vẽ bounding box và nhận diện cảm xúc cho mỗi khuôn mặt
        for (x1, y1, x2, y2, conf, cls) in detections:
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Cắt ảnh khuôn mặt
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                continue
                
            # Tiền xử lý ảnh khuôn mặt
            face_tensor = preprocess_face(face).to(device)
            
            # Nhận diện cảm xúc
            with torch.no_grad():
                emotion_output = self.resnet_model(face_tensor)
                emotion_probs = torch.softmax(emotion_output, dim=1)
                emotion_idx = torch.argmax(emotion_probs).item()
                emotion = map_emotion(emotion_idx)
                confidence = emotion_probs[0][emotion_idx].item()
            
            # Vẽ bounding box và hiển thị cảm xúc
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(img, f"{emotion} {confidence:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 255, 0), 4)
        
        return img

    def process_image(self, image_path):
        start_time = datetime.now()
        """Xử lý một ảnh tĩnh"""
        # Đọc ảnh
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Không thể đọc ảnh từ {image_path}")
            
        # Xử lý ảnh
        processed_img = self.process_frame(img)

        process_time = datetime.now() - start_time

        print(f"Thời gian xử lý ảnh : {process_time}")
        
        # Lưu ảnh đã xử lý
        output_path = image_path.rsplit('.', 1)[0] + '_processed.jpg'
        cv2.imwrite(output_path, processed_img)
        
        return output_path

    def process_video(self, video_path, output_path=None):
        """Xử lý video"""
        # Mở video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Không thể mở video từ {video_path}")
            
        # Lấy thông tin video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Tạo output path nếu không có
        if output_path is None:
            output_path = video_path.rsplit('.', 1)[0] + '_processed.mp4'
            
        # Tạo video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Xử lý frame
                processed_frame = self.process_frame(frame)
                
                # Ghi frame đã xử lý
                out.write(processed_frame)
                
        finally:
            # Giải phóng tài nguyên
            cap.release()
            out.release()
            
        return output_path

    def process_webcam(self):
        """Xử lý video từ webcam"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Không thể mở webcam")
            
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Xử lý frame
                processed_frame = self.process_frame(frame)
                
                # Hiển thị kết quả
                cv2.imshow("Emotion Detection", processed_frame)
                
                # Thoát nếu nhấn 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def stop(self):
        self.yolo_processor.stop()

def main():
    detector = EmotionDetector()
    
    # Menu chọn chế độ
    print("Chọn chế độ xử lý:")
    print("1. Xử lý ảnh")
    print("2. Xử lý video")
    print("3. Xử lý webcam")
    choice = input("Nhập lựa chọn (1-3): ")
    
    try:
        if choice == '1':
            # Xử lý ảnh
            image_path = input("Nhập đường dẫn ảnh: ")
            output_path = detector.process_image(image_path)
            print(f"Đã xử lý và lưu ảnh tại: {output_path}")
            
        elif choice == '2':
            # Xử lý video
            video_path = input("Nhập đường dẫn video: ")
            output_path = input("Nhập đường dẫn output (Enter để tự động tạo): ")
            if not output_path:
                output_path = None
            output_path = detector.process_video(video_path, output_path)
            print(f"Đã xử lý và lưu video tại: {output_path}")
            
        elif choice == '3':
            # Xử lý webcam
            print("Bắt đầu xử lý webcam. Nhấn 'q' để thoát.")
            detector.process_webcam()
            
        else:
            print("Lựa chọn không hợp lệ!")
            
    except Exception as e:
        print(f"Lỗi: {e}")
    finally:
        detector.stop()

if __name__ == "__main__":
    main() 