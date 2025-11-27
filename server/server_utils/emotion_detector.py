# Import các module
import threading
import sys
from pathlib import Path
import torch
import cv2
import time

# Thêm đường dẫn của các thư mục vào sys.path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

from AI_models.yolo.video_processor import VideoProcessor as YOLOVideoProcessor
from AI_models.yolo.models.common import YOLOV5m
from AI_models.resnet.models.resnet50 import ResNet50Plus
from AI_models.resnet.utils.preprocess import preprocess_face

# Thiết lập device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Các nhãn cảm xúc
EMOTIONS = ["Happy", "Negative", "Neutral", "Surprise"]

class EmotionDetector:
    def __init__(self):
        self.yolo_model = self.init_yolo()
        self.resnet_model = self.init_resnet()
        self.yolo_processor = YOLOVideoProcessor(self.yolo_model, device)
        self.yolo_processor.start()
        self._stop_event = threading.Event()

        print(f"Running on Device : {device}")
        
    def init_yolo(self):
        num_classes = 1
        anchors = [
            [[0.88, 1.13],  [1.38, 2.0],  [2.5, 2.75]],
            [[1.44, 2.31],  [2.56, 3.44],  [3.63, 4.88]],
            [[2.34, 3.44],  [3.75, 5.63],  [5.94, 9.06]]
        ]
        
        model = YOLOV5m(first_out=64, nc=num_classes, anchors=anchors, ch=[256, 512, 1024]).to(device)
        checkpoint_path = ROOT_DIR / 'AI_models' / "weights" / "yolo" / "yolov5m_epoch_50_loss_0.2591.pt"
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def init_resnet(self):
        model = ResNet50Plus().to(device)
        checkpoint_path = ROOT_DIR / 'AI_models' / "weights" / "resnet" / "resnet50_weights_20epoch.pth"
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        new_state_dict = {}
        for k, v in checkpoint.items():
            new_key = k.replace("module.", "")
            new_state_dict[new_key] = v

        model.load_state_dict(new_state_dict)
        model.eval()
        return model

    def process_frame(self, frame):
        if self._stop_event.is_set():
            return None, None
        
            
        # Resize frame về kích thước cố định
        img, img_tensor = self.yolo_processor.load_image(frame)
        with torch.no_grad():
            predictions = self.yolo_model(img_tensor)
        detections = self.yolo_processor.process_predictions(predictions)

        emotions = []
        for (x1, y1, x2, y2, conf, cls) in detections:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                continue
                
            face_tensor = preprocess_face(face).to(device)
            
            with torch.no_grad():
                emotion_output = self.resnet_model(face_tensor)
                emotion_probs = torch.softmax(emotion_output, dim=1)
                emotion_idx = torch.argmax(emotion_probs).item()
                emotion = EMOTIONS[emotion_idx]
                confidence = emotion_probs[0][emotion_idx].item()
                emotions.append(emotion)    

            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{emotion} {confidence:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img, emotions if len(detections) > 0 else None

    def stop(self):
        print("Stopping YOLO processor...")
        self._stop_event.set()
        self.yolo_processor.stop()
        time.sleep(1)
        print("YOLO processor stopped")
        