import sys
import os
import torch
import numpy as np
import cv2
import random

# Thêm đường dẫn đến thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.resnet50 import ResNet50Plus
from utils.data_utils import load_processed_data
from utils.visualize_utils import visualize_emotions

def resize_face(frame, bbox, target_size=(48, 48)):
    """Cắt và resize vùng khuôn mặt về kích thước 48x48."""
    x1, y1, x2, y2 = bbox
    face = frame[y1:y2, x1:x2]
    face = cv2.resize(face, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Chuyển thành grayscale
    face = face.reshape(48, 48, 1) / 255.0  # Chuẩn hóa
    return face

def test_on_sample_images(model, processed_dir):
    """Dự đoán trên 5 ảnh test ngẫu nhiên"""
    test_images, test_labels = load_processed_data(processed_dir, dataset_type="test")

    num_samples = min(5, len(test_images))  
    random_indices = random.sample(range(len(test_images)), num_samples)
    test_images_sample = test_images[random_indices]
    test_labels_sample = test_labels[random_indices]

    preds = model.predict(test_images_sample)
    
    # Đánh giá và trực quan hóa
    visualize_emotions(test_images_sample.numpy(), test_labels_sample.numpy(), preds)

def predict_from_camera(model):
    """Sử dụng webcam để dự đoán cảm xúc thời gian thực"""
    cap = cv2.VideoCapture(0)
    emotions = ["Happy", "Negative", "Neutral", "Surprise"]

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = resize_face(frame, (x, y, x + w, y + h))
            face = torch.tensor(face, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

            with torch.no_grad():
                pred = model.predict(face)
                emotion_text = emotions[pred.item()]

            # Vẽ bounding box và hiển thị nhãn dự đoán
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    processed_dir = "data/processed/"
    weights_path = "face_emotion_detection/models/resnet50_weights.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ResNet50Plus().to(device)

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Model weights loaded successfully.")
    else:
        raise FileNotFoundError("Weights file not found. Train the model first.")

    # Thử nghiệm với 5 ảnh test trước
    test_on_sample_images(model, processed_dir)

    # Sau đó chạy nhận diện bằng camera
    predict_from_camera(model)

if __name__ == "__main__":
    main()
