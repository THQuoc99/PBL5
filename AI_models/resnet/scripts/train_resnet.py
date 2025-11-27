# scripts/train_resnet.py
import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Thêm đường dẫn đến thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.resnet50 import ResNet50Plus
from utils.data_utils import load_processed_data
from utils.train_utils import train_model

def main():
    # Đường dẫn dữ liệu
    processed_dir = "data/processed/"
    os.makedirs(processed_dir, exist_ok=True)
    log_dir = "data/logs/"
    os.makedirs(log_dir, exist_ok=True)

    # Tải dữ liệu đã xử lý
    print("Loading processed training data...")
    train_images, train_labels = load_processed_data(processed_dir, dataset_type="train")

    print("Loading processed validation data...")
    val_images, val_labels = load_processed_data(processed_dir, dataset_type="validation")

    print("Loading processed test data...")
    test_images, test_labels = load_processed_data(processed_dir, dataset_type="test")

    # Tạo DataLoader
    train_dataset = TensorDataset(train_images, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    
    val_dataset = TensorDataset(val_images, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Khởi tạo mô hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = ResNet50Plus().to(device)

    # Huấn luyện
    print("Training ResNet-50+...")
    train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device=device, log_dir=log_dir, resume=True)
# Lưu trọng số
    weights_path = "face_emotion_detection/models/resnet50_weights.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"Model weights saved to {weights_path}")

if __name__ == "__main__":
    main()