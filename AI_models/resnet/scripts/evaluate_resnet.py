# scripts/evaluate_resnet.py
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

# Thêm đường dẫn đến thư mục gốc vào sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.resnet50 import ResNet50Plus
from utils.data_utils import load_processed_data

def evaluate_model(model, test_loader, device):
    """
    Đánh giá mô hình trên tập test và in ra độ chính xác.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            outputs = model(batch_images)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def main():
    # Đường dẫn dữ liệu
    processed_dir = "data/processed/"
    model_dir = "face_emotion_detection/models/"
    log_dir = "data/logs/"

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Tải dữ liệu test
    print("Loading processed test data...")
    test_images, test_labels = load_processed_data(processed_dir, dataset_type="test")

    # Tạo DataLoader cho tập test
    test_dataset = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Khởi tạo mô hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = ResNet50Plus().to(device)

    # Đánh giá best model
    best_model_path = os.path.join(log_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        best_acc = evaluate_model(model, test_loader, device)
        print(f"Best Model Accuracy: {best_acc:.2f}%")
    else:
        print("Best model not found.")

    # Đánh giá resnet50_weights.pth
    weights_path = os.path.join(model_dir, "resnet50_weights.pth")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        weights_acc = evaluate_model(model, test_loader, device)
        print(f"ResNet50 Weights Accuracy: {weights_acc:.2f}%")
    else:
        print("ResNet50 weights not found.")

if __name__ == "__main__":
    main()
