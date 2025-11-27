# scripts/check_processed_data.py
import sys
import os
import numpy as np  # Thêm import numpy vì sử dụng np.unique, np.min, np.max

# Thêm đường dẫn đến thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from face_emotion_detection.utils.data_utils_5emotion import load_processed_data

def main():
    # Tải dữ liệu đã xử lý
    train_images, train_labels = load_processed_data(train=True)
    test_images, test_labels = load_processed_data(train=False)

    # Kiểm tra shape và giá trị
    print("Training data:")
    print(f"Images shape: {train_images.shape}, Min: {train_images.min():.4f}, Max: {train_images.max():.4f}")
    print(f"Labels shape: {train_labels.shape}, Unique values: {np.unique(train_labels)}")
    
    print("\nTest data:")
    print(f"Images shape: {test_images.shape}, Min: {test_images.min():.4f}, Max: {test_images.max():.4f}")
    print(f"Labels shape: {test_labels.shape}, Unique values: {np.unique(test_labels)}")

if __name__ == "__main__":
    main()