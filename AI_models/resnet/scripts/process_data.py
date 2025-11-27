# scripts/process_data.py
import sys
import os

# Thêm đường dẫn đến thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_utils import load_and_process_data

def main():
    data_dir = "data"  # Thư mục chứa dữ liệu thô
    output_dir = output_dir = "E:/data"   # Thư mục lưu dữ liệu đã xử lý
    # Xử lý và lưu dữ liệu kiểm tra (không augmentation)
    print("Processing and saving test data...")
    load_and_process_data(
        os.path.join(data_dir, "test.csv"),
        output_dir=output_dir,
        dataset_type="test",
        augment=False
    )
    # Xử lý và lưu dữ liệu validation (không augmentation)
    print("Processing and saving validation data...")
    load_and_process_data(
        os.path.join(data_dir, "validation.csv"),
        output_dir=output_dir,
        dataset_type="validation",
        augment=False
    )
    # Xử lý và lưu dữ liệu huấn luyện (với augmentation)
    print("Processing and saving training data...")
    load_and_process_data(
        os.path.join(data_dir, "train.csv"),
        output_dir=output_dir,
        dataset_type="train",
        augment=True
    )
    
    

if __name__ == "__main__":
    main()