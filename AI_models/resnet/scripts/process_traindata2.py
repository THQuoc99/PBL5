# scripts/process_data.py
import sys
import os

# Thêm đường dẫn đến thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_utils import load_and_process_data

def main():
    data_dir = "data"  # Thư mục chứa dữ liệu thô
    output_dir = os.path.join(data_dir, "processed")  # Thư mục lưu dữ liệu đã xử lý
    # Xử lý và lưu dữ liệu huấn luyện (không augmentation)
    print("Processing and saving training data...")
    load_and_process_data(
        os.path.join(data_dir, "output_part2.csv"),
        output_dir=output_dir,
        dataset_type="train1",
        augment=False
    )
    
    

if __name__ == "__main__":
    main()