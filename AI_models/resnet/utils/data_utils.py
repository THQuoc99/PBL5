import numpy as np
import pandas as pd
import torch
import os
import gc
import random
from scipy.ndimage import rotate, shift
from tqdm import tqdm

def get_device():
    """Trả về thiết bị (GPU nếu có, nếu không dùng CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def map_labels(original_labels):
    """Chuyển đổi nhãn cảm xúc thành 4 nhóm chính."""
    new_labels = np.zeros_like(original_labels, dtype=int)
    for i, label in enumerate(original_labels):
        if label == 3:  # Happy
            new_labels[i] = 0
        elif label in [0, 1, 4, 2]:  # Negative (Angry, Disgust, Sad, Fear)
            new_labels[i] = 1
        elif label == 6:  # Neutral
            new_labels[i] = 2
        elif label == 5:  # Surprise
            new_labels[i] = 3
    return new_labels

def augment_image(img):
    """Áp dụng một augmentation ngẫu nhiên trên ảnh."""
    augmentation_choice = random.choice(['flip', 'rotate', 'shift', 'noise', 'none'])

    if augmentation_choice == 'flip':
        return np.flip(img, axis=1)
    elif augmentation_choice == 'rotate':
        return rotate(img, angle=random.choice([-10, 10]), reshape=False, mode='constant', cval=0.0)
    elif augmentation_choice == 'shift':
        return shift(img, shift=[random.randint(-10, 10), random.randint(-10, 10), 0], mode='constant', cval=0.0)
    elif augmentation_choice == 'noise':
        noise = np.random.normal(0, 0.02, img.shape)
        return np.clip(img + noise, 0, 1)
    
    return img  # Không áp dụng augmentation

def load_and_process_data(csv_path, output_dir="data/processed/", dataset_type="train", augment=False, batch_size=5000):
    """Load dữ liệu từ CSV, áp dụng augmentation khi load theo batch để tiết kiệm RAM."""
    device = get_device()
    
    # Đọc file CSV theo batch
    data_iterator = pd.read_csv(csv_path, chunksize=batch_size)
    
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, f"{dataset_type}_images.pt")
    lbl_path = os.path.join(output_dir, f"{dataset_type}_labels.pt")

    augmented_total = 0  # Biến đếm số lượng ảnh được augment

    for batch_num, data in enumerate(tqdm(data_iterator, desc="Processing batches")):
        labels = map_labels(data['emotion'].values)  # Map labels trước khi lưu
        pixels = data['pixels'].values

        images = np.array([np.fromstring(p, sep=" ").reshape(128, 128, 1) / 255.0 for p in pixels])

        # Áp dụng augmentation nếu cần
        if augment and dataset_type.startswith("train"):
            images, labels, augmented_count = augment_data(images, labels)
            augmented_total += augmented_count  # Cộng dồn số ảnh augment

        # Chuyển dữ liệu thành PyTorch tensors
        images = torch.FloatTensor(images).permute(0, 3, 1, 2).to(device)  # [N, 1, 128, 128]
        labels = torch.LongTensor(labels).to(device)

        # Nếu file tồn tại, load dữ liệu cũ và ghi tiếp vào
        if os.path.exists(img_path) and os.path.exists(lbl_path):
            old_images = torch.load(img_path, map_location=device)
            old_labels = torch.load(lbl_path, map_location=device)

            images = torch.cat((old_images, images), dim=0)
            labels = torch.cat((old_labels, labels), dim=0)

        # Lưu dữ liệu đã xử lý
        torch.save(images, img_path)
        torch.save(labels, lbl_path)

        # Giải phóng bộ nhớ sau mỗi batch
        del images, labels
        gc.collect()
        torch.cuda.empty_cache()  # Nếu sử dụng GPU

    print(f"✅ Dữ liệu đã lưu tại {output_dir}")
    print(f"🔹 Tổng số ảnh được augment: {augmented_total}")

def load_processed_data(data_dir="/kaggle/working/", dataset_type="train"):
    """Tải dữ liệu đã xử lý từ file .pt, giữ trên CPU."""
    images = torch.load(os.path.join(data_dir, f"{dataset_type}_images.pt"), map_location="cpu", weights_only=False)
    labels = torch.load(os.path.join(data_dir, f"{dataset_type}_labels.pt"), map_location="cpu", weights_only=False)
    print(f"{dataset_type} - Hình ảnh: shape {images.shape}, dtype {images.dtype}, device {images.device}")
    print(f"{dataset_type} - Nhãn: shape {labels.shape}, dtype {labels.dtype}, device {labels.device}")
    return images, labels

def add_noise(img, noise_level=0.02):
    """Thêm nhiễu Gaussian vào ảnh."""
    noise = np.random.normal(0, noise_level, img.shape)
    return np.clip(img + noise, 0, 1)

def shift_image(img, dx, dy):
    """Dịch chuyển ảnh theo hướng dx, dy."""
    return shift(img, shift=[dx, dy, 0], mode='constant', cval=0.0)

def augment_data(images, labels, augmentation_probability=0.2):
    """Thực hiện augmentation với xác suất nhất định."""
    augmented_images = []
    augmented_labels = []
    augmented_count = 0  # Biến đếm số lượng ảnh được augment
    
    for img, label in zip(images, labels):
        # Không augment tất cả ảnh, chỉ augment với xác suất
        if random.random() < augmentation_probability:
            augmented_images.append(img)
            augmented_labels.append(label)
            
            # Chọn một phương pháp augmentation ngẫu nhiên
            augmentation_choice = random.choice(['flip', 'rotate', 'shift', 'noise', 'none'])
            
            if augmentation_choice == 'flip':
                augmented_images.append(np.flip(img, axis=1))
                augmented_labels.append(label)
                augmented_count += 1
            
            elif augmentation_choice == 'rotate':
                augmented_images.append(rotate(img, 10, reshape=False, mode='constant', cval=0.0))
                augmented_labels.append(label)
                augmented_count += 1
                
            elif augmentation_choice == 'shift':
                augmented_images.append(shift_image(img, dx=10, dy=10))
                augmented_labels.append(label)
                augmented_count += 1
                
            elif augmentation_choice == 'noise':
                augmented_images.append(add_noise(img, noise_level=0.02))
                augmented_labels.append(label)
                augmented_count += 1
        else:
            # Nếu không augment, giữ nguyên ảnh
            augmented_images.append(img)
            augmented_labels.append(label)
    
    return np.array(augmented_images), np.array(augmented_labels), augmented_count
