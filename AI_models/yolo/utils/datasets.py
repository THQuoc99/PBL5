import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class YOLODataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, augment=True):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment
        
        # Tìm file ảnh trong thư mục con
        img_extensions = ('.jpg', '.jpeg', '.png')
        self.img_files = []
        for root, _, files in os.walk(img_dir):
            self.img_files.extend(os.path.join(root, f) for f in files if f.lower().endswith(img_extensions))
        self.img_files = sorted(self.img_files)
        if not self.img_files:
            raise ValueError(f"No images found in {img_dir}")

        # Định nghĩa augmentation
        self.resize = A.Resize(img_size, img_size)
        transforms = []
        if augment:
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Blur(p=0.1),
                A.CLAHE(p=0.1),
            ] + transforms
        transforms.append(ToTensorV2())
        self.transform = A.Compose(transforms, bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        # Lấy đường dẫn ảnh
        img_path = self.img_files[index]
        
        # Tính đường dẫn nhãn dựa trên img_path
        rel_path = os.path.relpath(img_path, self.img_dir)  # Đường dẫn tương đối
        label_path = os.path.join(self.label_dir, os.path.splitext(rel_path)[0] + '.txt')


        # Tải ảnh
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Không thể tải ảnh: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Tải nhãn
        if os.path.exists(label_path):
            try:
                labels = np.loadtxt(label_path).reshape(-1, 5)  # [class, x, y, w, h]
                labels = labels[(labels[:, 3] > 0) & (labels[:, 4] > 0)]
            except Exception as e:
                raise ValueError(f"Không thể tải nhãn từ {label_path}: {e}")
        else:
            labels = np.array([]).reshape(-1, 5)

        # Áp dụng augmentation hoặc resize
        resized = self.resize(image=img)
        img = resized['image']
        augmented = self.transform(image=img, bboxes=labels[:, 1:] if len(labels) > 0 else [], labels=labels[:, 0] if len(labels) > 0 else [])
        img = augmented['image']  # Đã là tensor [C, H, W]
        # Xử lý labels
        if len(augmented['bboxes']) > 0:
            labels = torch.tensor(np.column_stack((augmented['labels'], augmented['bboxes'])), dtype=torch.float32)
        
            
        # Chuyển img về float32 và chuẩn hóa
        img = img.to(torch.float32) / 255.0  # Giữ đúng kiểu dữ liệu cho mô hình
        
        return img, labels
    
def collate_fn(batch):
    imgs, labels = zip(*batch)

    # Stack ảnh
    imgs = torch.stack(imgs)

    # Gộp nhãn, thêm batch index
    batch_labels = []
    for i, label in enumerate(labels):
        if len(label) > 0:
            # Thêm batch index vào đầu nhãn
            batch_idx = torch.full((label.shape[0], 1), i, dtype=torch.float32)
            label_with_idx = torch.cat((batch_idx, label), dim=1)
            batch_labels.append(label_with_idx)

    # Ghép nhãn, xử lý trường hợp rỗng
    batch_labels = torch.cat(batch_labels, dim=0) if batch_labels else torch.zeros((0, 6))

    return imgs, batch_labels