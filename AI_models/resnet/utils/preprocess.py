import cv2
import torch
import numpy as np
from torchvision import transforms

# def preprocess_face(face_img):
#     """
#     Tiền xử lý ảnh khuôn mặt cho model ResNet
#     """
#     # Resize về kích thước 48x48
#     face_img = cv2.resize(face_img, (48, 48))
    
#     # Chuyển sang grayscale
#     face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
#     # Chuẩn hóa về [0, 1]
#     face_img = face_img.astype(np.float32) / 255.0
    
#     # Thêm channel và batch dimension
#     face_img = np.expand_dims(face_img, axis=0)
#     face_img = np.expand_dims(face_img, axis=0)
    
#     # Chuyển sang tensor
#     face_tensor = torch.from_numpy(face_img)
    
#     return face_tensor 

def preprocess_face(face_img):
    """
    Tiền xử lý ảnh khuôn mặt cho model ResNet
    """
    # Resize về kích thước 48x48
    face_img = cv2.resize(face_img, (128, 128))
    
    # Chuyển sang grayscale
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Chuẩn hóa về [0, 1]
    face_img = face_img.astype(np.float32) / 255.0
    
    # Thêm channel và batch dimension
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=0)
    
    # Chuyển sang tensor
    face_tensor = torch.from_numpy(face_img)
    
    return face_tensor 