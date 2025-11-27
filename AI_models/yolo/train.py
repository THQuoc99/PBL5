import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
import csv
from tqdm import tqdm  # Để hiển thị tiến trình

# Giả định các class đã được định nghĩa trước
from models.common import YOLOV5m  # File chứa YOLOV5m
from utils.datasets import YOLODataset, collate_fn  # File chứa YOLODataset và collate_fn
from models.loss import YOLOv5Loss  # File chứa YOLOv5Loss
# Giả định có hàm tính mAP (nếu không, bạn có thể thay bằng metric khác)
from utils.metrics import bbox_iou, compute_ap  # Hàm metric (tạm thời giả định)

# Thiết lập tham số
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
anchors = [
    [10, 13, 16, 30, 33, 23],  # P3/8
    [30, 61, 62, 45, 59, 119],  # P4/16
    [116, 90, 156, 198, 373, 326]  # P5/32
]
num_classes = 1  
img_size = 640
batch_size = 16
epochs = 50
learning_rate = 0.01

# Khởi tạo dataset và dataloader
train_dataset = YOLODataset(
    img_dir="path/to/train/images",
    label_dir="path/to/train/labels",
    img_size=img_size,
    augment=True
)
val_dataset = YOLODataset(
    img_dir="path/to/val/images",
    label_dir="path/to/val/labels",
    img_size=img_size,
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

# Khởi tạo mô hình và loss
model = YOLOV5m(
    first_out=64,
    nc=num_classes,
    anchors=anchors,
    ch=[256, 512, 1024]  # Kênh đầu ra của 3 mức FPN trong YOLOv5m
).to(device)

loss_fn = YOLOv5Loss(
    anchors=torch.tensor(anchors).float().to(device),
    anchor_t=4.0,
    balance=[4.0, 1.0, 0.4],
    lambda_box=0.05,
    lambda_obj=0.7,
    lambda_cls=0.3,
    label_smoothing=0.1
)

# Optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=0.9,
    weight_decay=0.0005
)

# Scheduler (giảm learning rate theo cosine annealing - giống YOLOv5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Đường dẫn lưu checkpoint
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Hàm huấn luyện
def train_loop(model, dataloader, loss_fn, optimizer, epoch):
    model.train()
    epoch_loss = 0.0
    box_loss_total = 0.0
    obj_loss_total = 0.0
    cls_loss_total = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} - Training")
    for batch_idx, (imgs, targets) in enumerate(pbar):
        imgs = imgs.to(device)
        targets = targets.to(device)
        
        # Forward
        optimizer.zero_grad()
        predictions = model(imgs)
        total_loss, loss_items = loss_fn(predictions, targets)
        
        # Backward
        total_loss.backward()
        optimizer.step()
        
        # Cộng dồn loss
        epoch_loss += total_loss.item()
        box_loss_total += loss_items[0].item()
        obj_loss_total += loss_items[1].item()
        cls_loss_total += loss_items[2].item()
        
        # Cập nhật progress bar
        pbar.set_postfix({
            "Total": f"{total_loss.item():.4f}",
            "Box": f"{loss_items[0].item():.4f}",
            "Obj": f"{loss_items[1].item():.4f}",
            "Cls": f"{loss_items[2].item():.4f}"
        })
    
    avg_loss = epoch_loss / len(dataloader)
    avg_box = box_loss_total / len(dataloader)
    avg_obj = obj_loss_total / len(dataloader)
    avg_cls = cls_loss_total / len(dataloader)
    
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f} "
          f"(Box: {avg_box:.4f}, Obj: {avg_obj:.4f}, Cls: {avg_cls:.4f})")
    return avg_loss

# Hàm đánh giá
def val_loop(model, dataloader, loss_fn, epoch):
    model.eval()
    epoch_loss = 0.0
    box_loss_total = 0.0
    obj_loss_total = 0.0
    cls_loss_total = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} - Validation")
        for imgs, targets in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)
            
            # Forward
            predictions = model(imgs)
            total_loss, loss_items = loss_fn(predictions, targets)
            
            # Cộng dồn loss
            epoch_loss += total_loss.item()
            box_loss_total += loss_items[0].item()
            obj_loss_total += loss_items[1].item()
            cls_loss_total += loss_items[2].item()
            
            # Lưu dự đoán và nhãn để tính mAP (giả định)
            for i in range(len(predictions)):
                pred = predictions[i].sigmoid()  # Chuyển về [0, 1]
                bs, na, gy, gx, _ = pred.shape
                pred = pred.view(bs, na * gy * gx, -1)  # Flatten
                all_preds.append(pred.cpu())
            
            all_targets.append(targets.cpu())
            
            pbar.set_postfix({
                "Total": f"{total_loss.item():.4f}",
                "Box": f"{loss_items[0].item():.4f}",
                "Obj": f"{loss_items[1].item():.4f}",
                "Cls": f"{loss_items[2].item():.4f}"
            })
    
    avg_loss = epoch_loss / len(dataloader)
    avg_box = box_loss_total / len(dataloader)
    avg_obj = obj_loss_total / len(dataloader)
    avg_cls = cls_loss_total / len(dataloader)
    
    # Tính mAP (giả định có hàm compute_ap)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    # Chuyển đổi dự đoán thành định dạng [x, y, w, h, conf, class_probs]
    # Đây là bước đơn giản hóa, bạn cần thêm NMS nếu muốn mAP chính xác
    mAP = 0.0  # Thay bằng compute_ap(all_preds, all_targets) nếu có
    
    print(f"Epoch {epoch+1}/{epochs} - Val Loss: {avg_loss:.4f} "
          f"(Box: {avg_box:.4f}, Obj: {avg_obj:.4f}, Cls: {avg_cls:.4f}, mAP: {mAP:.4f})")
    return avg_loss, mAP

# Vòng lặp chính
best_val_loss = float("inf")
for epoch in range(epochs):
    # Huấn luyện
    train_loss = train_loop(model, train_loader, loss_fn, optimizer, epoch)
    
    # Đánh giá
    val_loss, val_mAP = val_loop(model, val_loader, loss_fn, epoch)
    
    # Cập nhật scheduler
    scheduler.step()
    
    # Lưu checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint_path = os.path.join(checkpoint_dir, f"yolov5m_epoch_{epoch+1}_loss_{val_loss:.4f}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_loss,
            "mAP": val_mAP
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

print("Training completed!")