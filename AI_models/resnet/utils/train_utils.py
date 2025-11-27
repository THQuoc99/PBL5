import torch
import os
import csv
from tqdm import tqdm
def load_checkpoint(model, optimizer, log_dir, device):
    """
    Tải checkpoint gần nhất nếu có.
    """
    checkpoint_path = os.path.join(log_dir, "last_model.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint["best_accuracy"]
        print(f"Checkpoint loaded: Epoch {start_epoch}, Best Accuracy: {best_accuracy:.2f}%")
        return start_epoch, best_accuracy
    return 0, 0.0  # Nếu không có checkpoint, bắt đầu từ epoch 0

def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device='cpu', log_dir="data/logs/", resume=True):
    """
    Huấn luyện mô hình ResNet-50+ và hỗ trợ tiếp tục từ checkpoint hoặc bắt đầu lại từ đầu.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training_log.txt")
    csv_file = os.path.join(log_dir, "training_log.csv")

    # Nếu resume = True, tải checkpoint
    if resume:
        start_epoch, best_accuracy = load_checkpoint(model, optimizer, log_dir, device)
    else:
        start_epoch = 0
        best_accuracy = 0.0
        # Xóa log cũ nếu huấn luyện từ đầu
        open(log_file, 'w').close()
        open(csv_file, 'w').close()
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Val Loss", "Val Accuracy"])

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        # Thanh tiến trình cho quá trình huấn luyện
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_images, batch_labels in progress_bar:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            optimizer.zero_grad()

            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
            
            progress_bar.set_postfix(loss=loss.item(), acc=100 * correct / total)
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total

        # ----- VALIDATION -----
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            # Thanh tiến trình cho quá trình validation
            progress_bar_val = tqdm(val_loader, desc="Validating", leave=False)
            for batch_images, batch_labels in progress_bar_val:
                batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
                
                progress_bar_val.set_postfix(loss=loss.item(), acc=100 * correct / total)
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total

        # ----- LOG KẾT QUẢ -----
        log_text = (f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        print(log_text)
        with open(log_file, "a") as f:
            f.write(log_text + "\n")
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, train_accuracy, val_loss, val_accuracy])

        # ----- LƯU CHECKPOINT -----
        last_check_point_path = os.path.join(log_dir, "last_check_point.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_accuracy": best_accuracy
        }, last_check_point_path)
        print(f"Checkpoint saved at epoch {epoch+1}")

        # Lưu model tốt nhất nếu accuracy cao hơn trước
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_path = os.path.join(log_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with accuracy: {val_accuracy:.2f}%")
