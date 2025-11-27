import cv2
import torch
import torch.nn as nn
import numpy as np

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block để tái cân bằng trọng số của các channel."""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    """Spatial Attention Module để tập trung vào các vùng quan trọng của khuôn mặt."""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(x_cat))
        return x * attention_map

class ResidualBlock(nn.Module):
    """Residual Block với SE block và depthwise separable convolution."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.expansion = 4
        mid_channels = out_channels // self.expansion

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 
                              kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels,
                              kernel_size=3, stride=stride, padding=1, 
                              groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 
                              kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.se = SEBlock(out_channels, reduction=8)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

class ResNet50Plus(nn.Module):
    """ResNet50+ tối ưu cho ảnh grayscale 128x128."""
    def __init__(self, input_size=(128, 128, 1), num_classes=7):  # Thay đổi num_classes thành 7
        super(ResNet50Plus, self).__init__()
        
        self.conv1 = nn.Conv2d(input_size[-1], 32, kernel_size=7,
                              stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(32, 128, 3, stride=2)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=1)
        self.layer4 = self._make_layer(512, 1024, 3, stride=2)
        
        self.attention = SpatialAttention(kernel_size=5)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(1024, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.attention(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

    def predict(self, images):
        """Dự đoán cảm xúc trên ảnh test."""
        self.eval()
        with torch.no_grad():
            if isinstance(images, np.ndarray):
                images = torch.FloatTensor(images)
            elif isinstance(images, torch.Tensor):
                pass
            else:
                raise ValueError("Input must be numpy array or torch tensor")

            if images.ndim == 4:
                if images.shape[1] == 1:  # [N, 1, H, W]
                    pass
                elif images.shape[3] == 1:  # [N, H, W, 1]
                    images = images.permute(0, 3, 1, 2)
            else:  # Single image [H, W, 1]
                images = images.permute(2, 0, 1).unsqueeze(0)

            device = next(self.parameters()).device
            images = images.to(device)
            outputs = self(images)
            _, preds = torch.max(outputs, 1)
            return preds.cpu().numpy()

    def evaluate(self, test_images, test_labels, batch_size=32):
        """Đánh giá mô hình trên tập test với cơ chế batching."""
        self.eval()
        with torch.no_grad():
            # Chuyển test_labels sang NumPy nếu là tensor
            if isinstance(test_labels, torch.Tensor):
                test_labels = test_labels.cpu().numpy()
            
            # Chuẩn bị dữ liệu cho batching
            dataset = torch.utils.data.TensorDataset(test_images, torch.LongTensor(test_labels))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            
            preds = []
            for batch_images, _ in dataloader:
                batch_preds = self.predict(batch_images)
                preds.extend(batch_preds)
            
            preds = np.array(preds)
            
            # Kiểm tra phạm vi nhãn
            if not np.all((test_labels >= 0) & (test_labels < self.fc2.out_features)):
                raise ValueError(f"Nhãn trong test_labels phải nằm trong phạm vi [0, {self.fc2.out_features-1}]")
            
            # Tính độ chính xác
            accuracy = np.mean(preds == test_labels)
            
            # Tính ma trận nhầm lẫn
            conf_matrix = np.zeros((self.fc2.out_features, self.fc2.out_features), dtype=int)
            for true, pred in zip(test_labels, preds):
                conf_matrix[true, pred] += 1
            
            return accuracy, conf_matrix