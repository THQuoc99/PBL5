# models/resnet50_pytorch.py
import torch
import torch.nn as nn
import numpy as np

class ResidualBlock(nn.Module):
    """Residual Block với 3 tầng conv và skip connection."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels//4)
        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels//4)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out)) 
        out += self.shortcut(x)
        return self.relu(out)

class ResNet50Plus(nn.Module):
    """ResNet-50+ để nhận diện 4 cảm xúc từ ảnh 128x128."""
    def __init__(self, input_size=(48, 48, 1), num_classes=4):
        super(ResNet50Plus, self).__init__()
        self.conv1 = nn.Conv2d(input_size[-1], 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Stages
        self.layer1 = self._make_layer(64, 64, 3)  # 3 blocks
        self.layer2 = self._make_layer(64, 128, 4, stride=2)  # 4 blocks
        self.layer3 = self._make_layer(128, 256, 6, stride=2)  # 6 blocks
        self.layer4 = self._make_layer(256, 512, 3, stride=2)  # 3 blocks
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def predict(self, images):
        """Dự đoán cảm xúc trên ảnh test."""
        self.eval()
        with torch.no_grad():
            # Chuyển thành tensor nếu là numpy array
            if isinstance(images, np.ndarray):
                images = torch.FloatTensor(images)

            # Xử lý dữ liệu đầu vào
            if images.ndim == 4:
                if images.shape[1] == 1:  # [N, 1, H, W] (đã đúng định dạng PyTorch)
                    pass
                elif images.shape[3] == 1:  # [N, H, W, 1] (từ resize_face)
                    images = images.permute(0, 3, 1, 2)  # [N, 1, H, W]
                else:  # Các định dạng khác như [N, 128, 1, 128] hoặc [N, 128, 128, 1]
                    if images.shape[2] == 1 and images.shape[3] == 128:
                        images = images.permute(0, 3, 1, 2)
                    elif images.shape[1] == 128 and images.shape[3] == 128:
                        images = images.permute(0, 3, 1, 2)
            else:  # Single image [H, W, 1]
                images = images.permute(2, 0, 1).unsqueeze(0)  # [1, 1, H, W]

            # Chuyển sang device của mô hình
            device = next(self.parameters()).device
            images = images.to(device)

            # Dự đoán
            outputs = self(images)
            _, preds = torch.max(outputs, 1)
            return preds.cpu().numpy()

    def evaluate(self, test_images, test_labels):
        """Đánh giá mô hình trên tập test."""
        self.eval()
        with torch.no_grad():
            preds = self.predict(test_images)
            accuracy = np.mean(preds == test_labels)
            conf_matrix = np.zeros((self.fc.out_features, self.fc.out_features), dtype=int)
            for true, pred in zip(test_labels, preds):
                conf_matrix[true, pred] += 1
            return accuracy, conf_matrix