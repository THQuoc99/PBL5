import torch
import torch.nn as nn
from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode

class Conv(nn.Module):
    """Convolutional layer with SiLU activation and optional BatchNorm."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2  # Same padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """Residual bottleneck layer with CSP."""
    def __init__(self, in_channels, out_channels,width_multiple = 1, shortcut=True):
        super().__init__()
        hidden_channels = int(width_multiple * in_channels)
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1, 0)
        self.conv2 = Conv(hidden_channels, out_channels, 3, 1, 1)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return x + out if self.shortcut else out
    
class SPPF(nn.Module):
    """SPPF layer (Spatial Pyramid Pooling - Fast)"""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1, 0)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.conv2 = Conv(hidden_channels * 4, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        p1 = self.pool(x)
        p2 = self.pool(p1)
        p3 = self.pool(p2)
        return self.conv2(torch.cat([x, p1, p2, p3], dim=1))
    
class C3(nn.Module):
    def __init__(self,in_channels,out_channels,width_multiple=1,depth=1,backbone=True):
        super().__init__()
        hidden_channels = int(in_channels * width_multiple)
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1, 0)
        self.conv2 = Conv(in_channels, hidden_channels, 1, 1, 0)
        if backbone:
            self.seq = nn.Sequential(*[Bottleneck(hidden_channels, hidden_channels) for _ in range(depth)])
        else:
            self.seq = nn.Sequential(*[
                nn.Sequential(
                    Conv(hidden_channels, hidden_channels,1,1,0),
                    Conv(hidden_channels, hidden_channels,3,1,1)
                )   for _ in range(depth)
            ])

        self.conv3 = Conv(hidden_channels * 2, out_channels, 1, 1, 0)

    def forward(self, x):
        x = torch.cat([self.seq(self.conv1(x)), self.conv2(x)], dim = 1)
        return self.conv3(x) 


class Heads(nn.Module):
    def __init__(self,num_classes = 80, anchors = (), ch = ()):
        super().__init__()
        self.nc = num_classes
        self.nl = len(anchors)
        self.naxs = len(anchors[0])

        self.stride = [8, 16, 32]

        anchors_ = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', anchors_)

        self.out_convs = nn.ModuleList()
        for in_channels in ch:
            self.out_convs += [
                nn.Conv2d(in_channels,(5+self.nc) * self.naxs, 1)
            ]

    def forward(self, x):
        for i in range(self.nl):
            x[i] = self.out_convs[i](x[i])

            bs, _, grid_y, grid_x = x[i].shape
            x[i] = x[i].view(bs, self.naxs, (5+self.nc), grid_y, grid_x).permute(0, 1, 3, 4, 2).contiguous()

        return x
    
class YOLOV5m(nn.Module):
    def __init__(self, first_out ,nc=80, anchors = (), ch = (), inference = False):
        super().__init__()
        self.inference = inference
        self.backbone = nn.ModuleList()
        self.backbone += [
            Conv(3, first_out, 6, 2, 2),
            Conv(first_out, first_out*2, 3, 2, 1),
            C3(first_out*2, first_out*2,0.5,2),
            Conv(first_out*2, first_out*4, 3, 2, 1),
            C3(first_out*4, first_out*4,0.5,4),
            Conv(first_out*4, first_out*8, 3, 2, 1),
            C3(first_out*8, first_out*8,0.5,6),
            Conv(first_out*8, first_out*16, 3, 2, 1),
            C3(first_out*16, first_out*16,0.5,2),
            SPPF(first_out*16, first_out*16)
        ]
        self.neck = nn.ModuleList()
        self.neck += [
            Conv(first_out*16,first_out*8, 1, 1, 0),
            C3(first_out*16, first_out*8, 0.25, 2, False),
            Conv(first_out*8, first_out*4, 1, 1, 0),
            C3(first_out*8, first_out*4, 0.25, 2, False),
            Conv(first_out*4, first_out*4, 3, 2, 1),
            C3(first_out*8, first_out*8, 0.5, 2, False),
            Conv(first_out*8, first_out*8, 3, 2, 1),
            C3(first_out*16, first_out*16, 0.5, 2, False)
        ]
        self.head = Heads(nc, anchors, ch)

    def forward(self,x):
        assert x.shape[2] % 32 == 0 and x.shape[3] % 32 == 0, "Width and Height aren't divisible by 32!"
        backbone_connection = []
        neck_connection = []
        outputs = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [4, 6]:
                backbone_connection.append(x)

        for i, layer in enumerate(self.neck):
            if i in [0, 2]: 
                x = layer(x)
                neck_connection.append(x)
                x = Resize([x.shape[2]*2,x.shape[3]*2],interpolation=InterpolationMode.NEAREST)(x)
                x = torch.cat([x, backbone_connection.pop(-1)], dim=1)

            elif i in [4,6]:
                x = layer(x)
                x = torch.cat([x, neck_connection.pop(-1)], dim=1)
            elif isinstance(layer,C3) and i > 2:
                x = layer(x)
                outputs.append(x)

            else:
                x = layer(x)

        return self.head(outputs)
