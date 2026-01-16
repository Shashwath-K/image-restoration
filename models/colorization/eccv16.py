
import torch
import torch.nn as nn
import torch.nn.functional as F

# Standalone ResNet BasicBlock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ColorNet(nn.Module):
    """
    ResNet-based U-Net for Image Colorization (Standalone Implementation).
    Backbone: ResNet18-like (modified for 1-channel input)
    Input: L channel (1, H, W)
    Output: ab channels (2, H, W)
    """
    def __init__(self, pretrained=False):
        super(ColorNet, self).__init__()
        
        # 1. Encoder (ResNet18-like)
        self.in_planes = 64
        
        self.enc1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        # 2. Decoder with Skip Connections
        self.up4 = self._up_block(512, 256)
        self.up3 = self._up_block(256 + 256, 128)
        self.up2 = self._up_block(128 + 128, 64)
        self.up1 = self._up_block(64 + 64, 64)
        self.up0 = self._up_block(64 + 64, 32)
        
        self.final = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )
        
    def _align_size(self, x, ref):
        if x.size()[2:] != ref.size()[2:]:
            x = F.interpolate(x, size=ref.size()[2:], mode='bilinear', align_corners=True)
        return x

    def forward(self, x):
        # Encoder
        x0 = self.enc1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0) # (64, H/2, W/2)
        
        x_pool = self.maxpool(x0) # (64, H/4, W/4)
        
        x1 = self.layer1(x_pool) # (64, H/4, W/4)
        x2 = self.layer2(x1) # (128, H/8, W/8)
        x3 = self.layer3(x2) # (256, H/16, W/16)
        x4 = self.layer4(x3) # (512, H/32, W/32)
        
        # Decoder
        d4 = self.up4(x4) # -> 256
        d4 = self._align_size(d4, x3)
        d4 = torch.cat([d4, x3], dim=1)
        
        d3 = self.up3(d4) # -> 128
        d3 = self._align_size(d3, x2)
        d3 = torch.cat([d3, x2], dim=1)
        
        d2 = self.up2(d3) # -> 64
        d2 = self._align_size(d2, x1)
        d2 = torch.cat([d2, x1], dim=1)
        
        d1 = self.up1(d2) # -> 64
        d1 = self._align_size(d1, x0)
        d1 = torch.cat([d1, x0], dim=1)
        
        out = self.up0(d1)
        
        # Final safety for input size match
        if out.size()[2:] != x.size()[2:]:
             out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=True)
            
        return self.final(out)
