import torch
import torch.nn as nn

# ### Model Definition ###
class ResidualBlock(nn.Module):
    """Basic residual block with two 7x7 convolutions and a skip connection."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # First convolution and batch norm
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, 
                              stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution and batch norm
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=7, 
                              stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (if dimensions change)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Store input for skip connection
        identity = x
        
        # First convolution block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second convolution block
        out = self.conv2(out)  # BUG FIX: was using x instead of out!
        out = self.bn2(out)
        
        # Skip connection
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

class SmallResNet(nn.Module):
    """Custom ResNet architecture for small 32x32 images with 7x7 kernels."""
    def __init__(self, num_classes=10, dropout_rate=0.2):
        super(SmallResNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks with increasing channels
        self.res_block1 = ResidualBlock(32, 64, stride=2)  # 16x16
        self.res_block2 = ResidualBlock(64, 64, stride=1)
        self.res_block3 = ResidualBlock(64, 128, stride=2)  # 8x8
        self.res_block4 = ResidualBlock(128, 128, stride=1)
        self.res_block5 = ResidualBlock(128, 256, stride=2)  # 4x4
        self.res_block6 = ResidualBlock(256, 256, stride=1)
        
        # Global average pooling and fully connected layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(256, 512)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)
        
        # Global average pooling
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x
