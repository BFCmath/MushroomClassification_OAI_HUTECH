import torch
import torch.nn as nn
import torch.nn.functional as F
from DualBranch import SpaceToDepthConv  # Import the existing SPD implementation

class SPDBasicBlock(nn.Module):
    """
    Basic ResNet block that uses Space-to-Depth for downsampling instead of strided convolutions.
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, downsample=False):
        super(SPDBasicBlock, self).__init__()
        
        # First convolution
        if downsample:
            # Replace stride-2 conv with Space-to-Depth followed by 1x1 conv
            self.conv1 = SpaceToDepthConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                block_size=2,  # Equivalent to stride=2
                padding=1
            )
        else:
            # Regular convolution for non-downsampling blocks
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            
        # Second convolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection/shortcut
        if downsample or in_channels != out_channels:
            if downsample:
                # Use Space-to-Depth for shortcut when downsampling
                self.shortcut = nn.Sequential(
                    nn.PixelUnshuffle(2),  # This is essentially a space-to-depth operation
                    nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                # 1x1 conv for channel matching without downsampling
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        else:
            self.shortcut = nn.Identity()
            
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class SPDBottleneck(nn.Module):
    """
    Bottleneck ResNet block that uses Space-to-Depth for downsampling instead of strided convolutions.
    """
    expansion = 4
    
    def __init__(self, in_channels, out_channels, downsample=False):
        super(SPDBottleneck, self).__init__()
        bottleneck_channels = out_channels // self.expansion
        
        # First 1x1 bottleneck convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True)
        )
        
        # Middle 3x3 convolution
        if downsample:
            # Replace stride-2 conv with Space-to-Depth followed by 3x3 conv
            self.conv2 = SpaceToDepthConv(
                in_channels=bottleneck_channels,
                out_channels=bottleneck_channels,
                kernel_size=3,
                block_size=2,  # Equivalent to stride=2
                padding=1
            )
        else:
            # Regular 3x3 convolution for non-downsampling blocks
            self.conv2 = nn.Sequential(
                nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(bottleneck_channels),
                nn.ReLU(inplace=True)
            )
        
        # Final 1x1 expansion convolution
        self.conv3 = nn.Sequential(
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Skip connection/shortcut
        if downsample or in_channels != out_channels:
            if downsample:
                # Use Space-to-Depth for shortcut when downsampling
                self.shortcut = nn.Sequential(
                    nn.PixelUnshuffle(2),  # Space-to-depth operation
                    nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                # 1x1 conv for channel matching without downsampling
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        else:
            self.shortcut = nn.Identity()
            
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class SPDResNet(nn.Module):
    """
    ResNet architecture that uses Space-to-Depth operations instead of strided convolutions
    for all downsampling operations, preserving spatial information.
    """
    def __init__(self, block, layers, num_classes=10, dropout_rate=0.2):
        super(SPDResNet, self).__init__()
        self.in_channels = 64
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Replace initial pooling with SPD for more information preservation
        self.spd_initial = SpaceToDepthConv(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            block_size=2,  # Equivalent to stride=2 pooling
            padding=1
        )
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], downsample=True)
        self.layer3 = self._make_layer(block, 256, layers[2], downsample=True)
        self.layer4 = self._make_layer(block, 512, layers[3], downsample=True)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, block, out_channels, blocks, downsample=False):
        layers = []
        # First block may perform downsampling
        layers.append(block(self.in_channels, out_channels, downsample))
        
        # Update in_channels for subsequent blocks
        self.in_channels = out_channels
        
        # Add remaining blocks (no downsampling)
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial processing
        x = self.initial(x)
        x = self.spd_initial(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

# Pre-defined model configurations
def spdresnet18(num_classes=10, dropout_rate=0.2):
    return SPDResNet(SPDBasicBlock, [2, 2, 2, 2], num_classes=num_classes, dropout_rate=dropout_rate)