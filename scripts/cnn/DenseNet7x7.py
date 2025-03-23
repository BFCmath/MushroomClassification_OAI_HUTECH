import torch
import torch.nn as nn

# ### DenseNet with 7x7 Kernels Implementation ###
class DenseLayer(nn.Module):
    """Single layer in a DenseNet block with larger 7x7 kernels."""
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        # BN-ReLU-Conv structure
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=False)  # Changed from inplace=True
        # Using 7x7 kernels instead of standard 3x3
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=7, 
                             stride=1, padding=3, bias=False)
        
    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)  # This uses the non-inplace ReLU
        out = self.conv(out)
        return torch.cat([x, out], 1)  # Dense connection


class DenseBlock(nn.Module):
    """Block containing multiple densely connected layers with 7x7 kernels."""
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    """Transition layer between DenseBlocks without spatial reduction for small images."""
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=False)  # Changed from inplace=True
        # Use 1x1 conv to reduce channels but keep spatial dimensions
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)  # This uses the non-inplace ReLU
        x = self.conv(x)
        return x


class DenseNet7x7(nn.Module):
    """DenseNet implementation with 7x7 kernels and no spatial reduction for small 32x32 images."""
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), 
                 num_classes=10, dropout_rate=0.2):
        super(DenseNet7x7, self).__init__()
        
        # Initial convolution without spatial reduction
        self.features = nn.Sequential(
            # Use stride=1 instead of stride=2 to maintain spatial dimensions
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),  # Changed from inplace=True
        )
        
        # Current number of channels
        num_channels = 64
        
        # Add dense blocks and transition layers
        for i, num_layers in enumerate(block_config):
            # Add dense block
            block = DenseBlock(num_channels, growth_rate, num_layers)
            self.features.add_module(f'denseblock{i+1}', block)
            
            # Update number of channels after dense block
            num_channels += num_layers * growth_rate
            
            # Add transition layer except after the last block
            if i != len(block_config) - 1:
                # Reduce number of channels by half in transition
                trans = TransitionLayer(num_channels, num_channels // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_channels = num_channels // 2
        
        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm2d(num_channels))
        self.features.add_module('relu_final', nn.ReLU(inplace=False))  # Changed from inplace=True
        
        # Keep the global average pooling - still needed for classification
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(num_channels, num_classes)
        
        # Initialize weights - same as before
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Added proper weight initialization
                if m.bias is not None:  # Check if bias exists before initializing
                    nn.init.constant_(m.bias, 0)
    
    # Forward method remains the same
    def forward(self, x):
        features = self.features(x)
        out = self.avg_pool(features)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.classifier(out)
        return out