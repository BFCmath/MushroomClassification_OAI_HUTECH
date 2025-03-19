import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom ReLU with variable slopes
class VariableReLU(nn.Module):
    def __init__(self, slope):
        super(VariableReLU, self).__init__()
        self.slope = slope

    def forward(self, x):
        return torch.max(torch.zeros_like(x), x * self.slope)

class AdaptedLowNet(nn.Module):
    """
    Adaptation of LowNet for main.py framework.
    
    Modifications:
    - Supports RGB input (3 channels) instead of grayscale (1 channel)
    - Allows configurable number of output classes
    - Adds configurable dropout rate
    - Uses padding to maintain spatial dimensions compatibility
    """
    def __init__(self, num_classes=10, dropout_rate=0.3):
        super(AdaptedLowNet, self).__init__()
        
        # Low-Resolution Feature Extractor (3 Conv Layers)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = VariableReLU(slope=4)  # Slope = 4 for Layer 1
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu2 = VariableReLU(slope=2)  # Slope = 2 for Layer 2
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu3 = VariableReLU(slope=1)  # Slope = 1 for Layer 3
        self.dropout_conv = nn.Dropout(p=dropout_rate)  # Configurable dropout rate
        
        # Add pooling to make feature map size manageable
        # For 32x32 input, we need to reduce dimensions for efficiency
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4)  # 32x32 -> 8x8
        
        # Classifier (3 Fully-Connected Layers)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout_fc1 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout_fc2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)  # Configurable output classes
        
    def forward(self, x):
        # Feature Extractor
        x = self.relu1(self.conv1(x))    # Conv1 + ReLU(slope=4)
        x = self.relu2(self.conv2(x))    # Conv2 + ReLU(slope=2)
        x = self.relu3(self.conv3(x))    # Conv3 + ReLU(slope=1)
        x = self.pool(x)                 # Add pooling to reduce dimensions
        x = self.dropout_conv(x)         # Dropout
        
        # Flatten for fully-connected layers
        x = x.view(x.size(0), -1)  # Flatten: [batch_size, 128 * 8 * 8]
        
        # Classifier
        x = F.relu(self.fc1(x))        # FC1 + ReLU
        x = self.dropout_fc1(x)        # Dropout
        x = F.relu(self.fc2(x))        # FC2 + ReLU
        x = self.dropout_fc2(x)        # Dropout
        x = self.fc3(x)                # FC3 (no activation here, Softmax applied in loss)
        
        return x
