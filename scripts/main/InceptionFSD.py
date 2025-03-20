import torch
import torch.nn as nn
# ### Inception Module and InceptionFSD Model Implementation ###

# Add new FSDDownsample module
class FSDDownsample(nn.Module):
    """
    Feature Scale Detection (FSD) Downsampling Module.
    Performs feature-aware downsampling by learning the most important features
    to preserve while reducing spatial dimensions.
    """
    def __init__(self, in_channels, out_channels):
        super(FSDDownsample, self).__init__()
        
        # Parallel pathways with different receptive fields
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Process through parallel branches that preserve different aspects of features
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        branch3_output = self.branch3(x)
        branch4_output = self.branch4(x)
        
        # Concatenate to combine all feature aspects
        return torch.cat([branch1_output, branch2_output, branch3_output, branch4_output], dim=1)

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel-wise feature recalibration.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class InceptionModule(nn.Module):
    """
    Inception module with parallel pathways for multi-scale feature extraction.
    Enhanced with additional channels and optional SE attention.
    """
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj, use_se=True):
        super(InceptionModule, self).__init__()
        
        # 1x1 convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 conv -> 3x3 conv branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 conv -> 5x5 conv branch (implemented as two 3x3 convs for efficiency)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5red, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )
        
        # Max pooling -> 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )
        
        # Optional squeeze-and-excitation attention
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(ch1x1 + ch3x3 + ch5x5 + pool_proj)
    
    def forward(self, x):
        # Process input through each branch
        branch1_output = self.branch1(x)
        branch2_output = self.branch2(x)
        branch3_output = self.branch3(x)
        branch4_output = self.branch4(x)
        
        # Concatenate outputs along the channel dimension
        outputs = [branch1_output, branch2_output, branch3_output, branch4_output]
        combined = torch.cat(outputs, 1)
        
        # Apply channel attention if enabled
        if self.use_se:
            combined = self.se(combined)
            
        return combined

class InceptionFSD(nn.Module):
    """
    Enhanced Inception-based Feature Scale Detector with increased capacity.
    
    This model:
    1. Uses deeper Inception modules with more channels
    2. Adds squeeze-and-excitation attention for feature refinement
    3. Uses advanced multi-scale feature aggregation
    4. Enhanced feature fusion network
    """
    def __init__(self, num_classes=10, dropout_rate=0.4):
        super(InceptionFSD, self).__init__()
        
        # Initial feature extraction - maintain 32 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        # First inception block - maintain spatial dimensions for small images
        self.inception1 = InceptionModule(
            in_channels=32,
            ch1x1=16,
            ch3x3red=24,
            ch3x3=32,
            ch5x5red=8,
            ch5x5=16,
            pool_proj=16,
            use_se=True
        )  # Output: 80 channels (16+32+16+16)
        
        # Additional inception block at the same scale for more capacity
        self.inception1b = InceptionModule(
            in_channels=80,
            ch1x1=24,
            ch3x3red=32,
            ch3x3=48,
            ch5x5red=12,
            ch5x5=24,
            pool_proj=20,
            use_se=True
        )  # Output: 116 channels (24+48+24+20)
        
        # Reduction block 1 - use FSD downsampling
        self.reduction1 = FSDDownsample(116, 144)  # 116 -> 144 channels, spatial dim: 32x32 -> 16x16
        
        # Second inception block
        self.inception2 = InceptionModule(
            in_channels=144,
            ch1x1=40,
            ch3x3red=48,
            ch3x3=64,
            ch5x5red=16,
            ch5x5=32,
            pool_proj=32,
            use_se=True
        )  # Output: 168 channels (40+64+32+32)
        
        # Reduction block 2 - use FSD downsampling
        self.reduction2 = FSDDownsample(168, 192)  # 168 -> 192 channels, spatial dim: 16x16 -> 8x8
        
        # Third inception block
        self.inception3 = InceptionModule(
            in_channels=192,
            ch1x1=64,
            ch3x3red=64,
            ch3x3=96,
            ch5x5red=24,
            ch5x5=48,
            pool_proj=48,
            use_se=True
        )  # Output: 256 channels (64+96+48+48)
        
        # Multi-scale feature aggregation with additional multi-resolution pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global context
        self.mid_pool = nn.AdaptiveAvgPool2d((2, 2))     # Mid-level context
        self.local_pool = nn.AdaptiveAvgPool2d((4, 4))   # Local context
        
        # Calculate flattened feature dimensions
        global_features = 256  # From inception3
        mid_features = 256 * 4  # 2x2 spatial dimension
        local_features = 168 * 16  # 4x4 spatial dimension
        
        # Enhanced feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(global_features + mid_features + local_features, 768),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(768, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(384, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/2)  # Reduced dropout in final fusion layer
        )
        
        # Classifier
        self.classifier = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:  # Add check for None bias
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:  # Add check for None bias
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial feature extraction
        x = self.conv1(x)
        
        # Feature extraction at different scales
        inception1_out = self.inception1(x)
        inception1b_out = self.inception1b(inception1_out)
        
        x = self.reduction1(inception1b_out)
        inception2_out = self.inception2(x)
        
        x = self.reduction2(inception2_out)
        inception3_out = self.inception3(x)
        
        # Multi-scale feature aggregation
        global_features = self.global_pool(inception3_out)
        mid_features = self.mid_pool(inception3_out)
        local_features = self.local_pool(inception2_out)
        
        # Flatten features
        global_features = torch.flatten(global_features, 1)
        mid_features = torch.flatten(mid_features, 1)
        local_features = torch.flatten(local_features, 1)
        
        # Concatenate multi-scale features
        concat_features = torch.cat([global_features, mid_features, local_features], dim=1)
        
        # Feature fusion
        fused_features = self.fusion(concat_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits