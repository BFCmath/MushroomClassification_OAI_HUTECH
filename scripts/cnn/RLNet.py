import torch
import torch.nn as nn
import torch.nn.functional as F

class MKBlock(nn.Module):
    """
    PyTorch implementation of the Multi-Kernel (MK) Block from RL-Net.
    Adjusted for 32x32x3 input images with proper channel handling.
    """
    def __init__(self, in_channels):
        super(MKBlock, self).__init__()

        # Layer 1: Parallel Convolutions with different kernel sizes
        # Path 1 (3x3)
        self.conv1_1_3x3 = nn.Conv2d(in_channels=in_channels, out_channels=48, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(48)

        # Path 2 (5x5)
        self.conv1_2_5x5 = nn.Conv2d(in_channels=in_channels, out_channels=24, kernel_size=5, padding=2)
        self.bn1_2 = nn.BatchNorm2d(24)

        # Path 3 (7x7)
        self.conv1_3_7x7 = nn.Conv2d(in_channels=in_channels, out_channels=12, kernel_size=7, padding=3)
        self.bn1_3 = nn.BatchNorm2d(12)

        # Activation
        self.relu = nn.ReLU(inplace=True)

        # Layer 2: Internal Connections and Convolutions
        # Combine 3x3 and 5x5 paths -> Conv 5x5 -> Conv 3x3
        self.conv2_1_5x5 = nn.Conv2d(in_channels=48 + 24, out_channels=36, kernel_size=5, padding=2)
        self.bn2_1 = nn.BatchNorm2d(36)
        self.conv3_1_3x3 = nn.Conv2d(in_channels=36, out_channels=72, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(72)

        # Combine 5x5 and 7x7 paths -> Conv 7x7
        self.conv2_2_7x7 = nn.Conv2d(in_channels=24 + 12, out_channels=18, kernel_size=7, padding=3)
        self.bn2_2 = nn.BatchNorm2d(18)

        # Layer 3: Final Aggregation within block
        # Concatenate specific paths: path1_1 (48) + path2_1 (36) + path3_1 (72) + path2_2 (18) = 174 channels
        final_concat_channels = 48 + 36 + 72 + 18
        self.conv_bottleneck_1x1 = nn.Conv2d(in_channels=final_concat_channels, out_channels=24, kernel_size=1)
        self.bn_bottleneck = nn.BatchNorm2d(24)

        # Max Pooling
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Store output channels for connecting blocks
        self.output_channels = 24

    def forward(self, x):
        # Layer 1 Forward Pass
        path1_1 = self.relu(self.bn1_1(self.conv1_1_3x3(x)))
        path1_2 = self.relu(self.bn1_2(self.conv1_2_5x5(x)))
        path1_3 = self.relu(self.bn1_3(self.conv1_3_7x7(x)))

        # Layer 2 Forward Pass
        # Path originating from 3x3 & 5x5
        concat1 = torch.cat((path1_1, path1_2), dim=1)  # Concatenate along channel dim
        path2_1 = self.relu(self.bn2_1(self.conv2_1_5x5(concat1)))
        path3_1 = self.relu(self.bn3_1(self.conv3_1_3x3(path2_1)))

        # Path originating from 5x5 & 7x7
        concat2 = torch.cat((path1_2, path1_3), dim=1)
        path2_2 = self.relu(self.bn2_2(self.conv2_2_7x7(concat2)))

        # Layer 3 Final Aggregation
        final_concat = torch.cat((path1_1, path2_1, path3_1, path2_2), dim=1)

        # Bottleneck
        bottleneck = self.relu(self.bn_bottleneck(self.conv_bottleneck_1x1(final_concat)))

        # Pooling
        output = self.maxpool(bottleneck)

        return output

class RLNet(nn.Module):
    """
    PyTorch implementation of the adapted RL-Net model.
    Modified for 32x32x3 input images and 4 output classes.
    """
    def __init__(self, num_classes=4, input_channels=3, dropout_rate=0.2):
        super(RLNet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        # Three MK blocks in sequence
        self.mk_block1 = MKBlock(in_channels=input_channels)
        self.mk_block2 = MKBlock(in_channels=self.mk_block1.output_channels)
        self.mk_block3 = MKBlock(in_channels=self.mk_block2.output_channels)

        self.final_bn = nn.BatchNorm2d(self.mk_block3.output_channels)
        self.flatten = nn.Flatten()

        # Calculate the flattened size
        # Initial image size: 32x32
        # After 3 blocks with 2x2 pooling each, size becomes 32/(2^3) = 4x4
        final_size = 4  # 32 // (2*2*2)
        flattened_size = self.mk_block3.output_channels * final_size * final_size

        # Fully Connected Layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(flattened_size, 256)
        self.relu_fc1 = nn.ReLU(inplace=True)

        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.relu_fc2 = nn.ReLU(inplace=True)

        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, 128)
        self.relu_fc3 = nn.ReLU(inplace=True)

        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(128, 64)
        self.relu_fc4 = nn.ReLU(inplace=True)

        # Output Layer
        self.fc_out = nn.Linear(64, num_classes)
        
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
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Multi-Kernel blocks
        x = self.mk_block1(x)
        x = self.mk_block2(x)
        x = self.mk_block3(x)

        # Final processing
        x = self.final_bn(x)
        x = self.flatten(x)

        # Fully connected layers
        x = self.dropout1(x)
        x = self.relu_fc1(self.fc1(x))

        x = self.dropout2(x)
        x = self.relu_fc2(self.fc2(x))

        x = self.dropout3(x)
        x = self.relu_fc3(self.fc3(x))

        x = self.dropout4(x)
        x = self.relu_fc4(self.fc4(x))

        x = self.fc_out(x)  # Logits output
        return x

# Factory function to create the model with default parameters
def create_rlnet(num_classes=4, input_channels=3, dropout_rate=0.2):
    return RLNet(num_classes=num_classes, input_channels=input_channels, dropout_rate=dropout_rate)
