import torch
import torch.nn as nn
import torch.nn.functional as F

# Import DenseNet components
from scripts.cnn.DenseNet7x7 import DenseBlock, TransitionLayer

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=4, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = nn.AdaptiveAvgPool3d(1)(x.mean(dim=2, keepdim=True))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                in_max, _ = torch.max(x, dim=2)
                max_pool = nn.AdaptiveMaxPool3d(1)(in_max)
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5).expand_as(x)
        return (scale * x) + x, scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SpatialGate(nn.Module):
    def __init__(self, num_cluster, out_channels):
        super(SpatialGate, self).__init__()
        kernel_size = 3
        self.num_cluster = num_cluster
        self.out_channels = out_channels
        self.compress = ChannelPool()
        self.spatial = BasicConv(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1,
                                 padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale + x


class PrimaryCapsLayer(nn.Module):
    """
    Primary Capsule Layer that converts standard conv features to capsule format.
    This is the bridge between the CNN backbone and the capsule network.
    """
    def __init__(self, in_channels, out_caps, caps_dim):
        super(PrimaryCapsLayer, self).__init__()
        self.out_caps = out_caps
        self.caps_dim = caps_dim
        
        # Conv layers to convert features to capsule format
        # One conv per output capsule type
        self.conv_list = nn.ModuleList([
            nn.Conv2d(in_channels, caps_dim, kernel_size=3, padding=1)
            for _ in range(out_caps)
        ])
        
        # Add batch norms for stability
        self.bn_list = nn.ModuleList([
            nn.BatchNorm2d(caps_dim) 
            for _ in range(out_caps)
        ])
        
        # Activation
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        # Create capsules list
        caps_output = []
        
        # Process through each conv path to create different capsule types
        for i in range(self.out_caps):
            cap = self.relu(self.bn_list[i](self.conv_list[i](x)))
            caps_output.append(cap)
        
        # Return capsules as a list
        return caps_output


class HybridCapsLayer(nn.Module):
    """
    Hybrid Capsule Layer with combined attention mechanism.
    Simplified version of RoutingWithCombinedAttention for the DenseNet backbone approach.
    """
    def __init__(self, in_caps, in_dim, out_caps, out_dim, kernel_size, stride=1, r=4, if_bias=True):
        super(HybridCapsLayer, self).__init__()
        self.in_caps = in_caps
        self.in_dim = in_dim
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.r = r  # reduction ratio for attention
        self.if_bias = if_bias
        
        # Transformation convolutions
        self.trans_conv = nn.ModuleList([
            nn.Conv2d(in_dim, out_caps * out_dim, kernel_size=3, 
                     stride=stride, padding=1, bias=if_bias)
            for _ in range(in_caps)
        ])
        
        # Value convolutions for attention
        self.value_conv = nn.ModuleList([
            nn.Conv3d(in_channels=in_caps, out_channels=kernel_size * in_caps, 
                     kernel_size=(1, 1, 1), stride=(1, 1, 1))
            for _ in range(out_caps)
        ])
        
        # Channel attention module
        self.channel_att = ChannelGate(
            gate_channels=kernel_size, 
            reduction_ratio=r, 
            pool_types=['avg', 'max']
        )
        
        # Spatial attention module
        self.spatial_att = SpatialGate(
            num_cluster=kernel_size, 
            out_channels=out_caps
        )
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm([out_dim, 1, 1])
    
    def forward(self, caps):
        # Get dimensions
        batch_size = caps[0].shape[0]
        h, w = caps[0].shape[2], caps[0].shape[3]
        
        # Transform each input capsule
        votes = []
        for i in range(self.in_caps):
            votes.append(self.trans_conv[i](caps[i]))
        
        # Reshape votes for routing
        for i in range(len(votes)):
            votes[i] = votes[i].view(batch_size, self.out_caps, self.out_dim, h, w)
        
        # Group votes by output capsule
        votes_by_caps = []
        for i in range(self.out_caps):
            # Concatenate votes from all input capsules for this output capsule
            to_cat = [votes[j][:, i:(i+1), :, :, :] for j in range(self.in_caps)]
            votes_for_cap_i = torch.cat(to_cat, dim=1)
            votes_by_caps.append(votes_for_cap_i)
        
        # Apply attention and routing
        values = []
        spatial_values = []
        
        # Apply value convolution
        for i in range(self.out_caps):
            values.append(self.value_conv[i](votes_by_caps[i]))
        
        # Apply spatial attention
        for i in range(self.out_caps):
            spatial_values.append(
                self.spatial_att(values[i]).view(
                    batch_size, self.kernel_size, self.in_caps, self.out_dim, h, w
                )
            )
        
        # Apply channel attention and calculate output capsules
        output_caps = []
        for i in range(self.out_caps):
            # Apply channel attention and get weights
            weighted_votes, _ = self.channel_att(spatial_values[i])
            
            # Calculate statistics
            stds, means = torch.std_mean(weighted_votes, dim=1, unbiased=False)
            
            # Agreement calculation
            agreement = -torch.log(stds + 1e-8)  # add small epsilon for numerical stability
            
            # Softmax over clusters
            atts = F.softmax(agreement, dim=1)
            
            # Calculate output capsule as weighted sum
            output_cap = (atts * means).sum(dim=1)
            
            # Apply normalization
            output_caps.append(output_cap.squeeze())
        
        return output_caps


class DenseNetHybridCapsNet(nn.Module):
    """
    HybridCapsNet with DenseNet backbone for feature extraction.
    Uses a stronger ConvNet first and then a simplified capsule network.
    """
    def __init__(self, num_classes, input_img_dim=3, input_img_size=32, 
                 growth_rate=16, block_config=(3, 6, 8), dropout_rate=0.2,
                 C=4, K=10, D=32, if_bias=True, reduction_ratio=4):
        super(DenseNetHybridCapsNet, self).__init__()
        
        # DenseNet backbone (modified to keep spatial dimensions)
        self.features = self._make_densenet_features(input_img_dim, growth_rate, block_config)
        
        # Calculate features dimension after DenseNet backbone
        backbone_channels = 64 + sum([layers * growth_rate for layers in block_config])
        
        # Primary capsules layer - convert backbone features to capsules
        self.primary_caps = PrimaryCapsLayer(in_channels=backbone_channels, 
                                           out_caps=C, 
                                           caps_dim=D)
        
        # Just one capsule layer instead of multiple
        # This layer processes the output of primary_caps directly to class capsules
        # Instead of having multiple capsule layers as in the original HybridCapsNet
        self.caps_layer = HybridCapsLayer(in_caps=C, 
                                        in_dim=D,
                                        out_caps=num_classes,
                                        out_dim=D, 
                                        kernel_size=K,
                                        stride=1, 
                                        r=reduction_ratio, 
                                        if_bias=if_bias)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_densenet_features(self, input_channels, growth_rate, block_config):
        """Create DenseNet backbone feature extractor."""
        features = nn.Sequential()
        
        # Initial convolution without spatial reduction to maintain 32x32
        features.add_module('conv0', 
                         nn.Conv2d(input_channels, 64, kernel_size=7, 
                                  stride=1, padding=3, bias=False))
        features.add_module('norm0', nn.BatchNorm2d(64))
        features.add_module('relu0', nn.ReLU(inplace=False))
        
        # Current number of channels
        num_channels = 64
        
        # Add dense blocks and transition layers
        for i, num_layers in enumerate(block_config):
            # Add dense block
            block = DenseBlock(num_channels, growth_rate, num_layers)
            features.add_module(f'denseblock{i+1}', block)
            
            # Update number of channels
            num_channels += num_layers * growth_rate
            
            # Add transition layer except after the last block
            if i != len(block_config) - 1:
                # No reduction in channel count to maintain feature richness
                trans = TransitionLayer(num_channels, num_channels)
                features.add_module(f'transition{i+1}', trans)
        
        # Final batch norm and ReLU
        features.add_module('norm_final', nn.BatchNorm2d(num_channels))
        features.add_module('relu_final', nn.ReLU(inplace=False))
        
        return features
    
    def _initialize_weights(self):
        """Initialize weights for the DenseNet backbone."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features using DenseNet backbone
        x = self.features(x)
        
        # Convert to primary capsules
        x = self.primary_caps(x)
        
        # Pass through reduced capsule layer (just one instead of multiple)
        x = self.caps_layer(x)
        
        # x is now a list of capsules, one for each class
        # For each class capsule, we need to calculate its length as the class logit
        # This is the same approach used in HybridCapsNet to convert capsule vectors to class scores
        logits = []
        for capsule in x:
            # Calculate squared norm for numerical stability
            # For 4D tensors (batch_size, channels, height, width)
            if len(capsule.shape) == 4:
                # Average pool to reduce spatial dimensions
                capsule = F.adaptive_avg_pool2d(capsule, 1)
                capsule = capsule.view(capsule.size(0), -1)
            
            # For already flattened capsules
            squared_norm = torch.sum(capsule * capsule, dim=-1)
            logits.append(squared_norm)
        
        # Stack the logits into a single tensor [batch_size, num_classes]
        logits = torch.stack(logits, dim=1)
        
        return logits


# Factory function for easy creation
def create_densenet_hybridcapsnet(num_classes=10, input_img_dim=3, input_img_size=32,
                               growth_rate=16, block_config=(3, 6, 8), dropout_rate=0.2,
                               C=4, K=10, D=32, if_bias=True, reduction_ratio=4):
    """
    Create a DenseNetHybridCapsNet with the specified parameters.
    """
    return DenseNetHybridCapsNet(
        num_classes=num_classes,
        input_img_dim=input_img_dim,
        input_img_size=input_img_size,
        growth_rate=growth_rate,
        block_config=block_config,
        dropout_rate=dropout_rate,
        C=C,
        K=K,
        D=D,
        if_bias=if_bias,
        reduction_ratio=reduction_ratio
    )
