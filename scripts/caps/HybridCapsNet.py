import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.caps.DenseNetHybridCapsNet import DenseNetHybridCapsNet

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=4, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
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


class RoutingWithCombinedAttention(nn.Module):
    def __init__(self, C_in, C_out, K, D_in, D_out, B, out_S, stride, reduction_ratio=4):
        super(RoutingWithCombinedAttention, self).__init__()
        self.K = K
        self.C_in = C_in
        self.C_out = C_out
        self.D_in = D_in
        self.D_out = D_out
        self.conv_trans = nn.ModuleList()
        self.value_conv = nn.ModuleList()
        for i in range(self.C_in):
            self.conv_trans.append(nn.Conv2d(self.D_in, self.C_out * self.D_out, 3,
                                             stride=stride, padding=1, bias=B))
        for i in range(self.C_out):
            self.value_conv.append(
                nn.Conv3d(in_channels=self.C_in, out_channels=self.K * self.C_in, kernel_size=(1, 1, 1),
                          stride=(1, 1, 1)))
        self.channel_att = ChannelGate(gate_channels=self.K, reduction_ratio=reduction_ratio, pool_types=['avg', 'max'])
        self.spatial_att = SpatialGate(num_cluster=self.K, out_channels=self.C_out)
        self.acti = nn.LayerNorm([self.D_out, out_S, out_S])

    def cluster_routing(self, votes):
        batch_size, _, h, w = votes[0].shape
        for i in range(len(votes)):
            votes[i] = votes[i].view(batch_size, self.C_out, self.D_out, h, w)
        votes_for_next_layer = []
        for i in range(self.C_out):
            to_cat = [votes[j][:, i:(i + 1), :, :, :] for j in range(self.C_in)]
            votes_for_channel_i = torch.cat(to_cat, dim=1)
            votes_for_next_layer.append(votes_for_channel_i)

        values, channel_values, spatial_values = [], [], []
        for i in range(self.C_out):
            values.append(self.value_conv[i](votes_for_next_layer[i]))
        for i in range(self.C_out):
            spatial_values.append(self.spatial_att(values[i]).
                                  view(batch_size, self.K, self.C_in, self.D_out, h, w))

        caps_of_next_layer = []
        for i in range(self.C_out):
            weighted_votes, weights = self.channel_att(spatial_values[i])
            stds, means = torch.std_mean(weighted_votes, dim=1, unbiased=False)
            agreement = -torch.log(stds)
            atts_for_c1 = F.softmax(agreement, dim=1)
            caps_of_channel_i = (atts_for_c1 * means).sum(dim=1)
            caps_of_next_layer.append(caps_of_channel_i)

        return caps_of_next_layer

    def forward(self, caps):
        votes = []
        for i in range(self.C_in):
            if isinstance(caps, list):
                votes.append(self.conv_trans[i](caps[i]))
            else:
                votes.append(self.conv_trans[i](caps))
        caps_of_next_layer = self.cluster_routing(votes)
        for i in range(self.C_out):
            caps_of_next_layer[i] = self.acti(caps_of_next_layer[i])
        return caps_of_next_layer


class CapsNet(nn.Module):
    def __init__(self, num_classes=10, input_img_dim=3, input_img_size=32, C=4, K=10, D=32, if_bias=True, dropout_rate=0.0, reduction_ratio=4):
        super(CapsNet, self).__init__()
        self.caps_layer1 = RoutingWithCombinedAttention(C, C, K, input_img_dim, D,
                                                        if_bias,
                                                        out_S=input_img_size, stride=1, 
                                                        reduction_ratio=reduction_ratio)
        self.caps_layer2 = RoutingWithCombinedAttention(C, C, K, D, D, if_bias,
                                                        out_S=input_img_size, stride=1,
                                                        reduction_ratio=reduction_ratio)

        self.caps_layer3 = RoutingWithCombinedAttention(C, C, K, D, D, if_bias,
                                                        out_S=int(input_img_size / 2), stride=2,
                                                        reduction_ratio=reduction_ratio)
        self.caps_layer4 = RoutingWithCombinedAttention(C, C, K, D, D, if_bias,
                                                        out_S=int(input_img_size / 2), stride=1,
                                                        reduction_ratio=reduction_ratio)
        self.caps_layer5 = RoutingWithCombinedAttention(C, C, K, D, D, if_bias,
                                                        out_S=int(input_img_size / 2), stride=1,
                                                        reduction_ratio=reduction_ratio)

        self.caps_layer6 = RoutingWithCombinedAttention(C, C, K, D, D, if_bias,
                                                        out_S=int(input_img_size / 4), stride=2,
                                                        reduction_ratio=reduction_ratio)
        self.caps_layer7 = RoutingWithCombinedAttention(C, C, K, D, D, if_bias,
                                                        out_S=int(input_img_size / 4), stride=1,
                                                        reduction_ratio=reduction_ratio)
        self.caps_layer8 = RoutingWithCombinedAttention(C, C, K, D, D, if_bias,
                                                        out_S=int(input_img_size / 4), stride=1,
                                                        reduction_ratio=reduction_ratio)

        self.caps_layer9 = RoutingWithCombinedAttention(C, C, K, D, D, if_bias,
                                                        out_S=int(input_img_size / 8), stride=2,
                                                        reduction_ratio=reduction_ratio)
        self.caps_layer10 = RoutingWithCombinedAttention(C, C, K, D, D, if_bias,
                                                         out_S=int(input_img_size / 8), stride=1,
                                                         reduction_ratio=reduction_ratio)
        self.caps_layer11 = RoutingWithCombinedAttention(C, num_classes, K, D, D, if_bias,
                                                         out_S=int(input_img_size / 8), stride=1,
                                                         reduction_ratio=reduction_ratio)

        self.classifier = nn.Linear(D * int(input_img_size / 8) ** 2, 1)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        caps_1 = self.caps_layer1(x)
        caps_2 = self.caps_layer2(caps_1)

        caps_3 = self.caps_layer3(caps_2)
        caps_4 = self.caps_layer4(caps_3)
        caps_5 = self.caps_layer5(caps_4)

        caps_6 = self.caps_layer6(caps_5)
        caps_7 = self.caps_layer7(caps_6)
        caps_8 = self.caps_layer8(caps_7)

        caps_9 = self.caps_layer9(caps_8)
        caps_10 = self.caps_layer10(caps_9)
        caps_11 = self.caps_layer11(caps_10)

        caps = [c.view(c.shape[0], -1).unsqueeze(1) for c in caps_11]
        caps = torch.cat(caps, dim=1)
        
        if self.dropout is not None:
            caps = self.dropout(caps)
            
        pred = self.classifier(caps).squeeze()
        return pred


def create_hybridcapsnet(num_classes=10, input_img_dim=3, input_img_size=32, C=4, K=10, D=32, if_bias=True, dropout_rate=0.0, reduction_ratio=4, use_densenet_backbone=False):
    """
    Factory function to create a HybridCapsNet instance with specified parameters.
    This implementation honors the original authors by preserving their combined attention approach.
    
    Args:
        num_classes (int): Number of output classes
        input_img_dim (int): Number of input image channels (e.g., 3 for RGB)
        input_img_size (int): Input image size (assumes square images)
        C (int): Number of capsule channels
        K (int): Number of kernels used to form a cluster
        D (int): Capsule depth
        if_bias (bool): Whether to use bias in transforming capsules
        dropout_rate (float): Dropout rate for regularization
        reduction_ratio (int): Reduction ratio for channel attention mechanism
        use_densenet_backbone (bool): Whether to use DenseNet backbone
        
    Returns:
        CapsNet: Instantiated model
    """
    if use_densenet_backbone:
        return DenseNetHybridCapsNet(
            num_classes=num_classes,
            input_img_dim=input_img_dim,
            input_img_size=input_img_size,
            growth_rate=16,
            block_config=(3, 6, 8),
            dropout_rate=dropout_rate,
            C=C,
            K=K,
            D=D,
            if_bias=if_bias,
            reduction_ratio=reduction_ratio
        )
    else:
        return CapsNet(
            num_classes=num_classes,
            input_img_dim=input_img_dim,
            input_img_size=input_img_size,
            C=C,
            K=K,
            D=D,
            if_bias=if_bias,
            dropout_rate=dropout_rate,
            reduction_ratio=reduction_ratio
        )