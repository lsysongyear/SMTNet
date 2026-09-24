"""
BrainNetwork_v7 消融实验 — 依次去掉一个组件，保留其余。

  BrainMagic_NoSubjectAttn  : 去掉被试特异性空间注意力，换成单一共享 Spatial Attention
  BrainMagic_NoShortConv    : 去掉 short_conv_block (GatedDeepMultiScaleBlock) 分支
  BrainMagic_NoFeatureEncoder: 去掉 feature_encoder (FeatureEncoder) 分支
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 共享基础组件（与 BrainNetwork_v7 一致）
# ============================================================================

class GatedShortTermTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, dilation=1, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, 2 * out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=(kernel_size + (kernel_size-1)*(dilation-1) - 1) // 2,
                              dilation=dilation)
        self.bn = nn.BatchNorm1d(2 * out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        res = x if self.residual is None else self.residual(x)
        h = self.bn(self.conv(x))
        out = torch.tanh(h[:, :h.size(1)//2]) * torch.sigmoid(h[:, h.size(1)//2:])
        out = self.dropout(out)
        return res + out


class GatedMultiScaleTemporalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3,5,7],
                 dilation=1, dropout=0.2):
        super().__init__()
        assert out_channels % len(kernel_sizes) == 0
        sub_ch = out_channels // len(kernel_sizes)
        self.convs = nn.ModuleList([
            GatedShortTermTemporalConv(in_channels, sub_ch, kernel_size=ks,
                                       dilation=dilation, dropout=dropout)
            for ks in kernel_sizes
        ])

    def forward(self, x):
        return torch.cat([conv(x) for conv in self.convs], dim=1)


class GatedDeepMultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3,5,7],
                 num_layers=3, dropout=0.2, dilation_cycle=6):
        super().__init__()
        self.dilations = []
        self.layers = nn.ModuleList()
        self.skip_cons = nn.ModuleList()
        for i in range(num_layers):
            layer_in = in_channels if i == 0 else out_channels
            dilation = 2 ** (i % dilation_cycle)
            self.dilations.append(dilation)
            self.layers.append(
                GatedMultiScaleTemporalLayer(layer_in, out_channels,
                                             kernel_sizes, dilation, dropout))
            self.skip_cons.append(
                nn.Identity() if layer_in == out_channels else nn.Conv1d(layer_in, out_channels, 1))
        self.final_norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        residual = x
        for layer, skip_conv in zip(self.layers, self.skip_cons):
            x = layer(x)
            if residual is not None:
                x = skip_conv(residual) + x
                residual = x
        return self.final_norm(x)


class Feature_Block(nn.Module):
    def __init__(self, layer_index, channels, kernel_size=10, dropout=0.5):
        super().__init__()
        self.layer_index = layer_index
        if kernel_size == 3:
            dilation1 = int(2 ** ((2 * layer_index) % 5))
            dilation2 = int(2 ** ((2 * layer_index + 1) % 5))
        else:
            dilation1 = int(2 ** (layer_index % 6))
            dilation2 = int(2 ** ((layer_index + 1) % 6))

        self.conv1 = nn.Conv1d(channels[0], channels[1], kernel_size,
                               padding='same', dilation=dilation1)
        self.conv2 = nn.Conv1d(channels[1], channels[2], kernel_size,
                               padding='same', dilation=dilation2)
        self.conv3 = nn.Conv1d(channels[2], channels[3], kernel_size,
                               padding='same', dilation=2)
        self.bn1 = nn.BatchNorm1d(channels[1], eps=1e-5, momentum=0.1)
        self.bn2 = nn.BatchNorm1d(channels[2], eps=1e-5, momentum=0.1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, x):
        input_x = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv3(x)
        mid = x.size(1) // 2
        out = torch.tanh(x[:, :mid, :]) * torch.sigmoid(x[:, mid:, :])
        out = self.dropout(out)
        if self.layer_index > 1 and input_x.shape[1] == out.shape[1]:
            out = input_x + out
        return out, out


class FeatureEncoder(nn.Module):
    def __init__(self, attention_dim, kernel_size, dropout, num_blocks=3):
        super().__init__()
        self.num_blocks = num_blocks
        channels = [attention_dim, attention_dim, attention_dim, 2 * attention_dim]
        self.blocks = nn.ModuleList([
            Feature_Block(i + 1, channels, kernel_size, dropout)
            for i in range(num_blocks)
        ])

    def forward(self, x):
        skip_sum = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip
        return skip_sum * math.sqrt(1.0 / self.num_blocks)


# ============================================================================
# 消融模型 1: 去掉被试特异性空间注意力 → 单一共享 Spatial Attention
# ============================================================================
class BrainMagic_NoSubjectAttn(nn.Module):
    def __init__(self, input_dim=306, attention_dim=128, output_dim=1,
                 kernel_size=3, depthwise_kernel=3, num_blocks1=3, num_blocks2=12, dropout=0.02, dilation_cycle=12):
        super().__init__()

        # 单一共享空间注意力（无被试索引）
        self.spatial_attention = self._build_spatial_attention(input_dim, attention_dim)

        self.feature_encoder = FeatureEncoder(attention_dim, kernel_size, dropout, num_blocks1)
        self.short_conv_block = GatedDeepMultiScaleBlock(
            in_channels=attention_dim, out_channels=attention_dim,
            kernel_sizes=[3, 5, 7, 9], num_layers=num_blocks2,
            dropout=dropout, dilation_cycle=dilation_cycle)

        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(2 * attention_dim, 2 * attention_dim, depthwise_kernel,
                      padding='same', groups=2 * attention_dim, bias=False),
            nn.BatchNorm1d(2 * attention_dim),
            nn.GELU(),
            nn.Conv1d(2 * attention_dim, attention_dim, 1, bias=False),
            nn.BatchNorm1d(attention_dim),
            nn.GELU())
        self.final_conv1 = nn.Conv1d(attention_dim, 4 * attention_dim, kernel_size=1)
        self.final_conv2 = nn.Conv1d(4 * attention_dim, output_dim, kernel_size=1)

    @staticmethod
    def _build_spatial_attention(input_dim, attention_dim):
        hidden_dims = [4 * input_dim]
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dims[-1], attention_dim))
        layers.append(nn.GELU())
        return nn.Sequential(*layers)

    def forward(self, x, subject_ids=None):
        # 单一共享注意力，忽略 subject_ids
        x = x.transpose(1, 2)                      # [B, L, input_dim]
        x = self.spatial_attention(x)               # [B, L, attention_dim]  ← 全部被试共享
        x = x.transpose(1, 2)                      # [B, attention_dim, L]

        short_feat = self.short_conv_block(x)
        encoded_feat = self.feature_encoder(x)
        fused = torch.cat([encoded_feat, short_feat], dim=1)
        x = self.depthwise_conv(fused)
        x = self.final_conv1(x)
        x = self.final_conv2(x)
        return x.squeeze(1)


# ============================================================================
# 消融模型 2: 去掉 short_conv_block 分支，只保留 feature_encoder
# ============================================================================
class BrainMagic_NoShortConv(nn.Module):
    def __init__(self, input_dim=306, attention_dim=128, output_dim=1,
                 kernel_size=3, depthwise_kernel=3, num_blocks1=3, num_blocks2=12, dropout=0.02, dilation_cycle=12):
        super().__init__()
        self.spatial_attention = self._build_spatial_attention(input_dim, attention_dim)

        self.feature_encoder = FeatureEncoder(attention_dim, kernel_size, dropout, num_blocks1)

        # 只有一个分支，输入从 2*attention_dim 变为 attention_dim
        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(attention_dim, attention_dim, depthwise_kernel,
                      padding='same', groups=attention_dim, bias=False),
            nn.BatchNorm1d(attention_dim),
            nn.GELU(),
            nn.Conv1d(attention_dim, attention_dim, 1, bias=False),
            nn.BatchNorm1d(attention_dim),
            nn.GELU())
        self.final_conv1 = nn.Conv1d(attention_dim, 4 * attention_dim, kernel_size=1)
        self.final_conv2 = nn.Conv1d(4 * attention_dim, output_dim, kernel_size=1)

    @staticmethod
    def _build_spatial_attention(input_dim, attention_dim):
        hidden_dims = [4 * input_dim]
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dims[-1], attention_dim))
        layers.append(nn.GELU())
        return nn.Sequential(*layers)

    def forward(self, x, subject_ids=None):
        B = x.size(0)
        x = x.transpose(1, 2)
        # single shared attention (no subject routing)

        x = self.spatial_attention(x)
        x = x.transpose(1, 2)

        x = self.feature_encoder(x)               # 仅 feature_encoder 分支
        x = self.depthwise_conv(x)
        x = self.final_conv1(x)
        x = self.final_conv2(x)
        return x.squeeze(1)


# ============================================================================
# 消融模型 3: 去掉 feature_encoder 分支，只保留 short_conv_block
# ============================================================================
class BrainMagic_NoFeatureEncoder(nn.Module):
    def __init__(self, input_dim=306, attention_dim=128, output_dim=1,
                 kernel_size=3, depthwise_kernel=3, num_blocks1=3, num_blocks2=12, dropout=0.02, dilation_cycle=12):
        super().__init__()
        self.spatial_attention = self._build_spatial_attention(input_dim, attention_dim)

        self.short_conv_block = GatedDeepMultiScaleBlock(
            in_channels=attention_dim, out_channels=attention_dim,
            kernel_sizes=[3, 5, 7, 9], num_layers=num_blocks2,
            dropout=dropout, dilation_cycle=dilation_cycle)

        # 只有一个分支，输入从 2*attention_dim 变为 attention_dim
        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(attention_dim, attention_dim, depthwise_kernel,
                      padding='same', groups=attention_dim, bias=False),
            nn.BatchNorm1d(attention_dim),
            nn.GELU(),
            nn.Conv1d(attention_dim, attention_dim, 1, bias=False),
            nn.BatchNorm1d(attention_dim),
            nn.GELU())
        self.final_conv1 = nn.Conv1d(attention_dim, 4 * attention_dim, kernel_size=1)
        self.final_conv2 = nn.Conv1d(4 * attention_dim, output_dim, kernel_size=1)

    @staticmethod
    def _build_spatial_attention(input_dim, attention_dim):
        hidden_dims = [4 * input_dim]
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dims[-1], attention_dim))
        layers.append(nn.GELU())
        return nn.Sequential(*layers)

    def forward(self, x, subject_ids=None):
        B = x.size(0)
        x = x.transpose(1, 2)
        # single shared attention (no subject routing)

        x = self.spatial_attention(x)
        x = x.transpose(1, 2)

        x = self.short_conv_block(x)              # 仅 short_conv_block 分支
        x = self.depthwise_conv(x)
        x = self.final_conv1(x)
        x = self.final_conv2(x)
        return x.squeeze(1)


if __name__ == "__main__":
    device = 'cpu'
    x = torch.rand(4, 204, 3000).to(device)

    for name, Model in [
        ("NoSubjectAttn", BrainMagic_NoSubjectAttn),
        ("NoShortConv", BrainMagic_NoShortConv),
        ("NoFeatureEncoder", BrainMagic_NoFeatureEncoder),
    ]:
        model = Model(input_dim=204, attention_dim=128, kernel_size=25, dropout=0.1).to(device)
        y = model(x)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"BrainMagic_{name}: output {list(y.shape)}, params {params:.2f}M")
