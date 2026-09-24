"""
BrainNetwork_v5 — Brain_Magic_speech_v1 with AWaveNet dilated residual groups.

Key improvements over v4:
  - AWaveNet-style Residual_group with exponential dilation (1→2048, 36 layers)
  - weight_norm + sqrt(0.5) residual scaling for deep training stability
  - sqrt(1/n_layers) skip accumulation normalization
  - Retains v4's spatial attention, FeatureEncoder, and output head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================
# 1. AWaveNet-style dilated conv blocks
# ==============================

class AWaveNet_Conv(nn.Module):
    """weight_norm + kaiming dilated conv (from AWaveNet)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)


class AWaveNet_ResBlock(nn.Module):
    """Single dilated residual block with gated activation and skip output."""
    def __init__(self, res_channels, skip_channels, dilation):
        super().__init__()
        self.res_channels = res_channels
        self.dilated_conv = AWaveNet_Conv(res_channels, 2 * res_channels,
                                          kernel_size=3, dilation=dilation)
        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)

    def forward(self, x):
        h = self.dilated_conv(x)
        out = torch.tanh(h[:, :self.res_channels, :]) * torch.sigmoid(h[:, self.res_channels:, :])
        res = self.res_conv(out)
        skip = self.skip_conv(out)
        return (x + res) * math.sqrt(0.5), skip


class AWaveNet_ResGroup(nn.Module):
    """Stack of dilated residual blocks with exponential dilation cycle."""
    def __init__(self, res_channels, skip_channels, num_layers=36, dilation_cycle=12):
        super().__init__()
        self.num_layers = num_layers
        self.blocks = nn.ModuleList()
        for n in range(num_layers):
            self.blocks.append(
                AWaveNet_ResBlock(res_channels, skip_channels,
                                  dilation=2 ** (n % dilation_cycle)))

    def forward(self, x):
        h = x
        skip_sum = 0
        for block in self.blocks:
            h, skip = block(h)
            skip_sum = skip_sum + skip
        return skip_sum * math.sqrt(1.0 / self.num_layers)


# ==============================
# 2. Feature_Block — kept from v4 (gated + skip output)
# ==============================

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


# ==============================
# 3. Brain_Magic_speech_v1 (v5) — v4 + AWaveNet dilated group
# ==============================

class Brain_Magic_speech_v1(nn.Module):
    def __init__(self, input_dim=306, attention_dim=128, output_dim=1,
                 kernel_size=3, depthwise_kernel=3, num_blocks1=3,
                 num_res_layers=36, dilation_cycle=12,
                 dropout=0.02, n_subjects=25):
        super().__init__()

        # --- 被试特异化空间注意力 ---
        self.n_subjects = n_subjects
        self.spatial_attentions = nn.ModuleList([
            self._build_spatial_attention(input_dim, attention_dim)
            for _ in range(n_subjects)
        ])

        # --- 跳跃累积版编码器（保留自 v4）---
        self.feature_encoder = FeatureEncoder(attention_dim, kernel_size, dropout, num_blocks1)

        # --- AWaveNet 扩张残差组（替代 v4 的 GatedDeepMultiScaleBlock）---
        self.dilated_group = AWaveNet_ResGroup(
            res_channels=attention_dim,
            skip_channels=attention_dim,
            num_layers=num_res_layers,
            dilation_cycle=dilation_cycle,
        )

        # --- 深度可分离卷积 ---
        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(2 * attention_dim, 2 * attention_dim, depthwise_kernel,
                      padding='same', groups=2 * attention_dim, bias=False),
            nn.BatchNorm1d(2 * attention_dim),
            nn.GELU(),
            nn.Conv1d(2 * attention_dim, attention_dim, 1, bias=False),
            nn.BatchNorm1d(attention_dim),
            nn.GELU()
        )

        # --- 帧级 VAD 输出头 ---
        self.final_conv1 = nn.Conv1d(attention_dim, 4 * attention_dim, kernel_size=1)
        self.final_conv2 = nn.Conv1d(4 * attention_dim, output_dim, kernel_size=1)
        self.dropout = dropout

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
        B, _, L = x.shape

        # 1. 被试特异化空间注意力
        x = x.transpose(1, 2)
        if subject_ids is not None:
            attn_outputs = []
            for i in range(B):
                idx = subject_ids[i].item()
                out_i = self.spatial_attentions[idx](x[i:i+1])
                attn_outputs.append(out_i)
            x = torch.cat(attn_outputs, dim=0)
        else:
            x = self.spatial_attentions[0](x)
        x = x.transpose(1, 2)

        # 2. 两条并行分支
        awavenet_feat = self.dilated_group(x)       # AWaveNet: 指数扩张门控残差
        encoded_feat = self.feature_encoder(x)       # FeatureEncoder: 跳跃累积

        # 3. 拼接融合
        fused = torch.cat([encoded_feat, awavenet_feat], dim=1)

        # 4. 深度可分离卷积
        x = self.depthwise_conv(fused)

        # 5. 帧级分类
        x = self.final_conv1(x)
        x = self.final_conv2(x)

        return x.squeeze(1)


if __name__ == "__main__":
    device = 'cpu'
    model = Brain_Magic_speech_v1(input_dim=57, attention_dim=128, kernel_size=25,
                                   dropout=0.1, num_res_layers=12, dilation_cycle=6).to(device)
    x = torch.rand(2, 57, 3000).to(device)
    subj = torch.tensor([0, 1])
    y = model(x, subject_ids=subj)
    print(f"V5: {x.shape} -> {y.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
