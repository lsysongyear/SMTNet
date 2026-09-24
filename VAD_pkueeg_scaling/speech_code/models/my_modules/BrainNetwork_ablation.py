# ============================================================================
# BrainNetwork_ablation.py - Brain_Magic_speech 消融分析模型
# ============================================================================
# 基于 Brain_Magic_speech 依次去除三个分支：
#   - Feature Encoder (FE): 5 层 dilated conv 编码器
#   - MultiScale Block (MS): 12 层多尺度时间卷积
#   - BiLSTM: 双向 LSTM
#
# 消融变体（6 个）：
#   1. BrainMagic_NoFE         — 去除 Feature Encoder
#   2. BrainMagic_NoMS         — 去除 MultiScale Block
#   3. BrainMagic_NoBiLSTM     — 去除 BiLSTM
#   4. BrainMagic_NoFE_NoMS    — 仅保留 BiLSTM
#   5. BrainMagic_NoFE_NoBiLSTM— 仅保留 MultiScale Block
#   6. BrainMagic_NoMS_NoBiLSTM— 仅保留 Feature Encoder
#
# 各消融变体的融合维度对齐策略：
#   - BiLSTM 输出 256 维，卷积分支输出 128 维
#   - 单卷积分支时用 Conv1d(128, 256, 1) 投影对齐到 256
#   - 双卷积分支时 concat(128, 128) = 256 天然对齐
#   - 仅 BiLSTM 时输出 256 维无需投影
#   保证 depthwise_conv 输入始终为 256 维，后续结构不变
# ============================================================================

import torch.nn as nn
import torch
from speech_code.models.my_modules.BrainNetwork import (
    Feature_Block, DeepMultiScaleBlock
)


class BrainMagic_NoFE(nn.Module):
    """去除 Feature Encoder，保留 MultiScale + BiLSTM"""

    def __init__(self, input_dim=306, attention_dim=128, output_dim=1,
                 kernel_size=5, depthwise_kernel=15, dropout=0.1):
        super().__init__()

        hidden_dims = [input_dim, 4 * input_dim]
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dims[-1], attention_dim))
        self.spatial_attention = nn.Sequential(*layers)

        self.short_conv_block = DeepMultiScaleBlock(
            in_channels=attention_dim, out_channels=attention_dim,
            kernel_sizes=[3, 5, 7, 9], num_layers=12, dropout=dropout)

        self.lstm = nn.LSTM(input_size=attention_dim, hidden_size=attention_dim,
                            num_layers=1, batch_first=True, bidirectional=True,
                            dropout=dropout)

        self.short_proj = nn.Conv1d(attention_dim, 2 * attention_dim, kernel_size=1)

        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(2 * attention_dim, 2 * attention_dim, kernel_size=depthwise_kernel,
                       padding='same', groups=2 * attention_dim, bias=False),
            nn.BatchNorm1d(2 * attention_dim), nn.GELU(),
            nn.Conv1d(2 * attention_dim, attention_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(attention_dim), nn.GELU())

        self.final_conv1 = nn.Conv1d(attention_dim, 4 * attention_dim, kernel_size=1)
        self.final_conv2 = nn.Conv1d(4 * attention_dim, output_dim, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.spatial_attention(x)
        x = x.transpose(1, 2)

        short_features = self.short_conv_block(x)
        lstm_features, _ = self.lstm(x.transpose(1, 2))

        fused_features = self.short_proj(short_features) + lstm_features.transpose(1, 2)

        x = self.depthwise_conv(fused_features)
        x = self.final_conv1(x)
        x = self.final_conv2(x)
        return x.squeeze(1)


class BrainMagic_NoMS(nn.Module):
    """去除 MultiScale Block，保留 Feature Encoder + BiLSTM"""

    def __init__(self, input_dim=306, attention_dim=128, output_dim=1,
                 kernel_size=5, depthwise_kernel=15, dropout=0.1):
        super().__init__()

        hidden_dims = [input_dim, 4 * input_dim]
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dims[-1], attention_dim))
        self.spatial_attention = nn.Sequential(*layers)

        encoder_channels = [attention_dim, attention_dim, attention_dim, 2 * attention_dim]
        self.feature_encoder = nn.Sequential(
            Feature_Block(1, encoder_channels, kernel_size, dropout),
            Feature_Block(2, encoder_channels, kernel_size, dropout),
            Feature_Block(3, encoder_channels, kernel_size, dropout),
            Feature_Block(4, encoder_channels, kernel_size, dropout),
            Feature_Block(5, encoder_channels, kernel_size, dropout),
        )

        self.lstm = nn.LSTM(input_size=attention_dim, hidden_size=attention_dim,
                            num_layers=1, batch_first=True, bidirectional=True,
                            dropout=dropout)

        self.encoder_proj = nn.Conv1d(attention_dim, 2 * attention_dim, kernel_size=1)

        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(2 * attention_dim, 2 * attention_dim, kernel_size=depthwise_kernel,
                       padding='same', groups=2 * attention_dim, bias=False),
            nn.BatchNorm1d(2 * attention_dim), nn.GELU(),
            nn.Conv1d(2 * attention_dim, attention_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(attention_dim), nn.GELU())

        self.final_conv1 = nn.Conv1d(attention_dim, 4 * attention_dim, kernel_size=1)
        self.final_conv2 = nn.Conv1d(4 * attention_dim, output_dim, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.spatial_attention(x)
        x = x.transpose(1, 2)

        encoded_features = self.feature_encoder(x)
        lstm_features, _ = self.lstm(x.transpose(1, 2))

        fused_features = self.encoder_proj(encoded_features) + lstm_features.transpose(1, 2)

        x = self.depthwise_conv(fused_features)
        x = self.final_conv1(x)
        x = self.final_conv2(x)
        return x.squeeze(1)


class BrainMagic_NoBiLSTM(nn.Module):
    """去除 BiLSTM，保留 Feature Encoder + MultiScale Block"""

    def __init__(self, input_dim=306, attention_dim=128, output_dim=1,
                 kernel_size=5, depthwise_kernel=15, dropout=0.1):
        super().__init__()

        hidden_dims = [input_dim, 4 * input_dim]
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dims[-1], attention_dim))
        self.spatial_attention = nn.Sequential(*layers)

        encoder_channels = [attention_dim, attention_dim, attention_dim, 2 * attention_dim]
        self.feature_encoder = nn.Sequential(
            Feature_Block(1, encoder_channels, kernel_size, dropout),
            Feature_Block(2, encoder_channels, kernel_size, dropout),
            Feature_Block(3, encoder_channels, kernel_size, dropout),
            Feature_Block(4, encoder_channels, kernel_size, dropout),
            Feature_Block(5, encoder_channels, kernel_size, dropout),
        )

        self.short_conv_block = DeepMultiScaleBlock(
            in_channels=attention_dim, out_channels=attention_dim,
            kernel_sizes=[3, 5, 7, 9], num_layers=12, dropout=dropout)

        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(2 * attention_dim, 2 * attention_dim, kernel_size=depthwise_kernel,
                       padding='same', groups=2 * attention_dim, bias=False),
            nn.BatchNorm1d(2 * attention_dim), nn.GELU(),
            nn.Conv1d(2 * attention_dim, attention_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(attention_dim), nn.GELU())

        self.final_conv1 = nn.Conv1d(attention_dim, 4 * attention_dim, kernel_size=1)
        self.final_conv2 = nn.Conv1d(4 * attention_dim, output_dim, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.spatial_attention(x)
        x = x.transpose(1, 2)

        short_features = self.short_conv_block(x)
        encoded_features = self.feature_encoder(x)

        fused_features = torch.cat([encoded_features, short_features], dim=1)

        x = self.depthwise_conv(fused_features)
        x = self.final_conv1(x)
        x = self.final_conv2(x)
        return x.squeeze(1)


class BrainMagic_NoFE_NoMS(nn.Module):
    """仅保留 BiLSTM（去除 Feature Encoder 和 MultiScale Block）"""

    def __init__(self, input_dim=306, attention_dim=128, output_dim=1,
                 kernel_size=5, depthwise_kernel=15, dropout=0.1):
        super().__init__()

        hidden_dims = [input_dim, 4 * input_dim]
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dims[-1], attention_dim))
        self.spatial_attention = nn.Sequential(*layers)

        self.lstm = nn.LSTM(input_size=attention_dim, hidden_size=attention_dim,
                            num_layers=1, batch_first=True, bidirectional=True,
                            dropout=dropout)

        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(2 * attention_dim, 2 * attention_dim, kernel_size=depthwise_kernel,
                       padding='same', groups=2 * attention_dim, bias=False),
            nn.BatchNorm1d(2 * attention_dim), nn.GELU(),
            nn.Conv1d(2 * attention_dim, attention_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(attention_dim), nn.GELU())

        self.final_conv1 = nn.Conv1d(attention_dim, 4 * attention_dim, kernel_size=1)
        self.final_conv2 = nn.Conv1d(4 * attention_dim, output_dim, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.spatial_attention(x)
        x = x.transpose(1, 2)

        lstm_features, _ = self.lstm(x.transpose(1, 2))

        fused_features = lstm_features.transpose(1, 2)

        x = self.depthwise_conv(fused_features)
        x = self.final_conv1(x)
        x = self.final_conv2(x)
        return x.squeeze(1)


class BrainMagic_NoFE_NoBiLSTM(nn.Module):
    """仅保留 MultiScale Block（去除 Feature Encoder 和 BiLSTM）"""

    def __init__(self, input_dim=306, attention_dim=128, output_dim=1,
                 kernel_size=5, depthwise_kernel=15, dropout=0.1):
        super().__init__()

        hidden_dims = [input_dim, 4 * input_dim]
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dims[-1], attention_dim))
        self.spatial_attention = nn.Sequential(*layers)

        self.short_conv_block = DeepMultiScaleBlock(
            in_channels=attention_dim, out_channels=attention_dim,
            kernel_sizes=[3, 5, 7, 9], num_layers=12, dropout=dropout)

        self.short_proj = nn.Conv1d(attention_dim, 2 * attention_dim, kernel_size=1)

        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(2 * attention_dim, 2 * attention_dim, kernel_size=depthwise_kernel,
                       padding='same', groups=2 * attention_dim, bias=False),
            nn.BatchNorm1d(2 * attention_dim), nn.GELU(),
            nn.Conv1d(2 * attention_dim, attention_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(attention_dim), nn.GELU())

        self.final_conv1 = nn.Conv1d(attention_dim, 4 * attention_dim, kernel_size=1)
        self.final_conv2 = nn.Conv1d(4 * attention_dim, output_dim, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.spatial_attention(x)
        x = x.transpose(1, 2)

        short_features = self.short_conv_block(x)

        fused_features = self.short_proj(short_features)

        x = self.depthwise_conv(fused_features)
        x = self.final_conv1(x)
        x = self.final_conv2(x)
        return x.squeeze(1)


class BrainMagic_NoMS_NoBiLSTM(nn.Module):
    """仅保留 Feature Encoder（去除 MultiScale Block 和 BiLSTM）"""

    def __init__(self, input_dim=306, attention_dim=128, output_dim=1,
                 kernel_size=5, depthwise_kernel=15, dropout=0.1):
        super().__init__()

        hidden_dims = [input_dim, 4 * input_dim]
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dims[-1], attention_dim))
        self.spatial_attention = nn.Sequential(*layers)

        encoder_channels = [attention_dim, attention_dim, attention_dim, 2 * attention_dim]
        self.feature_encoder = nn.Sequential(
            Feature_Block(1, encoder_channels, kernel_size, dropout),
            Feature_Block(2, encoder_channels, kernel_size, dropout),
            Feature_Block(3, encoder_channels, kernel_size, dropout),
            Feature_Block(4, encoder_channels, kernel_size, dropout),
            Feature_Block(5, encoder_channels, kernel_size, dropout),
        )

        self.encoder_proj = nn.Conv1d(attention_dim, 2 * attention_dim, kernel_size=1)

        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(2 * attention_dim, 2 * attention_dim, kernel_size=depthwise_kernel,
                       padding='same', groups=2 * attention_dim, bias=False),
            nn.BatchNorm1d(2 * attention_dim), nn.GELU(),
            nn.Conv1d(2 * attention_dim, attention_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(attention_dim), nn.GELU())

        self.final_conv1 = nn.Conv1d(attention_dim, 4 * attention_dim, kernel_size=1)
        self.final_conv2 = nn.Conv1d(4 * attention_dim, output_dim, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.spatial_attention(x)
        x = x.transpose(1, 2)

        encoded_features = self.feature_encoder(x)

        fused_features = self.encoder_proj(encoded_features)

        x = self.depthwise_conv(fused_features)
        x = self.final_conv1(x)
        x = self.final_conv2(x)
        return x.squeeze(1)


if __name__ == "__main__":
    device = 'cpu'
    models = {
        "NoFE": BrainMagic_NoFE,
        "NoMS": BrainMagic_NoMS,
        "NoBiLSTM": BrainMagic_NoBiLSTM,
        "NoFE_NoMS": BrainMagic_NoFE_NoMS,
        "NoFE_NoBiLSTM": BrainMagic_NoFE_NoBiLSTM,
        "NoMS_NoBiLSTM": BrainMagic_NoMS_NoBiLSTM,
    }
    for name, cls in models.items():
        model = cls(input_dim=57, attention_dim=128, kernel_size=25, dropout=0.01).to(device)
        x = torch.rand(1, 57, 3000).to(device)
        y = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:20s}  output: {y.shape}  params: {params:,}")
