import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# ==============================
# 1. 基础组件（门控激活）
# ==============================
class GatedShortTermTemporalConv(nn.Module):
    """原 ShortTermTemporalConv，用门控激活替代 ReLU"""
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
    """原 MultiScaleTemporalLayer，使用门控版本的短时卷积"""
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
    """原 DeepMultiScaleBlock，使用门控多尺度层"""
    def __init__(self, in_channels, out_channels, kernel_sizes=[3,5,7],
                 num_layers=3, dropout=0.2, dilation_growth=1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.skip_cons = nn.ModuleList()
        for i in range(num_layers):
            layer_in = in_channels if i == 0 else out_channels
            dilation = dilation_growth ** i
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


# ==============================
# 2. Feature_Block：门控激活 + 跳跃输出
# ==============================
class Feature_Block(nn.Module):
    """
    保留原有卷积-BN-Dropout 骨架，将 GLU 替换为 WaveNet 门控，
    并额外返回一个跳跃输出用于外部累积。
    """
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
    """堆叠 num_blocks 个 Feature_Block，累积所有跳跃输出并归一化"""
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
# 3. Brain_Magic_speech_v2 — 添加 BiLSTM 时序建模（仿 SHINE）
# ==============================
class Brain_Magic_speech_v2(nn.Module):
    def __init__(self, input_dim=306, attention_dim=128, output_dim=1,
                 kernel_size=3, depthwise_kernel=3, num_blocks1=3, num_blocks2=12, dropout=0.02,
                 n_subjects=25, n_days=3):
        super().__init__()

        # --- 被试 + 天数特异化空间注意力（25×3=75 层）---
        self.n_subjects = n_subjects
        self.n_days = n_days
        self.spatial_attentions = nn.ModuleList([
            self._build_spatial_attention(input_dim, attention_dim)
            for _ in range(n_subjects * n_days)
        ])

        # --- 跳跃累积版编码器 ---
        self.feature_encoder = FeatureEncoder(attention_dim, kernel_size, dropout, num_blocks1)

        # --- 门控版多尺度短时卷积 ---
        self.short_conv_block = GatedDeepMultiScaleBlock(
            in_channels=attention_dim,
            out_channels=attention_dim,
            kernel_sizes=[3, 5, 7, 9],
            num_layers=num_blocks2,
            dropout=dropout
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

        # --- BiLSTM 时序建模（仿 SHINE）---
        self.lstm_pre_conv = nn.Conv1d(attention_dim, attention_dim, 101, padding=50)
        self.lstm_norm = nn.LayerNorm(attention_dim)
        self.lstm = nn.LSTM(attention_dim, attention_dim, batch_first=True, bidirectional=True)

        self.lstm_fc = nn.Linear(3 * attention_dim, attention_dim)
        self.lstm_fc2 = nn.Linear(attention_dim, output_dim)

        self.relu = nn.LeakyReLU()
        self.dropout_val = dropout

        self._initialize_weights()

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

    def _initialize_weights(self):
        def init_func(module):
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                if module.weight is not None:
                    init.ones_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)
        self.apply(init_func)

    def forward(self, x, subject_ids=None, day_ids=None):
        # x: [B, input_dim, L]
        # subject_ids: [B]  int tensor, 0-indexed (0-24)
        # day_ids:      [B]  int tensor, 0-indexed (0-2)
        B, _, L = x.shape

        # 1. 被试+天数特异化空间注意力
        x = x.transpose(1, 2)

        if subject_ids is not None and day_ids is not None:
            indices = subject_ids * self.n_days + day_ids
            attn_outputs = []
            for i in range(B):
                idx = indices[i].item()
                out_i = self.spatial_attentions[idx](x[i:i+1])
                attn_outputs.append(out_i)
            x = torch.cat(attn_outputs, dim=0)
        else:
            x = self.spatial_attentions[0](x)

        x = x.transpose(1, 2)

        # 2. 两条并行分支
        short_feat = self.short_conv_block(x)
        encoded_feat = self.feature_encoder(x)

        # 3. 拼接融合
        fused = torch.cat([encoded_feat, short_feat], dim=1)

        # 4. 深度可分离卷积
        x = self.depthwise_conv(fused)                     # [B, attention_dim, L]

        # 5. BiLSTM 时序建模（仿 SHINE）
        lstm_residual = x.transpose(1, 2)                  # [B, L, attention_dim]

        x = self.lstm_pre_conv(x)                          # [B, attention_dim, L]
        x = x.transpose(1, 2)                              # [B, L, attention_dim]
        x = self.lstm_norm(x)

        x, _ = self.lstm(x)                                # [B, L, 2*attention_dim]
        x = self.relu(x)

        x = torch.cat([x, lstm_residual], dim=2)           # [B, L, 3*attention_dim]

        x = self.lstm_fc(x)                                # [B, L, attention_dim]
        x = self.relu(x)
        x = self.lstm_fc2(x)                               # [B, L, output_dim]

        return x.squeeze(2)                                # [B, L]


if __name__ == "__main__":
    device = 'cpu'
    model = Brain_Magic_speech_v2(input_dim=57, attention_dim=128, kernel_size=25, dropout=0.1).to(device)
    x = torch.rand(1, 57, 3000).to(device)
    y = model(x)
    print(y.shape)  # 应为 [1, 3000]
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
