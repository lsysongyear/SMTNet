"""
BM (Brain Module) — EEG spatial attention + FeatureEncoder backbone.
Adapted from NIPS2025 for VAD: in_dim is configurable (default 57 for PKU EEG).
"""
import torch.nn as nn
import torch.nn.init as init


class FeatureEncoderBlock_v2(nn.Module):
    """Dilated residual conv block with GLU gating."""
    def __init__(self, k=1, channels=None, kernel_size=25, dropout=0):
        super().__init__()
        if channels is None:
            channels = [64, 64, 64, 128]
        self.k = k
        if kernel_size == 3:
            dilation1 = int(pow(2, (2 * self.k) % 5))
            dilation2 = int(pow(2, (2 * self.k + 1) % 5))
        else:
            dilation1 = int(pow(2, (self.k) % 5))
            dilation2 = int(pow(2, (self.k + 1) % 5))
        self.conv1 = nn.Conv1d(channels[0], channels[1], kernel_size,
                               padding='same', dilation=dilation1)
        self.conv2 = nn.Conv1d(channels[1], channels[2], kernel_size,
                               padding='same', dilation=dilation2)
        self.conv3 = nn.Conv1d(channels[2], channels[3], kernel_size,
                               padding='same', dilation=2)
        self.batch_norm1 = nn.BatchNorm1d(channels[1], eps=1e-05, momentum=0.1)
        self.batch_norm2 = nn.BatchNorm1d(channels[2], eps=1e-05, momentum=0.1)
        self.GELU = nn.GELU()
        self.GLU = nn.GLU(dim=-2)
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, x):
        if self.k == 1:
            x = self.conv1(x)
            x = self.batch_norm1(x)
            x = self.GELU(x)
            x = self.dropout(x)
            x = self.conv2(x)
            x = self.batch_norm2(x)
            x = self.GELU(x)
            x = self.dropout(x)
            x = self.conv3(x)
            x = self.GLU(x)
            x = self.dropout(x)
            return x
        else:
            residual = x
            x = self.conv1(x)
            x = self.batch_norm1(x)
            x = self.GELU(x)
            x = self.dropout(x)
            x = self.conv2(x)
            x = self.batch_norm2(x)
            x = self.GELU(x)
            x = self.dropout(x)
            x = self.conv3(x)
            x = self.GLU(x)
            x = self.dropout(x)
            return residual + x


class BM(nn.Module):
    def __init__(self, in_dim=57, att_out_dim=64, out_dim=1,
                 kernel_size=25, dropout=0):
        super().__init__()
        self.spatialattention1 = nn.Linear(in_dim, in_dim)
        self.spatialattention2 = nn.Linear(in_dim, att_out_dim)

        self.FeatureEncoder = nn.Sequential(
            FeatureEncoderBlock_v2(1, [att_out_dim, att_out_dim, att_out_dim, 2 * att_out_dim], kernel_size, dropout),
            FeatureEncoderBlock_v2(2, [att_out_dim, att_out_dim, att_out_dim, 2 * att_out_dim], kernel_size, dropout),
            FeatureEncoderBlock_v2(3, [att_out_dim, att_out_dim, att_out_dim, 2 * att_out_dim], kernel_size, dropout),
            FeatureEncoderBlock_v2(4, [att_out_dim, att_out_dim, att_out_dim, 2 * att_out_dim], kernel_size, dropout),
            FeatureEncoderBlock_v2(5, [att_out_dim, att_out_dim, att_out_dim, 2 * att_out_dim], kernel_size, dropout),
            FeatureEncoderBlock_v2(6, [att_out_dim, att_out_dim, att_out_dim, 2 * att_out_dim], kernel_size, dropout),
        )

        self.finconv1 = nn.Conv1d(att_out_dim, 4 * att_out_dim, kernel_size=1)
        self.finconv2 = nn.Conv1d(4 * att_out_dim, out_dim, kernel_size=1)

        self.d_conv1 = nn.Sequential(
            nn.Conv1d(att_out_dim, att_out_dim, kernel_size=kernel_size,
                      padding='same', groups=att_out_dim, bias=False),
            nn.Conv1d(att_out_dim, att_out_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(att_out_dim),
            nn.GELU(),
        )
        self.d_conv2 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()

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

    def forward(self, x):
        # x: [B, in_dim, L] → [B, L, att_out_dim]
        x = x.transpose(1, 2)
        x = self.spatialattention1(x)
        x = self.spatialattention2(x)
        x = x.transpose(1, 2)
        x = self.FeatureEncoder(x)
        x = self.d_conv1(x)
        x = x.transpose(1, 2)
        return x
