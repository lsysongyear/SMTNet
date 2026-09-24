import torch
import torch.nn as nn

class DilatedConv(nn.Module):
    def __init__(self,
                 input_channels=64,
                 output_dim=1,
                 sub_num=12,
                 n_layers=3,
                 kernel_size=3,
                 spatial_filters=64,
                 dilation_filters=64,
                 sub_dim=128):
        super().__init__()
        self.n_layers = n_layers
        self.input_channels = input_channels

        self.spatial_attention = self._build_spatial_attention(input_channels)

        self.eeg_projections = nn.ModuleList(
            [nn.Conv1d(input_channels, dilation_filters, kernel_size=kernel_size, padding='same')] +
            [nn.Conv1d(dilation_filters, dilation_filters, kernel_size=kernel_size,
                       dilation=kernel_size ** i, padding='same') for i in range(1, n_layers)]
        )
        self.activation = nn.ReLU()
        self.linear = nn.Linear(dilation_filters, output_dim)

    @staticmethod
    def _build_spatial_attention(input_dim):
        hidden = 4 * input_dim
        return nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, input_dim),
            nn.GELU(),
        )

    def forward(self, x, subject_ids=None):
        # x: (B, C, T)
        x = x.transpose(1, 2)                   # (B, T, C)
        x = self.spatial_attention(x)            # single shared attention
        x = x.transpose(1, 2)                   # (B, C, T)

        for i in range(self.n_layers):
            x = self.eeg_projections[i](x)
            x = self.activation(x)
        x = x.permute(0, 2, 1)                  # (B, T, dilation_filters)
        x = self.linear(x)                      # (B, T, output_dim)
        return x
