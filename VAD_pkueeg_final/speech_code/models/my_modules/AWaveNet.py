import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x):
    return x * torch.sigmoid(x)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)


class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        return self.conv(x)


class Residual_block(nn.Module):
    def __init__(self, res_channels, skip_channels, dilation):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels
        self.dilated_conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3, dilation=dilation)
        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)

    def forward(self, input_data):
        x = input_data
        h = x
        B, C, L = x.shape
        assert C == self.res_channels
        h = self.dilated_conv_layer(h)
        out = torch.tanh(h[:, :self.res_channels, :]) * torch.sigmoid(h[:, self.res_channels:, :])
        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)
        return (x + res) * math.sqrt(0.5), skip


class Residual_group(nn.Module):
    def __init__(self, res_channels, skip_channels, num_res_layers, dilation_cycle):
        super(Residual_group, self).__init__()
        self.num_res_layers = num_res_layers
        self.residual_blocks = nn.ModuleList()
        for n in range(self.num_res_layers):
            self.residual_blocks.append(Residual_block(res_channels, skip_channels,
                                                       dilation=2 ** (n % dilation_cycle)))

    def forward(self, input_data):
        h = input_data
        skip = 0
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n](h)
            skip = skip + skip_n
        return skip * math.sqrt(1.0 / self.num_res_layers)


class AWaveNet(nn.Module):
    def __init__(self, output_dim=1,
                 res_channels=64, skip_channels=64,
                 num_res_layers=36, dilation_cycle=12,
                 input_dim=None, n_subjects=25):
        super(AWaveNet, self).__init__()

        self.n_subjects = n_subjects
        self._input_dim = input_dim if input_dim is not None else res_channels

        self.spatial_attentions = nn.ModuleList([
            self._build_spatial_attention(self._input_dim)
            for _ in range(n_subjects)
        ])

        if input_dim is not None and input_dim != res_channels:
            self.input_proj = nn.Conv1d(input_dim, res_channels, kernel_size=1)
        else:
            self.input_proj = None

        self.residual_layer = Residual_group(res_channels=res_channels,
                                             skip_channels=skip_channels,
                                             num_res_layers=num_res_layers,
                                             dilation_cycle=dilation_cycle)
        self.fc = nn.Linear(skip_channels, output_dim)

    @staticmethod
    def _build_spatial_attention(input_dim):
        hidden = 4 * input_dim
        return nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, input_dim),
            nn.GELU(),
        )

    def forward(self, input_data, subject_ids=None):
        # input_data: (B, C, L)
        B = input_data.size(0)
        x = input_data.transpose(1, 2)            # (B, L, C)

        if subject_ids is not None:
            outs = []
            for i in range(B):
                idx = subject_ids[i].item()
                outs.append(self.spatial_attentions[idx](x[i:i+1]))
            x = torch.cat(outs, dim=0)
        else:
            x = self.spatial_attentions[0](x)

        x = x.transpose(1, 2)                     # (B, C, L)

        if self.input_proj is not None:
            x = self.input_proj(x)
        x = self.residual_layer(x)
        x = x.transpose(-1, -2)
        x = self.fc(x)
        return x
