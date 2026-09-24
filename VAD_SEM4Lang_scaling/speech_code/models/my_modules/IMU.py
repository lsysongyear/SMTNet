"""
IMU (Incremental Modeling Unit) — WaveNet-style dilated conv residual stack.
Adapted from NIPS2025 for VAD: input_channels is configurable (default 57 for PKU EEG).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x):
    return x * torch.sigmoid(x)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)


class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        return self.conv(x)


class Residual_block(nn.Module):
    def __init__(self, res_channels, skip_channels, dilation):
        super().__init__()
        self.res_channels = res_channels
        self.dilated_conv_layer = Conv(res_channels, 2 * res_channels,
                                       kernel_size=3, dilation=dilation)

        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)

        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)

    def forward(self, input_data):
        x = input_data
        B, C, L = x.shape
        assert C == self.res_channels

        h = self.dilated_conv_layer(x)

        # gated-tanh nonlinearity
        out = torch.tanh(h[:, :self.res_channels, :]) * torch.sigmoid(h[:, self.res_channels:, :])

        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip


class Residual_group(nn.Module):
    def __init__(self, res_channels, skip_channels, num_res_layers, dilation_cycle):
        super().__init__()
        self.num_res_layers = num_res_layers
        self.residual_blocks = nn.ModuleList()
        for n in range(self.num_res_layers):
            self.residual_blocks.append(
                Residual_block(res_channels, skip_channels,
                               dilation=2 ** (n % dilation_cycle)))

    def forward(self, input_data):
        x = input_data
        h = x
        skip = 0
        for block in self.residual_blocks:
            h, skip_n = block(h)
            skip = skip + skip_n
        return skip * math.sqrt(1.0 / self.num_res_layers)


class IMU(nn.Module):
    def __init__(self, input_channels=57, out_channels=1, sub_num=12,
                 res_channels=128, skip_channels=128,
                 num_res_layers=36, dilation_cycle=12,
                 sub_dim=128):
        super().__init__()
        self.spatialattention1 = nn.Linear(input_channels, input_channels)
        self.spatialattention2 = nn.Linear(input_channels, 128)
        self.relu = nn.LeakyReLU()

        self.residual_layer = Residual_group(
            res_channels=res_channels, skip_channels=skip_channels,
            num_res_layers=num_res_layers, dilation_cycle=dilation_cycle)

        self.final_conv = nn.Sequential(
            Conv(skip_channels, skip_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(skip_channels, out_channels, kernel_size=1))

        self.proj1 = nn.Linear(res_channels, res_channels)
        self.fc = nn.Linear(128, 64)

    def forward(self, input_data):
        # input_data: [B, input_channels, L] → output: [B, L, 64]
        x = input_data
        x = x.transpose(-1, -2)
        x = self.spatialattention1(x)
        x = self.relu(x)
        x = self.spatialattention2(x)
        x = x.transpose(-1, -2)
        x = self.residual_layer(x)
        x = x.transpose(-1, -2)
        x = self.fc(x)
        return x
