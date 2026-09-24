import torch.nn as nn
import torch


class Extractor(nn.Module):
    def __init__(self, filters=(256, 256, 256, 128, 128), kernels=(8,) * 5, input_channels=64):
        super(Extractor, self).__init__()
        self.layers = nn.ModuleList()
        for filter_, kernel in zip(filters, kernels):
            self.layers.append(nn.Conv1d(input_channels, filter_, kernel))
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.ConstantPad1d((0, kernel - 1), 0))
            input_channels = filter_

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = x.permute(0, 2, 1)
            if isinstance(layer, nn.LeakyReLU):
                norm_shape = x.shape[-1]
                x = nn.LayerNorm(norm_shape).to(x.device)(x)
            x = x.permute(0, 2, 1)
        return x


class OutputContext(nn.Module):
    def __init__(self, filter_=64, kernel=32, input_channels=64):
        super(OutputContext, self).__init__()
        self.pad = nn.ConstantPad1d((kernel - 1, 0), 0)
        self.conv = nn.Conv1d(input_channels, filter_, kernel)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        norm_shape = x.shape[1:]
        x = nn.LayerNorm(norm_shape).to(x.device)(x)
        return self.activation(x)


class VLAAI(nn.Module):
    def __init__(self, input_channels=64, output_dim=1, nb_blocks=4,
                 use_skip=True, n_subjects=25):
        super(VLAAI, self).__init__()

        self.n_subjects = n_subjects
        self.nb_blocks = nb_blocks
        self.use_skip = use_skip
        self.input_channels = input_channels

        self.spatial_attentions = nn.ModuleList([
            self._build_spatial_attention(input_channels)
            for _ in range(n_subjects)
        ])

        self.extractor = Extractor(input_channels=input_channels)
        self.dense = nn.Linear(128, input_channels)
        self.output_context = OutputContext(filter_=input_channels, input_channels=input_channels)
        self.final_dense = nn.Linear(input_channels, output_dim)

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
        B = x.size(0)
        x = x.transpose(1, 2)                   # (B, T, C)

        if subject_ids is not None:
            outs = []
            for i in range(B):
                idx = subject_ids[i].item()
                outs.append(self.spatial_attentions[idx](x[i:i+1]))
            x = torch.cat(outs, dim=0)
        else:
            x = self.spatial_attentions[0](x)

        x = x.transpose(1, 2)                   # (B, C, T)

        for _ in range(self.nb_blocks):
            skip = x if self.use_skip else 0
            x = self.extractor(x + skip)
            x = x.transpose(1, 2)
            x = self.dense(x)
            x = x.transpose(1, 2)
            x = self.output_context(x)
        x = x.transpose(1, 2)
        x = self.final_dense(x)
        return x
