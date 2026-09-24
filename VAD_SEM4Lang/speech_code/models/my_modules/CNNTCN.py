"""PNPL 2025 CNN+TCN with the project's subject-indexed spatial projection.

The backbone is the TCNBlock/CNN_TCN architecture from Parameter Team's
public model.py (2025-06-09), matching the pinned parameter-search adapter.
Only the inference-equivalent large-dilation optimization is added here.
"""

from torch import nn
from torch.nn import functional as F

from .public_baseline_utils import (
    SubjectSpatialProjection,
    dropout_probability,
    positive_integer,
)


class ExactLargeDilationConv1d(nn.Conv1d):
    """Avoid allocating padding when both outside taps multiply only zeros."""

    def forward(self, x):
        if (self.kernel_size == (3,) and self.stride == (1,)
                and self.padding == self.dilation
                and self.padding_mode == "zeros"
                and self.dilation[0] >= x.shape[-1]):
            return F.conv1d(x, self.weight[:, :, 1:2], self.bias, groups=self.groups)
        return super().forward(x)


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = ExactLargeDilationConv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.conv2 = ExactLargeDilationConv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(p=dropout)
        self.downsample = (nn.Conv1d(in_channels, out_channels, 1)
                           if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        residual = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        return self.relu(out + residual)


class CNN_TCN(nn.Module):
    """Pinned backbone; input and TCN channel widths are tied to model_dim."""

    def __init__(self, input_dim, model_dim, tcn_layers, dropout):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, model_dim, kernel_size=1, padding=0)
        self.batch_norm = nn.Identity()
        self.conv_dropout = nn.Dropout(p=dropout)
        self.tcn = nn.Sequential(*[
            TCNBlock(model_dim, model_dim, dilation=2 ** i, dropout=dropout)
            for i in range(tcn_layers)
        ])
        self.fc = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.conv_dropout(self.batch_norm(self.conv(x)))
        x = self.tcn(x).permute(0, 2, 1)
        return self.fc(x).squeeze(-1)


class CNNTCN(nn.Module):
    """Return raw frame logits [B,T] for inputs [B,C,T] and subject IDs [B].

    model_dim controls both the initial CNN and TCN channel widths. The input
    projection has no BatchNorm; each TCN block retains its two BatchNorms.
    Sigmoid and temporal smoothing belong to ClassificationModule, not here.
    """

    requires_subject_ids = True

    def __init__(self, input_dim=64, n_subjects=25, model_dim=100,
                 tcn_layers=20, dropout=0.3):
        super().__init__()
        positive_integer("model_dim", model_dim)
        positive_integer("tcn_layers", tcn_layers)
        dropout_probability("dropout", dropout)
        self.spatial_projection = SubjectSpatialProjection(input_dim, n_subjects)
        self.model = CNN_TCN(input_dim, model_dim, tcn_layers, dropout)

    def forward(self, x, subject_ids=None):
        return self.model(self.spatial_projection(x, subject_ids))
