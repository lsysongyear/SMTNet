"""
SHINE — Spatial-Hierarchical Iterative Neural Encoder for EEG VAD.

Adapted for PKU EEG (57ch): input_dim is configurable, BM and IMU receive
the same input_dim.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .BM import BM
from .IMU import IMU


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, \
            "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out


class Satt(nn.Module):
    def __init__(self, embed_size=64, heads=8, pool_stride=4):
        super(Satt, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.pool_stride = pool_stride

    def forward(self, E):
        # E: (B, L, C)  — 先降采样降低 self-attention 的 O(L²) 显存
        if self.pool_stride > 1:
            E_pooled = F.avg_pool1d(
                E.transpose(1, 2), kernel_size=self.pool_stride, stride=self.pool_stride
            ).transpose(1, 2)  # (B, L//s, C)
            M_s = self.attention(E_pooled, E_pooled, E_pooled)
            M_s = F.interpolate(
                M_s.transpose(1, 2), size=E.shape[1], mode='linear', align_corners=False
            ).transpose(1, 2)  # (B, L, C)
        else:
            M_s = self.attention(E, E, E)
        return M_s


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.convs1 = nn.Conv1d(64 * 3, 64, 1)
        self.convt1 = nn.Conv1d(64 * 4, 64 * 4, 12, groups=64 * 4)

        self.convs2 = nn.Conv1d(64 * 4, 64, 1)
        self.convt2 = nn.Conv1d(64 * 4, 64 * 4, 12, groups=64 * 4)

        self.convs3 = nn.Conv1d(64 * 4, 64, 1)
        self.convt3 = nn.Conv1d(64 * 4, 64 * 4, 12, groups=64 * 4)

        self.convs4 = nn.Conv1d(64 * 4, 64, 1)
        self.convt4 = nn.Conv1d(64 * 4, 64 * 4, 12, groups=64 * 4)

        self.conv5 = nn.Conv1d(64 * 4, 64 * 2, 12)

        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(64 * 2)
        self.norm3 = nn.LayerNorm(64 * 4)

        self.relu = nn.LeakyReLU()
        self.pad = nn.ZeroPad2d((0, 0, 0, 11))

    def forward(self, x):
        eeg = x

        x = x.permute(0, 2, 1)
        x = self.convs1(x)
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = self.relu(x)
        x = torch.cat((eeg, x), dim=2)

        x = x.permute(0, 2, 1)
        x = self.convt1(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pad(x)

        x = x.permute(0, 2, 1)
        x = self.convs2(x)
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = self.relu(x)
        x = torch.cat((eeg, x), dim=2)

        x = x.permute(0, 2, 1)
        x = self.convt2(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pad(x)

        x = x.permute(0, 2, 1)
        x = self.convs3(x)
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = self.relu(x)
        x = torch.cat((eeg, x), dim=2)

        x = x.permute(0, 2, 1)
        x = self.convt3(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pad(x)

        x = x.permute(0, 2, 1)
        x = self.convs4(x)
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = self.relu(x)
        x = torch.cat((eeg, x), dim=2)

        x = x.permute(0, 2, 1)
        x = self.convt4(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pad(x)

        x = x.permute(0, 2, 1)
        x = self.conv5(x)
        x = x.permute(0, 2, 1)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.pad(x)
        return x


class OutputContext(nn.Module):
    def __init__(self):
        super(OutputContext, self).__init__()
        self.pad = nn.ZeroPad2d((0, 0, 63, 0))
        self.conv = nn.Conv1d(64, 64, 64)
        self.norm = nn.LayerNorm(64)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.pad(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.relu(x)
        return x


class SHINE(nn.Module):
    """
    SHINE model for frame-level EEG VAD.

    Args:
        input_dim:   EEG channels (57 for PKU EEG after ch_transform)
        emb:         embedding dimension (64)
        nb_blocks:   number of iterative refinement blocks (6)
        output_dim:  output dimension (1 for binary VAD)
        dropout:     dropout rate (for BM submodule)
        kernel_size: conv kernel size (for BM submodule)
    """
    def __init__(self, input_dim=57, emb=64, nb_blocks=6, output_dim=1,
                 dropout=0.1, kernel_size=25):
        super(SHINE, self).__init__()
        self.spatialattention1 = nn.Linear(input_dim, input_dim)
        self.spatialattention2 = nn.Linear(input_dim, emb)
        self.extractor = Extractor()
        self.satt = Satt(embed_size=emb)
        self.output_context = OutputContext()
        self.nb_blocks = nb_blocks
        self.linear_layer = nn.Linear(emb * 2, emb)
        self.fc = nn.Linear(emb * 5, emb)
        self.fc2 = nn.Linear(emb, output_dim)

        self.relu = nn.LeakyReLU()

        self.BM = BM(in_dim=input_dim, att_out_dim=emb, out_dim=output_dim,
                      kernel_size=kernel_size, dropout=dropout)
        self.IMU = IMU(input_channels=input_dim, out_channels=output_dim)

        self.cnn = nn.Conv1d(emb * 3, emb, 101, padding=50)
        self.norm = nn.LayerNorm(emb)

        self.lstm = nn.LSTM(emb, emb, batch_first=True, bidirectional=True)
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
        # x: [B, input_dim, L]
        bm_out = self.BM(x)       # [B, L, emb]
        imu_out = self.IMU(x)     # [B, L, emb]

        eeg = x.permute(0, 2, 1)  # [B, L, input_dim]
        x = self.spatialattention1(eeg)
        eeg_x = self.relu(x)
        eeg_x = self.spatialattention2(eeg_x)  # [B, L, emb]

        eeg_out = torch.zeros_like(eeg_x)
        eeg_out_att = torch.zeros_like(eeg_x)

        for i in range(self.nb_blocks):
            eeg_out = torch.cat((eeg_x, eeg_out, eeg_out_att), dim=2)
            eeg_out = self.extractor(eeg_out)
            eeg_out = self.linear_layer(eeg_out)
            eeg_out = self.output_context(eeg_out)
            eeg_out_att = self.satt(eeg_out)
            eeg_out_att = eeg_out_att * eeg_out

        eeg_out = torch.cat((eeg_out, bm_out, imu_out), dim=2)
        eeg_out_raw = eeg_out

        eeg_out = self.cnn(eeg_out.permute(0, 2, 1))
        eeg_out = eeg_out.permute(0, 2, 1)
        eeg_out = self.norm(eeg_out)

        eeg_out, _ = self.lstm(eeg_out)
        eeg_out = self.relu(eeg_out)

        eeg_out = torch.cat((eeg_out, eeg_out_raw), dim=2)

        eeg_out = self.fc(eeg_out)
        eeg_out = self.relu(eeg_out)
        eeg_out = self.fc2(eeg_out)

        return eeg_out.squeeze(dim=2)  # [B, L]
