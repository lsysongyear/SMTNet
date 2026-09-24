"""
EEG Conformer — adapted for frame-level VAD (voice activity detection).

Input:  (B, input_dim, T)          — multi-channel EEG time series
Output: (B, T)                      — frame-level voice probability logits

Architecture:
  Conv1d embedding → AvgPool (reduce T) → Transformer encoder →
  Linear head → Interpolate upsample → output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size, num_heads=10, drop_p=0.5,
                 forward_expansion=4, forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p),
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p),
            )),
        )


class EEGConformer(nn.Module):
    def __init__(self,
                 input_dim=57,
                 emb_size=128,
                 depth=6,
                 num_heads=8,
                 output_dim=1,
                 kernel_size=25,
                 pool_size=8,
                 dropout=0.1,
                 forward_expansion=4,
                 **kwargs):
        super().__init__()

        self.pool_size = pool_size

        self.spatial_attention = self._build_spatial_attention(input_dim)

        self.conv_embed = nn.Sequential(
            nn.Conv1d(input_dim, emb_size, kernel_size, padding='same'),
            nn.BatchNorm1d(emb_size),
            nn.GELU(),
            nn.AvgPool1d(pool_size, pool_size),
            nn.Dropout(dropout),
        )

        self.transformer = nn.Sequential(*[
            TransformerEncoderBlock(
                emb_size, num_heads=num_heads,
                drop_p=dropout, forward_expansion=forward_expansion,
                forward_drop_p=dropout)
            for _ in range(depth)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, emb_size * forward_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size * forward_expansion, output_dim),
        )

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
        # x: (B, input_dim, T)
        B, _, T_in = x.shape

        # Single shared spatial attention
        x_t = x.transpose(1, 2)                   # (B, T, input_dim)
        x_t = self.spatial_attention(x_t)
        x = x_t.transpose(1, 2)                   # (B, input_dim, T)

        x = self.conv_embed(x)                  # (B, emb_size, T_pooled)
        x = x.transpose(1, 2)                   # (B, T_pooled, emb_size)
        x = self.transformer(x)                 # (B, T_pooled, emb_size)
        x = self.head(x)                        # (B, T_pooled, output_dim)

        if x.dim() == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)                   # (B, T_pooled)
        x = x.unsqueeze(1)                      # (B, 1, T_pooled)
        x = F.interpolate(x, size=T_in, mode='linear', align_corners=False)
        return x.squeeze(1)                     # (B, T_in)
