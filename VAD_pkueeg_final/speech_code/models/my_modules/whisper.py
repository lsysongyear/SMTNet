# ============================================================================
# whisper.py - OpenAI Whisper 风格的音频编码器（参考模型，当前未使用）
# ============================================================================
# 此文件实现了 OpenAI Whisper 模型的核心组件：
#   - MultiHeadAttention: 多头自注意力/交叉注意力机制
#   - ResidualAttentionBlock: 残差注意力块（Self-Attn + Cross-Attn + MLP）
#   - AudioEncoder: 音频编码器（Conv + PositionalEncoding + Transformer Blocks）
#   - 支持 scaled_dot_product_attention（PyTorch 内置的 SDPA 加速）
#
# 这是从 Whisper 项目中提取的参考实现，本项目主要使用 BrainNetwork.py
# 中的 Brain_Magic_speech 模型。此文件作为备选的模型架构保留。
# ============================================================================

import base64
import gzip
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


# ---- 尝试导入 PyTorch 的 scaled_dot_product_attention（1.13+ 版本）----
try:
    from torch.nn.functional import scaled_dot_product_attention
    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False


# ============================================================================
# ModelDimensions - 模型维度配置（Whisper 风格, 使用 dataclass 管理超参）
# ============================================================================
@dataclass
class ModelDimensions:
    """存储 Whisper 模型的各维度超参数"""
    n_mels: int            # Mel 频谱的频带数
    n_audio_ctx: int       # 音频上下文长度（时间帧数）
    n_audio_state: int     # 音频编码器的隐状态维度
    n_audio_head: int      # 注意力头数
    n_audio_layer: int     # Transformer 层数
    n_vocab: int           # 词汇表大小（文本解码器用）
    n_text_ctx: int        # 文本上下文长度
    n_text_state: int      # 文本解码器的隐状态维度
    n_text_head: int       # 文本解码器注意力头数
    n_text_layer: int      # 文本解码器层数


# ============================================================================
# 自定义 LayerNorm（始终使用 float32 精度，提高数值稳定性）
# ============================================================================
class LayerNorm(nn.LayerNorm):
    """精度感知的 LayerNorm：前向传播时先转为 float32 再转回原始 dtype"""
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


# ============================================================================
# 自定义 Linear 层（将权重转换为输入的 dtype，支持混合精度）
# ============================================================================
class Linear(nn.Linear):
    """精度感知的 Linear：确保权重与输入张量具有相同的数据类型"""
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


# ============================================================================
# 自定义 Conv1d 层（支持混合精度：权重重铸为输入的 dtype）
# ============================================================================
class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


# ============================================================================
# 正弦位置编码生成函数
# ============================================================================
def sinusoids(length, channels, max_timescale=10000):
    """
    生成正弦-余弦位置编码（Transformer 标准位置编码）。

    参数:
        length: 序列长度
        channels: 编码维度（必须为偶数）
        max_timescale: 最大时间尺度（控制频率范围）

    返回:
        torch.Tensor - [length, channels] 位置编码矩阵
    """
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    # 奇数位置用 sin，偶数位置用 cos
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


# ============================================================================
# 上下文管理器：临时禁用 SDPA（用于测试回退实现）
# ============================================================================
@contextmanager
def disable_sdpa():
    """临时禁用 scaled_dot_product_attention，使用手动实现的注意力"""
    prev_state = MultiHeadAttention.use_sdpa
    try:
        MultiHeadAttention.use_sdpa = False
        yield
    finally:
        MultiHeadAttention.use_sdpa = prev_state


# ============================================================================
# MultiHeadAttention - 多头注意力机制
# ============================================================================
class MultiHeadAttention(nn.Module):
    """
    多头注意力模块（支持自注意力和交叉注意力）。

    特性：
    - 自动选择 PyTorch SDPA（更快）或手动实现（兼容性更好）
    - 使用类级别标志 use_sdpa 全局控制
    - scale 使用 n_state/n_head 的 -0.25 次方（分散到 Q 和 K 各半）
    - 支持 KV 缓存（加速自回归推理）
    """
    use_sdpa = True  # 类级别标志：是否使用 PyTorch 的 SDPA 加速

    def __init__(self, n_state: int, n_head: int):
        """
        Args:
            n_state: 总的隐状态维度（必须能被 n_head 整除）
            n_head: 注意力头数
        """
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)


    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        """
        Args:
            x: 查询输入 [batch, seq_len, n_state]
            xa: 交叉注意力的 key/value 源（为 None 则是自注意力）
            mask: 注意力掩码
            kv_cache: KV 缓存字典（用于自回归解码）
        """
        q = self.query(x)

        # 如果有 KV 缓存且是交叉注意力，则复用缓存的 K/V（加速推理）
        if kv_cache is None or xa is None or self.key not in kv_cache:
            # 正常计算 K 和 V（自注意力用 x，交叉注意力用 xa）
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk


    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        计算缩放点积注意力（QK^T/sqrt(d) · V）。

        优先级：
        1. 如果 SDPA 可用 → 使用 PyTorch 内置的 scaled_dot_product_attention（更快）
        2. 否则 → 使用手动实现的注意力
        """
        n_batch, n_ctx, n_state = q.shape
        # 缩放因子：将 scale 分散到 Q 和 K 各半（乘积等于 1/sqrt(d_head)）
        scale = (n_state // self.n_head) ** -0.25
        # 重塑为 [batch, n_head, seq, d_head]
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa:
            # 使用 PyTorch 内置的 SDPA（支持 Flash Attention 等加速后端）
            a = scaled_dot_product_attention(
                q, k, v, is_causal=mask is not None and n_ctx > 1
            )
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = None
        else:
            # 手动实现注意力
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(q.dtype)
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            qk = qk.detach()

        return out, qk


# ============================================================================
# ResidualAttentionBlock - 残差注意力块
# ============================================================================
class ResidualAttentionBlock(nn.Module):
    """
    Transformer 的单个残差注意力块。

    结构（Pre-LN 风格）：
    x = x + SelfAttention(LayerNorm(x))
    x = x + CrossAttention(LayerNorm(x))   ← 可选
    x = x + MLP(LayerNorm(x))

    MLP 使用 4× 扩展比: n_state → 4*n_state → n_state
    """
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        """
        Args:
            n_state: 隐状态维度
            n_head: 注意力头数
            cross_attention: 是否包含交叉注意力层
        """
        super().__init__()

        # 自注意力
        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        # 交叉注意力（可选，用于 Encoder-Decoder 结构）
        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        # MLP 前馈网络（4× 扩展比 + GELU 激活）
        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)


    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        # 自注意力 + 残差
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        # 交叉注意力 + 残差（可选）
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        # MLP + 残差
        x = x + self.mlp(self.mlp_ln(x))
        return x


# ============================================================================
# AudioEncoder - Whisper 风格的音频编码器
# ============================================================================
class AudioEncoder(nn.Module):
    """
    Whisper 音频编码器。

    结构：
    1. Conv1d × 2（stem）：下采样 2×（n_ctx → n_ctx/2）
    2. 正弦位置编码
    3. n_layer 个 ResidualAttentionBlock（无交叉注意力）
    4. 最终 LayerNorm

    本项目中可作为备选的特征提取器替代 Brain_Magic_speech。
    """
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int
    ):
        """
        Args:
            n_mels: Mel 频带数（或等效的 MEG 通道数）
            n_ctx: 输入序列长度（时间帧数）
            n_state: Transformer 隐状态维度
            n_head: 注意力头数
            n_layer: Transformer 层数
        """
        super().__init__()
        # 两个卷积层：保持特征维度不变，第二层 stride=2 下采样
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        # 位置编码（注册为 buffer，随模型一起保存但不算可训练参数）
        self.register_buffer("positional_embedding", sinusoids(n_ctx//2, n_state))

        # Transformer 层堆叠
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)


    def forward(self, x: Tensor):
        """
        Args:
            x: [batch_size, n_mels, n_ctx] 如 Mel 频谱或 MEG 信号

        Returns:
            [batch_size, n_ctx//2, n_state] 编码后的特征
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))          # 下采样 2×
        x = x.permute(0, 2, 1)             # [batch, n_ctx//2, n_state]

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)  # 加入位置编码

        # 逐层 Transformer 处理
        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)  # 最终层归一化
        return x


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    device='cpu'
    model = AudioEncoder(n_mels=96, n_ctx=1000, n_state=128, n_head=8, n_layer=6).to(device)
    x = torch.rand(1, 96, 1000).to(device)
    y = model(x)
    print(y.shape)  # 预期输出: [1, 500, 128]
