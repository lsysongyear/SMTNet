# ============================================================================
# BrainNetwork.py - 核心模型定义：Brain_Magic_speech 及其子模块
# ============================================================================
# 模型架构概述：
#   Brain_Magic_speech 是一个多分支融合的脑信号回归/分类网络，专为 100Hz MEG 设计。
#
#   输入: [batch, 204（梯度计通道）, 1200（时间点，12s×100Hz）]
#   流程:
#     1. 空间注意力（MLP）: 将 204 个 MEG 通道映射到 128 维隐空间
#     2. 三个并行分支:
#        a) Feature Encoder: 5 层带 dilated conv 的 Feature_Block 堆叠（深层特征）
#        b) DeepMultiScaleBlock: 12 层多尺度时间卷积（多分辨率短期特征）
#        c) BiLSTM: 双向 LSTM 捕获长距离时序依赖
#     3. 特征融合: Concat(Encoder, MultiScale) + LSTM → 求和融合
#     4. 深度可分离卷积: 降维 256→128
#     5. 最终卷积: 128→512→1
#   输出: [batch, 1200] 逐时间点的语音概率
#
#   子模块:
#     - Feature_Block: 带空洞卷积和 GLU 激活的特征编码块
#     - ShortTermTemporalConv: 短期时间卷积（残差连接）
#     - MultiScaleTemporalLayer: 多尺度时间特征层（并行多 kernel）
#     - DeepMultiScaleBlock: 深层堆叠的多尺度时间卷积块
# ============================================================================

import torch.nn as nn
import torch


# ============================================================================
# Brain_Magic_speech - 核心模型类
# ============================================================================
class Brain_Magic_speech(nn.Module):
    """
    脑信号语音检测网络，针对 100Hz 采样率的 MEG 信号设计。

    关键设计理念：
    - 空间注意力：自动学习 204 通道之间的关联，降维到 128 维
    - 多分支并行：不同时间感受野的特征互补
      * Feature Encoder: 通过递增空洞率（dilation）指数级扩大感受野
      * MultiScale Block: 并行使用多个卷积核大小 [3,5,7,9] 捕获不同尺度模式
      * BiLSTM: 建模全局时间上下文
    - 特征融合：拼接 + 加法混合融合策略
    - 深度可分离卷积：高效降维，减少参数

    参数:
        input_dim: 输入通道数（默认 306, 实际使用 204 个梯度计通道）
        attention_dim: 空间注意力后的特征维度（默认 128）
        output_dim: 输出维度（默认 1，逐时间点的语音概率）
        kernel_size: Feature Encoder 中各卷积层的核大小（默认 25）
        depthwise_kernel: 深度可分离卷积的核大小（默认 19）
        dropout: Dropout 概率（默认 0.1）
    """
    def __init__(self, input_dim=306, attention_dim=128, output_dim=1,
                 kernel_size=5, depthwise_kernel=15, dropout=0.1):
        """
        初始化模型各子模块。

        Args:
            input_dim: 输入特征维度（如 306 个 MEG 通道）
            attention_dim: 空间注意力变换后的维度
            output_dim: 输出维度（如 1 表示语音概率）
            kernel_size: Feature Encoder 中的卷积核大小
            depthwise_kernel: 深度可分离卷积的核大小
            dropout: Dropout 丢弃概率
        """
        super().__init__()

        # ====================================================================
        # 1. 空间注意力模块（Spatial Attention / Channel Transformer）
        #    将输入 [batch, 204, T] 中的 204 个通道映射到 128 维隐空间
        #    使用 3 层 MLP: 204 → 816 → 128
        # ====================================================================
        hidden_dims = [input_dim, 4 * input_dim]  # [204, 816] 两层隐藏维度
        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))  # 全连接层
            layers.append(nn.GELU())                         # GELU 激活函数
            current_dim = hidden_dim

        # 最终投影到注意力维度
        layers.append(nn.Linear(current_dim, attention_dim))
        self.spatial_attention = nn.Sequential(*layers)

        # ====================================================================
        # 2. 深度特征编码器（Feature Encoder）
        #    5 层 Feature_Block 堆叠，每层使用递增的空洞卷积（dilated conv）
        #    通道配置: [128, 128, 128, 256]，最后一层经 GLU 后输出 128 维
        # ====================================================================
        encoder_channels = [attention_dim, attention_dim, attention_dim, 2 * attention_dim]
        self.feature_encoder = nn.Sequential(
            Feature_Block(1, encoder_channels, kernel_size, dropout),
            Feature_Block(2, encoder_channels, kernel_size, dropout),
            Feature_Block(3, encoder_channels, kernel_size, dropout),
            Feature_Block(4, encoder_channels, kernel_size, dropout),
            Feature_Block(5, encoder_channels, kernel_size, dropout),
        )

        # ====================================================================
        # 3. 多尺度时间卷积块（DeepMultiScaleBlock）
        #    12 层堆叠，每层并行使用 4 种 kernel [3,5,7,9]
        #    捕获不同时间尺度的短期特征模式
        # ====================================================================
        self.short_conv_block = DeepMultiScaleBlock(
            in_channels=attention_dim,
            out_channels=attention_dim,
            kernel_sizes=[3, 5, 7, 9],   # 4 种不同大小的卷积核
            num_layers=12,                 # 深度堆叠 12 层
            dropout=dropout
        )

        # ====================================================================
        # 4. 双向 LSTM 模块
        #    建模全局时间依赖关系，输出与输入同维度
        # ====================================================================
        self.lstm = nn.LSTM(
            input_size=attention_dim,    # 每个时间步输入的特征维度
            hidden_size=attention_dim,   # LSTM 隐状态维度
            num_layers=1,                # 单层 LSTM
            batch_first=True,            # 输入格式 (batch, seq, feature)
            bidirectional=True,          # 双向 LSTM，输出维度翻倍
            dropout=dropout
        )

        # ====================================================================
        # 5. 深度可分离卷积（Depthwise Separable Convolution）
        #    用于融合拼接后的特征并降维。
        #    步骤：Depthwise Conv → BN → GELU → Pointwise Conv → BN → GELU
        #    256 → 128
        # ====================================================================
        self.depthwise_conv = nn.Sequential(
            # -- 逐通道卷积（Depthwise） --
            nn.Conv1d(
                in_channels=2 * attention_dim,
                out_channels=2 * attention_dim,
                kernel_size=depthwise_kernel,
                padding='same',
                groups=2 * attention_dim,   # groups=in_channels: 每个通道独立卷积
                bias=False
            ),
            nn.BatchNorm1d(2 * attention_dim),
            nn.GELU(),

            # -- 逐点卷积（Pointwise）: 将分组卷积的输出混合 --
            nn.Conv1d(
                in_channels=2 * attention_dim,
                out_channels=attention_dim,
                kernel_size=1,              # 1×1 卷积实现通道间信息交互
                bias=False
            ),
            nn.BatchNorm1d(attention_dim),
            nn.GELU()
        )

        # ====================================================================
        # 6. 最终分类卷积
        #    128 → 512 → 1
        # ====================================================================
        self.final_conv1 = nn.Conv1d(attention_dim, 4 * attention_dim, kernel_size=1)
        self.final_conv2 = nn.Conv1d(4 * attention_dim, output_dim, kernel_size=1)


    def forward(self, x):
        """
        前向传播。

        Args:
            x: 输入张量 [batch, input_dim, sequence_length]
               例如 [batch, 204, 1200] 表示 204 通道 × 1200 时间点

        Returns:
            输出张量 [batch, sequence_length]，例如 [batch, 1200]
            每个时间点一个标量值（经过后续 sigmoid 后即为语音概率）
        """
        batch_size, input_dim, seq_len = x.shape

        # --- 步骤 1: 空间注意力 ---
        # 转置为 [batch, seq_len, input_dim] 以便对每个时间步的通道向量做 MLP
        x = x.transpose(1, 2)            # [batch, seq_len, 204]
        x = self.spatial_attention(x)     # [batch, seq_len, 128]
        x = x.transpose(1, 2)            # [batch, 128, seq_len]

        # --- 步骤 2: 三个分支并行处理 ---
        # 分支 A: 多尺度时间卷积（短期特征）
        short_features = self.short_conv_block(x)   # [batch, 128, seq_len]

        # 分支 B: 深层特征编码器（中期特征，空洞卷积）
        encoded_features = self.feature_encoder(x)  # [batch, 128, seq_len]

        # 分支 C: 双向 LSTM（长期时序依赖）
        lstm_features, _ = self.lstm(x.transpose(1,2))  # [batch, seq_len, 256]

        # --- 步骤 3: 特征融合 ---
        # 策略：拼接 Encoder 和 MultiScale 特征 → 与 LSTM 特征做加法融合
        # concat → [batch, 256, seq_len]
        # LSTM转置 → [batch, 256, seq_len]
        # 加法: 三个分支的信息通过加法融合
        fused_features = torch.cat([encoded_features, short_features], dim=1) + lstm_features.transpose(1,2)

        # --- 步骤 4: 深度可分离卷积（融合+降维） ---
        x = self.depthwise_conv(fused_features)  # [batch, 128, seq_len]

        # --- 步骤 5: 最终分类卷积 ---
        x = self.final_conv1(x)  # [batch, 512, seq_len]
        x = self.final_conv2(x)  # [batch, 1, seq_len]

        # 去掉通道维度
        output = x.squeeze(1)    # [batch, seq_len]

        return output


# ============================================================================
# Feature_Block - 特征编码块（空洞卷积 + 残差连接 + GLU）
# ============================================================================
class Feature_Block(nn.Module):
    """
    单层特征编码器模块。

    结构：
    Conv1d (dilated) → BN → GELU → Dropout
    → Conv1d (dilated) → BN → GELU → Dropout
    → Conv1d (dilated) → GLU（通道减半） → Dropout
    → 可选残差连接

    空洞率（dilation）根据层索引动态计算：
    - kernel_size=3 时: dilation ∈ {1,2,4,8,16}
    - 其他 kernel_size 时: dilation ∈ {1,2,4,8,16,32}
    使得不同层有不同大小的感受野，从局部到全局逐层扩大。
    """
    def __init__(self, layer_index, channels, kernel_size=10, dropout=0.5):
        """
        Args:
            layer_index: 层索引（从 1 开始），用于计算空洞率
            channels: 通道数列表 [in, hidden1, hidden2, out]
                     其中 out = 2 * hidden1（用于 GLU 门控后的通道减半）
            kernel_size: 卷积核大小
            dropout: Dropout 丢弃概率
        """
        super().__init__()
        self.layer_index = layer_index

        # ---- 根据层索引和核大小计算空洞率 ----
        # 目的：每层使用不同空洞率，实现指数级扩大的感受野
        if kernel_size == 3:
            dilation1 = int(2 ** ((2 * layer_index) % 5))      # 循环 1,2,4,8,16
            dilation2 = int(2 ** ((2 * layer_index + 1) % 5))  # 循环 2,4,8,16,1
        else:
            dilation1 = int(2 ** (layer_index % 6))             # 循环 1,2,4,8,16,32
            dilation2 = int(2 ** ((layer_index + 1) % 6))      # 循环 2,4,8,16,32,1

        # ---- 三个卷积层（前两个用 BatchNorm+GELU，最后一层用 GLU） ----
        self.conv1 = nn.Conv1d(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=kernel_size,
            padding='same',
            dilation=dilation1
        )
        self.conv2 = nn.Conv1d(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=kernel_size,
            padding='same',
            dilation=dilation2
        )
        self.conv3 = nn.Conv1d(
            in_channels=channels[2],
            out_channels=channels[3],   # 输出通道数 = 2 * hidden，供 GLU 使用
            kernel_size=kernel_size,
            padding='same',
            dilation=2
        )

        # ---- 归一化层 ----
        self.batch_norm1 = nn.BatchNorm1d(channels[1], eps=1e-05, momentum=0.1)
        self.batch_norm2 = nn.BatchNorm1d(channels[2], eps=1e-05, momentum=0.1)

        # ---- 激活函数和正则化 ----
        self.activation = nn.GELU()
        self.glu = nn.GLU(dim=-2)       # 沿通道维度执行门控线性单元（输出通道减半）
        self.dropout = nn.Dropout1d(dropout)


    def forward(self, x):
        """
        前向传播（支持可选的残差连接）。

        Args:
            x: [batch, channels[0], seq_len]

        Returns:
            [batch, channels[3]//2, seq_len]  (GLU 后通道减半)
        """
        input_x = x  # 保存输入用于残差连接

        # -- 第一个卷积块 --
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # -- 第二个卷积块 --
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation(x)
        x = self.dropout(x)

        # -- 第三个卷积块 + GLU 门控 --
        x = self.conv3(x)
        x = self.glu(x)       # GLU 将通道分成两半做门控，输出通道数减半
        x = self.dropout(x)

        # -- 残差连接（从第 2 层开始） --
        # 第 1 层输入输出维度相同（128→128），直接相加
        # 更深层由于 GLU 后维度可能变化，通过输入输出维度匹配性做残差
        if self.layer_index > 1:
            x = input_x + x

        return x


# ============================================================================
# ShortTermTemporalConv - 短期时间卷积（带残差连接）
# ============================================================================
class ShortTermTemporalConv(nn.Module):
    """
    单路短期时间卷积模块。

    结构：Conv1d → BatchNorm → ReLU → Dropout → 残差相加
    支持空洞卷积（dilation），用于扩大感受野而不增加参数。
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, dropout=0.2):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小（默认 3）
            stride: 步长（默认 1，保持时间分辨率不变）
            dilation: 空洞率（默认 1，即标准卷积）
            dropout: Dropout 概率（默认 0.2）
        """
        super(ShortTermTemporalConv, self).__init__()

        # 主卷积层（padding 计算确保输出序列长度不变）
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size + (kernel_size-1)*(dilation-1) - 1) // 2,  # 'same' padding
            dilation=dilation
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 残差连接：如果输入输出通道数不一致，用 1×1 卷积对齐维度
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None


    def forward(self, x):
        """
        输入: (batch_size, in_channels, seq_len)
        输出: (batch_size, out_channels, seq_len)
        """
        residual = x
        # 维度不匹配时用 1×1 卷积转换
        if self.residual is not None:
            residual = self.residual(residual)

        # 主路径：卷积 → BN → ReLU → Dropout
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        # 残差相加
        out = residual + out

        return out


# ============================================================================
# MultiScaleTemporalLayer - 多尺度时间特征层
# ============================================================================
class MultiScaleTemporalLayer(nn.Module):
    """
    并行多核卷积层：在同一层内使用多个不同大小的卷积核，
    每个核处理一部分通道，最后拼接起来。

    例如 kernel_sizes=[3,5,7,9]，out_channels=128：
    - 4 个 ShortTermTemporalConv 各输出 32 通道
    - 拼接后得到 128 通道
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7], dilation=1, dropout=0.2):
        super(MultiScaleTemporalLayer, self).__init__()
        # 确保 out_channels 能被 kernel 数量整除
        assert out_channels % len(kernel_sizes) == 0, "out_channels must be divisible by number of kernel sizes"
        self.convs = nn.ModuleList([
            ShortTermTemporalConv(
                in_channels,
                out_channels // len(kernel_sizes),  # 每个分支分配等量通道
                kernel_size=ks,
                dilation=dilation,
                dropout=dropout
            ) for ks in kernel_sizes
        ])


    def forward(self, x):
        """并行执行所有尺度的卷积，沿通道维拼接结果"""
        out = torch.cat([conv(x) for conv in self.convs], dim=1)
        return out


# ============================================================================
# DeepMultiScaleBlock - 深层堆叠的多尺度时间卷积块
# ============================================================================
class DeepMultiScaleBlock(nn.Module):
    """
    堆叠多层 MultiScaleTemporalLayer，构成深度多尺度特征提取器。

    特点：
    - 层间使用跳跃连接（skip connection）：从第 2 层开始，每层输入与前一层输出相加
    - 可选的空洞率增长（dilation_growth），但默认=1（不变）
    - 最终用 BatchNorm + ReLU 归一化
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7],
                 num_layers=3, dropout=0.2, dilation_growth=1):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 每层的输出通道数
            kernel_sizes: 使用的卷积核大小列表
            num_layers: 堆叠层数
            dropout: Dropout 概率
            dilation_growth: 空洞率增长因子（1 表示保持不变）
        """
        super(DeepMultiScaleBlock, self).__init__()

        self.layers = nn.ModuleList()
        self.skip_cons = nn.ModuleList()  # 跳跃连接（用于维度对齐）

        # 逐层构建
        for i in range(num_layers):
            # 第一层输入通道为 in_channels，后续为 out_channels
            layer_in_channels = in_channels if i == 0 else out_channels
            dilation = dilation_growth ** i  # 空洞率增长

            self.layers.append(
                MultiScaleTemporalLayer(
                    layer_in_channels,
                    out_channels,
                    kernel_sizes=kernel_sizes,
                    dilation=dilation,
                    dropout=dropout
                )
            )

            # 跳跃连接：如果输入输出通道不一致，用 1×1 卷积对齐
            if layer_in_channels != out_channels:
                self.skip_cons.append(nn.Conv1d(layer_in_channels, out_channels, 1))
            else:
                self.skip_cons.append(nn.Identity())

        # 最终融合和归一化
        self.final_norm = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU()


    def forward(self, x):
        """
        前向传播。
        从第 2 层开始，每层输出与经过 skip_conv 转换后的输入相加。
        """
        residual = x
        for i, (layer, skip_conv) in enumerate(zip(self.layers, self.skip_cons)):
            x = layer(x)
            if i > 0:  # 从第二层开始加入跳跃连接
                x = skip_conv(residual) + x
                residual = x  # 更新残差引用

        x = self.final_norm(x)
        x = self.final_relu(x)
        return x


# ============================================================================
# 测试代码：验证模型前向传播的输入输出维度
# ============================================================================
if __name__ == "__main__":
    device='cpu'
    # 创建模型实例（输入204个梯度计通道，注意力维度128）
    model = Brain_Magic_speech(input_dim=204, attention_dim=128, kernel_size=25, dropout=0.01).to(device)
    # 构造随机输入: [batch=1, channels=204, timesteps=1000]
    x = torch.rand(1, 204, 1000).to(device)
    y = model(x)
    # 输出 shape 应为 [1, 1000]（逐时间点的语音概率）
    print(y.shape)
