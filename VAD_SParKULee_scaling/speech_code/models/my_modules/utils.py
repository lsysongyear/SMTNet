# ============================================================================
# utils.py (my_modules) - 模型构建工厂函数 + 优化器工厂 + 损失工厂
# ============================================================================
# 内容概述：
#   1. modules_from_config: 从 YAML 配置动态构建模型层列表
#      - 支持 16 种层类型，包括线性层、卷积、激活、归一化、GRU、自定义模型等
#      - 核心的"配置到代码"桥梁
#   2. optimizer_from_config: 从配置创建优化器（支持学习率调度器）
#      - 支持 Adam / AdamW / SGD
#      - 支持 Linear / Step / Cosine / CosineWarmRestarts 调度器
#   3. loss_fn_from_config: 从配置创建损失函数
#      - 支持 CrossEntropyLoss 和 WeightedMSEByLabel
#   4. ResnetBlock: 可配置的残差连接块
#   5. WeightedMSEByLabel: 按标签加权的 MSE 损失
# ============================================================================

from torch.nn import Conv1d, ELU
from torch.nn import Softsign, GRU, Linear, ReLU, Sigmoid, GELU, BatchNorm1d
from torch import nn
import torch
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import LinearLR, StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from speech_code.models.util_layers import Permute
from torch.nn import LayerNorm
from speech_code.models.average_groups import AverageGroups
from speech_code.models.my_modules.BrainNetwork import Brain_Magic_speech
from speech_code.models.my_modules.model_EEG import ConcatCovNet, TransformerEncoder, CNN_baseline
from speech_code.models.my_modules.AWaveNet import AWaveNet
from speech_code.models.my_modules.CNNLSTM import CNNLSTM
from speech_code.models.my_modules.DilatedConv import DilatedConv
from speech_code.models.my_modules.VLAAI import VLAAI
from speech_code.models.my_modules.BrainNetwork_v1 import Brain_Magic_speech_v1
from speech_code.models.my_modules.BrainNetwork_v4 import Brain_Magic_speech_v1 as Brain_Magic_speech_v4
from speech_code.models.my_modules.BrainNetwork_v7 import Brain_Magic_speech_v1 as Brain_Magic_speech_v7
from speech_code.models.my_modules.SHINE import SHINE
from speech_code.models.conformer import EEGConformer
from speech_code.models.my_modules.BrainNetwork_ablation import (
    BrainMagic_NoFE, BrainMagic_NoMS, BrainMagic_NoBiLSTM,
    BrainMagic_NoFE_NoMS, BrainMagic_NoFE_NoBiLSTM, BrainMagic_NoMS_NoBiLSTM
)
from speech_code.models.my_modules.BrainNetwork_v7_ablation import (
    BrainMagic_NoSubjectAttn, BrainMagic_NoShortConv, BrainMagic_NoFeatureEncoder
)


# ============================================================================
# modules_from_config - 模型构建工厂
# ============================================================================
def modules_from_config(modules: list[tuple[str, dict]]):
    """
    从 YAML 配置字典动态构建 PyTorch 层/模块列表。

    支持的模块类型映射：
    - 线性层: "linear" → nn.Linear
    - 卷积层: "conv1d" → nn.Conv1d
    - 激活函数: "softsign" / "relu" / "elu" / "sigmoid" / "gelu"
    - 正则化: "dropout" / "dropout1d" / "batch_norm1d" / "layer_norm"
    - RNN: "gru"
    - 维度变换: "flatten" / "permute"
    - 自定义模块: "resnet_block" / "average_groups" / "brain_magic_speech"

    参数:
        modules: dict - 配置字典，如 {"brain_magic_speech": {...params...}}

    返回:
        list[nn.Module] - PyTorch 模块列表
    """
    modules_list = []
    for module in modules:
        module_type = module
        config = modules[module_type]
        # 根据模块类型名称创建对应的 PyTorch 层
        if module_type == "linear":
            module = Linear(**config)
        elif module_type == "conv1d":
            module = Conv1d(**config)
        elif module_type == "softsign":
            module = Softsign()
        elif module_type == "relu":
            module = ReLU()
        elif module_type == "elu":
            module = ELU()
        elif module_type == "sigmoid":
            module = Sigmoid()
        elif module_type == "gelu":
            module = GELU()
        elif module_type == "dropout":
            module = nn.Dropout(**config)
        elif module_type == "dropout1d":
            module = nn.Dropout1d(**config)
        elif module_type == "gru":
            module = GRU(**config)
        elif module_type == "flatten":
            module = nn.Flatten()
        elif module_type == "batch_norm1d":
            module = BatchNorm1d(**config)
        elif module_type == "permute":
            module = Permute(**config)
        elif module_type == "resnet_block":
            # 残差连接块：内部包含多层子模块，输入输出通过加法连接
            module = ResnetBlock(**config)
        elif module_type == "layer_norm":
            module = LayerNorm(**config)
        elif module_type == "average_groups":
            # 通道分组平均层：将输入通道分组后取平均
            module = AverageGroups(**config)
        elif module_type == "brain_magic_speech":
            # 本项目核心模型：脑信号语音检测网络
            module = Brain_Magic_speech(**config)
        elif module_type == "brain_magic_speech_v1":
            # WaveNet 门控 + 跳跃累积版 VAD 模型（无 BiLSTM）
            module = Brain_Magic_speech_v1(**config)
        elif module_type == "brain_magic_speech_v4":
            module = Brain_Magic_speech_v4(**config)
        elif module_type == "brain_magic_speech_v7":
            module = Brain_Magic_speech_v7(**config)
        elif module_type == "shine":
            # SHINE: 空间注意力 + 迭代精炼 + BM/IMU 双分支 + BiLSTM
            module = SHINE(**config)
        elif module_type == "concat_cov_net":
            # DenseNet 风格的密集连接卷积网络基线
            module = ConcatCovNet(**config)
        elif module_type == "transformer_encoder":
            # CNN + Transformer 混合编码器基线
            module = TransformerEncoder(**config)
        elif module_type == "cnn_baseline":
            # CNN 基线模型（5层 Conv1D + stride-2 降采样）
            module = CNN_baseline(**config)
        elif module_type == "awavenet":
            # AWaveNet: 膨胀残差卷积 + gated-tanh + skip connections
            module = AWaveNet(**config)
        elif module_type == "cnn_lstm":
            # CNNLSTM: CNN 特征提取 + 逐通道 LSTM + FC
            module = CNNLSTM(**config)
        elif module_type == "dilated_conv":
            # DilatedConv: 膨胀卷积堆叠（Accou et al 基线）
            module = DilatedConv(**config)
        elif module_type == "vlaai":
            # VLAAI: 多块堆叠 Conv 提取器 + skip connections
            module = VLAAI(**config)
        elif module_type == "brain_magic_nofe":
            module = BrainMagic_NoFE(**config)
        elif module_type == "brain_magic_noms":
            module = BrainMagic_NoMS(**config)
        elif module_type == "brain_magic_nobilstm":
            module = BrainMagic_NoBiLSTM(**config)
        elif module_type == "brain_magic_nofe_noms":
            module = BrainMagic_NoFE_NoMS(**config)
        elif module_type == "brain_magic_nofe_nobilstm":
            module = BrainMagic_NoFE_NoBiLSTM(**config)
        elif module_type == "brain_magic_noms_nobilstm":
            module = BrainMagic_NoMS_NoBiLSTM(**config)
        elif module_type == "brain_magic_no_subject_attn":
            module = BrainMagic_NoSubjectAttn(**config)
        elif module_type == "brain_magic_no_short_conv":
            module = BrainMagic_NoShortConv(**config)
        elif module_type == "brain_magic_no_feature_encoder":
            module = BrainMagic_NoFeatureEncoder(**config)
        elif module_type == "eeg_conformer":
            module = EEGConformer(**config)
        else:
            raise ValueError(f"Unsupported module_type: {module_type}")
        modules_list.append(module)
    return modules_list


# ============================================================================
# optimizer_from_config - 优化器工厂函数
# ============================================================================
def optimizer_from_config(parameters, config):
    """
    从配置字典创建优化器（可选的学习率调度器）。

    支持的优化器：
    - adam: Adam
    - adamw: AdamW（带权重衰减的 Adam）
    - sgd: SGD

    支持的学习率调度器（可选）：
    - linear: LinearLR（线性衰减）
    - step: StepLR（阶梯衰减）
    - cosine: CosineAnnealingLR（余弦退火）
    - cosine_warm: CosineAnnealingWarmRestarts（带 warmup 重启的余弦退火）

    参数:
        parameters: 模型参数迭代器（model.parameters()）
        config: 优化器配置字典

    返回:
        optimizer 或 {"optimizer": optimizer, "lr_scheduler": scheduler}
    """
    # --- 创建优化器 ---
    if (config["name"] == "adam"):
        optimizer = Adam(parameters, **config["config"])
    elif (config["name"] == "adamw"):
        optimizer = AdamW(parameters, **config["config"])
    elif (config["name"] == "sgd"):
        optimizer = SGD(parameters, **config["config"])
    else:
        raise ValueError(f"Unsupported optimizer: "
                         + f"{config['name']}")

    # --- 可选：创建学习率调度器 ---
    if ("scheduler" in config):
        if (config["scheduler"] == "linear"):
            scheduler = LinearLR(optimizer,
                                 **config["scheduler_config"])
        elif (config["scheduler"] == "step"):
            scheduler = StepLR(optimizer,
                               **config["scheduler_config"])
        elif (config["scheduler"] == "cosine"):
            scheduler = CosineAnnealingLR(optimizer,
                                          **config["scheduler_config"])
        elif (config["scheduler"] == "cosine_warm"):
            scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                    **config["scheduler_config"])
        else:
            raise ValueError(f"Unsupported scheduler: ",
                             config["scheduler"])
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

    return optimizer


# ============================================================================
# loss_fn_from_config - 损失函数工厂
# ============================================================================
def loss_fn_from_config(loss_config):
    """
    从配置创建损失函数。

    支持：
    - cross_entropy: 标准交叉熵（多分类）
    - mse: 均方误差（回归/二分类），支持类别权重

    当 "weight" 参数存在时，返回 WeightedMSEByLabel（按标签加权 MSE）
    """
    if loss_config["name"] == "cross_entropy":
        if ("config" not in loss_config or loss_config["config"] is None):
            return nn.CrossEntropyLoss()
        return nn.CrossEntropyLoss(**loss_config["config"])
    elif loss_config["name"] == "mse":
        # 判断是否提供了类别权重
        if "config" not in loss_config or loss_config["config"] is None:
            return nn.MSELoss()
        else:
            config = loss_config["config"]
            if "weight" in config:
                # 从配置列表转换为 WeightedMSEByLabel 所需参数
                weight_tensor = torch.tensor(config["weight"], dtype=torch.float32)
                return WeightedMSEByLabel(weight_0=weight_tensor[0].item(),
                                         weight_1=weight_tensor[1].item())
            else:
                return nn.MSELoss(**config)
    else:
        raise ValueError(f"Unsupported loss: {loss_config['name']}")


# ============================================================================
# ResnetBlock - 可配置的残差连接块
# ============================================================================
class ResnetBlock(nn.Module):
    """
    残差连接块：输入 x 经过一系列子模块处理后，与原始输入相加。

    公式: output = F(x) + x
    其中 F(x) 是内部子模块的组合。

    用途：
    - 缓解梯度消失问题
    - 让网络更容易学习恒等映射
    """
    def __init__(self, model_config: list[tuple[str, dict]]):
        super().__init__()
        self.model_config = model_config
        # 内部子模块列表
        self.module_list = nn.ModuleList()
        self.module_list.extend(modules_from_config(model_config))


    def forward(self, x):
        x_residual = x  # 保存原始输入
        for module in self.module_list:
            x = module(x)
        # 残差相加（要求输入输出形状一致）
        return x + x_residual


# ============================================================================
# WeightedMSEByLabel - 按标签类别的加权均方误差损失
# ============================================================================
class WeightedMSEByLabel(nn.Module):
    """
    按标签加权的 MSE 损失。

    对标签=0（静音）和标签=1（语音）的样本分别施加不同权重，
    用于缓解类别不平衡问题。

    用法示例：
        loss_fn = WeightedMSEByLabel(weight_0=1.0, weight_1=3.0)
        # 语音样本的错误惩罚是静音样本的 3 倍
    """
    def __init__(self, weight_0=1.0, weight_1=1.0, reduction='mean'):
        """
        Args:
            weight_0: 标签为 0 的样本的 MSE 权重
            weight_1: 标签为 1 的样本的 MSE 权重
            reduction: 'mean' | 'sum' | 'none'
        """
        super().__init__()
        self.weight_0 = weight_0
        self.weight_1 = weight_1
        self.reduction = reduction


    def forward(self, input, target):
        # 逐元素 squared error: (pred - target)^2
        mse = (input - target) ** 2

        # 根据目标标签值分配权重: target==1 → weight_1, target==0 → weight_0
        weights = torch.where(target == 1, self.weight_1, self.weight_0)
        weighted_mse = weights * mse

        if self.reduction == 'mean':
            return weighted_mse.mean()
        elif self.reduction == 'sum':
            return weighted_mse.sum()
        else:
            return weighted_mse
