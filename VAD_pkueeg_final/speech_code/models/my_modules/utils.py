import torch
from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    LinearLR, StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts,
)
from speech_code.models.my_modules.BrainNetwork_v7 import Brain_Magic_speech_v1 as BrainMagicV7
from speech_code.models.my_modules.BrainNetwork_v7_ablation import (
    BrainMagic_NoSubjectAttn, BrainMagic_NoShortConv, BrainMagic_NoFeatureEncoder,
)
from speech_code.models.my_modules.CNNLSTM import CNNLSTM
from speech_code.models.my_modules.DilatedConv import DilatedConv
from speech_code.models.my_modules.AWaveNet import AWaveNet
from speech_code.models.my_modules.CNNTCN import CNNTCN
from speech_code.models.conformer import EEGConformer


def modules_from_config(modules):
    constructors = {
        "brain_magic_speech_v7": BrainMagicV7,
        "brain_magic_no_subject_attn": BrainMagic_NoSubjectAttn,
        "brain_magic_no_short_conv": BrainMagic_NoShortConv,
        "brain_magic_no_feature_encoder": BrainMagic_NoFeatureEncoder,
        "cnn_lstm": CNNLSTM,
        "dilated_conv": DilatedConv,
        "awavenet": AWaveNet,
        "eeg_conformer": EEGConformer,
        "pnpl_cnn_tcn": CNNTCN,
    }
    return [constructors[name](**config) for name, config in modules.items()]


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
