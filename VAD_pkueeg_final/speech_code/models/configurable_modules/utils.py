# ============================================================================
# configurable_modules/utils.py
# 模型/优化器/损失函数的工厂函数（旧版）
# ============================================================================
# 此文件是早期版本的工厂函数实现，与 my_modules/utils.py 类似但有以下区别：
#   - 从 libribrain_experiments 包导入工具层（而非 speech_code）
#   - 不支持 brain_magic_speech 模块
#   - 损失函数仅支持 cross_entropy
#
# 当前项目中使用 my_modules/utils.py 替代此文件。
# ============================================================================

from torch.nn import Conv1d, ELU
from torch.nn import Softsign, GRU, Linear, ReLU, Sigmoid, GELU, BatchNorm1d
from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import LinearLR, StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from libribrain_experiments.models.util_layers import Permute
from torch.nn import LayerNorm
from libribrain_experiments.models.average_groups import AverageGroups


# ============================================================================
# modules_from_config - 模型构建工厂
# ============================================================================
def modules_from_config(modules: list[tuple[str, dict]]):
    """
    根据 YAML 配置列表动态创建 PyTorch 层/模块。

    支持的模块类型：
    - linear, conv1d
    - softsign, relu, elu, sigmoid, gelu (激活函数)
    - dropout, dropout1d, batch_norm1d, layer_norm (正则化)
    - gru (RNN)
    - flatten, permute (维度变换)
    - resnet_block, average_groups (自定义模块)
    """
    modules_list = []
    for module in modules:
        module_type = list(module)[0]
        config = module[module_type]
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
            module = ResnetBlock(**config)
        elif module_type == "layer_norm":
            module = LayerNorm(**config)
        elif module_type == "average_groups":
            module = AverageGroups(**config)
        else:
            raise ValueError(f"Unsupported module_type: {module_type}")
        modules_list.append(module)
    return modules_list


# ============================================================================
# optimizer_from_config - 优化器工厂
# ============================================================================
def optimizer_from_config(parameters, config):
    """从配置构建优化器 + 可选的学习率调度器"""
    if (config["name"] == "adam"):
        optimizer = Adam(parameters, **config["config"])
    elif (config["name"] == "adamw"):
        optimizer = AdamW(parameters, **config["config"])
    elif (config["name"] == "sgd"):
        optimizer = SGD(parameters, **config["config"])
    else:
        raise ValueError(f"Unsupported optimizer: "
                         + f"{config['name']}")

    # 可选的学习率调度
    if ("scheduler" in config):
        if (config["scheduler"] == "linear"):
            scheduler = LinearLR(optimizer, **config["scheduler_config"])
        elif (config["scheduler"] == "step"):
            scheduler = StepLR(optimizer, **config["scheduler_config"])
        elif (config["scheduler"] == "cosine"):
            scheduler = CosineAnnealingLR(optimizer, **config["scheduler_config"])
        elif (config["scheduler"] == "cosine_warm"):
            scheduler = CosineAnnealingWarmRestarts(optimizer, **config["scheduler_config"])
        else:
            raise ValueError(f"Unsupported scheduler: ", config["scheduler"])
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

    return optimizer


# ============================================================================
# loss_fn_from_config - 损失函数工厂（仅支持 cross_entropy）
# ============================================================================
def loss_fn_from_config(loss_config):
    """从配置构建损失函数"""
    if loss_config["name"] == "cross_entropy":
        if ("config" not in loss_config or loss_config["config"] is None):
            return nn.CrossEntropyLoss()
        return nn.CrossEntropyLoss(**loss_config["config"])
    else:
        raise ValueError(f"Unsupported loss: {loss_config['name']}")


# ============================================================================
# ResnetBlock - 可配置的残差连接块
# ============================================================================
class ResnetBlock(nn.Module):
    """残差连接块：输出 = F(x) + x"""
    def __init__(self, model_config: list[tuple[str, dict]]):
        super().__init__()
        self.model_config = model_config
        self.module_list = nn.ModuleList()
        self.module_list.extend(modules_from_config(model_config))


    def forward(self, x):
        x_residual = x  # 保存原始输入
        for module in self.module_list:
            x = module(x)
        return x + x_residual  # 残差相加
