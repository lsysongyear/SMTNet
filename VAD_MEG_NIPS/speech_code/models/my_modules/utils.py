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
from speech_code.models.my_modules.BrainNetwork_v1 import Brain_Magic_speech_v1
from speech_code.models.my_modules.BrainNetwork_v7 import Brain_Magic_speech_v7
from speech_code.models.my_modules.BrainNetwork_v7_ablation import (
    BrainMagic_NoSubjectAttn, BrainMagic_NoShortConv, BrainMagic_NoFeatureEncoder)
from speech_code.models.my_modules.AWaveNet import AWaveNet
from speech_code.models.my_modules.CNNLSTM import CNNLSTM
from speech_code.models.my_modules.DilatedConv import DilatedConv
from speech_code.models.my_modules.VLAAI import VLAAI
from speech_code.models.conformer import EEGConformer

def modules_from_config(modules: list[tuple[str, dict]]):
    modules_list = []
    for module in modules:
        module_type = module
        config = modules[module_type]
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
        elif module_type == "brain_magic_speech":
            if "n_subjects" in config:
                from speech_code.models.my_modules.BrainNetworkSubject import SubjectProjectedBrainMagic
                module = SubjectProjectedBrainMagic(**config)
            else:
                module = Brain_Magic_speech(**config)
        elif module_type == "brain_magic_speech_v1":
            module = Brain_Magic_speech_v1(**config)
        elif module_type == "brain_magic_speech_v7":
            module = Brain_Magic_speech_v7(**config)
        elif module_type == "brain_magic_no_subject_attn":
            module = BrainMagic_NoSubjectAttn(**config)
        elif module_type == "brain_magic_no_short_conv":
            module = BrainMagic_NoShortConv(**config)
        elif module_type == "brain_magic_no_feature_encoder":
            module = BrainMagic_NoFeatureEncoder(**config)
        elif module_type == "awavenet":
            module = AWaveNet(**config)
        elif module_type == "cnn_lstm":
            module = CNNLSTM(**config)
        elif module_type == "dilated_conv":
            module = DilatedConv(**config)
        elif module_type == "vlaai":
            module = VLAAI(**config)
        elif module_type == "eeg_conformer":
            module = EEGConformer(**config)
        elif module_type == "pnpl_cnn_tcn":
            from .CNNTCN import CNNTCN
            module = CNNTCN(**config)
        else:
            raise ValueError(f"Unsupported module_type: {module_type}")
        modules_list.append(module)
    return modules_list


def optimizer_from_config(parameters, config):
    if (config["name"] == "adam"):
        optimizer = Adam(parameters, **
                         config["config"])
    elif (config["name"] == "adamw"):
        optimizer = AdamW(parameters, **
                          config["config"])
    elif (config["name"] == "sgd"):
        optimizer = SGD(parameters, **
                        config["config"])
    else:
        raise ValueError(f"Unsupported optimizer: "
                         + f"{config['name']}")

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


def loss_fn_from_config(loss_config):
    if loss_config["name"] == "cross_entropy":
        if ("config" not in loss_config or loss_config["config"] is None):
            return nn.CrossEntropyLoss()
        return nn.CrossEntropyLoss(**loss_config["config"])
    elif loss_config["name"] == "mse":
        # 判断是否提供了权重
        if "config" not in loss_config or loss_config["config"] is None:
            return nn.MSELoss()
        else:
            config = loss_config["config"]
            if "weight" in config:
                weight_tensor = torch.tensor(config["weight"], dtype=torch.float32)
                return WeightedMSEByLabel(weight_0=weight_tensor[0].item(),
                                         weight_1=weight_tensor[1].item())
            else:
                return nn.MSELoss(**config)
    else:
        raise ValueError(f"Unsupported loss: {loss_config['name']}")


class ResnetBlock(nn.Module):
    def __init__(self, model_config: list[tuple[str, dict]]):
        super().__init__()
        self.model_config = model_config
        self.module_list = nn.ModuleList()
        self.module_list.extend(modules_from_config(model_config))

    def forward(self, x):
        x_residual = x
        for module in self.module_list:
            x = module(x)
        return x + x_residual

# 加权mse损失
class WeightedMSEByLabel(nn.Module):
    def __init__(self, weight_0=1.0, weight_1=1.0, reduction='mean'):
        super().__init__()
        self.weight_0 = weight_0
        self.weight_1 = weight_1
        self.reduction = reduction

    def forward(self, input, target):
        # 计算 squared error
        mse = (input - target) ** 2

        # 应用权重
        weights = torch.where(target == 1, self.weight_1, self.weight_0)
        weighted_mse = weights * mse

        if self.reduction == 'mean':
            return weighted_mse.mean()
        elif self.reduction == 'sum':
            return weighted_mse.sum()
        else:
            return weighted_mse
