import torch
from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    LinearLR, StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts,
)
from speech_code.models.my_modules.BrainNetwork_v7 import Brain_Magic_speech_v7 as BrainMagicV7
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
