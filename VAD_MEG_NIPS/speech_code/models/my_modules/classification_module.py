from torch import nn
from torchmetrics import Accuracy, Precision, Recall
from pytorch_lightning import LightningModule
from torchmetrics import F1Score
from .utils import modules_from_config, optimizer_from_config, loss_fn_from_config
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Dice loss for binary classification/multi-class segmentation.
    Args:
        pred: (N, *) predicted probabilities
        target: (N, *) ground truth (0 or 1)
        smooth: smoothing factor to avoid division by zero
    Returns:
        dice loss (scalar tensor)
    """
    pred = pred.float()
    target = target.float()
    
    # Flatten all dimensions except batch
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1. - dice.mean()

def iou_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    IoU (Jaccard) loss for binary classification/multi-class segmentation.
    Args:
        pred: (N, *) predicted probabilities
        target: (N, *) ground truth (0 or 1)
        smooth: smoothing factor to avoid division by zero
    Returns:
        IoU loss (scalar tensor)
    """
    pred = pred.float()
    target = target.float()
    
    # Flatten all dimensions except batch
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = (pred_flat + target_flat - pred_flat * target_flat).sum(dim=1)
    
    iou = (intersection + smooth) / (union + smooth)
    return 1. - iou.mean()

def confidence_margin_loss(y_pred: torch.Tensor, y_true: torch.Tensor, margin0: float = 0.3, margin1: float = 0.3) -> torch.Tensor:
    """
    Confidence margin loss that enforces:
    - For positive samples (y_true=1): y_pred > 0.5 + margin
    - For negative samples (y_true=0): y_pred < 0.5 - margin
    
    Args:
        y_pred: (N,) predicted probabilities
        y_true: (N,) ground truth labels (0 or 1)
        margin: confidence margin
    Returns:
        margin loss (scalar tensor)
    """
    y_pred = y_pred.float()
    y_true = y_true.float()
    
    # Create masks
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    
    # Penalties
    pos_penalty = F.relu(0.5 + margin1 - y_pred[pos_mask])  # Penalize if < 0.5+margin
    neg_penalty = F.relu(y_pred[neg_mask] - (0.5 - margin0))  # Penalize if > 0.5-margin
    
    # Handle cases where there are no positive/negative samples
    num_pos = max(1, pos_mask.sum().item())
    num_neg = max(1, neg_mask.sum().item())
    
    loss = (pos_penalty.sum() / num_pos + neg_penalty.sum() / num_neg) / 2
    return loss

class ClassificationModule(LightningModule):
    def __init__(self, model_config: list[tuple[str, dict]], n_classes: int, optimizer_config: dict, loss_config: dict, margin_weight: list):
        super().__init__()
        self.save_hyperparameters()
        self.modules_list = nn.ModuleList()
        self.modules_list.extend(modules_from_config(model_config))
        self.loss_mse = WeightedMSEByLabel(weight_0=loss_config['config']['weight'][0],
                                         weight_1=loss_config['config']['weight'][1])
        self.optimizer_config = optimizer_config
        self.margin0 = margin_weight[0]
        self.margin1 = margin_weight[1]
        self.sigmoid = nn.Sigmoid()
        self.smoother = nn.AvgPool1d(kernel_size=31, stride=1, padding=31 // 2)

    def forward(self, x, subject_ids=None):
        for module in self.modules_list:
            if getattr(module, "requires_subject_ids", False):
                x = module(x, subject_ids=subject_ids)
            else:
                x = module(x)
        if x.dim() == 3:
            x = x.squeeze(-1)  # (B, T, 1) → (B, T)
        x = x.unsqueeze(1)  # 增加通道维度 -> (batch, 1, time)
        x = self.smoother(x)  # 平滑处理
        x = x.squeeze(1)    # 移除通道维度 -> (batch, time)
        x = self.sigmoid(x)
        return x

    def configure_optimizers(self):
        return optimizer_from_config(self.parameters(), self.optimizer_config)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        mse = self.loss_mse(y_hat, y)
        dice = dice_loss(y_hat, y)
        iou = iou_loss(y_hat, y)
        margin = confidence_margin_loss(y_hat, y, self.margin0,self.margin1)
        loss = 1.0*mse + 0.0*dice + 0.0*iou + 0.0*margin
        # self.log('train_loss', loss)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("train_margin", margin, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        #print(f"y_hat: {y_hat.shape}, y: {y.shape}")
        mse = self.loss_mse(y_hat, y)
        dice = dice_loss(y_hat, y)
        iou = iou_loss(y_hat, y)
        margin = confidence_margin_loss(y_hat, y, self.margin0,self.margin1)
        loss = 1.0*mse + 0.0*dice + 0.0*iou + 0.0*margin
        # self.log('val_loss', loss)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        #self.log("val_margin", margin, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        mse = self.loss_mse(y_hat, y)
        dice = dice_loss(y_hat, y)
        iou = iou_loss(y_hat, y)
        margin = confidence_margin_loss(y_hat, y, self.margin0,self.margin1)
        loss = 1.0*mse + 0.0*dice + 0.0*iou + 0.0*margin
        # elf.log('test_loss', loss)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
class ClassificationModule_v2(LightningModule):
    def __init__(self, model_config: list[tuple[str, dict]], n_classes: int, optimizer_config: dict, loss_config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.modules_list = nn.ModuleList()
        self.modules_list.extend(modules_from_config(model_config))
        self.ce_loss = nn.CrossEntropyLoss()
        self.optimizer_config = optimizer_config

    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x

    def configure_optimizers(self):
        return optimizer_from_config(self.parameters(), self.optimizer_config)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        loss = self.ce_loss(y_hat, y)
        # self.log('train_loss', loss)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        loss = self.ce_loss(y_hat, y) 
        # self.log('val_loss', loss)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.ce_loss(y_hat, y) 
        # elf.log('test_loss', loss)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
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
        
