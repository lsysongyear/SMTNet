# ============================================================================
# classification_module.py - PyTorch Lightning 训练模块 + 损失函数
# ============================================================================
# 内容概述：
#   1. 损失函数：
#      - dice_loss: Dice 损失（常用于分割任务，衡量预测与标签的重叠度）
#      - iou_loss: IoU（Jaccard）损失（交并比损失）
#      - confidence_margin_loss: 置信度边距损失（鼓励输出远离决策边界 0.5）
#   2. ClassificationModule: 主训练模块（PyTorch Lightning LightningModule）
#      - 包含模型前向、后处理（平滑+上采样+sigmoid）、训练/验证/测试步骤
#      - 损失策略：当前使用纯 MSE，Dice/IoU/Margin 权重为 0（代码保留用于实验）
#   3. ClassificationModule_v2: 多分类版本（使用 CrossEntropyLoss，备用）
#   4. WeightedMSEByLabel: 逐类别加权的 MSE 损失函数
# ============================================================================

from torch import nn
from torchmetrics import Accuracy, Precision, Recall
from pytorch_lightning import LightningModule
from torchmetrics import F1Score
from .utils import modules_from_config, optimizer_from_config, loss_fn_from_config
from speech_code.models.my_modules.BrainNetwork_v7 import Brain_Magic_speech_v1 as Brain_Magic_speech_v7
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score


# ============================================================================
# Dice 损失函数
# ============================================================================
def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Dice 损失：衡量二值预测与目标之间的重叠程度。
    常用于医学图像分割等类别不平衡任务。

    公式: Dice = 2*|A∩B| / (|A| + |B|)
         Dice Loss = 1 - Dice

    Args:
        pred: (N, *) 预测概率值
        target: (N, *) 真实标签（0 或 1）
        smooth: 平滑因子，防止分母为零

    Returns:
        scalar - Dice 损失值（标量张量）
    """
    pred = pred.float()
    target = target.float()

    # 展平除 batch 外的所有维度
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    # 计算交集和并集
    intersection = (pred_flat * target_flat).sum(dim=1)          # 逐样本的交集
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)        # 逐样本的并集

    # Dice 系数 + 平滑
    dice = (2. * intersection + smooth) / (union + smooth)
    # Dice 损失 = 1 - Dice（最大化重叠 → 最小化损失）
    return 1. - dice.mean()


# ============================================================================
# IoU（Jaccard）损失函数
# ============================================================================
def iou_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    IoU（Jaccard）损失：衡量预测与目标的交并比。

    公式: IoU = |A∩B| / |A∪B| = |A∩B| / (|A| + |B| - |A∩B|)
         IoU Loss = 1 - IoU

    Args:
        pred: (N, *) 预测概率值
        target: (N, *) 真实标签（0 或 1）
        smooth: 平滑因子，防止分母为零

    Returns:
        scalar - IoU 损失值（标量张量）
    """
    pred = pred.float()
    target = target.float()

    # 展平除 batch 外的所有维度
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    # 交集和并集（IoU 风格的并集计算不同于 Dice）
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = (pred_flat + target_flat - pred_flat * target_flat).sum(dim=1)

    iou = (intersection + smooth) / (union + smooth)
    return 1. - iou.mean()


# ============================================================================
# 置信度边距损失（Confidence Margin Loss）
# ============================================================================
def confidence_margin_loss(y_pred: torch.Tensor, y_true: torch.Tensor, margin0: float = 0.3, margin1: float = 0.3) -> torch.Tensor:
    """
    置信度边距损失：鼓励模型对分类更有"信心"，输出远离 0.5 决策边界。

    约束目标：
    - 正样本（y_true=1）：预测概率 > 0.5 + margin1（鼓励 >0.5+margin）
    - 负样本（y_true=0）：预测概率 < 0.5 - margin0（鼓励 <0.5-margin）

    Args:
        y_pred: (N,) 预测概率值（范围 [0,1]）
        y_true: (N,) 真实标签（0 或 1）
        margin0: 负样本的置信度边距（默认 0.3，鼓励 <0.2）
        margin1: 正样本的置信度边距（默认 0.3，鼓励 >0.8）

    Returns:
        scalar - 边距损失值（标量张量）
    """
    y_pred = y_pred.float()
    y_true = y_true.float()

    # 构造正负样本的布尔掩码
    pos_mask = y_true == 1
    neg_mask = y_true == 0

    # 正样本惩罚：如果预测 < 0.5+margin1，则产生损失
    pos_penalty = F.relu(0.5 + margin1 - y_pred[pos_mask])
    # 负样本惩罚：如果预测 > 0.5-margin0，则产生损失
    neg_penalty = F.relu(y_pred[neg_mask] - (0.5 - margin0))

    # 处理没有正样本或负样本的极端情况（避免除以0）
    num_pos = max(1, pos_mask.sum().item())
    num_neg = max(1, neg_mask.sum().item())

    # 正负样本损失取平均
    loss = (pos_penalty.sum() / num_pos + neg_penalty.sum() / num_neg) / 2
    return loss


# ============================================================================
# ClassificationModule - PyTorch Lightning 主训练模块
# ============================================================================
class ClassificationModule(LightningModule):
    """
    PyTorch Lightning 训练模块，封装模型、优化器、损失函数和训练逻辑。

    前向传播流程：
    1. 通过可配置的模型模块列表（modules_list）提取特征
    2. AvgPool1d 平滑处理（kernel=31，保持时间分辨率不变）
    3. Linear 插值上采样 2.5×（使输出帧率与标注更好地对齐）
    4. Sigmoid 激活 → 输出概率值

    损失策略：
    当前实际使用: loss = 1.0*MSE + 0.0*Dice + 0.0*IoU + 0.0*Margin
    （Dice/IoU/Margin 目前权重为 0，保留用于后续实验）
    """
    def __init__(self, model_config: list[tuple[str, dict]], n_classes: int, optimizer_config: dict, loss_config: dict, margin_weight: list):
        """
        Args:
            model_config: 模型层配置列表
            n_classes: 类别数量
            optimizer_config: 优化器配置
            loss_config: 损失函数配置（包含类别权重）
            margin_weight: 置信度边距损失的权重 [margin0, margin1]
        """
        super().__init__()
        # 保存超参数到 checkpoint（用于后续加载和复现）
        self.save_hyperparameters()

        # 从配置构建模型层列表
        self.modules_list = nn.ModuleList()
        self.modules_list.extend(modules_from_config(model_config))

        # 加权 MSE 损失（对语音/静音类别可设置不同权重）
        self.loss_mse = WeightedMSEByLabel(weight_0=loss_config['config']['weight'][0],
                                         weight_1=loss_config['config']['weight'][1])
        self.optimizer_config = optimizer_config

        # 置信度边距
        self.margin0 = margin_weight[0]  # 负样本（静音）边距
        self.margin1 = margin_weight[1]  # 正样本（语音）边距

        # Sigmoid 激活（将原始输出映射到 [0,1] 概率）
        self.sigmoid = nn.Sigmoid()

        # 时间平滑器：1D 平均池化，窗口 31 点，保持长度不变
        self.smoother = nn.AvgPool1d(kernel_size=31, stride=1, padding=31 // 2)


    def forward(self, x, subject_ids=None, day_ids=None):
        """
        模型前向传播。

        Args:
            x: [batch, channels, time] EEG 数据
            subject_ids: [batch] 被试 ID (0-indexed, 0-24)，可选
            day_ids:      [batch] 天数 ID (0-indexed, 0-2)，可选

        Returns:
            [batch, target_time] 上采样后的语音概率（0~1）
        """
        # 依次通过所有模型层
        for module in self.modules_list:
            if isinstance(module, Brain_Magic_speech_v7) or getattr(module, "requires_subject_ids", False):
                x = module(x, subject_ids=subject_ids)
            else:
                x = module(x)

        # 处理模型输出 [batch, time, features] → [batch, time]
        if x.dim() == 3:
            if x.shape[-1] == 1:
                x = x.squeeze(-1)
            else:
                x = x[..., 0]

        # --- 后处理流程 ---
        x = x.unsqueeze(1)                             # [batch, time] → [batch, 1, time]
        x = self.smoother(x)                           # 均值平滑（减少噪声抖动）
        x = x.squeeze(1)                               # [batch, 1, time] → [batch, time]
        x = self.sigmoid(x)                            # 映射到 [0, 1]
        return x


    def configure_optimizers(self):
        """配置优化器（由 YAML 配置动态创建）"""
        return optimizer_from_config(self.parameters(), self.optimizer_config)

    def _extract_ids(self, batch):
        """从 batch[2] (list of info dicts, or collated dict) 中提取 subject_ids, day_ids。"""
        if len(batch) < 3:
            return None, None
        info = batch[2]
        # collated dict format
        if isinstance(info, dict):
            # Prefer subject_id (int) over subject (str like 'sub-01')
            if 'subject_id' in info:
                sids = info['subject_id']
                if isinstance(sids, torch.Tensor):
                    subj_ids = sids.to(self.device)
                else:
                    subj_ids = torch.tensor(sids, device=self.device)
            elif 'subject' in info:
                subjects = info['subject']
                subj_ids = torch.tensor([int(str(s).replace('sub-', '')) - 1 for s in subjects],
                                        device=self.device)
            else:
                return None, None
            days = info.get('day_id', [0] * len(subj_ids))
            if isinstance(days, torch.Tensor):
                day_ids = days.to(self.device)
            else:
                day_ids = torch.tensor(days, device=self.device)
            return subj_ids, day_ids
        # list of dicts format
        if isinstance(info, (list, tuple)) and len(info) > 0 and isinstance(info[0], dict):
            d0 = info[0]
            if 'subject_id' in d0:
                subject_ids = torch.tensor([d['subject_id'] for d in info], device=self.device)
            elif 'subject' in d0:
                subject_ids = torch.tensor([int(str(d['subject']).replace('sub-', '')) - 1 for d in info],
                                           device=self.device)
            else:
                return None, None
            day_ids = torch.tensor([d.get('day_id', 0) for d in info], device=self.device)
            return subject_ids, day_ids
        return None, None

    def training_step(self, batch, batch_idx):
        """
        单步训练。

        计算多种损失（用于监控），但实际回传的梯度只包含加权 MSE。
        """
        x, y = batch[0], batch[1]
        subj, day = self._extract_ids(batch)
        y_hat = self(x, subject_ids=subj, day_ids=day)

        # 计算各类损失（MSE 为主要损失，其余辅助监控）
        mse = self.loss_mse(y_hat, y)
        dice = dice_loss(y_hat, y)
        iou = iou_loss(y_hat, y)
        margin = confidence_margin_loss(y_hat, y, self.margin0, self.margin1)

        # 总损失 = 1.0*MSE + 0.3*Dice + 0.3*IoU + 0.0*Margin
        loss = 1.0*mse + 0.0*dice + 0.0*iou + 0.0*margin

        # 记录训练损失（同步到所有进程以支持 DDP）
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_mse", mse, on_epoch=True, sync_dist=True)
        self.log("train_dice", dice, on_epoch=True, sync_dist=True)
        self.log("train_iou", iou, on_epoch=True, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):
        """
        单步验证。

        计算与训练步骤相同的损失用于监控。
        """
        x, y = batch[0], batch[1]
        subj, day = self._extract_ids(batch)
        y_hat = self(x, subject_ids=subj, day_ids=day)

        mse = self.loss_mse(y_hat, y)
        dice = dice_loss(y_hat, y)
        iou = iou_loss(y_hat, y)
        margin = confidence_margin_loss(y_hat, y, self.margin0, self.margin1)
        loss = 1.0*mse + 0.0*dice + 0.0*iou + 0.0*margin

        # 记录验证损失
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_mse", mse, on_epoch=True, sync_dist=True)
        self.log("val_dice", dice, on_epoch=True, sync_dist=True)
        self.log("val_iou", iou, on_epoch=True, sync_dist=True)
        return loss


    def test_step(self, batch, batch_idx):
        """
        单步测试。

        计算与训练步骤相同的损失用于监控。
        """
        x, y = batch[0], batch[1]
        subj, day = self._extract_ids(batch)
        y_hat = self(x, subject_ids=subj, day_ids=day)

        mse = self.loss_mse(y_hat, y)
        dice = dice_loss(y_hat, y)
        iou = iou_loss(y_hat, y)
        margin = confidence_margin_loss(y_hat, y, self.margin0, self.margin1)
        loss = 1.0*mse + 0.0*dice + 0.0*iou + 0.0*margin

        # 记录测试损失
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_mse", mse, on_epoch=True, sync_dist=True)
        self.log("test_dice", dice, on_epoch=True, sync_dist=True)
        self.log("test_iou", iou, on_epoch=True, sync_dist=True)
        return loss


# ============================================================================
# ClassificationModule_v2 - 多分类版本的训练模块（备用，当前未使用）
# ============================================================================
class ClassificationModule_v2(LightningModule):
    """
    多分类版本的 Lightning 模块，使用 CrossEntropyLoss。
    用于原始的 LibriBrain 多分类任务（如音素分类），本项目中作为备用方案。
    """
    def __init__(self, model_config: list[tuple[str, dict]], n_classes: int, optimizer_config: dict, loss_config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.modules_list = nn.ModuleList()
        self.modules_list.extend(modules_from_config(model_config))
        self.ce_loss = nn.CrossEntropyLoss()  # 交叉熵损失（多分类标准损失）
        self.optimizer_config = optimizer_config


    def forward(self, x):
        """直接通过模块列表，不做后处理"""
        for module in self.modules_list:
            x = module(x)
        return x


    def configure_optimizers(self):
        return optimizer_from_config(self.parameters(), self.optimizer_config)


    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        loss = self.ce_loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        loss = self.ce_loss(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.ce_loss(y_hat, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss


# ============================================================================
# WeightedMSEByLabel - 按类别加权的 MSE 损失
# ============================================================================
class WeightedMSEByLabel(nn.Module):
    """
    按真实标签类别加权 MSE 损失。

    公式: loss = weight_i * (pred_i - target_i)^2
    其中 weight_i 取决于 target_i 是 0（静音）还是 1（语音）。

    用途：当正负样本严重不平衡时（如静音远多于语音），
    可通过调大语音类的权重来平衡两类对损失的贡献。
    """
    def __init__(self, weight_0=1.0, weight_1=1.0, reduction='mean'):
        """
        Args:
            weight_0: 静音类（标签=0）的 MSE 权重
            weight_1: 语音类（标签=1）的 MSE 权重
            reduction: 'mean'（平均）, 'sum'（求和）, 或 'none'（不归约）
        """
        super().__init__()
        self.weight_0 = weight_0
        self.weight_1 = weight_1
        self.reduction = reduction


    def forward(self, input, target):
        """
        Args:
            input: (N, *) 预测概率值
            target: (N, *) 真实标签（0 或 1）

        Returns:
            scalar - 加权后的 MSE 损失
        """
        # 逐元素 squared error
        mse = (input - target) ** 2

        # 根据真实标签选择对应的权重
        # target==1 的位置用 weight_1，target==0 的位置用 weight_0
        weights = torch.where(target == 1, self.weight_1, self.weight_0)
        weighted_mse = weights * mse

        if self.reduction == 'mean':
            return weighted_mse.mean()
        elif self.reduction == 'sum':
            return weighted_mse.sum()
        else:
            return weighted_mse
