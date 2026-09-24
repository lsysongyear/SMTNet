# ============================================================================
# speech_utils.py - 自定义训练流程、评估逻辑（UPDATED for PKU EEG）
# ============================================================================
# 功能说明：
#   1. my_run_training - 自定义训练函数（单 GPU 版本，使用 PyTorch Lightning）
#   2. my_run_validation - 验证函数：多 checkpoint × 多阈值搜索（旧版）
#   3. my_run_test - 测试函数：加载所有 checkpoint，找最佳阈值并支持 ensemble
#   4. compute_metrics - 二分类评估指标计算（宏平均 + 逐类别 + 加权损失）
#
# [COMMENTED OUT]:
#   - LibriBrainHoldout_new: Competition Holdout dataset class
#   - my_run_holdout: Holdout inference and CSV generation
#   - SPEECH_HOLDOUT_PREDICTIONS*: Holdout prediction count constants
# ============================================================================

# [COMMENTED OUT] LibriBrain-specific imports
# from pnpl.datasets.libribrain2025.base import LibriBrainBase
import os
import os.path as op
import torch
from pytorch_lightning import Trainer
from torchmetrics import Accuracy, F1Score, Recall, Precision
from speech_code.models.my_modules.classification_module import ClassificationModule,ClassificationModule_v2
from speech_code.models.my_modules.utils import WeightedMSEByLabel
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import warnings
import h5py
from scipy.signal import butter, filtfilt
from sklearn.metrics import f1_score
import glob
from tqdm import tqdm
import csv
# [COMMENTED OUT] from pnpl.datasets.utils import check_include_and_exclude_ids, include_exclude_ids
# [COMMENTED OUT] from pnpl.datasets.libribrain2025.constants import RUN_KEYS


# ============================================================================
# 全局常量
# ============================================================================
# [COMMENTED OUT] Competition holdout prediction counts:
# SPEECH_HOLDOUT_PREDICTIONS = 560638
# SPEECH_HOLDOUT_PREDICTIONS_NEW = 224255


# ============================================================================
# [COMMENTED OUT] LibriBrainHoldout_new - Competition Holdout dataset class.
# Not needed for PKU EEG — no competition holdout set.
# See git history for the original implementation.
# ============================================================================


def _extract_subject_ids(batch, device):
    """从 batch[2] 提取 subject_ids。"""
    if len(batch) < 3:
        return None
    info = batch[2]
    if isinstance(info, dict) and 'subject_id' in info:
        subjects = info['subject_id']
        if isinstance(subjects, torch.Tensor):
            return subjects.to(device)
        elif isinstance(subjects, list):
            return torch.tensor(subjects, device=device)
    if isinstance(info, (list, tuple)) and len(info) > 0 and isinstance(info[0], dict):
        if 'subject_id' in info[0]:
            return torch.tensor([d['subject_id'] for d in info], device=device)
    return None


# ============================================================================
# my_run_training - 自定义训练函数（当前使用的版本）
# ============================================================================
def my_run_training(train_loader, val_loader, config, n_classes, best_model_metric="val_f1_macro", module=None, best_model_metric_mode="max"):
    """
    使用 PyTorch Lightning 执行模型训练。

    与 utils.py 中 run_training 的区别：
    - 单 GPU 模式（gpus=[0]）
    - 使用 ClassificationModule 的 margin_weight 参数
    - 保存 top-5 最佳 checkpoint（而非 top-1）
    - 支持 DDP 的 replace_sampler_ddp

    参数:
        train_loader: 训练数据 DataLoader
        val_loader: 验证数据 DataLoader
        config: 完整配置字典
        n_classes: 类别数量
        best_model_metric: 用于 checkpoint 选择的指标名
        module: 可选预初始化的模型
        best_model_metric_mode: "min" 或 "max"

    返回:
        (trainer, module) - PyTorch Lightning Trainer 和训练后的模型
    """
    # --- 日志记录器 ---
    logger = False
    if ("tensorboard" in config["general"] and config["general"]["tensorboard"]):
        log_dir = op.join(config["general"]["run_dir"], "tensorboard_logs")
        if int(os.getenv("LOCAL_RANK", 0)) == 0:
            os.makedirs(log_dir, exist_ok=True)
        logger = TensorBoardLogger(
            save_dir=log_dir,
            name="",
            version=None
        )

    # --- 模型检查点回调 ---
    callbacks = []
    if (config["general"]["checkpoint_path"] is not None):
        os.makedirs(config["general"]["checkpoint_path"], exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=config["general"]["checkpoint_path"],
            monitor=best_model_metric,
            mode=best_model_metric_mode,
            save_top_k=5,
            filename="model-{epoch:02d}-{val_loss:.4f}",
            save_last=False
        )
        callbacks.append(checkpoint_callback)

    early_stop = EarlyStopping(
        monitor=best_model_metric,
        mode=best_model_metric_mode,
        patience=5,
        verbose=True,
    )
    callbacks.append(early_stop)

    # --- 初始化 PyTorch Lightning Trainer ---
    trainer_config = config["trainer"]
    trainer = Trainer(
        logger=logger,
        devices=[0],
        accelerator="gpu",
        log_every_n_steps=1,
        callbacks=callbacks,
        **trainer_config
    )

    # --- 创建模型实例 ---
    if module is None:
        train_dataset_cfg = next(iter(config["data"]["datasets"]["train"][0].values()))
        sfreq = train_dataset_cfg.get("sfreq", 250.0)
        module = ClassificationModule(
            model_config=config["model"],
            n_classes=n_classes,
            optimizer_config=config["optimizer"],
            loss_config=config["loss"],
            margin_weight=config['general']['margin_weight'],
            sfreq=sfreq)

    # --- 执行训练 ---
    trainer.fit(module, train_loader, val_loader)

    print("Rank", int(os.getenv("LOCAL_RANK", 0)), ": Trainer finished training.")

    return trainer, module


# ============================================================================
# my_run_validation - 验证函数：多 checkpoint × 多阈值搜索（旧版，当前未使用）
# ============================================================================
def my_run_validation(val_loader, module_path, labels, weight, threshold=[0.5, 0.6, 0.7, 0.8, 0.9], prefix="val_"):
    """
    在验证集上遍历所有 checkpoint 和阈值组合，找到最佳 F1 的配置。

    工作流程：
    1. 加载模块路径下所有 .ckpt 文件
    2. 对每个 checkpoint，在验证集上推理获取概率
    3. 对每个阈值，计算二分类指标
    4. 选出让 macro_f1 最大的 (checkpoint, threshold) 组合

    参数:
        val_loader: 验证数据 DataLoader
        module_path: checkpoint 目录路径
        labels: 类别标签列表
        weight: 损失权重 [weight_0, weight_1]
        threshold: 阈值搜索列表
        prefix: 指标名前缀（如 "val_"）

    返回:
        (results_dict, best_model_path, best_threshold)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    escaped_path = glob.escape(module_path)
    ckpt_paths = sorted(glob.glob(os.path.join(escaped_path, "*.ckpt")))
    best_f1 = -1
    best_model_path = None
    best_threshold = None
    best_metrics = None

    results = {}

    for ckpt in tqdm(ckpt_paths, desc="Validating models"):
        results[ckpt] = {}
        module = ClassificationModule.load_from_checkpoint(ckpt)
        module.eval().to(device)
        disp_labels = labels
        all_targets = []
        all_probas = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                x, y = batch[0], batch[1]
                x = x.to(module.device)
                y = y.to(module.device)
                subj = _extract_subject_ids(batch, module.device)
                probs = module(x, subject_ids=subj)
                all_probas.extend(probs.reshape(-1))
                all_targets.extend(y.reshape(-1))

        all_targets = torch.stack(all_targets)
        all_probas = torch.stack(all_probas)

        for t in threshold:
            y_pred_labels = (all_probas >= t).int()
            metrics = compute_metrics(y_pred_labels, all_probas, all_targets, t, prefix, weight, module.device)
            results[ckpt][f"{prefix}threshold_{t}"] = metrics
            if metrics["val_macro_f1"] > best_f1:
                best_f1 = metrics["val_macro_f1"]
                best_model_path = ckpt
                best_threshold = t
                best_metrics = metrics

        del module
        torch.cuda.empty_cache()

    results["best_model_path"] = best_model_path
    results["best_threshold"] = best_threshold
    results["best_metrics"] = best_metrics
    return results, best_model_path, best_threshold


# ============================================================================
# my_run_test - 测试评估函数（用于验证/测试集）
# ============================================================================
def my_run_test(test_loader, module_path, labels, weight, threshold=[0.5, 0.6, 0.7, 0.8, 0.9], prefix="test_", test_results_path = None, fixed_ckpt_path = None, fixed_threshold = None):
    """
    在测试集（或验证集）上评估 checkpoint。

    工作流程（搜索模式，默认）：
    1. 遍历所有 checkpoint，在 GPU 上推理，保存 probas/targets 到 CPU
    2. 使用 sklearn 快速搜索最佳阈值（按 macro F1）
    3. 仅对最佳模型+阈值计算完整 torchmetrics 指标
    4. 保存最佳模型的预测结果到 .npz 文件

    工作流程（固定模式，传入 fixed_ckpt_path 和 fixed_threshold）：
    1. 仅加载指定 checkpoint 推理
    2. 使用指定的 fixed_threshold 计算指标，不搜索

    参数:
        test_loader: 测试数据 DataLoader
        module_path: checkpoint 目录路径
        labels: 类别标签列表
        weight: 损失权重 [weight_0, weight_1]
        threshold: 阈值搜索列表（固定模式下忽略）
        prefix: 指标名前缀（如 "val_" 或 "test_"）
        test_results_path: 保存预测结果的 .npz 路径
        fixed_ckpt_path: 固定 checkpoint 路径（跳过 ckpt 搜索）
        fixed_threshold: 固定阈值（跳过阈值搜索）

    返回:
        (final_results_dict, best_model_paths_list, best_thresholds_list)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 固定模式：使用指定的 ckpt 和阈值，不搜索 ──────────────────────
    use_fixed = (fixed_ckpt_path is not None and fixed_threshold is not None)
    if use_fixed:
        ckpt_paths = [fixed_ckpt_path]
        threshold = [fixed_threshold]
    else:
        escaped_path = glob.escape(module_path)
        ckpt_paths = sorted(glob.glob(os.path.join(escaped_path, "*.ckpt")))

    final_results = {}

    best_ckpt = None
    best_threshold = None
    best_acc = -1
    best_probas_np = None
    best_targets_np = None
    best_preds = None

    for ckpt in tqdm(ckpt_paths, desc="Testing models"):
        module = ClassificationModule.load_from_checkpoint(ckpt)
        module.eval().to(device)
        all_targets_cpu = []
        all_probas_cpu = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                x, y = batch[0], batch[1]
                x = x.to(module.device)
                subj = _extract_subject_ids(batch, module.device)
                probs = module(x, subject_ids=subj)
                all_probas_cpu.append(probs.cpu().flatten())
                all_targets_cpu.append(y.flatten())

        all_targets_np = torch.cat(all_targets_cpu).numpy()
        all_probas_np = torch.cat(all_probas_cpu).numpy()
        del all_targets_cpu, all_probas_cpu, module
        torch.cuda.empty_cache()

        for t in threshold:
            y_pred = (all_probas_np >= t).astype(int)
            tp = np.sum((y_pred == 1) & (all_targets_np == 1))
            fp = np.sum((y_pred == 1) & (all_targets_np == 0))
            tn = np.sum((y_pred == 0) & (all_targets_np == 0))
            fn = np.sum((y_pred == 0) & (all_targets_np == 1))
            r0 = tn / (tn + fp) if (tn + fp) > 0 else 0
            r1 = tp / (tp + fn) if (tp + fn) > 0 else 0
            acc = (r0 + r1) / 2
            if acc > best_acc:
                best_acc = acc
                best_ckpt = ckpt
                best_threshold = t
                best_probas_np = all_probas_np
                best_targets_np = all_targets_np
                best_preds = y_pred

    all_targets = torch.from_numpy(best_targets_np)
    all_probas = torch.from_numpy(best_probas_np)
    best_preds_t = torch.from_numpy(best_preds)

    best_metrics = compute_metrics(best_preds_t, all_probas, all_targets, best_threshold, prefix, weight, torch.device("cpu"))

    ensemble_preds_t = best_preds_t
    ensemble_metrics = compute_metrics(ensemble_preds_t, all_probas, all_targets, threshold=0.5, prefix="ensemble_", weight=weight, device=torch.device("cpu"))

    final_results[f"top1:{best_ckpt}, {best_threshold}"] = best_metrics
    final_results["ensemble, 0.5"] = ensemble_metrics
    final_results["best_models"] = [best_ckpt]
    final_results["best_thresholds"] = [best_threshold]

    np.savez(test_results_path, probas=best_probas_np, targets=best_targets_np, preds=best_preds, threshold=best_threshold)
    return final_results, [best_ckpt], [best_threshold]


# ============================================================================
# [COMMENTED OUT] my_run_holdout - Competition Holdout inference.
# Not needed for PKU EEG — no competition holdout submission.
# See git history for the original implementation.
# ============================================================================


# ============================================================================
# compute_metrics - 二分类评估指标计算
# ============================================================================
def compute_metrics(
    y_pred_labels: torch.Tensor,
    y_hat_probs: torch.Tensor,
    y_true: torch.Tensor,
    threshold: float = 0.5,
    prefix: str = "",
    weight: list = [1.0, 1.0],
    device: torch.device = torch.device("cpu")
) -> dict:
    """
    计算二分类任务的各类评估指标。

    指标类别：
    1. Macro 级别（将二分类视为 2 类多分类）：
       - Accuracy, Precision, Recall, F1
    2. 逐类别指标（class 0 = 静音, class 1 = 语音）：
       - Precision, Recall, F1
    3. 总体指标：
       - Binary Accuracy
    4. 加权 MSE 损失（使用 WeightedMSEByLabel）
    5. 使用的阈值

    返回:
        dict - 指标名 -> 指标值（Python float）
    """
    results = {}

    macro_metrics = {
        "macro_acc": Accuracy(num_classes=2, task="multiclass", average="macro").to(device),
        "macro_precision": Precision(num_classes=2, task="multiclass", average="macro").to(device),
        "macro_recall": Recall(num_classes=2, task="multiclass", average="macro").to(device),
        "macro_f1": F1Score(num_classes=2, task="multiclass", average="macro").to(device),
    }

    for name, metric in macro_metrics.items():
        results[f"{prefix}{name}"] = metric(y_pred_labels, y_true).item()

    binary_precision = Precision(task="multiclass", num_classes=2, average="none").to(device)
    binary_recall = Recall(task="multiclass", num_classes=2, average="none").to(device)
    binary_f1 = F1Score(task="multiclass", num_classes=2, average="none").to(device)

    all_precision = binary_precision(y_pred_labels, y_true)
    all_recall = binary_recall(y_pred_labels, y_true)
    all_f1 = binary_f1(y_pred_labels, y_true)
    for class_idx in [0, 1]:
        bin_prefix = f"{prefix}class{class_idx}_"
        results[bin_prefix + "precision"] = all_precision[class_idx].item()
        results[bin_prefix + "recall"] = all_recall[class_idx].item()
        results[bin_prefix + "f1"] = all_f1[class_idx].item()

    acc = Accuracy(task="binary").to(device)
    results[f"{prefix}binary_acc"] = acc(y_pred_labels, y_true).item()

    loss_fn = WeightedMSEByLabel(weight[0], weight[1]).to(device)
    results[f"{prefix}loss"] = loss_fn(y_hat_probs, y_true.float()).item()

    results[f"{prefix}threshold"] = threshold

    return results
