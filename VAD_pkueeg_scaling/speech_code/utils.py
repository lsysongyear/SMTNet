# ============================================================================
# utils.py - 数据集类定义、数据加载工具、训练/评估工具函数
# ============================================================================
# 功能说明：
#   1. EEG_Speech_v1 类：自定义 EEG 语音检测 Dataset（PKU EEG 数据）
#      - 从 raw .npz 文件加载 64 通道 EEG 数据（250Hz）
#      - 首次访问时：ch_transform (64→57) + resample (250→100Hz) + 缓存
#      - 根据 segments.tsv 生成逐采样点的 voice/silence 标签
#      - 按滑动窗口切分样本，支持训练时的随机 jitter 数据增强
#   2. [COMMENTED OUT] Speech_v1 类：原 LibriBrain MEG 数据集类
#   3. get_datasets_from_config：从配置字典构建训练/验证/测试集的统一接口
#   4. run_training：基于 PyTorch Lightning 的训练封装函数（旧版）
#   5. run_validation：多指标评估函数，对比随机/朴素基线
#   6. adapt_config_to_data：根据实际数据调整配置（如自动计算类别权重）
#   7. 辅助工具：标签检查、类别统计、结果日志等
# ============================================================================

# [COMMENTED OUT] LibriBrain-specific imports (replaced by EEG dataset)
# from pnpl.datasets import LibriBrainPhoneme, LibriBrainSpeech
# from pnpl.datasets.libribrain2025.base import LibriBrainBase
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from pnpl.datasets.grouped_dataset import GroupedDataset
import json
import os
import os.path as op
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, F1Score, Recall, Precision
from torchmetrics.classification import MulticlassAUROC, BinaryAUROC
from torchmetrics import JaccardIndex
from speech_code.models.my_modules.classification_module import ClassificationModule
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import json
import numpy as np
import warnings
import h5py
import pandas as pd
from scipy.signal import butter, filtfilt
import random
import pickle
import zipfile
from tqdm import tqdm
# [COMMENTED OUT] LibriBrain-specific imports
# from pnpl.datasets.utils import check_include_and_exclude_ids, include_exclude_ids
# from pnpl.datasets.libribrain2025.constants import RUN_KEYS
# from torch.utils.data import Dataset
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'preprocess'))
from ch_transform import ch_transform
from scipy.signal import resample as scipy_resample
from scaling.duration import prefix_allocation

from itertools import groupby


def _npz_eeg_sample_count(path):
    with zipfile.ZipFile(path) as archive:
        with archive.open("eeg_data.npy") as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(handle)
            else:
                shape, _, _ = np.lib.format.read_array_header_2_0(handle)
    return int(shape[-1])


# ============================================================================
# 翻转过短的连续片段（用于后处理平滑，当前未使用）
# ============================================================================
def flip_short_segments(seq, v, threshold):
    """
    翻转序列中过短的连续片段。
    例如，将语音段中过短的静音片段纠正为语音。

    参数:
        seq: np.ndarray - 标签序列（0/1 二值）
        v: int - 要翻转的目标值（0 或 1）
        threshold: int - 长度阈值，小于此长度的片段会被翻转

    返回:
        np.ndarray - 处理后的标签序列
    """
    result = []
    for val, group in groupby(seq):
        group_list = list(group)
        length = len(group_list)

        if val == v and length < threshold:
            result.extend([1 - val] * length)
        else:
            result.extend(group_list)

    return np.array(result)


# [COMMENTED OUT] LibriBrain competition holdout prediction count
# SPEECH_HOLDOUT_PREDICTIONS = 560638


# ============================================================================
# EEG_Speech_v1 - 自定义 EEG 语音检测 Dataset 类 (NEW for PKU EEG)
# ============================================================================
class EEG_Speech_v1(Dataset):
    """
    针对 PKU EEG 语音活动检测任务的 PyTorch Dataset。

    工作流程：
    1. 根据 include_subjects / exclude_subjects / stories 确定要加载的数据
    2. 从 segments_{story_id}.tsv 中读取 voice/silence 时间区间
    3. 生成逐采样点的二值标签序列（0=silence, 1=voice，基于 250Hz 采样率）
    4. 按时间窗口（tmax-tmin）滑动切分样本，支持 stride 和 jitter
    5. __getitem__ 首次访问时从 raw .npz 加载 → ch_transform → (可选 resample) → 缓存

    核心参数:
        data_path: 数据集根目录 (包含 sub-XX/story_N.npz)
        label_path: segments tsv 文件目录
        tmin/tmax: 每个样本的时间窗口（相对起始点），默认 [0.0, 12.0] 秒
        include_subjects / exclude_subjects: 被试级数据划分
        stories: 要加载的 story 编号列表（默认 1-50）
        standardize: 是否对每个样本做 z-score 标准化
        data_type: 'train' / 'val' / 'test'，影响 stride 策略和 jitter 行为
    """
    def __init__(
        self,
        data_path: str,
        label_path: str,
        include_subjects: list = None,
        exclude_subjects: list = None,
        stories: list = None,
        tmin: float = 0.0,
        tmax: float = 12.0,
        standardize: bool = True,
        data_type: str = 'train',
        sfreq: float = 250.0,
        stride=None,
        delay: int = 0,
        channel_means: np.ndarray | None = None,
        channel_stds: np.ndarray | None = None,
        include_info: bool = False,
        recording_fraction: float = 1.0,
        **kwargs,
    ):
        self.data_path = data_path
        self.label_path = label_path
        self.tmin = tmin
        self.tmax = tmax
        self.standardize = standardize
        self.include_info = include_info
        self.channel_means = channel_means
        self.channel_stds = channel_stds
        self.recording_fraction = float(recording_fraction)
        self.scaling_metadata = {
            "recording_fraction": self.recording_fraction,
            "duration_sfreq": 250.0,
            "subjects": {},
        }

        # EEG 数据采样率（默认与原始数据一致 250Hz）
        self.sfreq = sfreq
        # 原始 npz 数据的采样率（250Hz）
        self.sfreq_raw = 250.0
        # 标签采样率：标签始终以原始采样率 250Hz 生成
        self.label_sfreq = self.sfreq_raw
        # 每个样本包含的 EEG 时间点数
        self.points_per_sample = int((tmax - tmin) * self.sfreq)
        # 每个样本的标签长度（250Hz 等效采样率下，匹配模型 upsampled 输出）
        self.points_per_sample_label = int((tmax - tmin) * self.label_sfreq)
        self.stride = stride
        self.samples = []
        self.delay = delay
        self.labels_sorted = [0, 1]  # 0=silence, 1=voice
        self.open_files = {}  # 缓存已加载+处理后的 EEG 数据
        self.data_type = data_type
        self.label_cache = {}  # 缓存已加载的 tsv 标签

        # --- 确定被试列表 ---
        if include_subjects is not None and len(include_subjects) > 0:
            self.subjects = list(include_subjects)
        else:
            # 扫描 data_path 下的 sub-XX 目录
            all_subs = sorted([
                d.replace('sub-', '') for d in os.listdir(data_path)
                if os.path.isdir(os.path.join(data_path, d)) and d.startswith('sub-')
            ])
            if exclude_subjects is not None:
                self.subjects = [s for s in all_subs if s not in exclude_subjects]
            else:
                self.subjects = all_subs

        if len(self.subjects) == 0:
            raise ValueError("No subjects found. Please check include_subjects / exclude_subjects / data_path.")

        # --- 确定 story 列表 ---
        if stories is not None:
            self.stories = stories
        else:
            self.stories = list(range(1, 51))

        # --- 遍历所有 (subject, story) 组合，收集样本元信息 ---
        for subject in self.subjects:
            subject_items = []
            for story_id in sorted(self.stories):
                npz_path = os.path.join(data_path, f"sub-{subject}", f"story_{story_id}.npz")
                if not os.path.exists(npz_path):
                    continue
                try:
                    speech_labels = self._get_labels_for_story(story_id)
                    if speech_labels is None:
                        continue
                    valid_samples = min(
                        len(speech_labels), _npz_eeg_sample_count(npz_path)
                    )
                    subject_items.append(
                        (story_id, speech_labels[:valid_samples], valid_samples)
                    )
                except Exception as e:
                    warnings.warn(f"Skipping sub-{subject} story_{story_id}: {e}")
                    continue

            target, allocations = prefix_allocation(
                [item[2] for item in subject_items], self.recording_fraction
            )
            unit_records = []
            for (story_id, speech_labels, full_samples), keep_samples in zip(
                subject_items, allocations
            ):
                unit_records.append(
                    {
                        "unit": str(story_id),
                        "full_samples": full_samples,
                        "selected_samples": keep_samples,
                    }
                )
                if keep_samples > 0:
                    self._collect_samples(
                        subject, story_id, speech_labels[:keep_samples]
                    )
            self.scaling_metadata["subjects"][str(subject)] = {
                "full_samples": sum(item[2] for item in subject_items),
                "selected_samples": target,
                "units": unit_records,
            }

        if len(self.samples) == 0:
            raise ValueError("No samples found. Please check data_path and label_path.")

    def _get_labels_for_story(self, story_id):
        """
        从 segments_{story_id}.tsv 生成逐采样点的二值标签序列（基于 label_sfreq=250Hz，
        匹配模型经过 2.5× 上采样后的输出）。

        标签定义：0 = silence, 1 = voice
        """
        if story_id in self.label_cache:
            return self.label_cache[story_id]

        tsv_path = os.path.join(self.label_path, f"segments_{story_id:02d}.tsv")
        if not os.path.exists(tsv_path):
            warnings.warn(f"Label file not found: {tsv_path}")
            return None

        seg_df = pd.read_csv(tsv_path, sep='\t')

        # 确定总时长：取最后一段的 end 时间
        max_time = seg_df['end'].max()
        total_samples = int(max_time * self.label_sfreq) + 1

        # 初始化全为 0（静音）
        labels = np.zeros(total_samples, dtype=int)

        # 将 voice 段标记为 1
        voice_df = seg_df[seg_df['label'] == 'voice']
        for _, row in voice_df.iterrows():
            start_sample = int(row['start'] * self.label_sfreq)
            end_sample = int(row['end'] * self.label_sfreq)
            labels[start_sample:end_sample] = 1

        self.label_cache[story_id] = labels
        return labels

    def _collect_samples(self, subject, story_id, speech_labels, stride=None):
        """
        将长标签序列按固定时间窗口切分为训练样本。

        样本存储格式: (subject, story_id, onset_seconds, label_window)

        Note: onset 使用 sfreq（100Hz）计算，因为 EEG 数据以此采样率切片；
              labels 使用 label_sfreq（250Hz）采样，匹配模型上采样后的输出。
        """
        time_window_samples = self.points_per_sample_label

        # 自适应 stride 策略（相对于 label_sfreq 采样率的步长）
        if stride is None:
            if self.data_type == 'train':
                stride = time_window_samples // 2  #50%重叠
            else:
                stride = time_window_samples  # val/test: 无重叠

        for i in range(0, len(speech_labels), stride):
            sample_labels = speech_labels[i:i + time_window_samples]
            if len(sample_labels) < time_window_samples:
                continue
            # 训练时引入 ±5 个采样点的随机 jitter（数据增强，约 ±20ms at 250Hz）
            onset_i = i  # 无 jitter 时的起始点
            if self.data_type == 'train':
                jitter = random.choice(list(range(-5, 5 + 1)))
                jittered_i = i + jitter
                if jittered_i < 0:
                    jittered_i = 0
                if jittered_i + time_window_samples > len(speech_labels):
                    jittered_i = i
                sample_labels = speech_labels[jittered_i:jittered_i + time_window_samples]
                onset_i = jittered_i  # 关键修复：onset 与 label 使用相同的 jitter 起始点
            # onset 以秒为单位（label_sfreq=250Hz），EEG 切片时再转到 sfreq=100Hz
            self.samples.append((subject, story_id, onset_i / self.label_sfreq, sample_labels))

    def _load_and_preprocess_eeg(self, subject, story_id):
        """
        加载 raw npz，执行通道对齐 + 降采样 + z-score，缓存结果。
        返回: np.ndarray (57, T_100Hz)
        """
        cache_key = (subject, story_id)
        if cache_key in self.open_files:
            return self.open_files[cache_key]

        npz_path = os.path.join(self.data_path, f"sub-{subject}", f"story_{story_id}.npz")
        data = np.load(npz_path, allow_pickle=True)
        eeg = data['eeg_data'].astype(np.float64)  # (C, T)

        # 步骤 1: 通道对齐 64 → 57（已是 57 通道则跳过）
        if eeg.shape[0] == 57:
            eeg_57 = eeg  # 已预处理为 57 通道（如 EA 对齐后数据）
        else:
            eeg_57 = ch_transform(eeg, data['ch_names'])  # (57, T)

        # 步骤 2: 降采样（仅当 sfreq < sfreq_raw 时执行）
        if self.sfreq < self.sfreq_raw:
            eeg_57 = scipy_resample(eeg_57, int(eeg_57.shape[1] * self.sfreq / self.sfreq_raw), axis=1)

        # 步骤 3: 全试次级别的 z-score 标准化（每个通道独立）
        ch_means = np.mean(eeg_57, axis=1, keepdims=True)
        ch_stds = np.std(eeg_57, axis=1, keepdims=True)
        ch_stds[ch_stds == 0] = 1.0
        eeg_norm = (eeg_57 - ch_means) / ch_stds

        eeg_norm = eeg_norm.astype(np.float32)
        self.open_files[cache_key] = eeg_norm
        return eeg_norm

    def __getitem__(self, idx):
        """
        获取第 idx 个样本。

        返回格式:
            [torch.Tensor (57, time), label, info_dict]
        """
        if idx >= len(self.samples):
            raise IndexError(
                f"Index {idx} is out of bounds for dataset of size {len(self.samples)}"
            )
        subject, story_id, onset, label = self.samples[idx]

        if self.include_info:
            # 根据 story_id 推断 day: D1(1-17)→0, D2(18-33)→1, D3(34-50)→2
            if story_id <= 17:
                day_id = 0
            elif story_id <= 33:
                day_id = 1
            else:
                day_id = 2
            info = {
                "dataset": "pkueeg",
                "subject": subject,
                "subject_id": int(subject) - 1,
                "story_id": story_id,
                "day_id": day_id,
                "onset": torch.tensor(onset, dtype=torch.float32),
            }

        # 延迟加载 + 缓存
        eeg_data = self._load_and_preprocess_eeg(subject, story_id)

        # 根据 onset 定位数据窗口
        delay_samples = self.delay
        start = max(0, int((onset + self.tmin) * self.sfreq) + delay_samples)
        end = start + self.points_per_sample
        data = eeg_data[:, start:end]

        # 边界处理：零填充
        if data.shape[1] < self.points_per_sample:
            padded = np.zeros((data.shape[0], self.points_per_sample), dtype=data.dtype)
            padded[:, :data.shape[1]] = data
            data = padded

        # 样本级 z-score 标准化
        if self.standardize:
            ch_means = np.mean(data, axis=1, keepdims=True)
            ch_stds = np.std(data, axis=1, keepdims=True)
            ch_stds[ch_stds == 0] = 1.0
            data = (data - ch_means) / ch_stds

        if self.include_info:
            return [torch.tensor(data, dtype=torch.float32), label, info]
        return [torch.tensor(data, dtype=torch.float32), label, {}]

    def __len__(self):
        return len(self.samples)


# ============================================================================
# [COMMENTED OUT] Speech_v1(LibriBrainBase) - Original MEG dataset class.
# Replaced by EEG_Speech_v1 above for PKU EEG data.
# See git history for the original implementation.
# ============================================================================


# ============================================================================
# 支持的数据集名称映射
# ============================================================================
DATASETS = {
    # [COMMENTED OUT] "libribrain_phoneme": LibriBrainPhoneme,
    # [COMMENTED OUT] "libribrain_speech": LibriBrainSpeech,
    # [COMMENTED OUT] "speech_v1": Speech_v1,
    "eeg_speech_v1": EEG_Speech_v1,  # [NEW] PKU EEG speech detection dataset
}


# ============================================================================
# 检查多个数据集是否具有相同的标签集
# ============================================================================
def check_labels(list_of_labels):
    """确保所有数据分区具有一致的标签列表，如果不一致则抛出异常"""
    reference_labels = list_of_labels[0]
    for labels in list_of_labels[1:]:
        if (labels != reference_labels):
            raise ValueError(
                f"Datasets have different labels: {labels} and {reference_labels}")


# ============================================================================
# 根据配置对数据集应用包装器（如样本分组/平均）
# ============================================================================
def apply_dataset_wrappers_from_data_config(dataset, data_config):
    """
    对数据集应用额外的包装器。
    支持两种分组方式（互斥）：
    - averaged_samples: 将连续多个样本平均为一个样本
    - grouped_samples: 将连续多个样本打包为一组
    """
    if ("averaged_samples" in data_config["general"] and "grouped_samples" in data_config["general"]):
        raise ValueError(
            "Only one grouping type can be used at a time. Please change data config")
    if ("averaged_samples" in data_config["general"] and data_config["general"]["averaged_samples"] > 1):
        dataset = GroupedDataset(
            dataset, grouped_samples=data_config["general"]["averaged_samples"], average_grouped_samples=True)
    if ("grouped_samples" in data_config["general"] and data_config["general"]["grouped_samples"] > 1):
        dataset = GroupedDataset(
            dataset, grouped_samples=data_config["general"]["grouped_samples"], average_grouped_samples=False, drop_remaining=True)
    return dataset


# ============================================================================
# 从配置字典加载单个数据分区（训练/验证/测试）
# ============================================================================
def get_dataset_partition_from_config(partition_config, channel_means=None, channel_stds=None):
    """
    根据配置加载一个数据分区。
    如果提供了 channel_means 和 channel_stds（来自训练集），
    则将其传递给验证/测试集以实现一致性标准化。

    返回:
        ConcatDataset - 可能包含多个子数据集的拼接
    """
    partition_dataset_names = [list(ds.keys())[0] for ds in partition_config]
    partition_dataset_configs = [list(ds.values())[0]
                                 for ds in partition_config]

    # 将训练集的统计量传递给非训练分区（实现标准化一致性）
    for config in partition_dataset_configs:
        if (config.get("standardize", True)):
            config['channel_means'] = channel_means.tolist(
            ) if channel_means is not None else None
            config['channel_stds'] = channel_stds.tolist(
            ) if channel_stds is not None else None

    # 逐个实例化数据集
    partition_datasets = []
    partition_dataset_labels = []
    for name, config in zip(partition_dataset_names, partition_dataset_configs):
        if (name not in DATASETS):
            raise ValueError(
                f"Dataset {name} not supported. Please change data config")
        dataset = DATASETS[name](**config)
        partition_datasets.append(dataset)
        partition_dataset_labels.append(dataset.labels_sorted)

    # 确保所有子数据集有相同的标签集
    check_labels(partition_dataset_labels)
    # 合并为一个数据集
    partition_dataset = ConcatDataset(partition_datasets)
    return partition_dataset


# ============================================================================
# 从完整数据配置构建训练/验证/测试数据集（主入口函数）
# ============================================================================
def get_datasets_from_config(data_config):
    """
    从数据配置字典中加载训练集、验证集、测试集。

    关键逻辑：
    - 训练集的通道统计量（均值/标准差）会被传递给验证集和测试集
    - 确保所有数据分区使用相同的标准化参数
    - 自动检查标签一致性

    返回:
        (train_dataset, val_dataset, test_dataset, labels_sorted)
    """
    datasets_config = data_config["datasets"]

    # --- 加载训练集 ---
    if "train" in datasets_config:
        train_dataset = get_dataset_partition_from_config(
            datasets_config["train"])
        # [MODIFIED] 安全提取 channel_means/stds：EEG 数据集不设这些属性时回退为 None
        train_channel_means = getattr(train_dataset.datasets[0], 'channel_means', None)
        train_channel_stds = getattr(train_dataset.datasets[0], 'channel_stds', None)
        train_labels_sorted = train_dataset.datasets[0].labels_sorted
        train_dataset = apply_dataset_wrappers_from_data_config(
            train_dataset, data_config)
    else:
        train_dataset = None
        train_labels_sorted = None
        train_channel_means = None
        train_channel_stds = None

    # --- 加载验证集（使用训练集的标准化参数） ---
    if "val" in datasets_config:
        val_dataset = get_dataset_partition_from_config(
            datasets_config["val"], train_channel_means, train_channel_stds)
        if train_labels_sorted is not None:
            check_labels(
                [train_labels_sorted, val_dataset.datasets[0].labels_sorted])
        val_dataset = apply_dataset_wrappers_from_data_config(
            val_dataset, data_config)
    else:
        val_dataset = None

    # 兼容处理：如果没有训练集标签，则使用验证集标签
    if train_labels_sorted is None:
        train_labels_sorted = val_dataset.datasets[0].labels_sorted

    # --- 加载测试集（使用训练集的标准化参数） ---
    if "test" in datasets_config:
        test_dataset = get_dataset_partition_from_config(
            datasets_config["test"], train_channel_means, train_channel_stds)
        if train_labels_sorted is not None:
            check_labels(
                [train_labels_sorted, test_dataset.datasets[0].labels_sorted])
        test_dataset = apply_dataset_wrappers_from_data_config(
            test_dataset, data_config)
    else:
        test_dataset = None

    return train_dataset, val_dataset, test_dataset, train_labels_sorted


# ============================================================================
# 保存评估结果到 JSON 文件（完整版，包含超参数配置和训练损失）
# ============================================================================
def log_results(result, y, preds, logits, output_path, run_name, hpo_config=None, trainer=None):
    """
    将评估结果保存到 results.json 文件，同时支持 Weights & Biases 在线记录。
    会处理 torch.Tensor 和 numpy array 的序列化问题。
    """
    if (hpo_config is not None):
        for conf in hpo_config:
            keys = [str(c) for c in conf[0]]
            key = "_".join(keys)
            value = conf[1]
            result[key] = value
    if (trainer is not None):
        result["train_loss"] = trainer.callback_metrics.get("train_loss")
    if (wandb.run is not None):
        wandb.log(result)
    result["targets"] = y
    result["preds"] = preds
    result["logits"] = logits
    del result["val_cm"]
    for key, value in result.items():
        if (isinstance(value, torch.Tensor)):
            result[key] = value.cpu().tolist()
        if (isinstance(value, np.ndarray)):
            result[key] = value.tolist()

    output_path = os.path.join(output_path, run_name)

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "results.json"), "w") as f:
        json.dump(result, f)


# ============================================================================
# 简化版结果保存函数：将结果字典写入 JSON 文件
# ============================================================================
def my_log_results(result, run_path, log_name):
    """将评估结果字典保存为带缩进的 JSON 文件（UTF-8 编码）"""
    with open(os.path.join(run_path, log_name), "w") as f:
        json.dump(result, f, indent=3, ensure_ascii=False)


# ============================================================================
# 统计各类别的样本数量
# ============================================================================
def get_label_counts(train_loader, n_classes):
    """
    遍历 DataLoader，统计每个类别的样本总数。

    参数:
        train_loader: DataLoader - 训练数据加载器
        n_classes: int - 类别数量

    返回:
        torch.Tensor - 每个类别的样本计数 (shape: [n_classes])
    """
    label_counts = torch.zeros(n_classes)
    for batch in train_loader:
        y = batch[1].flatten().long()
        label_counts += torch.bincount(y, minlength=n_classes)
    return label_counts


# ============================================================================
# 计算类别分布（频率）
# ============================================================================
def get_label_distribution(train_loader, n_classes):
    """计算类别分布频率，用于自动设置类别权重"""
    label_counts = get_label_counts(train_loader, n_classes)
    label_distribution = label_counts / label_counts.sum()
    return label_distribution


# ============================================================================
# run_training - PyTorch Lightning 训练封装函数（旧版，多 GPU 支持）
# ============================================================================
def run_training(train_loader, val_loader, config, n_classes, best_model_metric="val_f1_macro", module=None, best_model_metric_mode="max"):
    """
    使用 PyTorch Lightning 执行模型训练。

    参数:
        train_loader: 训练数据 DataLoader
        val_loader: 验证数据 DataLoader
        config: 完整配置字典
        n_classes: 类别数量
        best_model_metric: 用于选择最佳 checkpoint 的指标名
        module: 可选的已初始化模型（None 则从配置创建）
        best_model_metric_mode: "min" 或 "max"

    返回:
        (trainer, best_module, module)
    """
    if module is None:
        sfreq = config["data"]["datasets"]["train"][0]["eeg_speech_v1"].get("sfreq", 250.0)
        module = ClassificationModule(
            model_config=config["model"], n_classes=n_classes, optimizer_config=config["optimizer"], loss_config=config["loss"], sfreq=sfreq)

    logger = False
    if (config["general"]["wandb"]):
        logger = WandbLogger()
    elif ("tensorboard_logger" in config["general"] and config["general"]["tensorboard_logger"]):
        logger = TensorBoardLogger(
            save_dir=config["general"]["checkpoint_path"])

    callbacks = []
    if (config["general"]["checkpoint_path"] is not None):
        os.makedirs(config["general"]["checkpoint_path"], exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=config["general"]["checkpoint_path"],
            monitor=best_model_metric,
            mode=best_model_metric_mode,
            save_top_k=1,
            verbose=True,
            filename="best-" + best_model_metric +
            "-" + str(config["general"]["run_name"]) +
            "-{epoch:02d}-{val_f1_macro:.4f}",
            save_last=True
        )
        callbacks.append(checkpoint_callback)

    trainer_config = config["trainer"]
    trainer = Trainer(
        logger=logger,
        accelerator="gpu",
        devices=4,
        log_every_n_steps=1,
        callbacks=callbacks,
        **trainer_config
    )

    trainer.fit(module, train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    print("Debug message: loading: ", str(
        checkpoint_callback.best_model_path,))

    if trainer.global_rank == 0:
        best_module = ClassificationModule.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
        print("Loaded best model from:", checkpoint_callback.best_model_path)
    else:
        best_module = None
    best_module = trainer.strategy.broadcast(best_module)

    return trainer, best_module, module


# ============================================================================
# run_validation - 综合评估函数（多指标 + 随机基线 + 朴素基线）
# ============================================================================
def run_validation(val_loader, module, labels, samples_per_class=None):
    """
    在验证集上进行全面评估，计算多种指标并对比基线。

    基线对比：
    - 随机基线（Random）：完全随机的预测
    - 朴素基线（Naive）：按训练集类别分布随机采样

    评估指标包括：
    - Accuracy（Micro/Macro）
    - F1 Score（Micro/Macro/Weighted）
    - ROC-AUC
    - 二分类指标（Binary Accuracy, Precision, Recall, F1）
    - Jaccard Index（仅二分类）

    返回:
        (result_dict, targets, preds, logits)
    """
    disp_labels = labels
    module.eval()
    all_preds = []
    all_logits = []
    all_targets = []
    all_probas = []
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch[0], batch[1]
            x = x.to(module.device)
            y = y.to(module.device)
            outputs = module(x)
            all_logits.extend(outputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds)
            all_targets.extend(y)
            all_probas.extend(torch.nn.functional.softmax(outputs, dim=1))

    all_targets = torch.stack(all_targets)
    all_preds = torch.stack(all_preds)
    all_logits = torch.stack(all_logits)
    all_probas = torch.stack(all_probas)

    if (samples_per_class is not None):
        bincount = samples_per_class.to(module.device)
    else:
        import warnings
        warnings.warn(
            "No samples per class provided, using bincount of val dataset")
        bincount = torch.bincount(all_targets).to(module.device)
    most_common_class = torch.argmax(bincount)
    naive_acc = bincount[most_common_class] / len(all_targets)

    acc = Accuracy(task="multiclass", average="micro",
                num_classes=len(disp_labels)).to(module.device)
    bal_acc = Accuracy(task="multiclass", average="macro",
                    num_classes=len(disp_labels)).to(module.device)
    f1_macro = F1Score(task="multiclass", average="macro",
                    num_classes=len(disp_labels)).to(module.device)
    f1_micro = F1Score(task="multiclass", average="micro",
                    num_classes=len(disp_labels)).to(module.device)
    f1_weighted = F1Score(task="multiclass", average="weighted",
                        num_classes=len(disp_labels)).to(module.device)
    rocauc_macro = MulticlassAUROC(average="macro",
                                num_classes=len(disp_labels)).to(module.device)
    rocauc_micro = MulticlassAUROC(average="weighted",
                                num_classes=len(disp_labels)).to(module.device)
    loss = torch.nn.CrossEntropyLoss().to(module.device)(all_logits, all_targets)

    random_preds = torch.randint(
        0, len(disp_labels), (len(all_targets),), device=module.device)
    random_acc = acc(random_preds, all_targets)
    random_balanced_acc = bal_acc(random_preds, all_targets)
    random_f1_macro = f1_macro(random_preds, all_targets)
    random_f1_micro = f1_micro(random_preds, all_targets)
    random_f1_weighted = f1_weighted(random_preds, all_targets)

    naive_preds = torch.multinomial(
        bincount.float(), len(all_targets), replacement=True).to(module.device)
    naive_acc = acc(naive_preds, all_targets)
    naive_balanced_acc = bal_acc(naive_preds, all_targets)
    naive_f1_macro = f1_macro(naive_preds, all_targets)
    naive_f1_micro = f1_micro(naive_preds, all_targets)
    naive_f1_weighted = f1_weighted(naive_preds, all_targets)

    frequencies = bincount.float() / bincount.sum()
    naive_probas = [frequencies for _ in all_targets]
    naive_probas = torch.stack(naive_probas)
    naive_loss = torch.nn.NLLLoss()(torch.log(naive_probas), all_targets)

    naive_rocauc_macro = rocauc_macro(naive_probas, all_targets)
    naive_rocauc_micro = rocauc_micro(naive_probas, all_targets)

    acc_val = acc(all_preds, all_targets)
    bal_acc_val = bal_acc(all_preds, all_targets)
    f1_macro_val = f1_macro(all_preds, all_targets)
    f1_micro_val = f1_micro(all_preds, all_targets)
    rocauc_macro_val = rocauc_macro(all_probas, all_targets)
    rocauc_micro_val = rocauc_micro(all_probas, all_targets)
    f1_weighted_val = f1_weighted(all_preds, all_targets)

    result = {
        "val_cm": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_targets.cpu().numpy(),
            preds=all_preds.cpu().numpy(),
            class_names=disp_labels
        ),
        "val_naive_acc": naive_acc,
        "val_random_acc": random_acc,
        "val_random_bal_acc": random_balanced_acc,
        "val_random_f1_macro": random_f1_macro,
        "val_random_f1_micro": random_f1_micro,
        "val_random_f1_weighted": random_f1_weighted,
        "val_naive_bal_acc": naive_balanced_acc,
        "val_naive_f1_macro": naive_f1_macro,
        "val_naive_f1_micro": naive_f1_micro,
        "val_naive_f1_weighted": naive_f1_weighted,
        "val_naive_loss": naive_loss,
        "val_acc": acc_val,
        "val_bal_acc": bal_acc_val,
        "val_f1_macro": f1_macro_val,
        "val_f1_micro": f1_micro_val,
        "val_f1_weighted": f1_weighted_val,
        "val_loss": loss,
        "val_rocauc_macro": rocauc_macro_val,
        "val_rocauc_micro": rocauc_micro_val,
        "val_naive_rocauc_macro": naive_rocauc_macro,
        "val_naive_rocauc_micro": naive_rocauc_micro,
    }

    if (len(disp_labels) == 2):
        jaccard_index = JaccardIndex(
            task="multiclass", num_classes=2).to(module.device)
        jaccard_index_val = jaccard_index(all_preds, all_targets)
        jaccard_index_naive = jaccard_index(naive_preds, all_targets)
        result["val_jaccard_index"] = jaccard_index_val
        result["val_naive_jaccard_index"] = jaccard_index_naive

    binary_acc = Accuracy(task="binary").to(module.device)
    binary_bal_acc = Recall(task="multiclass", num_classes=2,
                            average="macro").to(module.device)
    binary_f1 = F1Score(task="binary").to(module.device)
    binary_rocauc = BinaryAUROC().to(module.device)
    classes = all_targets.unique()
    for c in classes:
        class_probas = all_probas[:, c]
        class_preds = all_preds == c
        class_targets = all_targets == c
        class_acc = binary_acc(class_preds, class_targets)
        class_f1 = binary_f1(class_preds, class_targets)
        class_bal_acc = binary_bal_acc(class_preds, class_targets)
        class_random_preds = random_preds == c
        class_random_acc = binary_acc(class_random_preds, class_targets)
        class_random_f1 = binary_f1(class_random_preds, class_targets)
        class_naive_preds = naive_preds == c
        class_naive_acc = binary_acc(class_naive_preds, class_targets)
        class_naive_bal_acc = binary_bal_acc(class_naive_preds, class_targets)
        class_naive_f1 = binary_f1(class_naive_preds, class_targets)
        class_rocauc = binary_rocauc(class_probas, class_targets)
        result[f"val_class_{c}_acc"] = class_acc
        result[f"val_class_{c}_f1"] = class_f1
        result[f"val_class_{c}_random_acc"] = class_random_acc
        result[f"val_class_{c}_random_f1"] = class_random_f1
        result[f"val_class_{c}_naive_acc"] = class_naive_acc
        result[f"val_class_{c}_naive_bal_acc"] = class_naive_bal_acc
        result[f"val_class_{c}_naive_f1"] = class_naive_f1
        result[f"val_class_{c}_bal_acc"] = class_bal_acc
        result[f"val_class_{c}_rocauc"] = class_rocauc
    return result, all_targets.cpu().numpy(), all_preds.cpu().numpy(), all_logits.cpu().numpy()


# ============================================================================
# adapt_config_to_data - 根据实际数据动态调整配置
# ============================================================================
def adapt_config_to_data(config, train_loader, labels):
    """
    在数据加载完成后调整配置：
    1. 如果损失权重设置为 "auto"，则根据类别分布自动计算权重
       （使用逆频率加权，样本少的类别获得更大权重）
    2. 如果模型中的 n_groups 设置为 "auto"，则自动匹配分组数

    参数:
        config: dict - 配置字典（原地修改）
        train_loader: DataLoader - 训练数据加载器
        labels: list - 类别标签列表
    """
    n_classes = len(labels)
    class_weights = None

    if ("loss" in config and "config" in config["loss"] and "weight" in config["loss"]["config"]):
        if (config["loss"]["config"]["weight"] == "auto"):
            if (class_weights is None):
                class_weights = get_label_distribution(train_loader, n_classes)
            inverse_class_freq = 1 / class_weights
            config["loss"]["config"]["weight"] = inverse_class_freq

    if "grouped_samples" in config["data"]["general"]:
        for layer in config["model"]:
            layer_name = list(layer.keys())[0]
            layer_dict = layer[layer_name]
            if (layer_dict is None):
                continue
            if ("n_groups" in layer_dict and layer_dict["n_groups"] == "auto"):
                layer_dict["n_groups"] = config["data"]["general"]["grouped_samples"]
