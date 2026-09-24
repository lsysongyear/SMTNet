# ============================================================================
# data_process.py - EEG 数据预处理脚本 (NEW for PKU EEG dataset)
# ============================================================================
# 功能说明：
#   将原始 PKU EEG .npz 文件进行以下预处理：
#   1. 加载 raw npz（eeg_data (64, T_250Hz), ch_names）
#   2. 通道对齐：ch_transform (64 → 57 common channels)
#   3. 降采样：250Hz → 100Hz（使用 MNE resample）
#   4. 全试次级 z-score 标准化（每个通道独立）
#   5. 保存为 float32 .npy 文件到 eeg_processed 目录
#
# 处理范围：
#   - 25 名被试 (sub-01 ~ sub-25)
#   - 50 个 story trial 每被试 (story_1 ~ story_50)
#
# 使用方法（在训练前运行一次）：
#   python speech_code/data_process.py
#
# 【注意】EEG_Speech_v1 dataset 类内置了 on-the-fly 预处理能力（首次加载时
# 自动做通道对齐+降采样+标准化+缓存），因此运行本脚本是可选的加速步骤。
# 如需跳过预处理直接训练，确保 config 中的 data_path 指向 raw npz 目录即可。
# ============================================================================

# ============================================================================
# [COMMENTED OUT] Original LibriBrain MEG preprocessing:
#   - Load HDF5 MEG data (306ch, 250Hz)
#   - Crop to valid range from events.tsv
#   - Resample 250→100Hz (MNE)
#   - z-score per channel
#   - float16 + grad channel selection (306→204)
#   - Save as .npy
# See git history for the original implementation.
# ============================================================================

import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'preprocess'))
import glob
import os.path as op
from tqdm import tqdm
import numpy as np
from ch_transform import ch_transform

if __name__ == "__main__":

    # ========================================================================
    # 全局配置
    # ========================================================================
    # 原始 EEG 数据目录
    input_dir = os.environ.get("PKU_RAW_DIR", "/gpfs/share/home/2201112028/lsycode/pkueeg")
    # 预处理输出目录
    output_dir = os.environ.get("PKU_CACHE_DIR", os.path.join(parent_dir, "eeg_processed"))
    # 原始采样率
    sfreq_raw = 250.0
    # 目标采样率
    sfreq_target = 100.0
    # 被试列表
    subjects = [f"{i:02d}" for i in range(1, 26)]
    # trial 列表（跳过 resting-state）
    story_ids = list(range(1, 51))

    try:
        from mne.filter import resample as mne_resample
        print("Using MNE resample")
    except ImportError:
        from scipy.signal import resample as mne_resample
        print("MNE not available, using scipy.signal.resample instead")

    os.makedirs(output_dir, exist_ok=True)

    # ========================================================================
    # 预处理所有被试和试次
    # ========================================================================
    for subject in tqdm(subjects, desc="Subjects"):
        subj_input = op.join(input_dir, f"sub-{subject}")
        subj_output = op.join(output_dir, f"sub-{subject}")
        os.makedirs(subj_output, exist_ok=True)

        for story_id in tqdm(story_ids, desc=f"Stories (sub-{subject})", leave=False):
            npz_path = op.join(subj_input, f"story_{story_id}.npz")
            if not op.exists(npz_path):
                print(f"  Warning: {npz_path} not found, skipping")
                continue

            # --- 步骤 1: 加载 raw npz ---
            data = np.load(npz_path, allow_pickle=True)
            eeg = data['eeg_data'].astype(np.float64)  # (64, T_250Hz)
            ch_names = data['ch_names']

            # --- 步骤 2: 通道对齐 64 → 57 ---
            eeg_57 = ch_transform(eeg, ch_names)  # (57, T_250Hz)

            # --- 步骤 3: 降采样 250Hz → 100Hz ---
            eeg_100hz = mne_resample(eeg_57, sfreq_target, sfreq_raw)  # (57, T_100Hz)

            # --- 步骤 4: 全试次级别 z-score 标准化 ---
            ch_means = np.mean(eeg_100hz, axis=1, keepdims=True)
            ch_stds = np.std(eeg_100hz, axis=1, keepdims=True)
            ch_stds[ch_stds == 0] = 1.0
            eeg_norm = (eeg_100hz - ch_means) / ch_stds

            # --- 步骤 5: 保存为 float32 ---
            output_path = op.join(subj_output, f"story_{story_id}_proc-eeg.npy")
            np.save(output_path, eeg_norm.astype(np.float32))

    print(f"\nDone! Preprocessed data saved to {output_dir}")
