#!/usr/bin/env python3
"""
EEG 预处理：512Hz → 250Hz 降采样 + 1-80Hz 带通滤波。

输入: raw_rename/eeg/sub-{XXX}_audio-{story}_NF_512Hz.npy  shape (64, T)
输出: preprocess/eeg/sub-{XXX}_audio-{story}_BP_250Hz.npy  shape (64, T)
"""

import os
import argparse
import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
from tqdm import tqdm

EEG_INPUT_DIR = "/dfs/share/chenjingLab/speech_tracking/dataset/SparKULee/raw_rename/eeg"
EEG_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eeg_1-80Hz")

FS_RAW = 512.0      # 原始采样率
FS_TARGET = 250.0    # 目标采样率
BP_LOW = 1.0         # 带通低频截止 (Hz)
BP_HIGH = 80.0       # 带通高频截止 (Hz)
BP_ORDER = 4         # Butterworth 滤波器阶数


def bandpass_filter(data, fs, lowcut, highcut, order=4):
    """零相位 Butterworth 带通滤波。"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    # 沿时间轴滤波 (axis=1 for (C, T))
    return filtfilt(b, a, data, axis=1)


def process_eeg(input_path, output_path):
    """降采样 + 带通滤波单个 EEG 文件。"""
    eeg = np.load(input_path).astype(np.float64)  # (64, T)

    # 1. 带通滤波 1-40 Hz
    eeg = bandpass_filter(eeg, FS_RAW, BP_LOW, BP_HIGH, BP_ORDER)

    # 2. 降采样 512Hz → 250Hz: 因子 512/250 = 256/125
    eeg_ds = resample_poly(eeg, up=125, down=256, axis=1)  # (64, T')

    eeg_ds = eeg_ds.astype(np.float32)

    np.save(output_path, eeg_ds)
    return eeg_ds.shape[1] / FS_TARGET  # duration in seconds


def main():
    parser = argparse.ArgumentParser(description="EEG preprocessing: 512→250Hz + 1-40Hz BP")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的文件")
    parser.add_argument("--input-dir", default=EEG_INPUT_DIR)
    parser.add_argument("--output-dir", default=EEG_OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    input_files = sorted(
        f for f in os.listdir(args.input_dir)
        if f.endswith("_NF_512Hz.npy") and f.startswith("sub-")
    )
    input_files = [os.path.join(args.input_dir, f) for f in input_files]
    print(f"Found {len(input_files)} EEG files")
    print(f"Filter: {BP_LOW}-{BP_HIGH} Hz bandpass, order={BP_ORDER}")
    print(f"Resample: {FS_RAW} → {FS_TARGET} Hz")

    processed = 0
    skipped = 0

    for input_path in tqdm(input_files, desc="Processing EEG"):
        basename = os.path.basename(input_path)
        # sub-001_audio-audiobook-1_NF_512Hz.npy → sub-001_audio-audiobook-1_BP_250Hz.npy
        output_name = basename.replace("_NF_512Hz.npy", "_BP_250Hz.npy")
        output_path = os.path.join(args.output_dir, output_name)

        if os.path.exists(output_path) and not args.overwrite:
            skipped += 1
            continue

        try:
            dur = process_eeg(input_path, output_path)
            processed += 1
        except Exception as e:
            print(f"Error processing {basename}: {e}")

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
