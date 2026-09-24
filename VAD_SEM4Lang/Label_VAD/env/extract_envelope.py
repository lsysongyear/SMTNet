#!/usr/bin/env python3
"""
从 WAV 音频文件中提取单通道宽带包络特征。

处理流程:
  1. 加载音频，去均值
  2. 计算宽带包络: |Hilbert(audio)|  → (T_raw,)
  3. 降采样到 512Hz
  4. 2-8Hz 带通滤波 (Butterworth 4阶, zero-phase)
  5. 降采样到 64Hz  → (T_64Hz, 1)

输入: raw_rename/audio/story_{XX}.wav
输出: preprocess/env/story-{XX}_envelope_64Hz.npy  shape (T,)

Usage:
    python extract_envelope.py
    python extract_envelope.py --overwrite
"""

import os
import sys
import glob
import argparse
import numpy as np
import scipy.signal
import librosa
from tqdm import tqdm

# ── 路径配置 ──────────────────────────────────────────────────
PROJ_ROOT = "/dfs/share/chenjingLab/speech_tracking/dataset/SEM4Lang/SEM4Lang"
AUDIO_DIR = os.path.join(PROJ_ROOT, "raw_rename", "audio")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ── 参数 ──────────────────────────────────────────────────────
TARGET_SR = 64           # 最终采样率 Hz
INTERMEDIATE_SR = 512    # 中间采样率 Hz（带通滤波前）
BP_LO = 2.0              # 带通低截止 Hz
BP_HI = 8.0              # 带通高截止 Hz
BP_ORDER = 4             # 带通滤波器阶数


# ── 单文件处理 ────────────────────────────────────────────────

def extract_envelope(audio_path, output_path):
    """提取宽带 Hilbert 包络，带通滤波，降采样到 64Hz。

    Returns
    -------
    n_samples : int
        输出数组的长度
    """
    # 1. 加载音频
    audio, sr = librosa.load(audio_path, sr=None)
    audio = audio.astype(np.float32)
    audio -= np.mean(audio)

    # 2. 宽带包络: |Hilbert(audio)|
    analytic = scipy.signal.hilbert(audio)
    env = np.abs(analytic).astype(np.float32)  # (T_raw,)

    # 3. 降采样到 512Hz
    env_512 = scipy.signal.resample_poly(env, INTERMEDIATE_SR, sr)
    env_512 = env_512.astype(np.float32)

    # 4. 2-8Hz 带通滤波 (zero-phase)
    sos = scipy.signal.butter(BP_ORDER, (BP_LO, BP_HI), "band",
                               fs=INTERMEDIATE_SR, output="sos")
    env_bp = scipy.signal.sosfiltfilt(sos, env_512)

    # 5. 降采样到 64Hz
    env_64 = scipy.signal.resample_poly(env_bp, TARGET_SR, INTERMEDIATE_SR)
    env_64 = env_64.astype(np.float32)

    # 6. 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, env_64)

    return len(env_64)


# ── 主函数 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="从 WAV 提取单通道宽带包络 (Hilbert → 2-8Hz BP → 64Hz)")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已存在的输出文件")
    parser.add_argument("--audio-dir", default=AUDIO_DIR)
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    audio_files = sorted(glob.glob(os.path.join(args.audio_dir, "story_*.wav")))
    print(f"音频目录: {args.audio_dir}")
    print(f"输出目录: {args.out_dir}")
    print(f"找到 {len(audio_files)} 个音频文件")
    print(f"参数: Hilbert 包络, {BP_LO}-{BP_HI}Hz BP, → {TARGET_SR}Hz")
    print()

    processed = 0
    skipped = 0
    errors = 0

    for audio_path in tqdm(audio_files, desc="提取包络"):
        basename = os.path.basename(audio_path)     # story_01.wav
        story_num = basename.replace("story_", "").replace(".wav", "")
        output_path = os.path.join(args.out_dir, f"story-{story_num}_envelope_64Hz.npy")

        if os.path.exists(output_path) and not args.overwrite:
            skipped += 1
            continue

        try:
            n = extract_envelope(audio_path, output_path)
            processed += 1
        except Exception as e:
            print(f"\n  错误 ({basename}): {e}", file=sys.stderr)
            errors += 1

    print(f"\n完成. 处理: {processed}, 跳过: {skipped}, 错误: {errors}")


if __name__ == "__main__":
    main()
