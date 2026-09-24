import sys
import os
# 获取 train.py 的绝对路径，并获取其父目录
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)  # 插入到开头，确保优先级最高
import glob
import os.path as op
from tqdm import tqdm
import numpy as np
import librosa
import pandas as pd
import h5py
from mne.filter import resample

if __name__ == "__main__":
    
    data_path = os.environ.get('LIBRIBRAIN_DATA_ROOT', op.join(parent_dir, 'data'))
    if not op.isdir(data_path):
        raise FileNotFoundError(
            f"Set LIBRIBRAIN_DATA_ROOT to a writable dataset copy before preprocessing: {data_path}"
        )
    sfreq = 250
    books = [f"Sherlock{i}" for i in range(1, 8)] 
    ch_types = np.load("ch_types.npz", allow_pickle=True)
    grad_ch = ch_types['grad_ch']

    for book in tqdm(books, desc="BOOKS"):
        events_dir = op.join(data_path, book, "derivatives", "events")
        megs_dir = op.join(data_path, book, "derivatives", "serialised")
        for filename in tqdm(os.listdir(events_dir), desc="FILES"):
            file = filename.replace("events.tsv", "")
            event_path = op.join(events_dir, filename)
            meg_path = op.join(megs_dir, file+"proc-bads+headpos+sss+notch+bp+ds_meg.h5")
            target_path = meg_path.replace("meg.h5","grad.npy")

            events_df = pd.read_csv(event_path, sep='\t')
            meg = h5py.File(meg_path, "r")["data"][:]

            events_df['timemeg_samples'] = (pd.to_numeric(
                events_df['timemeg'], errors='coerce') * sfreq).astype(int)
            events_df['duration_samples'] = (pd.to_numeric(
                events_df['duration'], errors='coerce') * sfreq).astype(int)
            silence_df = events_df[events_df['kind'] == 'silence']
            words_df = events_df[events_df['kind'] == 'word']
            max_word_sample_time = (words_df['timemeg_samples'] +
                      words_df['duration_samples']).max()
            max_silence_sample_time = (silence_df['timemeg_samples'] +
                        silence_df['duration_samples']).max()
            max_len = max(max_word_sample_time,max_silence_sample_time) + 1
            start_sample = int(max(0, events_df.loc[0,'timemeg_samples']))

            meg = meg[:,start_sample:max_len]
            meg = meg.astype(np.float64)
            meg = resample(meg, 100, 250)

            meg_mean = np.mean(meg, axis=1, keepdims=True)  # [306, 1]
            meg_std = np.std(meg, axis=1, keepdims=True)   # [306, 1]
            meg = (meg - meg_mean) / meg_std   # [306, T]

            meg = meg.astype(np.float16)
            meg_grad = meg[grad_ch, :]

            np.save(target_path, meg_grad)
            print(file, meg_grad.shape)
    
    holdout_meg_path = op.join(data_path, "COMPETITION_HOLDOUT", "derivatives", "serialised", "sub-0_ses-2025_task-COMPETITION_HOLDOUT_run-1_proc-bads+headpos+sss+notch+bp+ds_meg.h5")
    holdout_target_path = holdout_meg_path.replace("meg.h5", "grad.npy")

    holdout_meg = h5py.File(holdout_meg_path, "r")["data"][:]
    holdout_meg = holdout_meg.astype(np.float64)
    holdout_meg = resample(holdout_meg, 100, 250)

    holdout_meg_mean = np.mean(holdout_meg, axis=1, keepdims=True)  # [306, 1]
    holdout_meg_std = np.std(holdout_meg, axis=1, keepdims=True)   # [306, 1]
    holdout_meg = (holdout_meg - holdout_meg_mean) / holdout_meg_std   # [306, T]

    holdout_meg = holdout_meg.astype(np.float16)
    holdout_meg_grad = holdout_meg[grad_ch, :]

    np.save(holdout_target_path, holdout_meg_grad)
    print("holdout", holdout_meg_grad.shape)



