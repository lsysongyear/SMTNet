import torch
import numpy as np

def ch_transform(eeg_data, ch_names):
    common_channels = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'PZ', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'O1', 'OZ', 'O2']
    ch_num = 57
    eeg_57_ret = np.zeros((57, eeg_data.shape[1]))
    cnt = 0
    for channel in common_channels:
        index=0
        for ch_name in ch_names:
            if ch_name.upper() == channel:
                eeg_57_ret[cnt,:] = eeg_data[index,:]
                cnt += 1
                break
            index += 1
    return eeg_57_ret
