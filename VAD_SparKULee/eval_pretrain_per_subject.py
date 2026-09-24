"""
使用预训练模型在单名被试上评估（不微调，验证集阈值搜索 + 测试）。

用法:
  python eval_pretrain_per_subject.py --subj 001
  python eval_pretrain_per_subject.py --subj 001 --seed 42
  python eval_pretrain_per_subject.py --subj 001 --ckpt <path>
"""

import os, sys, argparse, json
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from speech_code.utils import get_datasets_from_config
from speech_code.models.my_modules.classification_module import ClassificationModule
from speech_code.data_split import DataSplitter, apply_split_to_config

CONFIG_PATH = "configs/speech/my_run/config.yaml"
RESULT_BASE = (
    "results/speech-detection/model:brain_magic_speech_v4-dataset:sparkulee_vad/"
    "split:per_sub/"
    "input_dim-64_attention_dim-128_output_dim-1_kernel_size-25_"
    "depthwise_kernel-30_num_blocks1-9_num_blocks2-15_dropout-0.1_"
    "n_subjects-85_weight-1.0_1.0_loss_type-mse_lr-0.001"
)

ALL_SUBJECTS = [f"{i:03d}" for i in range(1, 86)]
ALL_STORIES = None  # will be detected from data
THRESHOLDS = np.arange(0.01, 1.0, 0.01)


def get_all_stories(data_path):
    stories = set()
    subj_stories_map = {}
    for f in os.listdir(data_path):
        if f.endswith('.npy') and '_NF_64Hz' in f:
            parts = f[:-4].split('_')
            subj = parts[0].replace('sub-', '')
            story = '_'.join(parts[1:-2]).replace('audio-', '', 1)
            stories.add(story)
            subj_stories_map.setdefault(subj, set()).add(story)
    return sorted(stories), {s: sorted(v) for s, v in subj_stories_map.items()}


def build_loaders(subject_id, split_seed=42):
    """构建单被试的 val/test DataLoader (per_sub split)。"""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    data_path = config['data']['datasets']['train'][0]['sparkulee_vad']['data_path']
    all_stories, subj_stories_map = get_all_stories(data_path)

    splitter = DataSplitter(ALL_SUBJECTS, all_stories, n_folds=5, seed=split_seed,
                            subject_stories_map=subj_stories_map)
    split = splitter.get_split("per_sub", 0)
    subj_split = split[subject_id]

    for partition in ['train', 'val', 'test']:
        config['data']['datasets'][partition][0]['sparkulee_vad']['include_subjects'] = [subject_id]
        config['data']['datasets'][partition][0]['sparkulee_vad']['stories'] = subj_split[f'{partition}_stories']

    config['data']['dataloader']['batch_size'] = 64
    config['data']['dataloader']['num_workers'] = 4

    _, val_dataset, test_dataset, _ = get_datasets_from_config(config['data'])
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=64, num_workers=4)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64, num_workers=4)
    return val_loader, test_loader, subj_split


def run_inference(model, loader, device):
    model.eval()
    all_probas, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            subj = _extract_ids(batch, device)
            probas = model(x, subject_ids=subj)
            all_probas.append(probas.cpu().flatten())
            all_targets.append(batch[1].flatten())
    return torch.cat(all_probas).numpy(), torch.cat(all_targets).numpy()


def _extract_ids(batch, device, debug=False):
    if len(batch) < 3: return None
    info = batch[2]
    if isinstance(info, dict) and 'subject_id' in info:
        s = info['subject_id']
        result = s.to(device) if isinstance(s, torch.Tensor) else torch.tensor(s, device=device)
        if debug:
            print(f"  [DEBUG] collated dict: subject_id={result.tolist()}")
        return result
    if isinstance(info, list) and info and isinstance(info[0], dict):
        if 'subject_id' in info[0]:
            result = torch.tensor([d['subject_id'] for d in info], device=device)
            if debug:
                print(f"  [DEBUG] list of dicts: subject_id={result.tolist()}")
            return result
    if debug:
        print(f"  [DEBUG] No subject_id found in batch[2]")
    return None


def compute_metrics(targets, preds):
    tp = np.sum((preds == 1) & (targets == 1)); fp = np.sum((preds == 1) & (targets == 0))
    tn = np.sum((preds == 0) & (targets == 0)); fn = np.sum((preds == 0) & (targets == 1))
    p0 = tn/(tn+fn) if (tn+fn)>0 else 0; r0 = tn/(tn+fp) if (tn+fp)>0 else 0
    p1 = tp/(tp+fp) if (tp+fp)>0 else 0; r1 = tp/(tp+fn) if (tp+fn)>0 else 0
    f1_0 = 2*p0*r0/(p0+r0) if (p0+r0)>0 else 0
    f1_1 = 2*p1*r1/(p1+r1) if (p1+r1)>0 else 0
    return {
        'binary_acc': float(np.mean(preds == targets)),
        'macro_acc': float((r0 + r1) / 2),
        'macro_f1': float((f1_0 + f1_1) / 2),
        'class0_precision': float(p0), 'class0_recall': float(r0), 'class0_f1': float(f1_0),
        'class1_precision': float(p1), 'class1_recall': float(r1), 'class1_f1': float(f1_1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subj', type=str, required=True, help='Subject ID (e.g. 001)')
    parser.add_argument('--seed', type=int, default=42, help='Split seed')
    parser.add_argument('--ckpt', type=str, default=None, help='Pretrained checkpoint')
    args = parser.parse_args()

    subject_id = f"{int(args.subj):03d}"

    # Find checkpoint
    ckpt = args.ckpt
    if ckpt is None:
        ckpt_txt = os.path.join(RESULT_BASE, "pretrain", "best_ckpt_fold0.txt")
        if not os.path.exists(ckpt_txt):
            ckpt_txt = os.path.join(RESULT_BASE, "pretrain", "best_ckpt.txt")
        with open(ckpt_txt) as f:
            ckpt = f.read().strip()
        print(f"Auto-detected: {ckpt}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClassificationModule.load_from_checkpoint(ckpt)
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    val_loader, test_loader, split = build_loaders(subject_id, args.seed)
    print(f"Subject sub-{subject_id}")
    print(f"  Train stories: {len(split['train_stories'])}  Val: {split['val_stories']}  Test: {split['test_stories']}")

    # Validation threshold search
    val_probas, val_targets = run_inference(model, val_loader, device)
    best_f1, best_thresh = -1, 0.5
    for t in THRESHOLDS:
        preds = (val_probas >= t).astype(int)
        m = compute_metrics(val_targets, preds)
        if m['macro_f1'] > best_f1:
            best_f1 = m['macro_f1']
            best_thresh = t
    print(f"  Val best: thresh={best_thresh:.2f} macro_f1={best_f1:.4f}")

    # Test evaluation
    test_probas, test_targets = run_inference(model, test_loader, device)
    test_preds = (test_probas >= best_thresh).astype(int)
    test_m = compute_metrics(test_targets, test_preds)
    print(f"  Test: macro_acc={test_m['macro_acc']:.4f}  macro_f1={test_m['macro_f1']:.4f}")
    print(f"  Silence: P={test_m['class0_precision']:.4f} R={test_m['class0_recall']:.4f} F1={test_m['class0_f1']:.4f}")
    print(f"  Voice:   P={test_m['class1_precision']:.4f} R={test_m['class1_recall']:.4f} F1={test_m['class1_f1']:.4f}")

    # Save results in unified directory
    out_dir = os.path.join(RESULT_BASE, "pretrain_eval")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f'sub-{subject_id}_results.npz'),
             probas=test_probas.astype(np.float32),
             targets=test_targets.astype(np.int64),
             preds=test_preds.astype(np.int64),
             threshold=best_thresh)
    test_m['threshold'] = float(best_thresh)
    test_m['subject'] = subject_id
    test_m['val_best_f1'] = float(best_f1)
    with open(os.path.join(out_dir, f'sub-{subject_id}_test_log.json'), 'w') as f:
        json.dump(test_m, f, indent=2)
    print(f"\nSaved to: {out_dir}")


if __name__ == '__main__':
    main()
