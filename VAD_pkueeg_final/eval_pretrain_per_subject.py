"""
使用预训练模型逐被试评估（不微调，遍历 top-5 ckpt × 99 阈值，以 val_macro_acc 选最优组合）。

用法:
  python eval_pretrain_per_subject.py
  python eval_pretrain_per_subject.py --subjects 1-10
  python eval_pretrain_per_subject.py --ckpt-dir <path/to/checkpoints/>
"""

import os, sys, argparse, json, glob
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from speech_code.utils import get_datasets_from_config
from speech_code.models.my_modules.classification_module import ClassificationModule
from speech_code.data_split import DataSplitter, apply_split_to_config

# ============================================================================
# 常量
# ============================================================================
CONFIG_PATH = "configs/speech/my_run/config.yaml"

ALL_SUBJECTS = [f"{i:02d}" for i in range(1, 26)]
ALL_STORIES = list(range(1, 51))
THRESHOLD_SEARCH = np.arange(0.01, 1.0, 0.01)  # 0.01 ~ 0.99


def parse_range(spec):
    result = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            for i in range(int(lo), int(hi) + 1):
                result.add(f"{i:02d}")
        else:
            result.add(f"{int(part):02d}")
    return sorted(result)


def build_loaders(subject_id, split_seed=545):
    """为指定被试构建 val/test DataLoader。"""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    splitter = DataSplitter(ALL_SUBJECTS, ALL_STORIES, n_folds=5, seed=split_seed)
    split = splitter.get_split("per_sub_5t5v", 0)
    subj_split = split[subject_id]

    for partition in ['train', 'val', 'test']:
        config['data']['datasets'][partition][0]['eeg_speech_v1']['include_subjects'] = [subject_id]
        config['data']['datasets'][partition][0]['eeg_speech_v1']['stories'] = subj_split[f'{partition}_stories']

    config['data']['dataloader']['batch_size'] = 64
    config['data']['dataloader']['num_workers'] = 4

    _, val_dataset, test_dataset, _ = get_datasets_from_config(config['data'])
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=64, num_workers=4)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64, num_workers=4)
    return val_loader, test_loader, subj_split


def run_inference(model, loader, device):
    """推理并返回 probas, targets。"""
    model.eval()
    all_probas, all_targets = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            subj_ids = _extract_ids(batch, device)
            probas = model(x, subject_ids=subj_ids)
            all_probas.append(probas.cpu().flatten())
            all_targets.append(batch[1].flatten())
    return torch.cat(all_probas).numpy(), torch.cat(all_targets).numpy()


def _extract_ids(batch, device):
    if len(batch) < 3:
        return None
    info = batch[2]
    if isinstance(info, dict) and 'subject_id' in info:
        s = info['subject_id']
        return s.to(device) if isinstance(s, torch.Tensor) else torch.tensor(s, device=device)
    if isinstance(info, list) and len(info) > 0 and isinstance(info[0], dict):
        if 'subject_id' in info[0]:
            return torch.tensor([d['subject_id'] for d in info], device=device)
    return None


def compute_metrics(targets, preds):
    tp = np.sum((preds == 1) & (targets == 1))
    fp = np.sum((preds == 1) & (targets == 0))
    tn = np.sum((preds == 0) & (targets == 0))
    fn = np.sum((preds == 0) & (targets == 1))
    p0 = tn/(tn+fn) if (tn+fn) > 0 else 0; r0 = tn/(tn+fp) if (tn+fp) > 0 else 0
    p1 = tp/(tp+fp) if (tp+fp) > 0 else 0; r1 = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1_0 = 2*p0*r0/(p0+r0) if (p0+r0) > 0 else 0
    f1_1 = 2*p1*r1/(p1+r1) if (p1+r1) > 0 else 0
    return {
        'binary_acc': float(np.mean(preds == targets)),
        'macro_acc': float((r0 + r1) / 2),
        'macro_f1': float((f1_0 + f1_1) / 2),
        'class0_precision': float(p0), 'class0_recall': float(r0), 'class0_f1': float(f1_0),
        'class1_precision': float(p1), 'class1_recall': float(r1), 'class1_f1': float(f1_1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', type=str, default='1-25')
    parser.add_argument('--ckpt-dir', type=str, default=None,
                        help='Checkpoint directory (auto-detect if not set)')
    parser.add_argument('--seed', type=int, default=545, help='Split seed')
    args = parser.parse_args()

    subject_list = parse_range(args.subjects)
    print(f"Subjects: {subject_list}")

    # Find checkpoint directory
    ckpt_dir = args.ckpt_dir
    if ckpt_dir is None:
        # Read config to match the current model
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        model_name = next(iter(cfg["model"]))
        pattern = f"results/speech-detection/model:{model_name}-*/split:per_sub_5t5v/*/pretrain/run*/checkpoints"
        matches = sorted(glob.glob(pattern, recursive=True))
        if not matches:
            raise FileNotFoundError(
                f"No checkpoints/ found matching model '{model_name}'. "
                f"Run pretrain first or provide --ckpt-dir.")
        ckpt_dir = matches[-1]  # pick the most recent run
        print(f"Auto-detected ckpt dir (model={model_name}, {len(matches)} total): {ckpt_dir}")

    ckpt_paths = sorted(glob.glob(os.path.join(glob.escape(ckpt_dir), "*.ckpt")))
    if not ckpt_paths:
        raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")
    print(f"Found {len(ckpt_paths)} checkpoint(s)")

    # Derive result base from ckpt directory: everything before "/pretrain/"
    result_base = ckpt_dir.rsplit("/pretrain/", 1)[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_metrics = []
    for subj in subject_list:
        print(f"\n{'='*50}")
        print(f"Subject sub-{subj}")

        val_loader, test_loader, split = build_loaders(subj, args.seed)
        print(f"  Val stories: {split['val_stories']}  Test stories: {split['test_stories']}")

        # --- Search best (ckpt, threshold) on val set by macro_acc ---
        best_val_acc = -1
        best_ckpt = None
        best_thresh = 0.5

        for ckpt_path in ckpt_paths:
            model = ClassificationModule.load_from_checkpoint(ckpt_path)
            model = model.to(device)
            model.eval()

            val_probas, val_targets = run_inference(model, val_loader, device)
            ckpt_best_acc = -1
            ckpt_best_thr = 0.5
            for t in THRESHOLD_SEARCH:
                preds = (val_probas >= t).astype(int)
                m = compute_metrics(val_targets, preds)
                if m['macro_acc'] > ckpt_best_acc:
                    ckpt_best_acc = m['macro_acc']
                    ckpt_best_thr = t

            if ckpt_best_acc > best_val_acc:
                best_val_acc = ckpt_best_acc
                best_ckpt = ckpt_path
                best_thresh = ckpt_best_thr
                best_val_probas = val_probas
                best_val_targets = val_targets

            del model
            torch.cuda.empty_cache()

        ckpt_name = os.path.basename(best_ckpt)
        print(f"  Best (ckpt, thr): ({ckpt_name}, {best_thresh:.2f})  val_macro_acc={best_val_acc:.4f}")

        # --- Test: use best ckpt + threshold ---
        best_model = ClassificationModule.load_from_checkpoint(best_ckpt)
        best_model = best_model.to(device)
        best_model.eval()
        test_probas, test_targets = run_inference(best_model, test_loader, device)
        test_preds = (test_probas >= best_thresh).astype(int)
        test_m = compute_metrics(test_targets, test_preds)
        print(f"  Test: macro_acc={test_m['macro_acc']:.4f}  macro_f1={test_m['macro_f1']:.4f}")
        del best_model
        torch.cuda.empty_cache()

        # Save
        out_dir = os.path.join(result_base, f"sub-{subj}", "pretrain_eval")
        os.makedirs(out_dir, exist_ok=True)
        np.savez(os.path.join(out_dir, 'test_results.npz'),
                 probas=test_probas.astype(np.float32),
                 targets=test_targets.astype(np.int64),
                 preds=test_preds.astype(np.int64),
                 threshold=best_thresh)
        test_m['subject'] = subj
        test_m['threshold'] = float(best_thresh)
        test_m['best_ckpt'] = best_ckpt
        test_m['val_macro_acc'] = float(best_val_acc)
        with open(os.path.join(out_dir, 'test_log.json'), 'w') as f:
            json.dump(test_m, f, indent=2)
        all_metrics.append(test_m)

    # Aggregate
    print(f"\n{'='*60}")
    print(f"Summary ({len(all_metrics)} subjects)")
    print(f"{'='*60}")
    for mk in ['macro_acc', 'macro_f1', 'binary_acc']:
        vals = [m[mk] for m in all_metrics]
        print(f"  {mk}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


if __name__ == '__main__':
    main()
