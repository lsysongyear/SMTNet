"""
train_v1.py — 预训练 + 逐被试评估一站式脚本 (SparKULee 适配版)。

用法:
  python speech_code/train_v1.py \
    --config=configs/speech/my_run/config.yaml \
    --search-space=configs/speech/my_run/search-space.yaml \
    --split-strategy random_trial --fold 0 --split-seed 42 --pretrain
"""

import sys, os, argparse, json, glob, importlib, itertools, re, copy, random, time, shutil
from pathlib import Path

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import yaml
import numpy as np
import torch
import pytorch_lightning as lightning
from torch.utils.data import DataLoader

from speech_code.utils import my_log_results, get_datasets_from_config, adapt_config_to_data
from speech_code.data_split import (
    DataSplitter, apply_split_to_config, get_dataset_from_pairs, SUPPORTED_STRATEGIES)
from speech_code.models.my_modules.classification_module import ClassificationModule, WeightedMSEByLabel


# ============================================================================
# SparKULee-specific: scan data directory for subjects & stories
# ============================================================================
def get_dataset_key(config, partition="train"):
    return next(iter(config["data"]["datasets"][partition][0]))


def get_dataset_config(config, partition="train"):
    return config["data"]["datasets"][partition][0][get_dataset_key(config, partition)]


def infer_split_items(config):
    dataset_key = get_dataset_key(config, "train")
    if dataset_key == "sparkulee_vad":
        ds_cfg = get_dataset_config(config, "train")
        data_path = ds_cfg["data_path"]
        preproc = ds_cfg.get("preproc", "LP-30")
        sfreq = ds_cfg.get("sfreq", 64.0)
        eeg_fs_tag = ds_cfg.get("eeg_fs_tag", f"{int(sfreq)}Hz")
        subjects = set()
        stories = set()
        subject_stories_map = {}
        for filename in os.listdir(data_path):
            if not filename.endswith(".npy"):
                continue
            parts = filename[:-4].split("_")
            if len(parts) < 4:
                continue
            if parts[-2] != preproc or parts[-1] != eeg_fs_tag:
                continue
            subject = parts[0].replace("sub-", "", 1)
            audio_key = "_".join(parts[1:-2])
            if audio_key.startswith("audio-"):
                story = audio_key.replace("audio-", "", 1)
                subjects.add(subject)
                stories.add(story)
                subject_stories_map.setdefault(subject, set()).add(story)
        subjects = sorted([s for s in subjects if len(subject_stories_map.get(s, set())) >= 3])
        return subjects, sorted(stories), {s: sorted(subject_stories_map[s]) for s in subjects}
    return [f"{i:02d}" for i in range(1, 26)], list(range(1, 51)), None


# ============================================================================
# Config helpers (from PKU train.py)
# ============================================================================
def update_config_for_single_run(config, run_config):
    for key_list, value in run_config:
        current = config
        for key in key_list[:-1]:
            current = current[key]
        current[key_list[-1]] = value
    return config


def runs_configs_from_search_space(search_space):
    if len(search_space) == 0:
        return []
    keys, values = zip(*search_space.items())
    return [list(zip(keys, v)) for v in itertools.product(*values)]


def get_run(config, search_space, i):
    return update_config_for_single_run(config, search_space[i])


def load_search_space(path):
    with open(path, 'r') as f:
        search_space = yaml.safe_load(f)
    result = {}
    for key, value in search_space.items():
        result[eval(key)] = value
    return result


def get_runx(result_dir):
    existing = [d for d in os.listdir(result_dir) if d.startswith("run") and os.path.isdir(os.path.join(result_dir, d))]
    nums = []
    for r in existing:
        try:
            nums.append(int(r[3:].split(':')[0]))
        except (ValueError, IndexError):
            continue
    return max(nums) + 1 if nums else 0


def _parse_subject_range(spec, width=3):
    result = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            for i in range(int(lo), int(hi) + 1):
                result.add(f"{i:0{width}d}")
        else:
            result.add(f"{int(part):0{width}d}")
    return result


# ============================================================================
# Per-subject eval (adapted from eval_pretrain_per_subject.py)
# ============================================================================
THRESHOLD_SEARCH = np.arange(0.01, 1.0, 0.01)
CONFIG_PATH = "configs/speech/my_run/config.yaml"


def build_per_subject_loaders(subject_id, config, subj_split):
    for partition in ['train', 'val', 'test']:
        for ds_entry in config["data"]["datasets"][partition]:
            ds_cfg = list(ds_entry.values())[0]
            ds_cfg['include_subjects'] = [subject_id]
            ds_cfg['stories'] = subj_split[f'{partition}_stories']

    config["data"]["dataloader"]["batch_size"] = 64
    config["data"]["dataloader"]["num_workers"] = 4

    _, val_dataset, test_dataset, _ = get_datasets_from_config(config["data"])
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=64, num_workers=4)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64, num_workers=4)
    return val_loader, test_loader, subj_split


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


def run_inference(model, loader, device):
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


def _pretrain_result_base(checkpoint_dir):
    checkpoint_path = Path(checkpoint_dir)
    for parent in checkpoint_path.parents:
        if parent.name == "pretrain":
            return str(parent.parent)
    raise ValueError(
        f"Checkpoint directory is not below a pretrain directory: {checkpoint_dir}"
    )


def run_per_subject_eval(
        checkpoint_dir, split_seed, subject_list, subject_stories_map, config, device):
    ckpt_paths = sorted(glob.glob(os.path.join(glob.escape(checkpoint_dir), "*.ckpt")))
    if not ckpt_paths:
        print(f"[EVAL] No .ckpt files in {checkpoint_dir}, skipping.")
        return

    result_base = _pretrain_result_base(checkpoint_dir)
    print(f"[EVAL] {len(ckpt_paths)} ckpts, {len(subject_list)} subjects")

    story_pool = sorted({
        story
        for subject in subject_list
        for story in subject_stories_map[subject]
    })
    splitter = DataSplitter(
        subject_list,
        story_pool,
        n_folds=5,
        seed=split_seed,
        subject_stories_map=subject_stories_map,
    )
    subject_splits = splitter.get_split("per_sub", 0)

    all_metrics = []
    for subj in subject_list:
        print(f"\n{'='*50}")
        print(f"Subject sub-{subj}")

        # Build validation/test loaders from the exact split used for pretraining.
        subj_config = copy.deepcopy(config)
        split = subject_splits[subj]
        val_loader, test_loader, split = build_per_subject_loaders(
            subj, subj_config, split)
        print(f"  Val stories: {len(split['val_stories'])}  Test stories: {len(split['test_stories'])}")

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

            del model
            torch.cuda.empty_cache()

        ckpt_name = os.path.basename(best_ckpt)
        print(f"  Best (ckpt, thr): ({ckpt_name}, {best_thresh:.2f})  val_macro_acc={best_val_acc:.4f}")

        best_model = ClassificationModule.load_from_checkpoint(best_ckpt)
        best_model = best_model.to(device)
        best_model.eval()
        test_probas, test_targets = run_inference(best_model, test_loader, device)
        test_preds = (test_probas >= best_thresh).astype(int)
        test_m = compute_metrics(test_targets, test_preds)
        print(f"  Test: macro_acc={test_m['macro_acc']:.4f}  macro_f1={test_m['macro_f1']:.4f}")
        del best_model
        torch.cuda.empty_cache()

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

    print(f"\n{'='*60}")
    print(f"Per-subject summary ({len(all_metrics)} subjects)")
    summary = {"n_subjects": len(all_metrics)}
    for mk in ['macro_acc', 'macro_f1', 'binary_acc']:
        vals = [m[mk] for m in all_metrics]
        summary[mk] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        print(f"  {mk}: {np.mean(vals):.4f} +- {np.std(vals):.4f}")

    summary_path = os.path.join(result_base, "pretrain_eval_summary.json")
    os.makedirs(result_base, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


# ============================================================================
# Main (from PKU train_v1.py, adapted for SparKULee)
# ============================================================================
def main(args):
    if args.config is None or args.search_space is None:
        raise ValueError("--config and --search-space are required")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if "model_name" not in config["general"] or config["general"]["model_name"] is None:
        config["general"]["model_name"] = next(iter(config["model"]))
    config["general"]["dataset_name"] = next(iter(config["data"]["datasets"]["train"][0]))

    config["general"]["result_dir"] = os.path.join(
        config["general"]["output_path"],
        config["general"]["project_name"],
        f'model:{config["general"]["model_name"]}-dataset:{config["general"]["dataset_name"]}',
        f'split:{args.split_strategy}')

    torch.set_float32_matmul_precision('high')

    search_space = load_search_space(args.search_space)
    run_configs = runs_configs_from_search_space(search_space)

    if int(os.getenv("LOCAL_RANK", 0)) == 0:
        os.makedirs(config["general"]["result_dir"], exist_ok=True)

    config = get_run(config, run_configs, args.run_index)
    config_str = "_".join([f"{key[-1]}-{value}" for key, value in run_configs[args.run_index]])
    config_str = re.sub(r'seed-\d+_', '', config_str)
    config["general"]["run_name"] = config_str

    ds_key = get_dataset_key(config, "train")
    train_tmax = config["data"]["datasets"]["train"][0][ds_key].get("tmax", None)
    if train_tmax is not None:
        for part in ["val", "test"]:
            if part in config["data"]["datasets"]:
                for ds_entry in config["data"]["datasets"][part]:
                    list(ds_entry.values())[0]["tmax"] = train_tmax

    config["general"]["split_strategy"] = args.split_strategy
    config["general"]["split_fold"] = args.fold
    config["general"]["split_seed"] = args.split_seed
    config["general"]["pretrain"] = args.pretrain
    config["general"]["pretrained_ckpt"] = args.pretrained_ckpt
    config["general"]["finetune_lr"] = args.finetune_lr
    config["general"]["pretrain_subjects"] = args.pretrain_subjects

    # ========================================================================
    # Training
    # ========================================================================
    lightning.seed_everything(config["general"]["seed"], workers=True)

    utils_module = importlib.import_module("speech_code.speech_utils")
    my_run_training = utils_module.my_run_training
    my_run_test = utils_module.my_run_test

    split_strategy = config["general"].get("split_strategy", "cross_subject")
    split_fold = config["general"].get("split_fold", 0)
    split_seed = config["general"].get("split_seed", 42)

    result_config_dir = os.path.join(config["general"]["result_dir"], config["general"]["run_name"])
    is_pretrain = config["general"].get("pretrain", False)
    if is_pretrain:
        result_config_dir = os.path.join(result_config_dir, "pretrain")
    if int(os.getenv("LOCAL_RANK", 0)) == 0:
        os.makedirs(result_config_dir, exist_ok=True)

    next_run_num2 = get_runx(result_config_dir)
    run_dir = os.path.join(result_config_dir,
                           f"run{next_run_num2}:seed{config['general']['seed']}_fold{split_fold}")
    config["general"]["run_dir"] = run_dir
    if int(os.getenv("LOCAL_RANK", 0)) == 0:
        os.makedirs(run_dir, exist_ok=True)

    config["general"]["checkpoint_path"] = os.path.join(run_dir, "checkpoints")
    config["general"]["model_path"] = os.path.join(run_dir, "model.py")
    config_save_path = os.path.join(run_dir, "config.yaml")

    model_source_files = {
        "brain_magic_speech_v7": "speech_code/models/my_modules/BrainNetwork_v7.py",
        "brain_magic_no_subject_attn": "speech_code/models/my_modules/BrainNetwork_v7_ablation.py",
        "brain_magic_no_short_conv": "speech_code/models/my_modules/BrainNetwork_v7_ablation.py",
        "brain_magic_no_feature_encoder": "speech_code/models/my_modules/BrainNetwork_v7_ablation.py",
        "cnn_lstm": "speech_code/models/my_modules/CNNLSTM.py",
        "dilated_conv": "speech_code/models/my_modules/DilatedConv.py",
        "awavenet": "speech_code/models/my_modules/AWaveNet.py",
        "eeg_conformer": "speech_code/models/conformer.py",
        "pnpl_cnn_tcn": "speech_code/models/my_modules/CNNTCN.py",
    }
    model_key = config["general"]["model_name"]
    src_file = model_source_files[model_key]
    shutil.copy(src_file, config["general"]["model_path"])
    if model_key == "pnpl_cnn_tcn":
        shutil.copy("speech_code/models/my_modules/public_baseline_utils.py", run_dir)

    all_subjects, all_stories, subject_stories_map = infer_split_items(config)

    pretrain_subjects = config["general"].get("pretrain_subjects", None)
    if pretrain_subjects and is_pretrain:
        filtered = _parse_subject_range(pretrain_subjects, width=3)
        all_subjects = [s for s in all_subjects if s in filtered]
        print(f"[PRETRAIN] Subject filter: {len(all_subjects)} subjects ({all_subjects})")

    if args.test_mode:
        config["trainer"]["max_epochs"] = 2
        config["data"]["dataloader"]["num_workers"] = 0
        config["general"]["threshold"] = [0.1, 0.3, 0.5, 0.7, 0.9]
        all_subjects = all_subjects[:5]
        all_stories = all_stories[:5]

    splitter = DataSplitter(all_subjects, all_stories, n_folds=5, seed=split_seed,
                            subject_stories_map=subject_stories_map)
    split = splitter.get_split(split_strategy, split_fold)

    if is_pretrain:
        for partition, key in [('train', 'train'), ('val', 'val'), ('test', 'test')]:
            if partition not in config["data"]["datasets"]:
                continue
            partition_stories = sorted({
                story
                for subject in all_subjects
                for story in split[subject][f'{key}_stories']
            })
            for ds_entry in config["data"]["datasets"][partition]:
                ds_cfg = list(ds_entry.values())[0]
                ds_cfg['include_subjects'] = all_subjects
                ds_cfg['stories'] = partition_stories
                ds_cfg.pop('include_pairs', None)
        config['data']['split_strategy'] = split_strategy
        config['general']['split_story_audit'] = {
            'split_seed': split_seed,
            'train_stories': sorted({
                story for subject in all_subjects for story in split[subject]['train_stories']
            }),
            'val_stories': sorted({
                story for subject in all_subjects for story in split[subject]['val_stories']
            }),
            'test_stories': sorted({
                story for subject in all_subjects for story in split[subject]['test_stories']
            }),
        }
        print(
            f"[PRETRAIN {split_strategy}] {len(all_subjects)} subjects use a "
            "globally disjoint shared-story split"
        )

    if not is_pretrain:
        apply_split_to_config(config, split, split_strategy)

    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_dataset, val_dataset, test_dataset, labels = get_datasets_from_config(config["data"])

    train_loader = DataLoader(train_dataset, shuffle=True, **config["data"]["dataloader"])
    val_loader = DataLoader(val_dataset, **config["data"]["dataloader"])
    test_loader = DataLoader(test_dataset, shuffle=False, **config["data"]["dataloader"])

    adapt_config_to_data(config, train_loader, labels)

    best_model_metric = config["general"].get("best_model_metrics", "val_loss")
    best_model_metric_mode = "min" if best_model_metric == "val_loss" else "max"

    pretrained_module = None
    if config["general"].get("pretrained_ckpt") is not None:
        pckpt = config["general"]["pretrained_ckpt"]
        print(f"[FINETUNE] Loading pretrained: {pckpt}")
        pretrained_module = ClassificationModule.load_from_checkpoint(pckpt)
        if config["general"].get("finetune_lr") is not None:
            flr = config["general"]["finetune_lr"]
            config["optimizer"]["config"]["lr"] = flr
            pretrained_module.optimizer_config["config"]["lr"] = flr
        new_weight = config["loss"]["config"]["weight"]
        pretrained_module.loss_mse = WeightedMSEByLabel(weight_0=new_weight[0], weight_1=new_weight[1])
        os.makedirs(config["general"]["checkpoint_path"], exist_ok=True)
        pt_ckpt_path = os.path.join(config["general"]["checkpoint_path"], "pretrained.ckpt")
        checkpoint = {
            'epoch': 0, 'global_step': 0,
            'pytorch-lightning_version': lightning.__version__,
            'state_dict': pretrained_module.state_dict(),
            'hyper_parameters': dict(pretrained_module.hparams),
            'optimizer_states': [], 'lr_schedulers': [],
        }
        torch.save(checkpoint, pt_ckpt_path)

    # --- Train ---
    trainer, module = my_run_training(
        train_loader, val_loader, config, len(labels),
        best_model_metric=best_model_metric,
        best_model_metric_mode=best_model_metric_mode,
        module=pretrained_module)

    if trainer.is_global_zero:
        del module

        # --- Val: select best (ckpt, threshold) ---
        results, best_model_paths, best_thresholds = my_run_test(
            val_loader, config["general"]["checkpoint_path"], labels,
            config["loss"]["config"]["weight"], config["general"]["threshold"],
            test_results_path=os.path.join(run_dir, "val_results.npz"), prefix="val_")
        my_log_results(results, run_dir, "val_log.json")

        # --- Test: with fixed best (ckpt, threshold) ---
        test_results, _, _ = my_run_test(
            test_loader, config["general"]["checkpoint_path"], labels,
            config["loss"]["config"]["weight"], config["general"]["threshold"],
            test_results_path=os.path.join(run_dir, "test_results.npz"), prefix="test_",
            fixed_ckpt_path=best_model_paths[0], fixed_threshold=best_thresholds[0])
        my_log_results(test_results, run_dir, "test_log.json")

        if is_pretrain:
            best_ckpt_path = os.path.join(config["general"]["result_dir"],
                                          config["general"]["run_name"],
                                          "pretrain", f"best_ckpt_fold{split_fold}.txt")
            os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)
            with open(best_ckpt_path, 'w') as f:
                f.write(best_model_paths[0])

        # ====================================================================
        # Per-subject eval
        # ====================================================================
        if is_pretrain:
            print(f"\n{'='*60}")
            print(" Starting per-subject evaluation")
            print(f"{'='*60}")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            per_sub_subjects = config["general"].get("pretrain_subjects", None)
            eval_subjects = sorted(_parse_subject_range(per_sub_subjects, width=3)) if per_sub_subjects else all_subjects
            run_per_subject_eval(
                config["general"]["checkpoint_path"],
                split_seed,
                eval_subjects,
                subject_stories_map,
                config,
                device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--search-space", type=str, default=None)
    parser.add_argument("--holdout", type=bool, default=False)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--test-mode", action="store_true", default=False)
    parser.add_argument("--split-strategy", type=str, default="cross_subject", choices=SUPPORTED_STRATEGIES)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--pretrain", action="store_true", default=False)
    parser.add_argument("--pretrain-subjects", type=str, default=None)
    parser.add_argument("--pretrained-ckpt", type=str, default=None)
    parser.add_argument("--finetune-lr", type=float, default=None)
    args = parser.parse_args()
    main(args)
