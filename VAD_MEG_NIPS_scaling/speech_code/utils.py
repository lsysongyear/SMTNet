from pnpl.datasets import LibriBrainPhoneme, LibriBrainSpeech
from pnpl.datasets.libribrain2025.base import LibriBrainBase
from torch.utils.data import DataLoader, ConcatDataset
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
from tqdm import tqdm
from pnpl.datasets.utils import check_include_and_exclude_ids, include_exclude_ids
from pnpl.datasets.libribrain2025.constants import RUN_KEYS
from torch.utils.data import Dataset
import numpy as np
from scaling.duration import canonical_prefix_allocation

from itertools import groupby

def flip_short_segments(seq, v, threshold):
    result = []
    for val, group in groupby(seq):
        group_list = list(group)
        length = len(group_list)

        # 判断是否需要翻转
        if val == v and length < threshold:
            result.extend([1 - val] * length)  
        else:
            result.extend(group_list)

    return np.array(result)

SPEECH_HOLDOUT_PREDICTIONS = 560638

class Speech_v1(LibriBrainBase):
    def __init__(
        self,
        data_path: str,
        partition: str | None = None,
        preprocessing_str: str | None = "bads+headpos+sss+notch+bp+ds",
        tmin: float = 0.0,
        tmax: float = 0.5,
        include_run_keys: list[str] = [],
        exclude_run_keys: list[str] = [],
        exclude_tasks: list[str] = [],
        standardize: bool = True,
        clipping_boundary: float | None = 10,
        channel_means: np.ndarray | None = None,
        channel_stds: np.ndarray | None = None,
        include_info: bool = False,
        oversample_silence_jitter: int = 0,
        preload_files: bool = True,
        stride=None,
        download: bool = True,
        delay: int = 0,
        data_type: str = 'train',
        recording_fraction: float = 1.0,
    ):
        os.makedirs(data_path, exist_ok=True)
        self.data_path = data_path
        self.partition = partition
        self.preprocessing_str = preprocessing_str
        self.tmin = tmin
        self.tmax = tmax
        self.include_run_keys = include_run_keys
        self.exclude_run_keys = exclude_run_keys
        self.standardize = standardize
        self.include_info = include_info
        self.channel_means = channel_means
        self.channel_stds = channel_stds

        self.sfreq_meg = 100.0
        self.points_per_sample_meg = int((tmax - tmin) * self.sfreq_meg)
        self.sfreq = 250.0
        self.points_per_sample = int((tmax - tmin) * self.sfreq)
        self.stride = stride
        self.samples = []
        run_keys_missing = []
        self.run_keys = []
        self.delay = delay
        self.oversample_silence_jitter = oversample_silence_jitter
        self.labels_sorted = [0, 1]
        self.open_h5_datasets = {}
        self.data_type = data_type
        self.recording_fraction = float(recording_fraction)
        self.scaling_metadata = {
            "recording_fraction": self.recording_fraction,
            "duration_sfreq": float(self.sfreq),
            "subjects": {},
        }
 
        include_run_keys = [tuple(run_key) for run_key in include_run_keys]
        exclude_run_keys = [tuple(run_key) for run_key in exclude_run_keys]
        check_include_and_exclude_ids(
            include_run_keys, exclude_run_keys, RUN_KEYS)

        allowed_run_keys = [
            tuple(run_key)
            for run_key in include_exclude_ids(
                include_run_keys, exclude_run_keys, RUN_KEYS
            )
        ]
        if include_run_keys:
            allowed_run_keys = set(allowed_run_keys)
            intended_run_keys = [
                run_key for run_key in include_run_keys if run_key in allowed_run_keys
            ]
        else:
            intended_run_keys = allowed_run_keys
        self.intended_run_keys = [
            run_key for run_key in intended_run_keys if run_key[2] not in exclude_tasks]

        if len(self.intended_run_keys) == 0:
            raise ValueError(
                f"Your configuration does not allow any run keys to be included. Please check configuration: include_run_keys={include_run_keys}, exclude_run_keys={exclude_run_keys}, exclude_tasks={exclude_tasks}"
            )

        if not os.path.exists(data_path):
            raise ValueError(f"Path {data_path} does not exist.")

        run_items = []
        for run_key in self.intended_run_keys:
            try:
                subject, session, task, run = run_key
                speech_labels = self.get_speech_silence_labels_for_session(
                    subject, session, task, run)
                fname = f"sub-{subject}_ses-{session}_task-{task}_run-{run}"
                if self.preprocessing_str is not None:
                    fname += f"_proc-{self.preprocessing_str}"
                meg_path = os.path.join(
                    self.data_path,
                    task,
                    "derivatives",
                    "serialised",
                    fname + "_grad.npy",
                )
                meg = np.load(meg_path, mmap_mode="r")
                valid_samples = min(
                    len(speech_labels),
                    int(meg.shape[-1] * self.sfreq / self.sfreq_meg),
                )
                run_items.append((run_key, speech_labels[:valid_samples], valid_samples))
                self.run_keys.append(run_key)
            except FileNotFoundError:
                run_keys_missing.append(run_key)
                warnings.warn(
                    f"File not found for run key {run_key}. Skipping")
                continue

        target, allocation_by_key, allocation_order = canonical_prefix_allocation(
            [(item[0], item[2]) for item in run_items],
            [tuple(run_key) for run_key in RUN_KEYS],
            self.recording_fraction,
        )
        items_by_key = {item[0]: item for item in run_items}
        unit_records = [
            {
                "unit": list(run_key),
                "full_samples": items_by_key[run_key][2],
                "selected_samples": allocation_by_key[run_key],
            }
            for run_key in allocation_order
        ]
        for run_key, speech_labels, full_samples in run_items:
            keep_samples = allocation_by_key[run_key]
            if keep_samples > 0:
                self._collect_speech_samples(
                    *run_key, speech_labels[:keep_samples], stride=self.stride
                )
        subject_key = str(run_items[0][0][0]) if run_items else "0"
        self.scaling_metadata["subjects"][subject_key] = {
            "full_samples": sum(item[2] for item in run_items),
            "selected_samples": target,
            "units": unit_records,
        }

        if len(run_keys_missing) > 0:
            warnings.warn(
                f"Run keys {run_keys_missing} not found in dataset. Present run keys: {self.run_keys}")

        if len(self.samples) == 0:
            raise ValueError("No samples found.")

    def get_speech_silence_labels_for_session(self, subject, session, task, run):
        df = self._load_events(subject, session, task, run)

        # Convert times to samples, handling errors
        df['timemeg_samples'] = (pd.to_numeric(
            df['timemeg'], errors='coerce') * self.sfreq).astype(int)
        df['duration_samples'] = (pd.to_numeric(
            df['duration'], errors='coerce') * self.sfreq).astype(int)

        # Filter for silence entries
        silence_df = df[df['kind'] == 'silence']

        if silence_df.empty or silence_df['timemeg_samples'].isnull().all() or silence_df['duration_samples'].isnull().all():
            print("Warning: No valid silence entries found. Returning None.")
            return None

        words_df = df[df['kind'] == 'word']

        max_word_sample_time = (words_df['timemeg_samples'] +
                      words_df['duration_samples']).max()

        max_silence_sample_time = (silence_df['timemeg_samples'] +
                      silence_df['duration_samples']).max()

        # Create the array, initialize with 1s (assuming everything is speech initially)
        speech_labels = np.ones(max(max_word_sample_time,max_silence_sample_time) + 1, dtype=int)

        # Fill in 0s for silence spans
        for index, row in silence_df.iterrows():
            start_sample = row['timemeg_samples']
            duration_samples = row['duration_samples']
            if not np.isnan(start_sample) and not np.isnan(duration_samples):
                end_sample = start_sample + duration_samples
                speech_labels[start_sample:end_sample] = 0
        start_sample = int(max(0, df.loc[0,'timemeg_samples']))
        speech_labels[:start_sample] = 0
        #if self.data_type == 'train':
        #    speech_labels = flip_short_segments(speech_labels, 1, 150)
        #    speech_labels = flip_short_segments(speech_labels, 0, 50)
        speech_labels = speech_labels[start_sample:]
        return speech_labels

    def _collect_speech_samples(self, subject, session, task, run, speech_labels, stride = None):
        # Calculate the number of samples in the time window
        time_window_samples = int((self.tmax - self.tmin) * self.sfreq)

        if stride is None:
            if self.data_type == 'train':
                stride = time_window_samples // 2
            else:
                stride = time_window_samples

        for i in range(0, len(speech_labels), stride):
            sample_labels = speech_labels[i:i+time_window_samples]
            if len(sample_labels) < time_window_samples:
                continue
            if self.data_type == 'train':
                jitter = random.choice(list(range(-2, 2 + 1)))
                jittered_i = i + jitter
                # 边界检查，防止越界
                if jittered_i < 0:
                    jittered_i = 0
                if jittered_i + time_window_samples > len(speech_labels):
                    jittered_i = i
                sample_labels = speech_labels[jittered_i : jittered_i + time_window_samples]
            self.samples.append(
                (subject, session, task, run, i / self.sfreq, sample_labels))

    def __getitem__(self, idx):
        # returns channels x time
        if idx >= len(self.samples):
            raise IndexError(
                f"Index {idx} is out of bounds for dataset of size {len(self.samples)}"
            )
        sample = self.samples[idx]
        subject, session, task, run, onset, label = sample
        # Downsample labels from 250Hz to 100Hz (ratio 5:2)
        label_len = len(label)
        target_len = int(label_len * self.sfreq_meg / self.sfreq)
        idx = (np.arange(target_len + 1) * label_len / target_len).astype(int)
        label = np.array([label[idx[i]:idx[i+1]].max() for i in range(target_len)], dtype=label.dtype)

        if self.include_info:
            info = {
                "dataset": "libribrain2025",
                "subject": subject,
                "session": session,
                "task": task,
                "run": run,
                "onset": torch.tensor(onset, dtype=torch.float32),
            }

        if (subject, session, task, run) not in self.open_h5_datasets:
            fname = f"sub-{subject}_ses-{session}_task-{task}_run-{run}"
            if self.preprocessing_str is not None:
                fname += f"_proc-{self.preprocessing_str}"
            fname += "_grad.npy"
            meg_path = os.path.join(self.data_path, task, "derivatives", "serialised", fname)
            meg_dataset = np.load(meg_path)
            self.open_h5_datasets[(subject, session, task, run)] = meg_dataset
        else:
            meg_dataset = self.open_h5_datasets[(subject, session, task, run)]

        delay_samples = self.delay
        start = max(0, int((onset + self.tmin) * self.sfreq_meg) + delay_samples)
        end = start + self.points_per_sample_meg
        data = meg_dataset[:, start:end].astype(np.float32)

        # 若数据不足 points_per_sample，补均值
        if data.shape[1] < self.points_per_sample_meg:
            final_data = np.zeros((data.shape[0], self.points_per_sample_meg), dtype=data.dtype)  # shape: [channels, points_per_sample]
            valid_len = data.shape[1]
            final_data[:, :valid_len] = data  # 填充已有数据
            data = final_data  # 更新 data 为填充后的数据

        if self.standardize:
            # for the edge case in which the last samples are smaller than points_per_sample,
            ch_means = np.mean(data, axis=1, keepdims=True)  # shape: (204, 1)
            ch_stds = np.std(data, axis=1, keepdims=True)    # shape: (204, 1)
            # 防止除以0
            ch_stds[ch_stds == 0] = 1.0
            data = (data - ch_means) / ch_stds
        #if self.data_type == 'train':
        #    noise_std = 0.01
        #    noise = np.random.normal(0, noise_std, size=data.shape).astype(np.float32)
        #    data += noise

        if self.include_info:
            return [torch.tensor(data, dtype=torch.float32), label, info]
        return [torch.tensor(data, dtype=torch.float32), label, {}]
        

DATASETS = {
    "libribrain_phoneme": LibriBrainPhoneme,
    "libribrain_speech": LibriBrainSpeech,
    "speech_v1": Speech_v1,
}


def check_labels(list_of_labels):
    reference_labels = list_of_labels[0]
    for labels in list_of_labels[1:]:
        if (labels != reference_labels):
            raise ValueError(
                f"Datasets have different labels: {labels} and {reference_labels}")


def apply_dataset_wrappers_from_data_config(dataset, data_config):
    # applies dataset wrappers from data config
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


def get_dataset_partition_from_config(partition_config, channel_means=None, channel_stds=None):
    # loads datasets from config
    # returns concatenated dataset
    partition_dataset_names = [list(ds.keys())[0] for ds in partition_config]
    partition_dataset_configs = [list(ds.values())[0]
                                 for ds in partition_config]

    for config in partition_dataset_configs:
        # for simplicity we standardize using the first training dataset
        if (config.get("standardize", True)):
            config['channel_means'] = channel_means.tolist(
            ) if channel_means is not None else None
            config['channel_stds'] = channel_stds.tolist(
            ) if channel_stds is not None else None

    partition_datasets = []
    partition_dataset_labels = []
    for name, config in zip(partition_dataset_names, partition_dataset_configs):
        if (name not in DATASETS):
            raise ValueError(
                f"Dataset {name} not supported. Please change data config")
        dataset = DATASETS[name](**config)
        partition_datasets.append(dataset)
        partition_dataset_labels.append(dataset.labels_sorted)
    # ensure all datasets have the same set of labels
    check_labels(partition_dataset_labels)
    partition_dataset = ConcatDataset(partition_datasets)
    return partition_dataset


def get_datasets_from_config(data_config):
    datasets_config = data_config["datasets"]

    if "train" in datasets_config:
        train_dataset = get_dataset_partition_from_config(
            datasets_config["train"])
        train_channel_means = train_dataset.datasets[0].channel_means
        train_channel_stds = train_dataset.datasets[0].channel_stds
        train_labels_sorted = train_dataset.datasets[0].labels_sorted
        train_dataset = apply_dataset_wrappers_from_data_config(
            train_dataset, data_config)
    else:
        train_dataset = None
        train_labels_sorted = None
        train_channel_means = None
        train_channel_stds = None
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
    if train_labels_sorted is None:  # HACKY FOR ARMENI COMPARISON
        train_labels_sorted = val_dataset.datasets[0].labels_sorted
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


def log_results(result, y, preds, logits, output_path, run_name, hpo_config=None, trainer=None):
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

def my_log_results(result, run_path, log_name):
    with open(os.path.join(run_path, log_name), "w") as f:
        json.dump(result, f, indent=3, ensure_ascii=False)

def get_label_counts(train_loader, n_classes):
    label_counts = torch.zeros(n_classes)
    for batch in train_loader:
        y = batch[1].flatten().long()
        label_counts += torch.bincount(y, minlength=n_classes)
    return label_counts


def get_label_distribution(train_loader, n_classes):
    label_counts = get_label_counts(train_loader, n_classes)
    label_distribution = label_counts / label_counts.sum()
    return label_distribution

def run_training(train_loader, val_loader, config, n_classes, best_model_metric="val_f1_macro", module=None, best_model_metric_mode="max"):
    if module is None:
        module = ClassificationModule(
            model_config=config["model"], n_classes=n_classes, optimizer_config=config["optimizer"], loss_config=config["loss"])

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
            monitor=best_model_metric,  # Metric to monitor
            mode=best_model_metric_mode,          # Higher is better
            save_top_k=1,        # Save only the best checkpoint
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
    
    if trainer.global_rank == 0:  # 或 trainer.is_global_zero
        best_module = ClassificationModule.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
        print("Loaded best model from:", checkpoint_callback.best_model_path)
    else:
        best_module = None
   # best_module = ClassificationModule.load_from_checkpoint(
   #     checkpoint_callback.best_model_path,
   # )
    best_module = trainer.strategy.broadcast(best_module)

    return trainer, best_module, module


def run_validation(val_loader, module, labels, samples_per_class=None):
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
    # Compare with Naive Baseline
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
    random_f1_macro = f1_macro(
        random_preds, all_targets)
    random_f1_micro = f1_micro(
        random_preds, all_targets)
    random_f1_weighted = f1_weighted(
        random_preds, all_targets)
    naive_preds = torch.multinomial(
        bincount.float(), len(all_targets), replacement=True).to(module.device)
    naive_acc = acc(naive_preds, all_targets)
    naive_balanced_acc = bal_acc(
        naive_preds, all_targets)
    naive_f1_macro = f1_macro(
        naive_preds, all_targets)
    naive_f1_micro = f1_micro(
        naive_preds, all_targets)
    naive_f1_weighted = f1_weighted(
        naive_preds, all_targets)

    # calculate loss
    frequencies = bincount.float() / bincount.sum()
    naive_probas = [frequencies for _ in all_targets]
    naive_probas = torch.stack(naive_probas)
    naive_loss = torch.nn.NLLLoss()(torch.log(naive_probas), all_targets)

    naive_rocauc_macro = rocauc_macro(
        naive_probas, all_targets)
    naive_rocauc_micro = rocauc_micro(
        naive_probas, all_targets)

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
        "val_naive_acc": naive_acc,
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

def adapt_config_to_data(config, train_loader, labels):
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
