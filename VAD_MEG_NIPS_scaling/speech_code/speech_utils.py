from pnpl.datasets.libribrain2025.base import LibriBrainBase
import os
import os.path as op
import torch
from pytorch_lightning import Trainer
from torchmetrics import Accuracy, F1Score, Recall, Precision
from speech_code.models.my_modules.classification_module import ClassificationModule,ClassificationModule_v2
from speech_code.models.my_modules.utils import WeightedMSEByLabel
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import warnings
import h5py
from scipy.signal import butter, filtfilt
import glob
from tqdm import tqdm
import csv
from pnpl.datasets.utils import check_include_and_exclude_ids, include_exclude_ids
from pnpl.datasets.libribrain2025.constants import RUN_KEYS

SPEECH_HOLDOUT_PREDICTIONS = 560638
SPEECH_HOLDOUT_PREDICTIONS_NEW = 224255
    
class LibriBrainHoldout_new(LibriBrainBase):
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
        preload_files: bool = False,
        stride=None,
        download: bool = True,
        highpass: bool = False,
        delay: int = 0,
    ):
        """
        data_path: path to serialized dataset. 
        preprocessing_str: Preprocessing string in the file name. Indicates Preprocessing steps applied to the data.
        tmin: start time of the sample in seconds in reference to the onset of the phoneme.
        tmax: end time of the sample in seconds in reference to the onset of the phoneme.
        standardize: Whether to standardize the data. Uses channel_means and channel_stds if provided. Otherwise it calculates mean and std for each channel of the dataset. 
        clipping_boundary: Min and max values to clip the data by.
        channel_means: Standardize using these channel means.
        channel_stds: Standardize using these channel stds.
        include_info: Whether to include info dict in the output. Info dict contains dataset name, subject, session, task, run, onset time of the sample, and full phoneme label that indicates if a phoneme is at the onset or offset of a word.
        oversample_silence_jitter: Over sample silence by this factor.
        preload_files: If true start parallel downloads of all sessions and runs into data_path. Otherwise it will download files as they are needed.
        download: Whether to download files from HuggingFace if not found locally (True) or throw an error if the file is not found locally (False).

        returns Channels x Time
        """
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

        self.highpass = highpass
        self.stride = stride
        self.samples = []
        run_keys_missing = []
        self.run_keys = []
        self.delay = delay
        self.sfreq = 100.0
        self.oversample_silence_jitter = oversample_silence_jitter
        self.open_h5_datasets = {}

        include_run_keys = [tuple(run_key) for run_key in include_run_keys]
        exclude_run_keys = [tuple(run_key) for run_key in exclude_run_keys]
        check_include_and_exclude_ids(
            include_run_keys, exclude_run_keys, RUN_KEYS)

        intended_run_keys = include_exclude_ids(
            include_run_keys, exclude_run_keys, RUN_KEYS)
        self.intended_run_keys = [
            run_key for run_key in intended_run_keys if run_key[2] not in exclude_tasks]

        if len(self.intended_run_keys) == 0:
            raise ValueError(
                f"Your configuration does not allow any run keys to be included. Please check configuration: include_run_keys={include_run_keys}, exclude_run_keys={exclude_run_keys}, exclude_tasks={exclude_tasks}"
            )

        if not os.path.exists(data_path):
            raise ValueError(f"Path {data_path} does not exist.")

        self.points_per_sample = int((tmax - tmin) * self.sfreq)
        for run_key in self.intended_run_keys:
            try:
                subject, session, task, run = run_key
                self._collect_speech_samples(
                    subject, session, task, run, SPEECH_HOLDOUT_PREDICTIONS_NEW, stride=self.stride)
                self.run_keys.append(run_key)
            except FileNotFoundError:
                run_keys_missing.append(run_key)
                warnings.warn(
                    f"File not found for run key {run_key}. Skipping")
                continue

        if len(run_keys_missing) > 0:
            warnings.warn(
                f"Run keys {run_keys_missing} not found in dataset. Present run keys: {self.run_keys}")

        if len(self.samples) == 0:
            raise ValueError("No samples found.")

    def _collect_speech_samples(self, subject, session, task, run, speech_segments, stride = None):
        # Calculate the number of samples in the time window
        time_window_samples = int((self.tmax - self.tmin) * self.sfreq)

        if stride is None:
            stride = time_window_samples

        for i in range(0, speech_segments - time_window_samples + 1, stride):
            self.samples.append((subject, session, task, run, i / self.sfreq, []))
        # 处理最后不足 time_window_samples 的部分：从尾部截取一个完整窗口
        if (speech_segments % stride) != 0:
            last_onset = speech_segments - time_window_samples
            self.samples.append((subject, session, task, run, last_onset / self.sfreq, []))

    def __getitem__(self, idx):
        # returns channels x time
        if idx >= len(self.samples):
            raise IndexError(
                f"Index {idx} is out of bounds for dataset of size {len(self.samples)}"
            )
        sample = self.samples[idx]
        subject, session, task, run, onset, label = sample
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

        # ==== 计算采样窗口 ====
        delay_samples = self.delay
        start_idx = max(0, int((onset + self.tmin) * self.sfreq) + delay_samples) 
        end_idx = start_idx + self.points_per_sample
        data = meg_dataset[:, start_idx:end_idx].astype(np.float32)

        # ==== Padding if not enough time points ====
        if data.shape[1] < self.points_per_sample:
            final_data = np.zeros((data.shape[0], self.points_per_sample), dtype=data.dtype)  # shape: [channels, points_per_sample]
            valid_len = data.shape[1]
            final_data[:, :valid_len] = data  # 填充已有数据
            data = final_data  # 更新 data 为填充后的数据

        if self.standardize:
            ch_means = np.mean(data, axis=1, keepdims=True)  # shape: (204, 1)
            ch_stds = np.std(data, axis=1, keepdims=True)    # shape: (204, 1)
            # 防止除以0
            ch_stds[ch_stds == 0] = 1.0
            data = (data - ch_means) / ch_stds

        if self.include_info:
            return [torch.tensor(data, dtype=torch.float32), info]
        return [torch.tensor(data, dtype=torch.float32), {}]

def my_run_training(train_loader, val_loader, config, n_classes, best_model_metric="val_f1_macro", module=None, best_model_metric_mode="max"):
    logger = False
    if ("tensorboard" in config["general"] and config["general"]["tensorboard"]):
        log_dir = op.join(config["general"]["run_dir"], "tensorboard_logs")
        if int(os.getenv("LOCAL_RANK", 0)) == 0:
            os.makedirs(log_dir, exist_ok=True)
        # 1.8.6版本 TensorBoardLogger 初始化
        logger = TensorBoardLogger(
            save_dir=log_dir,
            name="",          # 根目录
            version=None      # 版本号，自定义或None自动编号
        )
        #logger = TensorBoardLogger(
        #    save_dir=log_dir)
    callbacks = []
    if (config["general"]["checkpoint_path"] is not None):
        os.makedirs(config["general"]["checkpoint_path"], exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=config["general"]["checkpoint_path"],
            monitor=best_model_metric,  # Metric to monitor
            mode=best_model_metric_mode,          # Higher is better
            save_top_k=5,        # Save the best 5 checkpoint
            # verbose=True,
            filename="model-{epoch:02d}-{val_loss:.4f}",
            save_last=False
        )
        callbacks.append(checkpoint_callback)

    early_stop = EarlyStopping(
        monitor=best_model_metric,
        mode=best_model_metric_mode,
        patience=5,
        verbose=True,
    )
    callbacks.append(early_stop)

    trainer_config = config["trainer"]
    trainer = Trainer(
        logger=logger,
        devices=[0],
        accelerator="gpu",
        log_every_n_steps=1,
        callbacks=callbacks,
        **trainer_config
    )

    if module is None:
        module = ClassificationModule(
            model_config=config["model"], n_classes=n_classes, optimizer_config=config["optimizer"], loss_config=config["loss"],margin_weight=config['general']['margin_weight'])

    trainer.fit(module, train_dataloaders=train_loader,
                val_dataloaders=val_loader)
    
    print("Rank", int(os.getenv("LOCAL_RANK", 0)), ": Trainer finished training.")
        
   # best_module = ClassificationModule.load_from_checkpoint(
   #     checkpoint_callback.best_model_path,
   # )

    return trainer, module

def my_run_validation(val_loader, module_path, labels, weight, threshold=[0.5, 0.6, 0.7, 0.8, 0.9], prefix="val_"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    escaped_path = glob.escape(module_path)
    ckpt_paths = sorted(glob.glob(os.path.join(escaped_path, "*.ckpt")))
    best_f1 = -1
    best_model_path = None
    best_threshold = None
    best_metrics = None

    results = {}

    for ckpt in tqdm(ckpt_paths, desc="Validating models"):
        results[ckpt] = {}
        module = ClassificationModule.load_from_checkpoint(ckpt)
        module.eval().to(device)
        disp_labels = labels
        all_targets = []
        all_probas = []

        with torch.no_grad():
            for batch in tqdm(val_loader,desc="Validating"):
                x, y = batch[0], batch[1]
                x = x.to(module.device)
                y = y.to(module.device)
                probs = module(x)
                all_probas.extend(probs.reshape(-1))
                all_targets.extend(y.reshape(-1))
    # Compare with Naive Baseline
        all_targets = torch.stack(all_targets)
        all_probas = torch.stack(all_probas)

        for t in threshold:
            y_pred_labels = (all_probas >= t).int()
            metrics = compute_metrics(y_pred_labels, all_probas, all_targets, t, prefix, weight, module.device)
            results[ckpt][f"{prefix}threshold_{t}"] = metrics
            if metrics["val_macro_f1"] > best_f1:
                best_f1 = metrics["val_macro_f1"]
                best_model_path = ckpt
                best_threshold = t
                best_metrics = metrics
        # 清理当前模型和缓存，释放显存
        del module
        torch.cuda.empty_cache()
    results["best_model_path"] = best_model_path
    results["best_threshold"] = best_threshold
    results["best_metrics"] = best_metrics
    return results, best_model_path, best_threshold

def my_run_test(test_loader, module_path, labels, weight, threshold=[0.5, 0.6, 0.7, 0.8, 0.9], prefix="test_", test_results_path=None, fixed_ckpt_path=None, fixed_threshold=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Fixed mode: use given ckpt + threshold, no search
    use_fixed = (fixed_ckpt_path is not None and fixed_threshold is not None)
    if use_fixed:
        escaped_path = glob.escape(module_path)
        ckpt_paths = [fixed_ckpt_path]
        threshold = [fixed_threshold]
    else:
        escaped_path = glob.escape(module_path)
        ckpt_paths = sorted(glob.glob(os.path.join(escaped_path, "*.ckpt")))

    results = []
    final_results = {}
    for ckpt in tqdm(ckpt_paths, desc="Testing models"):
        module = ClassificationModule.load_from_checkpoint(ckpt)
        module.eval().to(device)
        disp_labels = labels
        all_targets = []
        all_probas = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader,desc="Testing"):
                x, y = batch[0], batch[1]
                x = x.to(module.device)
                y = y.to(module.device)
                probs = module(x)
                all_probas.extend(probs.reshape(-1))
                all_targets.extend(y.reshape(-1))
    # Compare with Naive Baseline
        all_targets = torch.stack(all_targets)
        all_probas = torch.stack(all_probas)

        for t in threshold:
            y_pred_labels = (all_probas >= t).int()
            metrics = compute_metrics(y_pred_labels, all_probas, all_targets, t, prefix, weight, module.device)
            results.append({
                "checkpoint": ckpt,
                "threshold": t,
                "f1_macro": metrics[prefix+"macro_f1"],
                "macro_acc": metrics[prefix+"macro_acc"],
                "preds": y_pred_labels.clone(),
                "metrics": metrics,
                "probas": all_probas.cpu().numpy(),
                "targets": all_targets.cpu().numpy()
            })
        # 清理当前模型和缓存，释放显存
        del module
        torch.cuda.empty_cache()
    # 找出 f1_macro 前 k 的（模型+阈值）组合
    top_k = 1
    top_results = sorted(results, key=lambda x: x["macro_acc"], reverse=True)[:top_k]

    # 取这k个预测概率，加权平均（soft-voting）
    ensemble_probs = torch.stack([entry["preds"].float() for entry in top_results]).mean(dim=0)
    ensemble_preds = (ensemble_probs >= 0.5).int()

    best_model_paths = [entry["checkpoint"] for entry in top_results]
    best_thresholds = [entry["threshold"] for entry in top_results]

    # 计算 ensemble 的指标
    ensemble_metrics = compute_metrics(ensemble_preds, ensemble_probs, all_targets, threshold=0.5, prefix="ensemble_", weight=weight, device=device)
    final_results[f"top1:{top_results[0]['checkpoint']}, {top_results[0]['threshold']}"] = top_results[0]["metrics"]
    #final_results[f"top10:{top_results[9]['checkpoint']}, {top_results[9]['threshold']}"] = top_results[9]["metrics"]
    final_results["ensemble, 0.5"] = ensemble_metrics
    final_results["best_models"] = best_model_paths
    final_results["best_thresholds"] = best_thresholds
    data_dict = {
        'probas': top_results[0]["probas"],
        'targets': top_results[0]["targets"],
        'preds': top_results[0]["preds"].cpu().numpy(),
        'threshold': top_results[0]["threshold"],
    }
    np.savez(test_results_path, probas=data_dict['probas'], targets=data_dict['targets'], preds=data_dict['preds'], threshold=data_dict['threshold'])
    return final_results, best_model_paths, best_thresholds

def my_run_holdout(holdout_loader, best_model_paths, best_thresholds, labels, holdout_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_models = len(best_model_paths)
    all_model_probs = []  # 每个模型一份预测
    all_model_preds = []

    for i in range(num_models):
        model_path = best_model_paths[i]
        threshold = best_thresholds[i]
        module = ClassificationModule.load_from_checkpoint(model_path)
        module.eval().to(device)

        current_len = 0
        all_probas = []
        with torch.no_grad():
            for batch in tqdm(holdout_loader, desc="HOLDOUT batches", ncols=100):
                data, info = batch
                data = data.to(device)
                probs = module(data)  # 输出 logits，形状应为 (B, 1) 或 (B, 2)
                if current_len + probs.shape[1] > SPEECH_HOLDOUT_PREDICTIONS:
                    valid_len = SPEECH_HOLDOUT_PREDICTIONS - current_len
                    all_probas.extend(probs.reshape(-1)[-valid_len:])
                else:
                    all_probas.extend(probs.reshape(-1))
                current_len += probs.shape[1]
        all_probas = torch.stack(all_probas)
        y_pred_labels = (all_probas >= threshold).int()
        all_model_preds.append(y_pred_labels.float())
        all_model_probs.append(all_probas.float())
    # 对所有模型的预测结果做 ensemble（平均）
    ensemble_preds = torch.stack(all_model_preds).mean(dim=0)
    ensemble_probs = torch.stack(all_model_probs).mean(dim=0)

    # 最终用 0.5 作为 threshold 得到预测标签
    y_pred_labels = (ensemble_preds >= 0.5).int()

    
    with open(holdout_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["idx", "speech_prob"])

        for idx, tensor in enumerate(y_pred_labels):
            # Ensure we extract the scalar float from tensor
            speech_prob = float(tensor.item()) if isinstance(
                tensor, torch.Tensor) else float(tensor)
            writer.writerow([idx, speech_prob])
            
    np.savez(holdout_path[:-4]+'_probas.npz', probas=ensemble_probs.cpu().numpy())
        

def compute_metrics(
    y_pred_labels: torch.Tensor,  # 模型输出的标签 [N]，值为 0 或 1
    y_hat_probs: torch.Tensor,  # 模型输出的概率值 [N]
    y_true: torch.Tensor,       # 真实标签 [N]，值为 0 或 1
    threshold: float = 0.5,
    prefix: str = "",
    weight: list = [1.0, 1.0],
    device: torch.device = torch.device("cpu")
) -> dict:
    """
    支持 macro 和逐类指标的二分类评估函数。
    """

    # Step 1: 转换为 0/1 标签
    # y_pred_labels = (y_hat_probs >= threshold).int()

    # Step 2: 初始化 metric 容器
    results = {}

    # Step 3: macro-level (多分类方式视作 2 类)
    macro_metrics = {
        "macro_acc": Accuracy(num_classes=2, task="multiclass", average="macro").to(device),
        "macro_precision": Precision(num_classes=2, task="multiclass", average="macro").to(device),
        "macro_recall": Recall(num_classes=2, task="multiclass", average="macro").to(device),
        "macro_f1": F1Score(num_classes=2, task="multiclass", average="macro").to(device),
    }

    for name, metric in macro_metrics.items():
        results[f"{prefix}{name}"] = metric(y_pred_labels, y_true).item()

    # Step 4: per-class binary metrics
    binary_precision = Precision(task="multiclass", num_classes=2, average="none").to(device)
    binary_recall = Recall(task="multiclass", num_classes=2, average="none").to(device)
    binary_f1 = F1Score(task="multiclass", num_classes=2, average="none").to(device)

    # 计算所有类别的指标（返回长度为 2 的张量）
    all_precision = binary_precision(y_pred_labels, y_true)
    all_recall = binary_recall(y_pred_labels, y_true)
    all_f1 = binary_f1(y_pred_labels, y_true)
    for class_idx in [0, 1]:
        bin_prefix = f"{prefix}class{class_idx}_"
        results[bin_prefix + "precision"] = all_precision[class_idx].item()
        results[bin_prefix + "recall"] = all_recall[class_idx].item()
        results[bin_prefix + "f1"] = all_f1[class_idx].item()
        
    # Step 5: overall binary accuracy
    acc = Accuracy(task="binary").to(device)
    results[f"{prefix}binary_acc"] = acc(y_pred_labels, y_true).item()

    # Step 6: 加入 loss
    loss_fn = WeightedMSEByLabel(weight[0], weight[1]).to(device)
    results[f"{prefix}loss"] = loss_fn(y_hat_probs, y_true.float()).item()

    # Step 7: 添加阈值信息
    results[f"{prefix}threshold"] = threshold

    return results


