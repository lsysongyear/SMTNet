import sys
import os
# 获取 train.py 的绝对路径，并获取其父目录
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)  # 插入到开头，确保优先级最高
from argparse import ArgumentParser
import itertools
from speech_code.utils import my_log_results, get_datasets_from_config, adapt_config_to_data
import yaml
import pytorch_lightning as lightning
import torch
from torch.utils.data import DataLoader
import time
from speech_code.utils import get_label_counts
from numpy.lib.stride_tricks import sliding_window_view
import os.path as op
import importlib
import shutil
from pnpl.datasets.libribrain2025.constants import RUN_KEYS
import copy
import random
from speech_code.models.my_modules.classification_module import ClassificationModule
import re

def update_config_for_single_run(config: dict, run_config: list[tuple[tuple, list]]):
    """
        config: dict
        run_config: list of updates to the config. Each entry maps a keylist to a value e.g. (("optimizer", "config", "lr"), 0.01)
    """
    for key_list, value in run_config:
        current = config
        try:
            for key in key_list[:-1]:
                current = current[key]
            current[key_list[-1]] = value
        except KeyError:
            raise KeyError(
                f"Key list {key_list} not found in config. Config: {config}")
    return config


def runs_configs_from_search_space(search_space: dict[tuple, list]):
    """
        search_space: dict that maps key_list to the list of values to try for that key
        returns: list where each element describes all the hyperparameter updates for a single run
    """
    if (len(search_space) == 0):
        return []
    keys, values = zip(*search_space.items())
    result = []
    for v in itertools.product(*values):
        config = list(zip(keys, v))
        result.append(config)
    return result


def get_run(config, search_space, i):
    run_config = search_space[i]
    return update_config_for_single_run(config, run_config)


def load_search_space(path: str):
    try:
        with open(path, 'r') as f:
            search_space = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Search space file not found. Please provide a valid path")
    search_space = parse_search_space(search_space)
    return search_space

def parse_search_space(search_space: dict):
    result = {}
    for key, value in search_space.items():
        new_key = eval(key)
        result[new_key] = value
    return result

def get_runx(result_dir):
    existing_runs = [d for d in os.listdir(result_dir) if d.startswith("run") and os.path.isdir(op.join(result_dir, d))]
    run_numbers = []
    for run in existing_runs:
        try:
            num = int(run[3:].split(':')[0])  # Extract number from "runX:..." format
            run_numbers.append(num)
        except (ValueError, IndexError):
            continue
    next_run_num = max(run_numbers) + 1 if run_numbers else 0
    return next_run_num

def main(args):
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Config file not found. Please provide a valid path")
    config["general"]["model_name"] = next(iter(config["model"]))
    config["general"]["dataset_name"] = next(iter(config["data"]["datasets"]["train"][0]))
    config["general"]["result_dir"] = op.join(config["general"]["output_path"], config["general"]["project_name"], f'model:{config["general"]["model_name"]}-dataset:{config["general"]["dataset_name"]}')
    config["general"]["holdout"] = args.holdout
    torch.set_float32_matmul_precision('high')  # 推荐设置
    
    search_space = load_search_space(args.search_space)
    run_configs = runs_configs_from_search_space(search_space)
    if int(os.getenv("LOCAL_RANK", 0)) == 0:
        os.makedirs(config["general"]["result_dir"], exist_ok=True)
    config = get_run(config, run_configs, args.run_index)
    print("rank", int(os.getenv("LOCAL_RANK", 0)), ": Running config: ", args.run_index)
    print("rank", int(os.getenv("LOCAL_RANK", 0)), ": Config: ", run_configs[args.run_index])
    config_str = "_".join([f"{key[-1]}-{value}" for key, value in run_configs[args.run_index]])
    # 移除 seed-x_ 模式
    config_str = re.sub(r'seed-\d+_', '', config_str)
    run_name = f"{config_str}"
    config["general"]["run_name"] = run_name
    train(config, args)

def train(config, args):
    lightning.seed_everything(config["general"]["seed"], workers=True)
#    module = ClassificationModule(model_config=config["model"], n_classes=2, optimizer_config=config["optimizer"], loss_config=config["loss"],margin_weight=config['general']['margin_weight'])
    project_name = config["general"]["project_name"]
    module_path = "speech_code.speech_utils"
    utils_module = importlib.import_module(module_path)
    my_run_training = utils_module.my_run_training
    my_run_validation = utils_module.my_run_validation
    my_run_test = utils_module.my_run_test
    my_run_holdout = utils_module.my_run_holdout
    LibriBrainHoldout_new = utils_module.LibriBrainHoldout_new

    start_time = time.time()
    result_config_dir = op.join(config["general"]["result_dir"], config["general"]["run_name"])
    if int(os.getenv("LOCAL_RANK", 0)) == 0:
        os.makedirs(result_config_dir, exist_ok=True)
    run_dir = op.join(result_config_dir, f"run{args.run_index}:seed{config['general']['seed']}")
    config["general"]["run_dir"] = run_dir
    if int(os.getenv("LOCAL_RANK", 0)) == 0:
        os.makedirs(run_dir, exist_ok=True)

    config["general"]["holdout_path"] = op.join(config["general"]["run_dir"], "holdout_speech.csv")
    config["general"]["checkpoint_path"] = op.join(config["general"]["run_dir"], "checkpoints")
    config["general"]["model_path"] = op.join(config["general"]["run_dir"], "model.py")
    config_save_path = op.join(config["general"]["run_dir"], "config.yaml")
    config["general"]["config_path"] = config_save_path

    model_source_files = {
        "brain_magic_speech": "speech_code/models/my_modules/BrainNetwork.py",
        "brain_magic_speech_v1": "speech_code/models/my_modules/BrainNetwork_v1.py",
        "brain_magic_speech_v7": "speech_code/models/my_modules/BrainNetwork_v7.py",
        "brain_magic_no_subject_attn": "speech_code/models/my_modules/BrainNetwork_v7_ablation.py",
        "brain_magic_no_short_conv": "speech_code/models/my_modules/BrainNetwork_v7_ablation.py",
        "brain_magic_no_feature_encoder": "speech_code/models/my_modules/BrainNetwork_v7_ablation.py",
        "awavenet": "speech_code/models/my_modules/AWaveNet.py",
        "cnn_lstm": "speech_code/models/my_modules/CNNLSTM.py",
        "dilated_conv": "speech_code/models/my_modules/DilatedConv.py",
        "vlaai": "speech_code/models/my_modules/VLAAI.py",
        "eeg_conformer": "speech_code/models/conformer.py",
    }
    model_key = config["general"]["model_name"]
    src_file = model_source_files.get(model_key, "speech_code/models/my_modules/BrainNetwork.py")
    shutil.copy(src_file, config["general"]["model_path"])

    run_keys_all = [list(x) for x in copy.deepcopy(RUN_KEYS)]
    split_seed = config["general"].get("split_seed", config["general"]["seed"])
    rng = random.Random(split_seed)
    rng.shuffle(run_keys_all)
    n_total = len(run_keys_all)
    n_test = max(1, n_total // 10)
    n_val = max(1, n_total // 10)
    n_train = n_total - n_test - n_val
    test_include = run_keys_all[:n_test]
    val_include = run_keys_all[n_test:n_test + n_val]
    train_include = run_keys_all[n_test + n_val:]
    config["data"]["datasets"]["train"][0]["speech_v1"].pop("exclude_run_keys", None)
    config["data"]["datasets"]["train"][0]["speech_v1"]["include_run_keys"] = train_include
    config["data"]["datasets"]["val"][0]["speech_v1"]["include_run_keys"] = val_include
    config["data"]["datasets"]["test"][0]["speech_v1"]["include_run_keys"] = test_include
    print("rank", int(os.getenv("LOCAL_RANK", 0)), ": Train sessions:", len(train_include),
          "Val sessions:", val_include, "Test sessions:", test_include)
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)  # sort_keys=False保持原始顺序

    print("rank", int(os.getenv("LOCAL_RANK", 0)), ": SEEDED EVERYTHING in ", time.time() - start_time, " seconds")
    start_time = time.time()

    train_dataset, val_dataset, test_dataset, labels = get_datasets_from_config(
        config["data"])
    print("rank", int(os.getenv("LOCAL_RANK", 0)), ": LOADED DATASETS in ", time.time() - start_time, " seconds")
    start_time = time.time()

    if ("train_fraction" in config["data"]["general"]):
        train_fraction = config["data"]["general"]["train_fraction"]
        train_size = int(len(train_dataset) * train_fraction)
        remaining_size = len(train_dataset) - train_size
        train_dataset, _ = torch.utils.data.random_split(
            train_dataset, [train_size, remaining_size])
    if ("val_fraction" in config["data"]["general"]):
        val_fraction = config["data"]["general"]["val_fraction"]
        val_size = int(len(val_dataset) * val_fraction)
        remaining_size = len(val_dataset) - val_size
        val_dataset, _ = torch.utils.data.random_split(
            val_dataset, [val_size, remaining_size])
        
    print("rank", int(os.getenv("LOCAL_RANK", 0)), ": TRAIN SIZE: ", len(train_dataset))
    print("rank", int(os.getenv("LOCAL_RANK", 0)), ": VAL SIZE: ", len(val_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, **config["data"]["dataloader"])
    val_loader = torch.utils.data.DataLoader(
        val_dataset, **config["data"]["dataloader"])
    test_loader = torch.utils.data.DataLoader(
        test_dataset, **config["data"]["dataloader"])
    adapt_config_to_data(config, train_loader, labels)

    print("rank", int(os.getenv("LOCAL_RANK", 0)), ": ADAPTED CONFIG TO DATA in ", time.time() - start_time, " seconds")

    if "best_model_metrics" in config["general"]:
        best_model_metric = config["general"]["best_model_metrics"]
    else:
        best_model_metric = "val_loss"

    if best_model_metric == "val_loss":
        best_model_metric_mode = "min"
    else:
        best_model_metric_mode = "max"

    start_time = time.time()
    trainer, module = my_run_training(
        train_loader, val_loader, config, len(labels), best_model_metric=best_model_metric, best_model_metric_mode=best_model_metric_mode)
    print("rank", int(os.getenv("LOCAL_RANK", 0)), ": TRAINED MODEL in ", time.time() - start_time, " seconds")

    if trainer.is_global_zero:
        del module
        
        #start_time = time.time()
        #results, best_model_path, best_threshold = my_run_validation(
        #    val_loader, config["general"]["checkpoint_path"], labels, config["loss"]["config"]["weight"], config["general"]["threshold"])
        #print("VALIDATED MODEL in ", time.time() - start_time, " seconds")
        #start_time = time.time()
        #my_log_results(results,
        #            config["general"]["run_dir"], "val_log.json")
        #print("LOGGED VAL RESULTS in ", time.time() - start_time, " seconds")

        print("Testing on val set")
        start_time = time.time()
        results, best_model_paths, best_thresholds = my_run_test(
            val_loader, config["general"]["checkpoint_path"], labels, config["loss"]["config"]["weight"], config["general"]["threshold"],test_results_path=op.join(config["general"]["run_dir"], "val_results.npz"),prefix="val_")
        print("VALIDATED MODEL in ", time.time() - start_time, " seconds")
        start_time = time.time()
        my_log_results(results,
                    config["general"]["run_dir"], "val_log.json")
        print("LOGGED VAL RESULTS in ", time.time() - start_time, " seconds")

        print("Testing on test set")
        start_time = time.time()
        results, _, _ = my_run_test(
            test_loader, config["general"]["checkpoint_path"], labels, config["loss"]["config"]["weight"], config["general"]["threshold"],
            test_results_path=op.join(config["general"]["run_dir"], "test_results.npz"), prefix="test_",
            fixed_ckpt_path=best_model_paths[0], fixed_threshold=best_thresholds[0])
        print("TESTED MODEL in ", time.time() - start_time, " seconds")
        start_time = time.time()
        my_log_results(results,
                    config["general"]["run_dir"], "test_log.json")
        print("LOGGED TEST RESULTS in ", time.time() - start_time, " seconds")

        if config["general"]["holdout"]:
            print("Generate holdout results")
            holdout_dataset =  LibriBrainHoldout_new(
                data_path='libribrain',  # Same as in the other LibriBrain dataset - this is where we'll store the data
                include_run_keys=[("0", "2025", "COMPETITION_HOLDOUT", "1")],
                tmax=12.0,             # Also identical to the other datasets - how many samples to return/group together
                standardize=True,
                preprocessing_str="bads+headpos+sss+notch+bp+ds",
                preload_files=False,
                include_info=True,
                delay=0,
            )
            holdout_loader = DataLoader(
                holdout_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=4
            )
            start_time = time.time()
            my_run_holdout(holdout_loader, best_model_paths, best_thresholds, labels, config["general"]["holdout_path"])
            print("Finish HOLDOUT in ", time.time() - start_time, " seconds")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--search-space", type=str, required=True)
    parser.add_argument("--holdout", type=bool, default=False)
    parser.add_argument("--run-index", type=int, required=True)
    args = parser.parse_args()
    main(args)