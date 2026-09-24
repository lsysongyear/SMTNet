# ============================================================================
# train.py - 模型训练主入口脚本
# ============================================================================
# 功能说明：
#   1. 加载 YAML 配置文件（模型配置、数据配置、训练配置）
#   2. 加载超参数搜索空间，支持网格搜索（Grid Search）
#   3. 动态更新配置以支持单次实验运行
#   4. 初始化数据加载器（训练集/验证集/测试集/Holdout集）
#   5. 调用 PyTorch Lightning 进行模型训练
#   6. 在验证集/测试集上评估最佳模型（多阈值搜索）
#   7. 对竞赛 Holdout 集生成预测结果
#
# 用法示例：
#   python speech_code/train.py \
#     --config=configs/speech/my_run/config.yaml \
#     --search-space=configs/speech/my_run/search-space.yaml \
#     --holdout=True \
#     --run-index=0
# ============================================================================

import sys
import os
# 获取 train.py 的绝对路径，并获取其父目录（即项目根目录），添加到 sys.path 中
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)  # 插入到开头，确保优先级最高
from argparse import ArgumentParser
import itertools
from speech_code.utils import my_log_results, get_datasets_from_config, adapt_config_to_data
from speech_code.data_split import (
    DataSplitter, apply_split_to_config, get_dataset_from_pairs, SUPPORTED_STRATEGIES)
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
# [COMMENTED OUT] LibriBrain-specific import:
# from pnpl.datasets.libribrain2025.constants import RUN_KEYS
import copy
import random
from speech_code.models.my_modules.classification_module import ClassificationModule, WeightedMSEByLabel
import re


# ============================================================================
# 配置更新函数：将超参数搜索空间中的单个配置注入到基础配置中
# ============================================================================
def update_config_for_single_run(config: dict, run_config: list[tuple[tuple, list]]):
    """
    根据 run_config 的描述，递归更新 config 字典中的值。
    用于将超参数搜索空间的某一组特定取值应用到基础配置中。

    参数:
        config: dict - 基础配置字典（会被原地修改）
        run_config: list[tuple[tuple, list]] - 更新列表，每个条目是
                    ((key1, key2, ...), value) 形式，按 key 路径定位到要修改的配置项
                    例如: (("optimizer", "config", "lr"), 0.01)

    返回:
        dict - 修改后的配置字典

    异常:
        KeyError - 当 key 路径在配置中不存在时抛出
    """
    for key_list, value in run_config:
        current = config
        try:
            # 逐级深入字典，到达目标 key 的父级
            for key in key_list[:-1]:
                current = current[key]
            # 修改目标 key 的值
            current[key_list[-1]] = value
        except KeyError:
            raise KeyError(
                f"Key list {key_list} not found in config. Config: {config}")
    return config


# ============================================================================
# 超参数搜索空间展开：将搜索空间转化为所有运行配置的列表（笛卡尔积）
# ============================================================================
def runs_configs_from_search_space(search_space: dict[tuple, list]):
    """
    将超参数搜索空间展开为多个独立运行配置。
    对搜索空间中所有参数的取值做笛卡尔积，每个组合对应一次独立实验。

    参数:
        search_space: dict - 搜索空间字典，key 为参数路径元组，value 为参数取值列表
                      例如: {("model","lr"): [0.001, 0.01], ("model","dropout"): [0.1, 0.2]}

    返回:
        list - 每个元素是一次实验的所有参数更新描述，包含所有超参数组合
    """
    if (len(search_space) == 0):
        return []
    keys, values = zip(*search_space.items())
    result = []
    # itertools.product 计算所有取值的笛卡尔积
    for v in itertools.product(*values):
        config = list(zip(keys, v))
        result.append(config)
    return result


# ============================================================================
# 根据 run_index 获取单次运行的完整配置
# ============================================================================
def get_run(config, search_space, i):
    """从搜索空间中取出第 i 个配置组合，并应用到基础 config 上"""
    run_config = search_space[i]
    return update_config_for_single_run(config, run_config)


# ============================================================================
# 加载并解析搜索空间 YAML 文件
# ============================================================================
def load_search_space(path: str):
    """
    从 YAML 文件加载超参数搜索空间。
    YAML 中的 key 是字符串形式的元组（因为 YAML 不支持元组作为 key），
    需要用 eval 解析为 Python 元组。

    参数:
        path: str - 搜索空间 YAML 文件的路径

    返回:
        dict - 解析后的搜索空间字典，key 为 Python 元组
    """
    try:
        with open(path, 'r') as f:
            search_space = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Search space file not found. Please provide a valid path")
    search_space = parse_search_space(search_space)
    return search_space


def parse_search_space(search_space: dict):
    """
    将 YAML 中的字符串 key（如 '("model", "lr")'）解析为 Python 元组。
    这是因为 YAML 不支持元组作为 key，只能存储为字符串后用 eval 还原。
    """
    result = {}
    for key, value in search_space.items():
        new_key = eval(key)  # 将字符串形式的元组转为 Python tuple
        result[new_key] = value
    return result


# ============================================================================
# 自动获取下一个可用的 run 编号（用于结果目录命名，避免覆盖之前的实验）
# ============================================================================
def get_runx(result_dir):
    """
    扫描 result_dir 中已有的 runX:seedY 格式目录，返回下一个可用的编号。

    参数:
        result_dir: str - 存放运行结果的目录路径

    返回:
        int - 下一个可用的 run 编号（比现有最大编号 +1，如果没有则为 0）
    """
    existing_runs = [d for d in os.listdir(result_dir) if d.startswith("run") and os.path.isdir(op.join(result_dir, d))]
    run_numbers = []
    for run in existing_runs:
        try:
            num = int(run[3:].split(':')[0])  # 从 "runX:seedY" 格式中提取数字 X
            run_numbers.append(num)
        except (ValueError, IndexError):
            continue
    next_run_num = max(run_numbers) + 1 if run_numbers else 0
    return next_run_num


# ============================================================================
# 主函数：解析命令行参数，组装配置，启动训练
# ============================================================================
def main(args):
    """
    训练主入口，完成以下步骤：
    1. 加载基础配置 YAML 文件
    2. 加载并展开超参数搜索空间
    3. 根据 run_index 选择本次运行的超参数组合
    4. 创建结果目录结构
    5. 调用 train() 启动完整的训练+评估流程

    如果指定了 --test-only，则跳过训练，直接从已有 run_dir 加载模型在测试集评估。
    """
    # --- 仅测试模式：跳过所有配置加载，直接评估 ---
    if args.test_only:
        test_only(args.test_only)
        return

    # --- 加载基础配置文件 ---
    if args.config is None:
        raise ValueError("--config is required for training mode")
    if args.search_space is None:
        raise ValueError("--search-space is required for training mode")
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Config file not found. Please provide a valid path")

    # 若用户未显式指定 model_name，则从 model 最后一项自动推断
    if "model_name" not in config["general"] or config["general"]["model_name"] is None:
        config["general"]["model_name"] = next(iter(config["model"]))
    config["general"]["dataset_name"] = next(iter(config["data"]["datasets"]["train"][0]))

    # 构建结果保存路径：output_path/project_name/model:模型名-dataset:数据集名/split:划分策略
    config["general"]["result_dir"] = op.join(
        config["general"]["output_path"],
        config["general"]["project_name"],
        f'model:{config["general"]["model_name"]}-dataset:{config["general"]["dataset_name"]}',
        f'split:{args.split_strategy}')

    config["general"]["holdout"] = args.holdout

    # 设置 PyTorch float32 矩阵乘法精度为高精度（利用 TF32 等加速）
    torch.set_float32_matmul_precision('high')

    # --- 加载并展开超参数搜索空间 ---
    search_space = load_search_space(args.search_space)
    run_configs = runs_configs_from_search_space(search_space)

    # 只在主进程（rank 0）创建结果目录（DDP 多卡训练时避免多进程冲突）
    if int(os.getenv("LOCAL_RANK", 0)) == 0:
        os.makedirs(config["general"]["result_dir"], exist_ok=True)

    # 根据 run_index 获取本次运行的超参数配置
    config = get_run(config, run_configs, args.run_index)
    print("rank", int(os.getenv("LOCAL_RANK", 0)), ": Running config: ", args.run_index)
    print("rank", int(os.getenv("LOCAL_RANK", 0)), ": Config: ", run_configs[args.run_index])

    # 根据超参数组合生成运行名称（用于目录和日志标识）
    config_str = "_".join([f"{key[-1]}-{value}" for key, value in run_configs[args.run_index]])
    # 移除 seed-x_ 模式，避免命名冗余
    config_str = re.sub(r'seed-\d+_', '', config_str)
    run_name = f"{config_str}"
    config["general"]["run_name"] = run_name

    # 同步 tmax 从 train 到 val/test（搜索空间仅设 train，三区需一致）
    ds_key = list(config["data"]["datasets"]["train"][0].keys())[0]
    train_tmax = config["data"]["datasets"]["train"][0][ds_key].get("tmax", None)
    if train_tmax is not None:
        for part in ["val", "test"]:
            if part in config["data"]["datasets"]:
                for ds_entry in config["data"]["datasets"][part]:
                    list(ds_entry.values())[0]["tmax"] = train_tmax

    # --- 嵌入数据划分参数到 config ---
    config["general"]["split_strategy"] = args.split_strategy
    config["general"]["split_fold"] = args.fold
    config["general"]["split_seed"] = args.split_seed
    config["general"]["single_subject_id"] = args.single_subject_id
    config["general"]["pretrain"] = args.pretrain
    config["general"]["pretrained_ckpt"] = args.pretrained_ckpt
    config["general"]["finetune_lr"] = args.finetune_lr
    config["general"]["pretrain_subjects"] = args.pretrain_subjects

    if args.test_only:
        test_only(args.test_only)
    else:
        train(config, test_mode=args.test_mode)


# ============================================================================
# 仅测试函数：跳过训练，直接加载已保存的模型在测试集上评估
# ============================================================================
def test_only(run_dir):
    """
    仅测试模式：从已完成训练的 run_dir 加载保存的模型和配置，在测试集上进行评估。

    参数:
        run_dir: str - 已完成训练的 run 目录路径（如 results/.../run0:seed1）
    """
    # 加载训练时保存的配置（包含被试划分等）
    saved_config_path = op.join(run_dir, "config.yaml")
    if op.exists(saved_config_path):
        with open(saved_config_path, 'r') as f:
            saved_config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Saved config not found at {saved_config_path}")

    # 使用保存的配置覆盖基础配置
    config = saved_config
    config["general"]["run_dir"] = run_dir
    config["general"]["checkpoint_path"] = op.join(run_dir, "checkpoints")

    # 设置随机种子
    lightning.seed_everything(config["general"]["seed"], workers=True)

    # 导入评估函数
    utils_module = importlib.import_module("speech_code.speech_utils")
    my_run_test = utils_module.my_run_test

    print(f"[TEST-ONLY] Loading test dataset from run_dir: {run_dir}")
    start_time = time.time()

    # 只加载测试集
    split_strategy = config["data"].get("split_strategy", "cross_subject")
    if split_strategy == 'whole_data' and 'split_pairs' in config.get('data', {}):
        base_cfg = config["data"]["datasets"]["test"][0]["eeg_speech_v1"]
        test_dataset = get_dataset_from_pairs(
            config["data"]["split_pairs"]["test"], base_cfg, 'test')
        labels = [0, 1]
    else:
        _, _, test_dataset, labels = get_datasets_from_config(config["data"])
    print(f"[TEST-ONLY] Test samples: {len(test_dataset)}")
    print(f"[TEST-ONLY] LOADED TEST DATASET in {time.time() - start_time:.2f} seconds")

    # 创建测试 DataLoader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, **config["data"]["dataloader"])

    # 确定阈值搜索列表（test-only 使用精简列表以减少内存占用）
    threshold = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

    checkpoint_path = config["general"]["checkpoint_path"]
    weight = config["loss"]["config"]["weight"]

    # 在测试集上评估所有 checkpoint，搜索最佳阈值
    print(f"[TEST-ONLY] Evaluating on test set with checkpoints from {checkpoint_path}")
    start_time = time.time()
    results, best_model_paths, best_thresholds = my_run_test(
        test_loader,
        checkpoint_path,
        labels,
        weight,
        threshold,
        test_results_path=op.join(run_dir, "test_results.npz"),
        prefix="test_")
    print(f"[TEST-ONLY] TEST EVALUATION DONE in {time.time() - start_time:.2f} seconds")

    # 保存测试结果
    my_log_results(results, run_dir, "test_log.json")
    print(f"[TEST-ONLY] Test results saved to {run_dir}/test_log.json")


# ============================================================================
# 训练核心函数：执行完整的训练+评估+Holdout推理流程
# ============================================================================
def train(config, test_mode=False):
    """
    完整的训练流程。

    参数:
        config: dict - 完整的实验配置字典
        test_mode: bool - 测试模式，仅加载少量数据以快速验证代码正确性
    """
    """
    完整的训练流程：
    1. 设置随机种子（保证可复现性）
    2. 动态导入自定义训练/验证/测试/Holdout 函数（来自 speech_utils.py）
    3. 创建结果目录并保存模型定义和配置文件
    4. 加载训练集/验证集/测试集并创建 DataLoader
    5. 使用 PyTorch Lightning Trainer 执行模型训练
    6. 在验证集上评估所有 checkpoint，选择最佳模型+阈值组合
    7. （可选）对竞赛 Holdout 集生成预测结果

    参数:
        config: dict - 完整的实验配置字典
    """

    # --- 设置随机种子：确保训练可复现 ---
    lightning.seed_everything(config["general"]["seed"], workers=True)

    # --- 动态导入自定义训练流程模块 ---
    # 使用 importlib 动态加载 speech_utils.py 中的函数，便于灵活替换训练逻辑
    project_name = config["general"]["project_name"]
    module_path = "speech_code.speech_utils"
    utils_module = importlib.import_module(module_path)
    # 将自定义函数绑定到局部变量，后续可直接调用
    my_run_training = utils_module.my_run_training
    my_run_validation = utils_module.my_run_validation
    my_run_test = utils_module.my_run_test
    # [COMMENTED OUT] Holdout — not needed for PKU EEG
    # my_run_holdout = utils_module.my_run_holdout
    # LibriBrainHoldout_new = utils_module.LibriBrainHoldout_new

    start_time = time.time()

    # --- 提取数据划分参数 ---
    split_strategy = config["general"].get("split_strategy", "cross_subject")
    split_fold = config["general"].get("split_fold", 0)
    split_seed = config["general"].get("split_seed", 42)
    single_subject_id = config["general"].get("single_subject_id", None)

    # --- 创建运行结果目录结构 ---
    # 第一级：超参数组合目录（single_subject 策略下嵌套被试目录）
    result_config_dir = op.join(config["general"]["result_dir"], config["general"]["run_name"])
    is_pretrain = config["general"].get("pretrain", False)
    if is_pretrain:
        result_config_dir = op.join(result_config_dir, "pretrain")
    elif split_strategy in ('single_subject', 'daily_stratified', 'daily_stratified_shared', 'cross_day', 'per_sub_5t5v'):
        result_config_dir = op.join(result_config_dir, f"sub-{single_subject_id}")
    if int(os.getenv("LOCAL_RANK", 0)) == 0:
        os.makedirs(result_config_dir, exist_ok=True)

    # 第二级：具体 run 编号 + 随机种子 + fold 信息
    next_run_num2 = get_runx(result_config_dir)
    run_dir = op.join(result_config_dir,
                      f"run{next_run_num2}:seed{config['general']['seed']}_fold{split_fold}")
    config["general"]["run_dir"] = run_dir
    if int(os.getenv("LOCAL_RANK", 0)) == 0:
        os.makedirs(run_dir, exist_ok=True)

    # --- 配置各类输出路径 ---
    config["general"]["holdout_path"] = op.join(config["general"]["run_dir"], "holdout_speech.csv")  # Holdout 预测 CSV
    config["general"]["checkpoint_path"] = op.join(config["general"]["run_dir"], "checkpoints")       # 模型检查点目录
    config["general"]["model_path"] = op.join(config["general"]["run_dir"], "model.py")               # 模型定义备份
    config_save_path = op.join(config["general"]["run_dir"], "config.yaml")                           # 配置备份
    config["general"]["config_path"] = config_save_path

    # 将模型定义文件复制到结果目录（方便后续复现实验结果）
    # 根据模型类型映射到对应的源文件
    model_source_files = {
        "brain_magic_speech": "speech_code/models/my_modules/BrainNetwork.py",
        "brain_magic_speech_v1": "speech_code/models/my_modules/BrainNetwork_v1.py",
        "brain_magic_speech_v2": "speech_code/models/my_modules/BrainNetwork_v2.py",
        "brain_magic_speech_v3": "speech_code/models/my_modules/BrainNetwork_v3.py",
        "brain_magic_speech_v4": "speech_code/models/my_modules/BrainNetwork_v4.py",
        "brain_magic_speech_v5": "speech_code/models/my_modules/BrainNetwork_v5.py",
        "brain_magic_speech_v6": "speech_code/models/my_modules/BrainNetwork_v6.py",
        "brain_magic_speech_v7": "speech_code/models/my_modules/BrainNetwork_v7.py",
        "shine": "speech_code/models/my_modules/SHINE.py",
        "awavenet": "speech_code/models/my_modules/AWaveNet.py",
        "cnn_baseline": "speech_code/models/my_modules/model_EEG.py",
        "cnn_baseline0": "speech_code/models/my_modules/model_EEG.py",
        "transformer_encoder": "speech_code/models/my_modules/model_EEG.py",
        "concat_cov_net": "speech_code/models/my_modules/model_EEG.py",
        "cnn_lstm": "speech_code/models/my_modules/CNNLSTM.py",
        "dilated_conv": "speech_code/models/my_modules/DilatedConv.py",
        "vlaai": "speech_code/models/my_modules/VLAAI.py",
        "eeg_conformer": "speech_code/models/conformer.py",
        "brain_magic_nofe": "speech_code/models/my_modules/BrainNetwork_ablation.py",
        "brain_magic_noms": "speech_code/models/my_modules/BrainNetwork_ablation.py",
        "brain_magic_nobilstm": "speech_code/models/my_modules/BrainNetwork_ablation.py",
        "brain_magic_nofe_noms": "speech_code/models/my_modules/BrainNetwork_ablation.py",
        "brain_magic_nofe_nobilstm": "speech_code/models/my_modules/BrainNetwork_ablation.py",
        "brain_magic_noms_nobilstm": "speech_code/models/my_modules/BrainNetwork_ablation.py",
        "brain_magic_no_subject_attn": "speech_code/models/my_modules/BrainNetwork_v7_ablation.py",
        "brain_magic_no_short_conv": "speech_code/models/my_modules/BrainNetwork_v7_ablation.py",
        "brain_magic_no_feature_encoder": "speech_code/models/my_modules/BrainNetwork_v7_ablation.py",
    }
    model_key = config["general"]["model_name"]
    src_file = model_source_files.get(model_key, "speech_code/models/my_modules/BrainNetwork.py")
    shutil.copy(src_file, config["general"]["model_path"])

    # --- 配置数据划分策略 ---
    # [COMMENTED OUT] Original LibriBrain run_key-based split:
    # run_keys_copy = [list(x) for x in copy.deepcopy(RUN_KEYS)]
    # valid_include = [['0', '11', 'Sherlock1', '2'], ['0', '12', 'Sherlock1', '2']]
    # train_exclude = valid_include
    # config["data"]["datasets"]["train"][0]["speech_v1"]["exclude_run_keys"] = train_exclude
    # config["data"]["datasets"]["val"][0]["speech_v1"]["include_run_keys"] = valid_include

    # --- 数据划分：使用 DataSplitter 统一管理 4 种策略 + K 折交叉验证 ---
    # 全量被试和试次集合
    all_subjects = [f"{i:02d}" for i in range(1, 26)]
    all_stories = list(range(1, 51))

    # --- 预训练被试过滤：仅使用指定被试 ---
    pretrain_subjects = config["general"].get("pretrain_subjects", None)
    if pretrain_subjects and is_pretrain:
        filtered = _parse_subject_range(pretrain_subjects)
        all_subjects = [s for s in all_subjects if s in filtered]
        print(f"[PRETRAIN] Subject filter applied: {len(all_subjects)} subjects ({all_subjects})")

    # --- 测试模式：根据划分策略缩减数据规模以快速验证 ---
    if test_mode:
        print("[TEST MODE] Overriding config for quick sanity check...")
        config["trainer"]["max_epochs"] = 2
        config["data"]["dataloader"]["num_workers"] = 0
        config["general"]["threshold"] = [0.1, 0.3, 0.5, 0.7, 0.9]

        # 各策略需要保证被划分的维度至少有 n_folds 个元素
        if split_strategy == 'cross_subject':
            # 按被试划分 → 需要 ≥ 5 名被试
            all_subjects = [f"{i:02d}" for i in range(1, 11)]   # 10 名被试
            all_stories = [1, 2, 3, 4, 5]                        # 5 个 story
        elif split_strategy == 'cross_trial':
            # 按试次划分 → 需要 ≥ 5 个 story
            all_subjects = [f"{i:02d}" for i in range(1, 6)]    # 5 名被试
            all_stories = list(range(1, 11))                      # 10 个 story
        elif split_strategy == 'whole_data':
            # 按 (被试, story) pair 划分 → 需要足够多的 pair
            all_subjects = [f"{i:02d}" for i in range(1, 7)]    # 6 名被试
            all_stories = list(range(1, 6))                       # 5 个 story → 30 pairs
        elif split_strategy == 'single_subject':
            # 被试内按试次划分 → 每名被试需要 ≥ 5 个 story
            all_subjects = [f"{i:02d}" for i in range(1, 5)]    # 4 名被试
            all_stories = list(range(1, 11))                      # 10 个 story
        elif split_strategy in ('daily_stratified', 'daily_stratified_shared'):
            # 每天分层随机/共享划分 → 每天需要 ≥ 5 个 story (3 test + 2 val)
            all_subjects = [f"{i:02d}" for i in range(1, 3)]    # 2 名被试
            all_stories = list(range(1, 51))
        elif split_strategy == 'per_sub_5t5v':
            all_subjects = [f"{i:02d}" for i in range(1, 3)]
            all_stories = list(range(1, 51))
        elif split_strategy == 'cross_day':
            # 跨天划分 → Day 1-2 训练, Day 3 随机 7 val + 10 test
            all_subjects = [f"{i:02d}" for i in range(1, 3)]    # 2 名被试
            all_stories = list(range(1, 51))                      # 全部 50 个 story

        print(f"[TEST MODE] strategy={split_strategy}, "
              f"subjects={all_subjects}, stories={all_stories}, max_epochs=2")

    # 获取划分
    splitter = DataSplitter(all_subjects, all_stories, n_folds=5, seed=split_seed)
    split = splitter.get_split(split_strategy, split_fold)

    if split_strategy in ('daily_stratified', 'daily_stratified_shared') and is_pretrain:
        # --- 预训练模式：所有被试共享同一份 story 划分 ---
        ref_subj = all_subjects[0]
        ref_split = split[ref_subj]
        for partition, key in [('train', 'train'), ('val', 'val'), ('test', 'test')]:
            if partition not in config["data"]["datasets"]:
                continue
            for ds_entry in config["data"]["datasets"][partition]:
                ds_cfg = list(ds_entry.values())[0]
                ds_cfg['include_subjects'] = all_subjects
                ds_cfg['stories'] = ref_split[f'{key}_stories']
        print(f"[PRETRAIN] All {len(all_subjects)} subjects share same story split")
        print(f"  train_stories ({len(ref_split['train_stories'])}): {ref_split['train_stories']}")
        print(f"  val_stories ({len(ref_split['val_stories'])}):   {ref_split['val_stories']}")
        print(f"  test_stories ({len(ref_split['test_stories'])}):  {ref_split['test_stories']}")
        config['data']['split_strategy'] = split_strategy
    elif split_strategy == 'per_sub_5t5v' and is_pretrain:
        # 预训练：所有被试共享同一份 story 划分，直接写入 include_subjects + stories
        ref_subj = all_subjects[0]
        ref_split = split[ref_subj]
        for partition, key in [('train', 'train'), ('val', 'val'), ('test', 'test')]:
            if partition not in config["data"]["datasets"]:
                continue
            for ds_entry in config["data"]["datasets"][partition]:
                ds_cfg = list(ds_entry.values())[0]
                ds_cfg['include_subjects'] = all_subjects
                ds_cfg['stories'] = ref_split[f'{key}_stories']
        config['data']['split_strategy'] = split_strategy
        print(f"[PRETRAIN per_sub_5t5v] All {len(all_subjects)} subjects share same story split")
        print(f"  train_stories ({len(ref_split['train_stories'])}): {ref_split['train_stories']}")
        print(f"  val_stories ({len(ref_split['val_stories'])}):   {ref_split['val_stories']}")
        print(f"  test_stories ({len(ref_split['test_stories'])}):  {ref_split['test_stories']}")
    elif split_strategy in ('single_subject', 'daily_stratified', 'daily_stratified_shared', 'cross_day', 'per_sub_5t5v'):
        if single_subject_id is None:
            single_subject_id = sorted(split.keys())[0]
        if single_subject_id not in split:
            raise ValueError(
                f"Subject {single_subject_id} not in split. "
                f"Available: {sorted(split.keys())}")
        split = split[single_subject_id]
        if split_strategy == 'daily_stratified':
            tag = 'DAILY-STRATIFIED'
        elif split_strategy == 'daily_stratified_shared':
            tag = 'DAILY-STRATIFIED-SHARED'
        elif split_strategy == 'cross_day':
            tag = 'CROSS-DAY'
        elif split_strategy == 'per_sub_5t5v':
            tag = 'PER-SUB-5T5V'
        else:
            tag = 'SINGLE-SUBJECT'
        print(f"[{tag}] subject={single_subject_id}, fold/run={split_fold}")
        print(f"  train_stories: {split['train_stories']} ({len(split['train_stories'])} stories)")
        print(f"  val_stories:   {split['val_stories']} ({len(split['val_stories'])} stories)")
        print(f"  test_stories:  {split['test_stories']} ({len(split['test_stories'])} stories)")
    else:
        print(f"[SPLIT] strategy={split_strategy}, fold={split_fold}, seed={split_seed}")
        print(f"  train_subjects: {split.get('train_subjects', 'N/A')}")
        print(f"  val_subjects:   {split.get('val_subjects', 'N/A')}")
        print(f"  test_subjects:  {split.get('test_subjects', 'N/A')}")
        print(f"  train_stories:  {split.get('train_stories', 'N/A')}")
        print(f"  val_stories:    {split.get('val_stories', 'N/A')}")
        print(f"  test_stories:   {split.get('test_stories', 'N/A')}")

    # 将划分结果写入 config
    if not (split_strategy in ('daily_stratified', 'daily_stratified_shared', 'per_sub_5t5v') and is_pretrain):
        apply_split_to_config(config, split, split_strategy)
    # 保存完整配置到 YAML 文件（方便追溯实验设置）
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)  # sort_keys=False 保持原始顺序

    print("rank", int(os.getenv("LOCAL_RANK", 0)), ": SEEDED EVERYTHING in ", time.time() - start_time, " seconds")
    start_time = time.time()

    # --- 加载数据集 ---
    _data_split_strategy = config.get('data', {}).get('split_strategy',
                                                      split_strategy)
    if split_strategy == 'whole_data':
        base_cfg = config["data"]["datasets"]["train"][0]["eeg_speech_v1"]
        train_dataset = get_dataset_from_pairs(config["data"]["split_pairs"]["train"], base_cfg, 'train')
        val_dataset = get_dataset_from_pairs(config["data"]["split_pairs"]["val"], base_cfg, 'val')
        test_dataset = get_dataset_from_pairs(config["data"]["split_pairs"]["test"], base_cfg, 'test')
        labels = [0, 1]
    else:
        train_dataset, val_dataset, test_dataset, labels = get_datasets_from_config(
            config["data"])
    print("rank", int(os.getenv("LOCAL_RANK", 0)), ": LOADED DATASETS in ", time.time() - start_time, " seconds")
    start_time = time.time()

    # --- 可选：使用部分训练数据进行快速实验 ---
    if ("train_fraction" in config["data"]["general"]):
        train_fraction = config["data"]["general"]["train_fraction"]
        train_size = int(len(train_dataset) * train_fraction)
        remaining_size = len(train_dataset) - train_size
        train_dataset, _ = torch.utils.data.random_split(
            train_dataset, [train_size, remaining_size])

    # --- 可选：使用部分验证数据进行快速实验 ---
    if ("val_fraction" in config["data"]["general"]):
        val_fraction = config["data"]["general"]["val_fraction"]
        val_size = int(len(val_dataset) * val_fraction)
        remaining_size = len(val_dataset) - val_size
        val_dataset, _ = torch.utils.data.random_split(
            val_dataset, [val_size, remaining_size])

    print("rank", int(os.getenv("LOCAL_RANK", 0)), ": TRAIN SIZE: ", len(train_dataset))
    print("rank", int(os.getenv("LOCAL_RANK", 0)), ": VAL SIZE: ", len(val_dataset))

    # --- 创建 DataLoader ---
    # 训练集开启 shuffle 以随机打乱数据顺序
    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, **config["data"]["dataloader"])
    # 验证集不 shuffle，保持顺序以利于评估
    val_loader = torch.utils.data.DataLoader(
        val_dataset, **config["data"]["dataloader"])
    # 测试集 DataLoader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, **config["data"]["dataloader"])

    # 根据实际数据调整配置（如自动计算类别权重用于损失函数）
    adapt_config_to_data(config, train_loader, labels)

    print("rank", int(os.getenv("LOCAL_RANK", 0)), ": ADAPTED CONFIG TO DATA in ", time.time() - start_time, " seconds")

    # --- 确定模型选择指标 ---
    if "best_model_metrics" in config["general"]:
        best_model_metric = config["general"]["best_model_metrics"]
    else:
        best_model_metric = "val_loss"  # 默认使用验证集损失

    # 确定指标的优化方向：val_loss 越小越好，其他指标越大越好
    if best_model_metric == "val_loss":
        best_model_metric_mode = "min"
    else:
        best_model_metric_mode = "max"

    # ========================================================================
    # 阶段 0：加载预训练 checkpoint（微调模式）
    # ========================================================================
    pretrained_module = None
    if config["general"].get("pretrained_ckpt") is not None:
        pretrained_ckpt = config["general"]["pretrained_ckpt"]
        print(f"[FINETUNE] Loading pretrained checkpoint: {pretrained_ckpt}")
        pretrained_module = ClassificationModule.load_from_checkpoint(pretrained_ckpt)
        print("[FINETUNE] Pretrained weights loaded successfully.")

        # 微调时使用更低的学习率
        if config["general"].get("finetune_lr") is not None:
            finetune_lr = config["general"]["finetune_lr"]
            config["optimizer"]["config"]["lr"] = finetune_lr
            pretrained_module.optimizer_config["config"]["lr"] = finetune_lr
            print(f"[FINETUNE] Overriding learning rate: {finetune_lr}")

        # 用当前配置中的权重替换损失权重
        new_weight = config["loss"]["config"]["weight"]
        pretrained_module.loss_mse = WeightedMSEByLabel(
            weight_0=new_weight[0], weight_1=new_weight[1])

        # 将预训练模型保存为候选 checkpoint，确保验证搜索时可以回退到预训练权重
        os.makedirs(config["general"]["checkpoint_path"], exist_ok=True)
        pretrained_ckpt_path = op.join(config["general"]["checkpoint_path"], "pretrained.ckpt")
        checkpoint = {
            'epoch': 0,
            'global_step': 0,
            'pytorch-lightning_version': lightning.__version__,
            'state_dict': pretrained_module.state_dict(),
            'hyper_parameters': dict(pretrained_module.hparams),
            'optimizer_states': [],
            'lr_schedulers': [],
        }
        torch.save(checkpoint, pretrained_ckpt_path)
        print(f"[FINETUNE] Saved pretrained model as candidate: {pretrained_ckpt_path}")

    # ========================================================================
    # 阶段 1：模型训练
    # ========================================================================
    start_time = time.time()
    trainer, module = my_run_training(
        train_loader, val_loader, config, len(labels),
        best_model_metric=best_model_metric,
        best_model_metric_mode=best_model_metric_mode,
        module=pretrained_module)
    print("rank", int(os.getenv("LOCAL_RANK", 0)), ": TRAINED MODEL in ", time.time() - start_time, " seconds")

    # ========================================================================
    # 阶段 2：验证集评估（仅在主进程执行）
    # ========================================================================
    if trainer.is_global_zero:
        del module  # 释放训练模块占用的 GPU 显存

        print("Testing on val set")
        start_time = time.time()
        # 加载所有保存的 checkpoint，对每个阈值计算 F1 等指标
        # 选择表现最好的模型+阈值组合
        results, best_model_paths, best_thresholds = my_run_test(
            val_loader,
            config["general"]["checkpoint_path"],  # checkpoint 存放路径
            labels,
            config["loss"]["config"]["weight"],     # 损失权重
            config["general"]["threshold"],          # 阈值搜索列表
            test_results_path=op.join(config["general"]["run_dir"], "val_results.npz"),
            prefix="val_")
        print("TESTED MODEL in ", time.time() - start_time, " seconds")

        # 记录验证结果日志（JSON 格式）
        start_time = time.time()
        my_log_results(results, config["general"]["run_dir"], "val_log.json")
        print("LOGGED VAL RESULTS in ", time.time() - start_time, " seconds")

        # ====================================================================
        # 阶段 3：测试集评估（使用验证集上选出的最佳 ckpt + 阈值，不在测试集上搜索）
        # 预训练模式下跳过测试集评估（测试数据为所有被试的 test 划分，评估无意义）
        # ====================================================================
        print("Testing on test set")
        start_time = time.time()
        test_results, test_best_paths, test_best_thresholds = my_run_test(
            test_loader,
            config["general"]["checkpoint_path"],
            labels,
            config["loss"]["config"]["weight"],
            config["general"]["threshold"],
            test_results_path=op.join(config["general"]["run_dir"], "test_results.npz"),
            prefix="test_",
            fixed_ckpt_path=best_model_paths[0],
            fixed_threshold=best_thresholds[0])
        print("TESTED ON TEST SET in ", time.time() - start_time, " seconds")

        start_time = time.time()
        my_log_results(test_results, config["general"]["run_dir"], "test_log.json")
        print("LOGGED TEST RESULTS in ", time.time() - start_time, " seconds")

        # --- 预训练：保存最佳 checkpoint 路径供微调阶段使用 ---
        if is_pretrain:
            best_ckpt_path = op.join(config["general"]["result_dir"],
                                     config["general"]["run_name"],
                                     "pretrain",
                                     f"best_ckpt_fold{split_fold}.txt")
            os.makedirs(op.dirname(best_ckpt_path), exist_ok=True)
            with open(best_ckpt_path, 'w') as f:
                f.write(best_model_paths[0])
            print(f"[PRETRAIN] Best checkpoint path saved to: {best_ckpt_path}")

        # ====================================================================
        # [COMMENTED OUT] Stage 4: Holdout inference (competition-specific)
        # Not applicable to PKU EEG dataset — no competition holdout set.
        # ====================================================================
        # if config["general"]["holdout"]:
        #     print("Generate holdout results")
        #     holdout_dataset = LibriBrainHoldout_new(...)
        #     holdout_loader = DataLoader(holdout_dataset, ...)
        #     my_run_holdout(holdout_loader, best_model_paths, best_thresholds,
        #                   labels, config["general"]["holdout_path"])
        #     print("Finish HOLDOUT in ", time.time() - start_time, " seconds")


def _parse_subject_range(spec):
    """Parse subject range string like '1-15,18-25' into a set of zero-padded IDs."""
    result = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            for i in range(int(lo), int(hi) + 1):
                result.add(f"{i:02d}")
        else:
            result.add(f"{int(part):02d}")
    return result


# ============================================================================
# 脚本入口：解析命令行参数并调用主函数
# ============================================================================
if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                       help="基础配置文件路径（YAML 格式，test-only 模式下可省略）")
    parser.add_argument("--search-space", type=str, default=None,
                       help="超参数搜索空间配置文件路径（YAML 格式，test-only 模式下可省略）")
    parser.add_argument("--holdout", type=bool, default=False,
                       help="是否在训练后对竞赛 Holdout 集生成预测")
    parser.add_argument("--run-index", type=int, default=0,
                       help="本次运行使用的超参数组合在搜索空间中的索引")
    parser.add_argument("--test-mode", action="store_true", default=False,
                       help="测试模式：仅加载少量被试和试次以快速验证代码正确性")
    parser.add_argument("--test-only", type=str, default=None,
                       help="仅测试模式：跳过训练，从指定 run_dir 加载已保存的模型在测试集上评估")
    parser.add_argument("--split-strategy", type=str, default="cross_subject",
                       choices=SUPPORTED_STRATEGIES,
                       help="数据划分策略：cross_subject(跨被试) | cross_trial(跨试次) | "
                            "whole_data(整体划分) | single_subject(单被试)")
    parser.add_argument("--fold", type=int, default=0,
                       help="K 折交叉验证的折号 (0 ~ 4)")
    parser.add_argument("--split-seed", type=int, default=42,
                       help="数据划分的随机种子（控制 K 折划分的 shuffle 顺序）")
    parser.add_argument("--single-subject-id", type=str, default=None,
                       help="单被试策略中指定被试 ID（如 '01'），仅 --split-strategy=single_subject 时生效")
    parser.add_argument("--pretrain", action="store_true", default=False,
                       help="预训练模式：合并所有被试的训练数据进行预训练")
    parser.add_argument("--pretrain-subjects", type=str, default=None,
                       help="预训练时仅使用指定被试（如 '1-15' 或 '1,3,5-10'），默认使用全部 25 名")
    parser.add_argument("--pretrained-ckpt", type=str, default=None,
                       help="预训练 checkpoint 路径（用于微调阶段加载预训练权重）")
    parser.add_argument("--finetune-lr", type=float, default=None,
                       help="微调阶段的学习率（覆盖配置文件中的 lr）")
    args = parser.parse_args()
    main(args)
