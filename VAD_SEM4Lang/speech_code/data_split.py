# ============================================================================
# data_split.py - 数据划分策略与 K 折交叉验证
# ============================================================================
# 支持 4 种数据划分策略，每种均支持 5 折交叉验证：
#
#   1. cross_subject  (跨被试)   — 若干被试的全部试次为训练/验证集，
#                                   剩余被试的全部试次为测试集
#   2. cross_trial    (跨试次)   — 所有被试的某些试次为训练/验证集，
#                                   所有被试的剩余试次为测试集
#   3. whole_data     (整体划分) — 将所有被试 × 所有试次作为整体，
#                                   随机划分为训练/验证/测试集
#   4. single_subject (单被试)   — 每名被试单独训练，被试内试次划分
#
# 用法：
#   splitter = DataSplitter(subjects, stories, n_folds=5, seed=42)
#   split = splitter.get_split('cross_subject', fold=0)
# ============================================================================

import random
import os
import os.path as op
import json
import torch
from torch.utils.data import ConcatDataset

SUPPORTED_STRATEGIES = ['cross_subject', 'cross_trial', 'whole_data',
                       'single_subject', 'daily_stratified', 'daily_stratified_shared',
                       'cross_day', 'per_sub_5t5v']

# Day-to-story mapping for PKU EEG experiment
DAY_STORIES = {
    1: list(range(1, 18)),    # Day 1: stories 1-17
    2: list(range(18, 34)),   # Day 2: stories 18-33
    3: list(range(34, 51)),   # Day 3: stories 34-50
}
TEST_PER_DAY = 3   # 3 test stories per day → 9 total (for daily_stratified)
VAL_PER_DAY = 2    # 2 val stories per day → 6 total (for daily_stratified)
# Train: remaining 35 stories (50 - 9 - 6)

# cross_day 策略常量
CROSS_DAY_TRAIN = list(range(1, 34))    # Day 1-2: stories 1-33 (train)
CROSS_DAY_DAY3 = list(range(34, 51))    # Day 3: stories 34-50 (val + test)
CROSS_DAY_VAL = 7    # 7 val stories from Day 3
CROSS_DAY_TEST = 10  # 10 test stories from Day 3


# ============================================================================
# DataSplitter — 统一的数据划分接口
# ============================================================================
class DataSplitter:
    """
    统一管理 4 种划分策略的 K 折交叉验证。

    参数:
        subjects: 所有被试 ID 列表
        stories:  所有 story ID 列表
        n_folds:  交叉验证折数（默认 5）
        seed:     随机种子（保证可复现）
        val_ratio: 训练/验证划分中验证集占比（默认 0.15）
    """

    def __init__(self, subjects, stories, n_folds=5, seed=42, val_ratio=0.15):
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")
        if len(subjects) < n_folds and len(stories) < n_folds:
            raise ValueError(
                f"Neither subjects ({len(subjects)}) nor stories ({len(stories)}) "
                f"have enough items for {n_folds}-fold CV")

        self.subjects = sorted(subjects)
        self.stories = sorted(stories)
        self.n_folds = n_folds
        self.seed = seed
        self.val_ratio = val_ratio

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def get_split(self, strategy, fold):
        """
        返回指定策略和折号的划分结果。

        返回值统一为 dict，包含：
            train_subjects, val_subjects, test_subjects : list[str]
            train_stories,  val_stories,  test_stories  : list[int]

        对于 whole_data 策略额外包含：
            train_pairs, val_pairs, test_pairs : list[(subj, story)]

        对于 single_subject 策略返回嵌套 dict：
            {subject_id: {train_subjects, ..., test_stories}}

        参数:
            strategy: 'cross_subject' | 'cross_trial' | 'whole_data' | 'single_subject'
            fold:     折号 (0 ~ n_folds-1)
        """
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from {SUPPORTED_STRATEGIES}")
        if not (0 <= fold < self.n_folds):
            raise ValueError(f"fold must be in [0, {self.n_folds - 1}], got {fold}")

        if strategy == 'cross_subject':
            return self._cross_subject_split(fold)
        elif strategy == 'cross_trial':
            return self._cross_trial_split(fold)
        elif strategy == 'whole_data':
            return self._whole_data_split(fold)
        elif strategy == 'single_subject':
            return self._single_subject_split(fold)
        elif strategy == 'daily_stratified':
            return self._daily_stratified_split(fold)
        elif strategy == 'daily_stratified_shared':
            return self._daily_stratified_shared_split(fold)
        elif strategy == 'cross_day':
            return self._cross_day_split(fold)
        else:  # per_sub_5t5v
            return self._per_sub_5t5v_split(fold)

    def get_all_folds(self, strategy):
        """返回某策略下所有折的划分列表，方便批量提交"""
        return [self.get_split(strategy, f) for f in range(self.n_folds)]

    def save_splits(self, strategy, output_dir):
        """将某策略的所有折划分保存为 JSON，便于追溯"""
        os.makedirs(output_dir, exist_ok=True)
        all_folds = {}
        for fold in range(self.n_folds):
            split = self.get_split(strategy, fold)
            # 将 tuple keys (for whole_data pairs) 转为 string 以便 JSON 序列化
            all_folds[str(fold)] = _make_json_safe(split)
        path = op.join(output_dir, f"splits_{strategy}_seed{self.seed}.json")
        with open(path, 'w') as f:
            json.dump(all_folds, f, indent=2, ensure_ascii=False)
        print(f"Saved {self.n_folds}-fold splits for '{strategy}' to {path}")
        return path

    # ------------------------------------------------------------------
    # 策略 1 — 跨被试划分
    # ------------------------------------------------------------------
    def _cross_subject_split(self, fold):
        """按被试划分：shuffle 后等分成 n_folds 份，取 1 份为测试集"""
        rng = random.Random(self.seed)
        subs = self.subjects.copy()
        rng.shuffle(subs)

        test_subs, train_val_subs = _kfold_select(subs, self.n_folds, fold)

        rng.shuffle(train_val_subs)
        n_val = max(1, int(len(train_val_subs) * self.val_ratio))
        val_subs = train_val_subs[:n_val]
        train_subs = train_val_subs[n_val:]

        return {
            'train_subjects': sorted(train_subs),
            'val_subjects':   sorted(val_subs),
            'test_subjects':  sorted(test_subs),
            'train_stories':  self.stories,
            'val_stories':    self.stories,
            'test_stories':   self.stories,
        }

    # ------------------------------------------------------------------
    # 策略 2 — 跨试次划分
    # ------------------------------------------------------------------
    def _cross_trial_split(self, fold):
        """按 story 划分：shuffle 后等分成 n_folds 份，取 1 份为测试集"""
        rng = random.Random(self.seed)
        trials = self.stories.copy()
        rng.shuffle(trials)

        test_trials, train_val_trials = _kfold_select(trials, self.n_folds, fold)

        rng.shuffle(train_val_trials)
        n_val = max(1, int(len(train_val_trials) * self.val_ratio))
        val_trials = train_val_trials[:n_val]
        train_trials = train_val_trials[n_val:]

        return {
            'train_subjects': self.subjects,
            'val_subjects':   self.subjects,
            'test_subjects':  self.subjects,
            'train_stories':  sorted(train_trials),
            'val_stories':    sorted(val_trials),
            'test_stories':   sorted(test_trials),
        }

    # ------------------------------------------------------------------
    # 策略 3 — 整体数据随机划分
    # ------------------------------------------------------------------
    def _whole_data_split(self, fold):
        """将所有 (subject, story) 对作为原子单元进行 K 折划分"""
        rng = random.Random(self.seed)
        pairs = [(s, t) for s in self.subjects for t in self.stories]
        rng.shuffle(pairs)

        test_pairs, train_val_pairs = _kfold_select(pairs, self.n_folds, fold)

        rng.shuffle(train_val_pairs)
        n_val = max(1, int(len(train_val_pairs) * self.val_ratio))
        val_pairs = train_val_pairs[:n_val]
        train_pairs = train_val_pairs[n_val:]

        return {
            'train_subjects': self.subjects,
            'val_subjects':   self.subjects,
            'test_subjects':  self.subjects,
            'train_stories':  self.stories,
            'val_stories':    self.stories,
            'test_stories':   self.stories,
            'train_pairs':    train_pairs,
            'val_pairs':      val_pairs,
            'test_pairs':     test_pairs,
        }

    # ------------------------------------------------------------------
    # 策略 4 — 单被试独立划分
    # ------------------------------------------------------------------
    def _single_subject_split(self, fold):
        """为每名被试独立做试次级 K 折划分"""
        result = {}
        base_seed = self.seed + fold * 1000

        for subject in self.subjects:
            # 每名被试使用固定但独立的随机种子（int(subject) 确保确定性）
            subj_seed = base_seed + int(subject)
            rng = random.Random(subj_seed)
            trials = self.stories.copy()
            rng.shuffle(trials)

            test_trials, train_val_trials = _kfold_select(
                trials, self.n_folds, fold)

            rng.shuffle(train_val_trials)
            n_val = max(1, int(len(train_val_trials) * self.val_ratio))
            val_trials = train_val_trials[:n_val]
            train_trials = train_val_trials[n_val:]

            result[subject] = {
                'train_subjects': [subject],
                'val_subjects':   [subject],
                'test_subjects':  [subject],
                'train_stories':  sorted(train_trials),
                'val_stories':    sorted(val_trials),
                'test_stories':   sorted(test_trials),
            }

        return result

    # ------------------------------------------------------------------
    # 策略 5 — 每日分层随机划分
    # ------------------------------------------------------------------
    def _daily_stratified_split(self, run_idx):
        """
        每日分层随机划分：每名被试独立随机选取测试和验证试次。

        测试集: 每天 3 个 story → 共 9 个
        验证集: 每天 2 个 story → 共 6 个
        训练集: 剩余 35 个 story

        参数 run_idx (0 ~ 4) 作为重复运行的索引，控制每次的随机选取。
        种子推导: subj_seed = base_seed + int(subject) * 100 + run_idx * 1000
        """
        result = {}
        base_seed = self.seed

        for subject in self.subjects:
            subj_seed = base_seed + int(subject) * 100 + run_idx * 1000
            rng = random.Random(subj_seed)

            test_stories = []
            val_stories = []

            for day in [1, 2, 3]:
                day_pool = DAY_STORIES[day][:]  # copy
                rng.shuffle(day_pool)

                picked_test = day_pool[:TEST_PER_DAY]
                remaining = day_pool[TEST_PER_DAY:]
                picked_val = remaining[:VAL_PER_DAY]

                test_stories.extend(picked_test)
                val_stories.extend(picked_val)

            all_held_out = set(test_stories + val_stories)
            train_stories = sorted(
                [s for s in self.stories if s not in all_held_out])

            result[subject] = {
                'train_subjects': [subject],
                'val_subjects':   [subject],
                'test_subjects':  [subject],
                'train_stories':  sorted(train_stories),
                'val_stories':    sorted(val_stories),
                'test_stories':   sorted(test_stories),
            }

        return result

    # ------------------------------------------------------------------
    # 策略 6 — 每日分层共享划分（所有被试使用相同的 story 划分）
    # ------------------------------------------------------------------
    def _daily_stratified_shared_split(self, run_idx):
        """
        与 daily_stratified 逻辑一致，但所有被试共享相同的 story 划分。

        每名被试的训练/验证/测试试次完全一致，避免不同被试使用不同划分
        导致预训练时数据泄露问题。

        测试集: 每天 3 个 story → 共 9 个
        验证集: 每天 2 个 story → 共 6 个
        训练集: 剩余 35 个 story

        种子推导: 所有被试共用 base_seed + run_idx * 1000
        """
        result = {}
        base_seed = self.seed
        shared_seed = base_seed + run_idx * 1000
        rng = random.Random(shared_seed)

        # 所有被试共享同一份 story 划分
        test_stories = []
        val_stories = []

        for day in [1, 2, 3]:
            day_pool = DAY_STORIES[day][:]
            rng.shuffle(day_pool)

            picked_test = day_pool[:TEST_PER_DAY]
            remaining = day_pool[TEST_PER_DAY:]
            picked_val = remaining[:VAL_PER_DAY]

            test_stories.extend(picked_test)
            val_stories.extend(picked_val)

        all_held_out = set(test_stories + val_stories)
        train_stories = sorted(
            [s for s in self.stories if s not in all_held_out])

        for subject in self.subjects:
            result[subject] = {
                'train_subjects': [subject],
                'val_subjects':   [subject],
                'test_subjects':  [subject],
                'train_stories':  sorted(train_stories),
                'val_stories':    sorted(val_stories),
                'test_stories':   sorted(test_stories),
            }

        return result

    # ------------------------------------------------------------------
    # 策略 7 — 跨天划分 (Day 1-2 训练, Day 3 验证+测试)
    # ------------------------------------------------------------------
    def _cross_day_split(self, run_idx):
        """
        跨天划分：第 1-2 天数据（stories 1-33）固定用于训练，
        第 3 天数据（stories 34-50）随机划分为验证集和测试集。

        训练集: 固定 33 个 story（Day 1-2: 1-33）
        验证集: Day 3 中随机 7 个 story
        测试集: Day 3 中剩余 10 个 story

        参数 run_idx (0 ~ 4) 作为重复运行的索引，控制每次的随机选取。
        种子推导: subj_seed = base_seed + int(subject) * 100 + run_idx * 1000
        """
        result = {}
        base_seed = self.seed

        for subject in self.subjects:
            subj_seed = base_seed + int(subject) * 100 + run_idx * 1000
            rng = random.Random(subj_seed)

            day3_pool = CROSS_DAY_DAY3[:]  # stories 34-50
            rng.shuffle(day3_pool)

            val_stories = day3_pool[:CROSS_DAY_VAL]
            test_stories = day3_pool[CROSS_DAY_VAL:CROSS_DAY_VAL + CROSS_DAY_TEST]
            train_stories = CROSS_DAY_TRAIN[:]  # always 1-33

            result[subject] = {
                'train_subjects': [subject],
                'val_subjects':   [subject],
                'test_subjects':  [subject],
                'train_stories':  sorted(train_stories),
                'val_stories':    sorted(val_stories),
                'test_stories':   sorted(test_stories),
            }

        return result


    # ------------------------------------------------------------------
    # 策略 8 — 每被试独立随机 5 验证 + 5 测试
    # ------------------------------------------------------------------
    def _per_sub_5t5v_split(self, fold):
        """
        所有被试共享相同的随机 5 验证 + 5 测试故事划分。

        设计目的：预训练 + 微调 pipeline（防止数据泄露）。
        - 所有被试使用完全相同的 test/val/train story 划分。
        - 确保同一 story 不会同时出现在某被试的训练集和另一被试的测试集中。
        - 微调时各被试使用相同的 story 划分，与预训练时完全一致。
        """
        result = {}
        rng = random.Random(self.seed)

        pool = self.stories.copy()
        rng.shuffle(pool)

        test_stories = pool[:5]
        val_stories = pool[5:10]
        train_stories = pool[10:]

        for subject in self.subjects:
            result[subject] = {
                'train_subjects': [subject],
                'val_subjects':   [subject],
                'test_subjects':  [subject],
                'train_stories':  sorted(train_stories),
                'val_stories':    sorted(val_stories),
                'test_stories':   sorted(test_stories),
            }

        return result


# ============================================================================
# 辅助函数
# ============================================================================

def _kfold_select(items, n_folds, fold):
    """
    将 items 等分成 n_folds 份，返回 (第 fold 份, 其余份)。
    余数均匀分配到前面的折（前 remainder 折各多 1 个元素）。
    """
    n = len(items)
    fold_size = n // n_folds
    remainder = n % n_folds

    # 计算各折边界
    boundaries = []
    pos = 0
    for f in range(n_folds):
        sz = fold_size + (1 if f < remainder else 0)
        boundaries.append((pos, pos + sz))
        pos += sz

    test_start, test_end = boundaries[fold]
    test_items = items[test_start:test_end]
    train_val_items = items[:test_start] + items[test_end:]

    return test_items, train_val_items


def _make_json_safe(obj):
    """递归转换，使对象可 JSON 序列化（tuple key → string）"""
    if isinstance(obj, dict):
        return {str(k) if isinstance(k, tuple) else k: _make_json_safe(v)
                for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return list(obj)
    return obj


# ============================================================================
# 从 (subject, story) pair 列表构造数据集
# ============================================================================

def get_dataset_from_pairs(pairs, base_config, data_type='train'):
    """
    根据 (subject_id, story_id) pair 列表创建拼接数据集。

    参数:
        pairs:       list[(subject_id, story_id)]
        base_config: 单个数据集的基础配置 dict（含 data_path, label_path,
                     tmin, tmax, standardize, sfreq 等）
        data_type:   'train' | 'val' | 'test'
    """
    from speech_code.utils import EEG_Speech_v1

    # 按被试分组
    subj_to_stories = {}
    for subj, story in pairs:
        subj_to_stories.setdefault(subj, []).append(story)

    datasets = []
    for subj, stories in subj_to_stories.items():
        cfg = dict(base_config)
        cfg['include_subjects'] = [subj]
        cfg['stories'] = sorted(stories)
        cfg['data_type'] = data_type
        # remove keys not accepted by EEG_Speech_v1
        cfg.pop('include_info', None)
        ds = EEG_Speech_v1(**cfg)
        if len(ds) > 0:
            datasets.append(ds)

    if len(datasets) == 0:
        raise ValueError("No samples found for the given pairs")
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def apply_split_to_config(config, split, strategy):
    """
    将 DataSplitter 返回的划分结果写入 config 的 data 部分。

    参数:
        config:   完整配置 dict（原地修改）
        split:    DataSplitter.get_split() 的返回值
        strategy: 划分策略名

    返回:
        config (已原地修改)
    """
    ds_configs = config['data']['datasets']

    for partition, key in [('train', 'train'), ('val', 'val'), ('test', 'test')]:
        if partition not in ds_configs:
            continue
        for ds_entry in ds_configs[partition]:
            ds_cfg = list(ds_entry.values())[0]
            ds_cfg['include_subjects'] = split[f'{key}_subjects']
            ds_cfg['stories'] = split[f'{key}_stories']

    # whole_data 附加 pairs 信息
    if strategy == 'whole_data':
        config['data']['split_pairs'] = {
            'train': split['train_pairs'],
            'val':   split['val_pairs'],
            'test':  split['test_pairs'],
        }

    config['data']['split_strategy'] = strategy
    return config
