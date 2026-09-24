import os
import random
from dataclasses import dataclass


SCALES = (0.05, 0.10, 0.20, 0.50, 0.80, 1.00)
ALL_SUBJECTS = tuple(f"{subject:02d}" for subject in range(1, 13))


@dataclass(frozen=True)
class StorySplit:
    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]


def infer_split_items(config):
    dataset_key = list(config["data"]["datasets"]["train"][0].keys())[0]
    ds_cfg = config["data"]["datasets"]["train"][0][dataset_key]
    data_path = ds_cfg["data_path"]
    if dataset_key == "sem4lang_meg":
        subjects, stories = set(), set()
        preproc = ds_cfg.get("preproc", "LP-30")
        fs_tag = ds_cfg.get("eeg_fs_tag", "64Hz")
        for filename in os.listdir(data_path):
            if not filename.endswith(".npy"):
                continue
            parts = filename[:-4].split("_")
            if len(parts) < 4:
                continue
            if parts[2] != preproc or parts[3] != fs_tag:
                continue
            subjects.add(parts[0].replace("sub-", ""))
            stories.add(int(parts[1].replace("story-", "")))
        return sorted(subjects), sorted(stories)
    return [f"{i:02d}" for i in range(1, 26)], list(range(1, 51))


def normalize_scale(scale: float) -> float:
    for candidate in SCALES:
        if abs(scale - candidate) < 1e-9:
            return candidate
    allowed = ", ".join(str(value) for value in SCALES)
    raise ValueError(f"Unsupported scale {scale}. Choose from: {allowed}")


def scale_tag(scale: float) -> str:
    return f"{int(round(normalize_scale(scale) * 100)):03d}"


def build_story_split(stories, split_seed: int = 5) -> StorySplit:
    pool = sorted({int(story) for story in stories})
    if len(pool) < 11:
        raise ValueError("At least 11 stories are required for a 5-val/5-test split")
    random.Random(split_seed).shuffle(pool)
    split = StorySplit(
        train=tuple(sorted(pool[10:])),
        val=tuple(sorted(pool[5:10])),
        test=tuple(sorted(pool[:5])),
    )
    assert_disjoint_split(split)
    return split


def assert_disjoint_split(split: StorySplit):
    train = set(split.train)
    val = set(split.val)
    test = set(split.test)
    if train & val or train & test or val & test:
        raise RuntimeError("Train, validation, and test stories must be disjoint")
