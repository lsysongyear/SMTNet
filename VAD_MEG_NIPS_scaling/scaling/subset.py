import random
from dataclasses import dataclass


SCALES = (0.05, 0.10, 0.20, 0.50, 0.80, 1.00)


@dataclass(frozen=True)
class RunSplit:
    train: tuple[tuple[str, ...], ...]
    val: tuple[tuple[str, ...], ...]
    test: tuple[tuple[str, ...], ...]


def normalize_scale(scale):
    for candidate in SCALES:
        if abs(scale - candidate) < 1e-9:
            return candidate
    allowed = ", ".join(str(value) for value in SCALES)
    raise ValueError(f"Unsupported scale {scale}. Choose from: {allowed}")


def scale_tag(scale):
    return f"{int(round(normalize_scale(scale) * 100)):03d}"


def build_run_split(run_keys, split_seed=46):
    ordered = [tuple(str(value) for value in run_key) for run_key in run_keys]
    shuffled = ordered.copy()
    random.Random(split_seed).shuffle(shuffled)
    heldout_count = max(1, len(shuffled) // 10)
    test = tuple(shuffled[:heldout_count])
    val = tuple(shuffled[heldout_count : 2 * heldout_count])
    train = tuple(shuffled[2 * heldout_count :])
    split = RunSplit(train=train, val=val, test=test)
    assert_disjoint_split(split)
    return split


def assert_disjoint_split(split):
    train = set(split.train)
    val = set(split.val)
    test = set(split.test)
    if train & val or train & test or val & test:
        raise RuntimeError("Train, validation, and test run keys must be disjoint")
