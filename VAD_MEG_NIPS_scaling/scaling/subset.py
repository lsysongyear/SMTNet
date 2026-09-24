import random
from dataclasses import dataclass


SCALES = (0.05, 0.10, 0.20, 0.40, 0.50, 0.60, 0.80, 1.00)


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


def nested_run_subset(train_run_keys, scale, subset_seed=1001):
    scale = normalize_scale(scale)
    train_run_keys = tuple(tuple(run_key) for run_key in train_run_keys)
    if not train_run_keys:
        raise ValueError("Training run pool cannot be empty")
    if scale == 1.0:
        return train_run_keys

    permutation = list(range(len(train_run_keys)))
    random.Random(subset_seed).shuffle(permutation)
    count = max(1, int(round(len(train_run_keys) * scale)))
    selected_indices = set(permutation[:count])
    return tuple(
        run_key for index, run_key in enumerate(train_run_keys) if index in selected_indices
    )


def assert_disjoint_split(split):
    train = set(split.train)
    val = set(split.val)
    test = set(split.test)
    if train & val or train & test or val & test:
        raise RuntimeError("Train, validation, and test run keys must be disjoint")


def assert_nested_subsets(train_run_keys, subset_seed=1001):
    previous = set()
    for scale in SCALES:
        current = set(nested_run_subset(train_run_keys, scale, subset_seed))
        if not previous.issubset(current):
            raise RuntimeError(f"Run subsets are not nested at scale={scale}")
        previous = current
