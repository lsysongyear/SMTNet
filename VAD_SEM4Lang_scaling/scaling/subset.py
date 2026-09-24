import random
from dataclasses import dataclass


SCALES = (0.05, 0.10, 0.20, 0.40, 0.50, 0.60, 0.80, 1.00)
ALL_SUBJECTS = tuple(f"{subject:02d}" for subject in range(1, 13))


@dataclass(frozen=True)
class StorySplit:
    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]


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


def nested_story_subset(train_stories, scale: float, subset_seed: int = 1001):
    scale = normalize_scale(scale)
    train_stories = tuple(sorted(int(story) for story in train_stories))
    if not train_stories:
        raise ValueError("Training story pool cannot be empty")
    if scale == 1.0:
        return train_stories

    permutation = list(range(len(train_stories)))
    random.Random(subset_seed).shuffle(permutation)
    count = max(1, int(round(len(train_stories) * scale)))
    selected_indices = set(permutation[:count])
    return tuple(
        story for index, story in enumerate(train_stories)
        if index in selected_indices
    )


def assert_disjoint_split(split: StorySplit):
    train = set(split.train)
    val = set(split.val)
    test = set(split.test)
    if train & val or train & test or val & test:
        raise RuntimeError("Train, validation, and test stories must be disjoint")


def assert_nested_subsets(train_stories, subset_seed: int = 1001):
    previous = set()
    for scale in SCALES:
        current = set(nested_story_subset(train_stories, scale, subset_seed))
        if not previous.issubset(current):
            raise RuntimeError(f"Story subsets are not nested at scale={scale}")
        previous = current
