import random
from dataclasses import dataclass


SCALES = (0.05, 0.10, 0.20, 0.50, 0.80, 1.00)
ALL_SUBJECTS = tuple(f"{i:02d}" for i in range(1, 26))
ALL_STORIES = tuple(range(1, 51))


@dataclass(frozen=True)
class ScalingSplit:
    scale: float
    subset_seed: int
    train_pool: tuple[int, ...]
    train_stories: tuple[int, ...]
    val_stories: tuple[int, ...]
    test_stories: tuple[int, ...]
    permutation: tuple[int, ...]


def normalize_scale(scale: float) -> float:
    for candidate in SCALES:
        if abs(scale - candidate) < 1e-9:
            return candidate
    allowed = ", ".join(str(value) for value in SCALES)
    raise ValueError(f"Unsupported scale {scale}. Choose from: {allowed}")


def scale_tag(scale: float) -> str:
    return f"{int(round(normalize_scale(scale) * 100)):03d}"


def build_scaling_split(
    scale: float,
    subset_seed: int,
    split_seed: int = 545,
) -> ScalingSplit:
    scale = normalize_scale(scale)
    split_permutation = list(ALL_STORIES)
    random.Random(split_seed).shuffle(split_permutation)
    test_stories = sorted(split_permutation[:5])
    val_stories = sorted(split_permutation[5:10])
    train_pool = sorted(split_permutation[10:])

    # Scaling is applied later as a chronological recording-duration prefix.
    # Keep the complete train and validation unit pools at every scale.
    permutation = train_pool.copy()
    selected = train_pool

    if set(selected) & set(val_stories) or set(selected) & set(test_stories):
        raise RuntimeError("Training subset overlaps validation or test stories")

    return ScalingSplit(
        scale=scale,
        subset_seed=subset_seed,
        train_pool=tuple(train_pool),
        train_stories=tuple(selected),
        val_stories=tuple(val_stories),
        test_stories=tuple(test_stories),
        permutation=tuple(permutation),
    )


def assert_nested_subsets(subset_seed: int, split_seed: int = 545) -> None:
    previous: set[int] = set()
    for scale in SCALES:
        current = set(build_scaling_split(scale, subset_seed, split_seed).train_stories)
        if not previous.issubset(current):
            raise RuntimeError(f"Scaling subsets are not nested at scale={scale}")
        previous = current
