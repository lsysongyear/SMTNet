import random
from dataclasses import dataclass
from pathlib import Path


SCALES = (0.05, 0.10, 0.20, 0.40, 0.50, 0.60, 0.80, 1.00)
ALL_SUBJECTS = (
    "001", "002", "003", "005", "006", "007", "008", "009",
    "010", "011", "012", "013", "014", "015", "016", "018",
    "019", "020", "021", "022", "023", "024", "026",
)


@dataclass(frozen=True)
class SubjectSplit:
    train_stories: tuple[str, ...]
    val_stories: tuple[str, ...]
    test_stories: tuple[str, ...]


def normalize_scale(scale: float) -> float:
    for candidate in SCALES:
        if abs(scale - candidate) < 1e-9:
            return candidate
    allowed = ", ".join(str(value) for value in SCALES)
    raise ValueError(f"Unsupported scale {scale}. Choose from: {allowed}")


def scale_tag(scale: float) -> str:
    return f"{int(round(normalize_scale(scale) * 100)):03d}"


def discover_subject_stories(
    data_path,
    subjects=ALL_SUBJECTS,
    preproc="BP",
    eeg_fs_tag="250Hz",
):
    requested = set(subjects)
    result = {subject: set() for subject in subjects}
    for path in sorted(Path(data_path).iterdir()):
        if path.suffix != ".npy":
            continue
        parts = path.stem.split("_")
        if len(parts) < 4 or parts[-2] != preproc or parts[-1] != eeg_fs_tag:
            continue
        subject = parts[0].replace("sub-", "", 1)
        if subject not in requested:
            continue
        audio_key = "_".join(parts[1:-2])
        if not audio_key.startswith("audio-"):
            continue
        result[subject].add(audio_key.replace("audio-", "", 1))

    missing = [subject for subject, stories in result.items() if len(stories) < 3]
    if missing:
        raise ValueError(f"Subjects with fewer than three trials: {missing}")
    return {subject: tuple(sorted(stories)) for subject, stories in result.items()}


def build_subject_splits(subject_stories, split_seed=84):
    pools = {}
    for subject in ALL_SUBJECTS:
        pool = set(subject_stories.get(subject, ()))
        if len(pool) < 3:
            raise ValueError(f"Subject {subject} needs at least three trials")
        pools[subject] = pool

    common_stories = sorted(set.intersection(*(pools[s] for s in ALL_SUBJECTS)))
    if len(common_stories) < 2:
        raise ValueError("At least two stories must be shared by every subject")
    random.Random(split_seed).shuffle(common_stories)
    test_stories = tuple(common_stories[:1])
    val_stories = tuple(common_stories[1:2])
    held_out = set(test_stories + val_stories)

    splits = {}
    for subject in ALL_SUBJECTS:
        train_stories = tuple(sorted(pools[subject] - held_out))
        splits[subject] = SubjectSplit(train_stories, val_stories, test_stories)
    assert_no_story_leakage(splits)
    return splits


def assert_paper_split(splits):
    for subject, split in splits.items():
        if split.val_stories != ("audiobook-5-1",) or split.test_stories != ("audiobook-5-3",):
            raise RuntimeError(
                f"Subject {subject} does not match the candidate-18 paper split: "
                f"val={split.val_stories}, test={split.test_stories}"
            )


def partition_pairs(splits, partition):
    attribute = f"{partition}_stories"
    if partition not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported partition: {partition}")
    return tuple(
        (subject, story)
        for subject in ALL_SUBJECTS
        for story in getattr(splits[subject], attribute)
    )


def legacy_partition_pairs(subject_stories, splits, partition, reference_subject="001"):
    attribute = f"{partition}_stories"
    if partition not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported partition: {partition}")
    shared_stories = set(getattr(splits[reference_subject], attribute))
    return tuple(
        (subject, story)
        for subject in ALL_SUBJECTS
        for story in subject_stories[subject]
        if story in shared_stories
    )


def assert_no_story_leakage(splits):
    for subject, split in splits.items():
        train = set(split.train_stories)
        val = set(split.val_stories)
        test = set(split.test_stories)
        if train & val or train & test or val & test:
            raise RuntimeError(f"Partition overlap for subject {subject}")

    train = {
        story for split in splits.values() for story in split.train_stories
    }
    val = {
        story for split in splits.values() for story in split.val_stories
    }
    test = {
        story for split in splits.values() for story in split.test_stories
    }
    if train & val or train & test or val & test:
        raise RuntimeError("Cross-subject story leakage detected")


def nested_subset_indices(dataset_size, scale, subset_seed):
    scale = normalize_scale(scale)
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    permutation = list(range(dataset_size))
    random.Random(subset_seed).shuffle(permutation)
    count = max(1, int(round(dataset_size * scale)))
    return tuple(sorted(permutation[:count]))


def assert_nested_subsets(dataset_size, subset_seed):
    previous = set()
    for scale in SCALES:
        current = set(nested_subset_indices(dataset_size, scale, subset_seed))
        if not previous.issubset(current):
            raise RuntimeError(f"Window subsets are not nested at scale={scale}")
        previous = current
