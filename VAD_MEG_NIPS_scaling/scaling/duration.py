from fractions import Fraction


def prefix_allocation(lengths, fraction):
    """Allocate an exact chronological prefix over non-negative sample lengths."""
    fraction = Fraction(str(fraction))
    if fraction <= 0 or fraction > 1:
        raise ValueError("recording fraction must be in (0, 1]")

    lengths = tuple(int(length) for length in lengths)
    if any(length < 0 for length in lengths):
        raise ValueError("recording lengths must be non-negative")

    total = sum(lengths)
    target = total * fraction.numerator // fraction.denominator
    remaining = target
    selected = []
    for length in lengths:
        keep = min(length, remaining)
        selected.append(keep)
        remaining -= keep

    if remaining != 0:
        raise RuntimeError("failed to allocate the requested recording prefix")
    return target, tuple(selected)


def canonical_prefix_allocation(keyed_lengths, canonical_keys, fraction):
    """Allocate a prefix canonically while retaining caller-controlled collection order."""
    keyed_lengths = tuple((tuple(key), int(length)) for key, length in keyed_lengths)
    lengths_by_key = dict(keyed_lengths)
    if len(lengths_by_key) != len(keyed_lengths):
        raise ValueError("run keys must be unique")

    canonical_keys = tuple(tuple(key) for key in canonical_keys)
    canonical_set = set(canonical_keys)
    unknown = set(lengths_by_key) - canonical_set
    if unknown:
        raise ValueError(f"run keys absent from canonical order: {sorted(unknown)}")

    allocation_order = tuple(key for key in canonical_keys if key in lengths_by_key)
    target, selected = prefix_allocation(
        [lengths_by_key[key] for key in allocation_order], fraction
    )
    return target, dict(zip(allocation_order, selected)), allocation_order
