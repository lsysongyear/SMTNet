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
