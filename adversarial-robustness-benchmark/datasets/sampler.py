"""Class-balanced sampling with a fixed seed.

Two entry points:

* ``ReservoirPerClass`` — single-pass, low-memory streaming selection. Used by
  ``scripts/generate_datasets.py`` while iterating the ImageNet-val parquet so
  the full 50k-image split never has to be held in memory.
* ``class_balanced_indices`` — in-memory selection when all labels are already
  known. Used wherever a label array is available up front.

Both are deterministic given the seed and the iteration order of the data.
"""

from __future__ import annotations

import random
from collections import defaultdict


class ReservoirPerClass:
    """Keep a seeded uniform-random sample of ``per_class`` items per class.

    Reservoir sampling lets us pick, say, 1 image per class out of a stream
    without knowing how many images each class has or storing them all. After
    the stream is exhausted, every class's kept items are a uniform random
    subset of everything seen for that class.
    """

    def __init__(self, per_class: int = 1, seed: int = 42):
        self.per_class = per_class
        self._rng = random.Random(seed)
        self._counts: dict[int, int] = defaultdict(int)
        self._kept: dict[int, list] = defaultdict(list)

    def offer(self, label: int, item) -> None:
        """Offer one ``(label, item)`` from the stream to the reservoir."""
        self._counts[label] += 1
        k = self._counts[label]
        kept = self._kept[label]
        if len(kept) < self.per_class:
            kept.append(item)
        else:
            # Replace a random kept item with probability per_class / k.
            j = self._rng.randint(0, k - 1)
            if j < self.per_class:
                kept[j] = item

    def result(self) -> dict[int, list]:
        """Return ``{label: [items...]}`` for every class seen."""
        return {label: list(items) for label, items in self._kept.items()}

    def is_complete(self, num_classes: int) -> bool:
        """True once every class has its full quota — lets the caller stop early."""
        if len(self._kept) < num_classes:
            return False
        return all(len(v) >= self.per_class for v in self._kept.values())


def class_balanced_indices(
    labels: list[int], per_class: int = 1, seed: int = 42
) -> list[int]:
    """Pick ``per_class`` indices for each label, seeded and sorted by label.

    Args:
        labels: ground-truth integer label for every item, by index.
        per_class: how many indices to keep per label.
        seed: RNG seed for reproducibility.

    Returns:
        Selected indices, ordered by (label, original index).
    """
    rng = random.Random(seed)
    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        by_label[label].append(idx)

    selected: list[int] = []
    for label in sorted(by_label):
        pool = by_label[label]
        if len(pool) <= per_class:
            chosen = pool
        else:
            chosen = sorted(rng.sample(pool, per_class))
        selected.extend(chosen)
    return selected
