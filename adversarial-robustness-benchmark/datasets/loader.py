"""Load the clean benchmark set and the Phase-4 training set.

The clean set is materialised on disk by ``scripts/generate_datasets.py`` as
plain image files plus a JSON manifest. This loader has no network or
HuggingFace dependency — it only reads what the generator wrote.
"""

from __future__ import annotations

import json
import os

from PIL import Image

# Repo root = parent of this file's directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CleanImageDataset:
    """The original (unmodified) ImageNet-val benchmark subset.

    Iterating yields ``(PIL.Image, int_label)`` pairs. Ground-truth integer
    labels are exposed so accuracy can be computed directly.
    """

    def __init__(self, clean_dir: str = "data/clean"):
        root = clean_dir if os.path.isabs(clean_dir) else os.path.join(_REPO_ROOT, clean_dir)
        self.root = root
        manifest_path = os.path.join(root, "manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"No clean-set manifest at {manifest_path}. "
                "Run: python scripts/generate_datasets.py --clean"
            )
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        self.records = self.manifest["images"]

        classes_path = os.path.join(root, "imagenet_classes.json")
        with open(classes_path, "r", encoding="utf-8") as f:
            self.categories: list[str] = json.load(f)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> tuple[Image.Image, int]:
        rec = self.records[i]
        path = os.path.join(self.root, rec["filename"])
        image = Image.open(path).convert("RGB")
        return image, int(rec["label"])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def labels(self) -> list[int]:
        return [int(r["label"]) for r in self.records]


def load_clean_dataset(clean_dir: str = "data/clean") -> CleanImageDataset:
    """Convenience wrapper returning the clean benchmark dataset."""
    return CleanImageDataset(clean_dir)


def load_eval_subset(
    n: int,
    seed: int = 42,
    clean_dir: str = "data/clean",
) -> tuple["CleanImageDataset", list[int]]:
    """Return (dataset, indices) for a class-balanced subset of size n.

    Always produces the same indices for a given (n, seed) — deterministic.
    Labels are read from the clean-set manifest; no separate file is needed.
    The caller iterates ``dataset[i]`` for i in indices to get (image, label)
    pairs — the label is already bundled with each image, so there is no
    alignment problem.

    Usage in the pipeline for slow attacks (AutoAttack, Square):
        dataset, indices = load_eval_subset(cfg["attack"]["autoattack_eval_subset"])
        images = torch.stack([to_tensor(dataset[i][0]) for i in indices])
        labels = [dataset[i][1] for i in indices]
    """
    from .sampler import class_balanced_indices

    dataset = CleanImageDataset(clean_dir)
    indices = class_balanced_indices(dataset.labels, per_class=max(1, n // len(dataset.categories)), seed=seed)
    # Trim to exactly n in case rounding gave slightly more.
    indices = indices[:n]
    return dataset, indices


def load_cifar100(train: bool = True, download: bool = True, root: str = "data/cifar100"):
    """Load CIFAR-100 for the Phase-4 defense fine-tuning.

    Thin wrapper over ``torchvision.datasets.CIFAR100``. Imported lazily so the
    clean-set build does not require torchvision.
    """
    from torchvision import datasets as tv_datasets

    abs_root = root if os.path.isabs(root) else os.path.join(_REPO_ROOT, root)
    return tv_datasets.CIFAR100(root=abs_root, train=train, download=download)
