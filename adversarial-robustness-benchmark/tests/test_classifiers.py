"""Classifier smoke tests (Phase 0).

Verifies, for each of the 7 benchmarked models:
  * it loads;
  * preprocess returns a [0, 1] tensor;
  * logits returns finite (1, 1000) scores;
  * logits stays differentiable w.r.t. the input (white-box attacks need this);
  * sample top-1 accuracy on clean images is in a sane range.

Run directly for a readable summary table:
    python tests/test_classifiers.py
Or via pytest:
    pytest -v tests/test_classifiers.py

Model weights download on first run; on a GPU box this is fast.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets.loader import load_clean_dataset  # noqa: E402
from models.classifiers import build_classifier  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_SIZE = 50

with open(os.path.join(_REPO_ROOT, "config.yaml"), "r", encoding="utf-8") as _f:
    MODEL_KEYS = yaml.safe_load(_f)["models"]

# Build each model once and reuse across tests (weights are large).
_CLASSIFIER_CACHE: dict = {}


def _get_classifier(key: str):
    if key not in _CLASSIFIER_CACHE:
        _CLASSIFIER_CACHE[key] = build_classifier(key, DEVICE)
    return _CLASSIFIER_CACHE[key]


@pytest.fixture(scope="session")
def dataset():
    return load_clean_dataset()


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_loads_and_runs(key, dataset):
    """Model loads; preprocess is [0,1]; logits is finite (1, 1000)."""
    clf = _get_classifier(key)
    image, _ = dataset[0]

    x = clf.preprocess(image)
    assert x.shape[0] == 3, f"{key}: preprocess must return (3, H, W)"
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0001, f"{key}: not in [0,1]"

    out = clf.logits(x.unsqueeze(0))
    assert tuple(out.shape) == (1, 1000), f"{key}: logits shape {tuple(out.shape)}"
    assert torch.isfinite(out).all(), f"{key}: non-finite logits"


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_logits_differentiable(key, dataset):
    """logits must be differentiable w.r.t. the input (white-box attacks)."""
    clf = _get_classifier(key)
    image, _ = dataset[0]
    x = clf.preprocess(image).unsqueeze(0).to(DEVICE).requires_grad_(True)
    clf.logits(x).sum().backward()
    assert x.grad is not None, f"{key}: no input gradient"
    assert torch.isfinite(x.grad).all(), f"{key}: non-finite input gradient"


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_sample_accuracy(key, dataset):
    """Sample top-1 accuracy should be well above chance (catches preprocess bugs)."""
    clf = _get_classifier(key)
    correct = 0
    for i in range(SAMPLE_SIZE):
        image, label = dataset[i]
        with torch.no_grad():
            pred = int(clf.logits(clf.preprocess(image).unsqueeze(0)).argmax(dim=1))
        correct += int(pred == label)
    acc = correct / SAMPLE_SIZE
    assert acc > 0.5, f"{key}: sample top-1 {acc:.2f} too low — check preprocessing"


def _main() -> None:
    """Readable summary table when run as a script."""
    dataset = load_clean_dataset()
    image0, label0 = dataset[0]
    print(f"device: {DEVICE}   |   sample size: {SAMPLE_SIZE}\n")
    header = f"{'model':<18}{'loads':<8}{'logits':<10}{'grad':<8}{'top1':<8}prediction[img0]"
    print(header)
    print("-" * len(header))
    for key in MODEL_KEYS:
        try:
            clf = build_classifier(key, DEVICE)
            x = clf.preprocess(image0).unsqueeze(0)
            shape_ok = tuple(clf.logits(x).shape) == (1, 1000)

            xg = clf.preprocess(image0).unsqueeze(0).to(DEVICE).requires_grad_(True)
            clf.logits(xg).sum().backward()
            grad_ok = xg.grad is not None

            correct = 0
            for i in range(SAMPLE_SIZE):
                im, lb = dataset[i]
                with torch.no_grad():
                    correct += int(clf.logits(clf.preprocess(im).unsqueeze(0)).argmax(1)) == lb
            acc = correct / SAMPLE_SIZE

            pred, conf = clf.predict(image0)
            print(
                f"{key:<18}{'ok':<8}{str(shape_ok):<10}{str(grad_ok):<8}"
                f"{acc:<8.2f}{pred[:28]} ({conf:.2f})"
            )
        except Exception as exc:  # noqa: BLE001 - smoke test: report and continue
            print(f"{key:<18}FAILED: {type(exc).__name__}: {exc}")
    print(f"\nground-truth label for img0: {label0} = {dataset.categories[label0]}")


if __name__ == "__main__":
    _main()
