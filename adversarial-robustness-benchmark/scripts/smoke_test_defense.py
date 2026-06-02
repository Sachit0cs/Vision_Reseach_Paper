"""CLI: clean-accuracy smoke test for the defense model — run this FIRST.

Purpose: before spending hours on the 4 benchmark axes, confirm the defense
checkpoint actually classifies *clean* images at an accuracy comparable to the
baseline models. If clean accuracy is at the floor (the old 100-class-trained /
1000-class-scored run sat at 0.06), studying its robustness is meaningless —
robustness is only interesting relative to a model that works on clean inputs.

This does inference only — no attacks, no gradients — so it is fast (one forward
pass over the 1000-image clean set). It:
  1. loads the defense checkpoint through the same DefenseModel wrapper the
     benchmark uses (resolves DEFENSE_CHECKPOINT_PATH / models/checkpoints/...),
  2. measures top-1 clean accuracy on the committed clean benchmark set,
  3. prints it next to the baseline ResNet-50 clean accuracy (read from the
     committed results/gradient/accuracy_table.csv, if present), and
  4. exits non-zero if clean accuracy is below --threshold, so a calling script
     / notebook can STOP before running the full benchmark.

Usage:
    python scripts/smoke_test_defense.py
    python scripts/smoke_test_defense.py --threshold 0.55
    python scripts/smoke_test_defense.py --limit 200            # quick partial check
    python scripts/smoke_test_defense.py --checkpoint /kaggle/input/defense-resnet50-ckpt/defense_final.pt
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys

import numpy as np
import torch
import yaml
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets.loader import load_clean_dataset  # noqa: E402
from models.classifiers import build_classifier  # noqa: E402

# Gate default: Gate C in plan.md required clean >= 0.55 on the trained classes.
# The six supervised baselines sit at 0.62-0.79 clean; a healthy defense
# fine-tune should land in roughly the same band.
_DEFAULT_THRESHOLD = 0.55


def load_config() -> dict:
    with open(os.path.join(_REPO_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def baseline_clean_accuracy(model_key: str = "resnet50") -> float | None:
    """Read a reference clean accuracy from the committed gradient CSV, if present."""
    csv_path = os.path.join(_REPO_ROOT, "results", "gradient", "accuracy_table.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("model") == model_key:
                    return float(row["clean_accuracy"])
    except (KeyError, ValueError, OSError):
        return None
    return None


@torch.no_grad()
def clean_accuracy(clf, dataset, indices, batch_size=64) -> tuple[int, int]:
    """Top-1 clean accuracy over dataset[indices] — same protocol as the runners."""
    tensors, labels = [], []
    for i in indices:
        image, label = dataset[i]
        tensors.append(clf.preprocess(image))
        labels.append(int(label))
    batch = torch.stack(tensors)
    labels = torch.tensor(labels, dtype=torch.long)

    correct = 0
    total = batch.shape[0]
    for start in tqdm(range(0, total, batch_size), desc="clean eval", leave=False):
        x = batch[start:start + batch_size].to(clf.device)
        y = labels[start:start + batch_size].to(clf.device)
        preds = clf.logits(x).argmax(dim=1)
        correct += int((preds == y).sum().item())
    return correct, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="defense_resnet50",
                        help="Model key to smoke-test (default: defense_resnet50).")
    parser.add_argument("--threshold", type=float, default=_DEFAULT_THRESHOLD,
                        help=f"Minimum clean accuracy to PASS (default: {_DEFAULT_THRESHOLD}).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N clean images (quick partial check).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Defense checkpoint path; sets DEFENSE_CHECKPOINT_PATH for the wrapper.")
    args = parser.parse_args()

    if args.checkpoint:
        os.environ["DEFENSE_CHECKPOINT_PATH"] = args.checkpoint

    cfg = load_config()
    set_seed(int(cfg.get("seed", 42)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 64)
    print(f"  CLEAN-ACCURACY SMOKE TEST — {args.model}")
    print("=" * 64)
    print(f"  device     : {device}")

    clf = build_classifier(args.model, device)
    ckpt = getattr(clf, "checkpoint_path", None)
    if ckpt:
        print(f"  checkpoint : {ckpt}")

    dataset = load_clean_dataset()
    n = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    indices = list(range(n))
    print(f"  clean set  : {n} images")

    correct, total = clean_accuracy(clf, dataset, indices)
    acc = correct / total if total else 0.0

    baseline = baseline_clean_accuracy("resnet50")
    print("-" * 64)
    print(f"  CLEAN ACCURACY: {correct}/{total} = {acc:.4f}")
    if baseline is not None:
        print(f"  baseline resnet50 clean (reference): {baseline:.4f}")
        print(f"  defense / baseline ratio           : {acc / baseline:.2f}" if baseline else "")
    print(f"  PASS threshold : {args.threshold:.4f}")
    print("-" * 64)

    if acc >= args.threshold:
        print(f"  RESULT: PASS — clean accuracy {acc:.4f} >= {args.threshold:.4f}.")
        print("  Robustness is worth measuring. Proceed to the 4 benchmark axes.")
        print("=" * 64)
        sys.exit(0)
    else:
        print(f"  RESULT: FAIL — clean accuracy {acc:.4f} < {args.threshold:.4f}.")
        print("  The model is at/near the floor on CLEAN images. Do NOT run the")
        print("  benchmark — its robustness numbers would be a floor artifact, not")
        print("  a result. Report this number back and re-train (check the per-epoch")
        print("  clean_acc in results/defense/training_log.json and the train/eval")
        print("  label-space match in config.yaml: defense.num_classes must be 1000).")
        print("=" * 64)
        sys.exit(1)


if __name__ == "__main__":
    main()
