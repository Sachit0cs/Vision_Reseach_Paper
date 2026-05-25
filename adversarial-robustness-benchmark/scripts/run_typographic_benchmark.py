"""CLI: evaluate the 7 models on the typographic-poisoned set.

Phase 1, Axis C runner — semantic, model-agnostic attack. The poisoned set
was already generated once by ``scripts/generate_datasets.py --typographic``
and committed under ``data/poisoned/typographic/``. This script does inference
only: one round of clean-set eval + one round of poisoned-set eval per model,
no adversarial generation, no gradients required.

Outputs one JSON per model at ``results/typographic/<model>.json`` containing
clean accuracy, robust accuracy (untargeted: did the model flip away from the
true label?), targeted-attack success rate (TASR: did the model predict the
overlay's target class?), and fooling rate (originally-correct images that
flipped). Per-model JSONs are the resumption unit — re-running skips finished
models unless ``--force`` is passed.

Usage:
    python scripts/run_typographic_benchmark.py
    python scripts/run_typographic_benchmark.py --models resnet50,clip_vit_b16
    python scripts/run_typographic_benchmark.py --force
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import yaml
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets.loader import load_clean_dataset, load_typographic_dataset  # noqa: E402
from models.classifiers import build_classifier  # noqa: E402

# Inference batch sizes — typographic eval is forward-only so we can go wider
# than the gradient runner. T4-safe.
_INFER_BATCH = {
    "resnet50": 64,
    "vgg16": 32,
    "convnext_tiny": 32,
    "vit_b_16": 32,
    "swin_t": 32,
    "efficientnet_b0": 64,
    "clip_vit_b16": 32,
}


def load_config() -> dict:
    with open(os.path.join(_REPO_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _preprocess_dataset(clf, dataset, get_image) -> torch.Tensor:
    """Run the classifier's PIL preprocess over a dataset.

    ``get_image(dataset, i)`` returns a PIL.Image for index i. We pass this
    function in so the same routine handles the clean dataset (which yields
    (image, label) tuples) and the typographic dataset (which yields
    (image, label, target_label) triples).
    """
    tensors = []
    for i in range(len(dataset)):
        img = get_image(dataset, i)
        tensors.append(clf.preprocess(img))
    return torch.stack(tensors, dim=0)


@torch.no_grad()
def _predict(clf, batch_0_1, batch_size=32) -> torch.Tensor:
    """Argmax labels (long, on CPU) over a [0,1] image batch, mini-batched."""
    preds = []
    total = batch_0_1.shape[0]
    for start in range(0, total, batch_size):
        x = batch_0_1[start:start + batch_size].to(clf.device)
        preds.append(clf.logits(x).argmax(dim=1).cpu())
    return torch.cat(preds, dim=0).long()


def run_one(model_key: str, cfg: dict, out_dir: str, force: bool) -> str:
    """Run a single model end-to-end and write its JSON to out_dir."""
    out_path = os.path.join(out_dir, f"{model_key}.json")
    if not force and os.path.exists(out_path):
        print(f"  skip  {model_key}  (cached at {out_path})")
        return out_path

    set_seed(int(cfg["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{model_key}] loading model on {device} ...")
    clf = build_classifier(model_key, device)

    clean_ds = load_clean_dataset()
    typo_ds = load_typographic_dataset()

    if len(clean_ds) != len(typo_ds):
        raise RuntimeError(
            f"Dataset size mismatch: clean has {len(clean_ds)} images, "
            f"typographic has {len(typo_ds)}. Re-generate the typographic set."
        )
    if clean_ds.labels != typo_ds.labels:
        raise RuntimeError(
            "Per-image label mismatch between clean and typographic manifests. "
            "Re-generate the typographic set from the current clean set."
        )

    print(f"  clean      : {len(clean_ds)} images")
    print(f"  typographic: {len(typo_ds)} images (seed={typo_ds.seed}, "
          f"font_size_frac={typo_ds.config.get('font_size_frac')})")

    # Preprocess both sets in CPU memory — 1k images at 224^2 ≈ ~600 MB float32,
    # well within reach. Move per-batch in _predict.
    t0 = time.perf_counter()
    clean_batch = _preprocess_dataset(clf, clean_ds, lambda d, i: d[i][0])
    typo_batch = _preprocess_dataset(clf, typo_ds, lambda d, i: d[i][0])
    t_prep = time.perf_counter() - t0

    labels = torch.tensor(clean_ds.labels, dtype=torch.long)
    target_labels = torch.tensor(typo_ds.target_labels, dtype=torch.long)
    bs = _INFER_BATCH.get(model_key, 32)

    # Clean predictions
    t0 = time.perf_counter()
    clean_preds = _predict(clf, clean_batch, batch_size=bs)
    t_clean = time.perf_counter() - t0
    clean_correct = int((clean_preds == labels).sum().item())
    clean_acc = clean_correct / len(labels)
    print(f"  clean acc      : {clean_correct}/{len(labels)} = {clean_acc:.3f}  ({t_clean:.1f}s)")

    # Poisoned predictions
    t0 = time.perf_counter()
    typo_preds = _predict(clf, typo_batch, batch_size=bs)
    t_typo = time.perf_counter() - t0
    robust_correct = int((typo_preds == labels).sum().item())
    robust_acc = robust_correct / len(labels)
    print(f"  robust acc     : {robust_correct}/{len(labels)} = {robust_acc:.3f}  ({t_typo:.1f}s)")

    # Targeted attack success rate (TASR): model predicted the overlay's class.
    tasr_correct = int((typo_preds == target_labels).sum().item())
    tasr = tasr_correct / len(labels)
    print(f"  TASR (target hit): {tasr_correct}/{len(labels)} = {tasr:.3f}")

    # Fooling rate on originally-correct images: among samples where the clean
    # prediction was right, what fraction flipped under the attack?
    orig_correct_mask = clean_preds == labels
    n_orig_correct = int(orig_correct_mask.sum().item())
    flipped = int(((clean_preds != typo_preds) & orig_correct_mask).sum().item())
    fooling_rate = flipped / n_orig_correct if n_orig_correct else 0.0
    print(f"  fooling rate   : {flipped}/{n_orig_correct} = {fooling_rate:.3f}  "
          f"(of originally-correct images that flipped)")

    record = {
        "model": model_key,
        "attack": "typographic",
        "seed": int(cfg["seed"]),
        "dataset_size": len(labels),
        "clean_correct": clean_correct,
        "robust_correct": robust_correct,
        "tasr_correct": tasr_correct,
        "clean_accuracy": clean_acc,
        "robust_accuracy": robust_acc,
        "targeted_attack_success_rate": tasr,
        "fooling_rate": fooling_rate,
        "n_originally_correct": n_orig_correct,
        "n_flipped_from_correct": flipped,
        "typographic_config": typo_ds.config,
        "wall_clock_s": {
            "preprocess": t_prep,
            "clean_eval": t_clean,
            "typographic_eval": t_typo,
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"  wrote {out_path}")

    del clf, clean_batch, typo_batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Re-run models whose JSON already exists.")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    models = [m.strip() for m in args.models.split(",")] if args.models else list(cfg["models"])
    out_dir = args.output_dir or os.path.join(_REPO_ROOT, "results", "typographic")
    os.makedirs(out_dir, exist_ok=True)

    print(f"models : {models}")
    print(f"output : {out_dir}")

    failures = []
    t0 = time.perf_counter()
    for model_key in models:
        try:
            run_one(model_key, cfg, out_dir, args.force)
        except Exception as exc:
            print(f"  FAIL  {model_key}: {exc}")
            failures.append((model_key, repr(exc)))

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed/60:.1f} min  failures: {len(failures)}")
    for m, e in failures:
        print(f"  - {m}: {e}")


if __name__ == "__main__":
    main()
