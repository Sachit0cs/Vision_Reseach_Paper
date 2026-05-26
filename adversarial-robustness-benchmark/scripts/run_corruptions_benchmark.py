"""CLI: evaluate the 7 models on the common-corruptions poisoned sets.

Phase 1, Axis A runner — non-adversarial, model-agnostic. The 15
ImageNet-C-style poisoned sets were already generated once by
``scripts/generate_datasets.py --corruptions`` and live under
``data/poisoned/corruptions/<corruption>/``. This script does inference
only: one clean-set eval + one eval per (model, corruption) pair. No
gradients, no adversarial generation — pure forward passes, T4-friendly.

Outputs one JSON per (model, corruption) at
``results/corruptions/<model>__<corruption>.json``. Per-(model, corruption)
JSONs are the resumption unit — re-running skips finished pairs unless
``--force`` is passed.

Usage:
    python scripts/run_corruptions_benchmark.py
    python scripts/run_corruptions_benchmark.py --models resnet50,clip_vit_b16
    python scripts/run_corruptions_benchmark.py --corruptions gaussian_noise,fog
    python scripts/run_corruptions_benchmark.py --force
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

from attacks.corruptions import CORRUPTION_TYPES  # noqa: E402
from datasets.loader import load_clean_dataset, load_corruptions_dataset  # noqa: E402
from models.classifiers import build_classifier  # noqa: E402

# Inference batch sizes — forward-only, same wider regime as typographic.
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


def _preprocess_dataset(clf, dataset) -> torch.Tensor:
    """Run the classifier's PIL preprocess over a (PIL, label) dataset."""
    tensors = []
    for i in range(len(dataset)):
        img = dataset[i][0]
        tensors.append(clf.preprocess(img))
    return torch.stack(tensors, dim=0)


@torch.no_grad()
def _predict(clf, batch_0_1: torch.Tensor, batch_size: int = 32) -> torch.Tensor:
    preds = []
    total = batch_0_1.shape[0]
    for start in range(0, total, batch_size):
        x = batch_0_1[start:start + batch_size].to(clf.device)
        preds.append(clf.logits(x).argmax(dim=1).cpu())
    return torch.cat(preds, dim=0).long()


def _host_env() -> dict:
    env = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda_available": bool(torch.cuda.is_available()),
        "torch": torch.__version__,
    }
    try:
        import torchvision
        env["torchvision"] = torchvision.__version__
    except ImportError:
        pass
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except ImportError:
        pass
    try:
        import PIL
        env["pillow"] = PIL.__version__
    except ImportError:
        pass
    return env


def run_one(
    model_key: str,
    corruption_type: str,
    cfg: dict,
    out_dir: str,
    force: bool,
    clf_cache: dict,
    clean_batch_cache: dict,
) -> str:
    """Run a single (model, corruption) pair end-to-end. Writes one JSON."""
    out_path = os.path.join(out_dir, f"{model_key}__{corruption_type}.json")
    if not force and os.path.exists(out_path):
        print(f"  skip  {model_key} / {corruption_type}  (cached)")
        return out_path

    set_seed(int(cfg["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Lazy model build, cached across corruptions so we pay it once per model.
    if model_key not in clf_cache:
        print(f"[{model_key}] loading on {device} ...")
        clf_cache[model_key] = build_classifier(model_key, device)
    clf = clf_cache[model_key]

    # Lazy clean-batch preprocess, cached across corruptions per model.
    if model_key not in clean_batch_cache:
        clean_ds = load_clean_dataset()
        t0 = time.perf_counter()
        clean_batch_cache[model_key] = (
            _preprocess_dataset(clf, clean_ds),
            torch.tensor(clean_ds.labels, dtype=torch.long),
            time.perf_counter() - t0,
        )
    clean_batch, labels, t_clean_prep = clean_batch_cache[model_key]

    corrupt_ds = load_corruptions_dataset(corruption_type)
    if len(clean_batch) != len(corrupt_ds):
        raise RuntimeError(
            f"Dataset size mismatch: clean has {len(clean_batch)} images, "
            f"corruption '{corruption_type}' has {len(corrupt_ds)}. "
            "Re-generate the corruption set."
        )
    if labels.tolist() != corrupt_ds.labels:
        raise RuntimeError(
            f"Per-image label mismatch between clean and corruption '{corruption_type}'. "
            "Re-generate the corruption set from the current clean set."
        )

    t0 = time.perf_counter()
    corrupt_batch = _preprocess_dataset(clf, corrupt_ds)
    t_corrupt_prep = time.perf_counter() - t0

    bs = _INFER_BATCH.get(model_key, 32)

    t0 = time.perf_counter()
    clean_preds = _predict(clf, clean_batch, batch_size=bs)
    t_clean_eval = time.perf_counter() - t0
    clean_correct = int((clean_preds == labels).sum().item())
    clean_acc = clean_correct / len(labels)

    t0 = time.perf_counter()
    corrupt_preds = _predict(clf, corrupt_batch, batch_size=bs)
    t_corrupt_eval = time.perf_counter() - t0
    robust_correct = int((corrupt_preds == labels).sum().item())
    robust_acc = robust_correct / len(labels)

    orig_correct_mask = clean_preds == labels
    n_orig_correct = int(orig_correct_mask.sum().item())
    flipped = int(((clean_preds != corrupt_preds) & orig_correct_mask).sum().item())
    fooling_rate = flipped / n_orig_correct if n_orig_correct else 0.0

    print(f"  {model_key:>16} / {corruption_type:>18}  "
          f"clean {clean_acc:.3f}  robust {robust_acc:.3f}  "
          f"drop {clean_acc - robust_acc:+.3f}  fool {fooling_rate:.3f}  "
          f"({t_corrupt_eval:.1f}s)")

    record = {
        "model": model_key,
        "attack": "corruptions",
        "corruption_type": corruption_type,
        "severity": int(corrupt_ds.severity),
        "seed": int(cfg["seed"]),
        "host_env": _host_env(),
        "dataset_size": len(labels),
        "clean_correct": clean_correct,
        "robust_correct": robust_correct,
        "clean_accuracy": clean_acc,
        "robust_accuracy": robust_acc,
        "accuracy_drop": clean_acc - robust_acc,
        "fooling_rate": fooling_rate,
        "n_originally_correct": n_orig_correct,
        "n_flipped_from_correct": flipped,
        "corruption_config": corrupt_ds.config,
        "wall_clock_s": {
            "preprocess_clean": t_clean_prep,
            "preprocess_corrupt": t_corrupt_prep,
            "clean_eval": t_clean_eval,
            "corrupt_eval": t_corrupt_eval,
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated list of model keys (default: all 7 from config).",
    )
    parser.add_argument(
        "--corruptions", type=str, default=None,
        help="Comma-separated subset of corruption types (default: all 15).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run (model, corruption) pairs whose JSON already exists.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    models = [m.strip() for m in args.models.split(",")] if args.models else list(cfg["models"])
    if args.corruptions:
        corruptions = [c.strip() for c in args.corruptions.split(",")]
        unknown = [c for c in corruptions if c not in CORRUPTION_TYPES]
        if unknown:
            parser.error(f"unknown corruption types: {unknown}")
    else:
        corruptions = list(CORRUPTION_TYPES)
    out_dir = args.output_dir or os.path.join(_REPO_ROOT, "results", "corruptions")
    os.makedirs(out_dir, exist_ok=True)

    print(f"models      : {models}")
    print(f"corruptions : {corruptions}  ({len(corruptions)} of {len(CORRUPTION_TYPES)})")
    print(f"pairs       : {len(models) * len(corruptions)}")
    print(f"output      : {out_dir}\n")

    failures: list[tuple[str, str, str]] = []
    t0 = time.perf_counter()
    # Outer loop = model so we pay the load + clean-preprocess cost once per model.
    for model_key in models:
        clf_cache: dict = {}
        clean_batch_cache: dict = {}
        for corruption_type in corruptions:
            try:
                run_one(model_key, corruption_type, cfg, out_dir, args.force,
                        clf_cache, clean_batch_cache)
            except Exception as exc:
                print(f"  FAIL  {model_key}/{corruption_type}: {exc}")
                failures.append((model_key, corruption_type, repr(exc)))
        # Release VRAM between models.
        for clf in clf_cache.values():
            del clf
        clf_cache.clear()
        clean_batch_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed / 60:.1f} min.  failures: {len(failures)}")
    for m, c, e in failures:
        print(f"  - {m}/{c}: {e}")


if __name__ == "__main__":
    main()
