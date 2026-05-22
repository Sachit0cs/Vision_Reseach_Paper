"""CLI: export adversarial ("poisoned") images to disk for inspection.

WHAT THIS FILE IS (read first):
  * It is the GENERAL attack-export tool. It imports the `attacks/` module and
    writes the poisoned image dataset for the attacks it covers. Today it
    covers the gradient attacks (FGSM, PGD, AutoAttack, Square); patch,
    typographic, corruptions and transfer extend the SAME folder structure
    later without changing the design.
  * It stores the poisoned images in a gitignored folder (`attack_previews/`)
    with a fixed, per-attack directory structure (see the layout below).
  * It is NOT part of the benchmark. `scripts/run_benchmark.py` /
    `evaluation/pipeline.py` regenerate adversarial images in memory at run
    time and never read this folder. This exporter exists ONLY so the exact
    images the benchmark uses can be browsed or shown to others — the folder
    is fully regenerable and safe to delete.

Generates the per-model gradient-attack images (FGSM, PGD, AutoAttack, Square)
and saves them as image files.

Output layout (gitignored ``attack_previews/``):

    attack_previews/
      README.txt
      gradient/<model>/<attack>/<label>_<id>.png   (+ manifest.json per folder)
      # shared/ (patch, typographic, corruptions) and transfer/ are added later.

Per-model gradient attacks: FGSM/PGD on all 1000 images; AutoAttack and Square
on the 200-image class-balanced subset specified in config.yaml (they are far
too slow for the full set). Default total: 7 models x (1000 + 1000 + 200 + 200)
= 16,800 files.

Usage:
    python scripts/generate_attack_previews.py                       # everything
    python scripts/generate_attack_previews.py --attacks fgsm,pgd     # subset
    python scripts/generate_attack_previews.py --models resnet50 --limit 20
    python scripts/generate_attack_previews.py --zip                  # + attack_previews.zip

Run on a GPU (Kaggle). AutoAttack across all 7 models is the expensive part —
budget for it. A completed (model, attack) folder is skipped on re-run, so a
crashed/interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch
import yaml
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets.loader import load_clean_dataset, load_eval_subset  # noqa: E402
from models.classifiers import build_classifier  # noqa: E402
from attacks.gradient import build_attack  # noqa: E402

_GRADIENT_ATTACKS = ["fgsm", "pgd", "autoattack", "square"]
# Attacks whose default eval set is a class-balanced subset (config key suffix).
_SUBSET_ATTACKS = {"autoattack", "square"}

_README = """\
attack_previews/
====================================================================
Adversarial images exported for inspection.

These are the exact images fed to the models during the benchmark.
The benchmark regenerates them in memory; this folder is an on-disk
export for visual inspection only and is NOT tracked by git.

Layout
------
gradient/<model>/<attack>/<label>_<id>.png
    Per-model white-box / black-box attacks. An adversarial image
    depends on (clean image, model, attack), so every model has its
    own copy. <attack> is fgsm | pgd | autoattack | square.
    Each folder also holds manifest.json (labels, epsilon, seed).

    Image counts per (model, attack):
      fgsm, pgd          : all 1000 clean images (cheap to attack).
      autoattack, square : 200-image class-balanced subset (config:
                           autoattack_eval_subset / square_eval_subset).
                           Same 200 indices on every run — seeded by
                           datasets.loader.load_eval_subset().

(added later)
shared/<attack>/...   model-agnostic attacks (patch, typographic,
                      corruptions) — one set reused by all 7 models.
transfer/...          surrogate-based transfer-attack images.

Originals
---------
The unmodified clean images are in  data/clean/images/  with matching
filenames, so clean vs adversarial can be compared side by side.

Note: fgsm / pgd / autoattack / square perturbations are bounded by a
small epsilon (see each manifest.json) and are usually imperceptible —
the images look identical to the originals to the human eye. That
invisibility is precisely the point of an adversarial example.
"""


def load_config() -> dict:
    with open(os.path.join(_REPO_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_attack(name: str, cfg: dict, epsilon: float):
    """Build a gradient attack with hyperparameters from config.yaml."""
    a = cfg["attack"]
    seed = cfg["seed"]
    if name == "fgsm":
        return build_attack("fgsm", epsilon=epsilon)
    if name == "pgd":
        return build_attack(
            "pgd", epsilon=epsilon, step_size=a["pgd_step"],
            num_steps=a["pgd_iters"], seed=seed,
        )
    if name == "autoattack":
        return build_attack("autoattack", epsilon=epsilon, seed=seed)
    if name == "square":
        return build_attack("square", epsilon=epsilon, seed=seed)
    raise ValueError(f"Unknown gradient attack: {name}")


def export_attack(
    clf, attack, model_key, attack_name, dataset, indices, out_root,
    *, batch_size, epsilon, seed, img_format,
):
    """Generate and save every adversarial image for one (model, attack).

    ``indices`` is the list of dataset indices to evaluate — it carries the
    per-attack sampling decision (full 1000 for FGSM/PGD, class-balanced
    subset for AutoAttack/Square), so this function is sampling-agnostic.
    """
    n = len(indices)
    out_dir = os.path.join(out_root, "gradient", model_key, attack_name)
    manifest_path = os.path.join(out_dir, "manifest.json")

    # Resume: skip a (model, attack) that already finished.
    if os.path.exists(manifest_path):
        done = len([f for f in os.listdir(out_dir) if f.endswith("." + img_format)])
        if done >= n:
            print(f"  skip  {model_key}/{attack_name}  (already {done} images)")
            return
    os.makedirs(out_dir, exist_ok=True)

    records = []
    for start in tqdm(range(0, n, batch_size), desc=f"  {model_key}/{attack_name}", leave=False):
        batch_indices = indices[start:start + batch_size]
        tensors, labels, recs = [], [], []
        for i in batch_indices:
            image, label = dataset[i]
            tensors.append(clf.preprocess(image))
            labels.append(label)
            recs.append(dataset.records[i])

        batch = torch.stack(tensors)
        label_tensor = torch.tensor(labels)
        adv = attack.apply(clf, batch, label_tensor).detach().cpu().clamp(0.0, 1.0)

        for j, rec in enumerate(recs):
            stem = os.path.splitext(os.path.basename(rec["filename"]))[0]
            fname = f"{stem}.{img_format}"
            to_pil_image(adv[j]).save(os.path.join(out_dir, fname))
            records.append(
                {"filename": fname, "label": rec["label"], "class_name": rec["class_name"]}
            )

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model_key,
                "attack": attack_name,
                "epsilon": epsilon,
                "seed": seed,
                "num_images": len(records),
                "images": records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"  done  {model_key}/{attack_name}  ({len(records)} images)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export gradient-attack images for inspection.")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model keys (default: all in config.yaml).")
    parser.add_argument("--attacks", type=str, default=None,
                        help=f"Comma-separated attacks (default: {','.join(_GRADIENT_ATTACKS)}).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap images per (model, attack). Default: 1000 for "
                             "FGSM/PGD, autoattack_eval_subset / square_eval_subset "
                             "(200) for the slow attacks.")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="L-infinity epsilon in [0,1] (default: config.yaml).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--format", choices=["png", "jpg"], default="png",
                        help="png = lossless/exact (default); jpg = ~5x smaller, viewing only.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output root (default: <repo>/attack_previews).")
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu (default: auto).")
    parser.add_argument("--zip", action="store_true", help="Also write attack_previews.zip.")
    args = parser.parse_args()

    cfg = load_config()
    models = [m.strip() for m in args.models.split(",")] if args.models else list(cfg["models"])
    attacks = [a.strip() for a in args.attacks.split(",")] if args.attacks else list(_GRADIENT_ATTACKS)
    epsilon = args.epsilon if args.epsilon is not None else cfg["attack"]["epsilon"]
    seed = cfg["seed"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_root = args.output_dir or os.path.join(_REPO_ROOT, "attack_previews")

    dataset = load_clean_dataset()

    # Per-attack indices. FGSM/PGD: all 1000 (cheap). AutoAttack/Square: the
    # class-balanced subset from config (expensive). --limit caps either case.
    def indices_for(attack_name: str) -> list[int]:
        if attack_name in _SUBSET_ATTACKS:
            subset_n = cfg["attack"].get(f"{attack_name}_eval_subset")
            if subset_n is None:
                subset_n = len(dataset)
            if args.limit is not None:
                subset_n = min(args.limit, subset_n)
            _, idx = load_eval_subset(subset_n, seed=seed)
            return idx
        n = min(args.limit, len(dataset)) if args.limit else len(dataset)
        return list(range(n))

    attack_indices = {a: indices_for(a) for a in attacks}

    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "README.txt"), "w", encoding="utf-8") as f:
        f.write(_README)

    print(f"device : {device}")
    print(f"models : {models}")
    print(f"attacks: {attacks}")
    per_attack_counts = ", ".join(f"{a}={len(attack_indices[a])}" for a in attacks)
    total = len(models) * sum(len(idx) for idx in attack_indices.values())
    print(f"images : per (model, attack): {per_attack_counts}  ->  {total} total")
    print(f"epsilon: {epsilon:.6f}   output: {out_root}")
    if device == "cpu":
        print("WARNING: running on CPU - this will be very slow. A GPU is strongly recommended.")
    if "autoattack" in attacks:
        print("NOTE: AutoAttack is the 4-attack ensemble - expect it to dominate the runtime.\n")

    for model_key in models:
        print(f"[{model_key}] loading...")
        clf = build_classifier(model_key, device)
        for attack_name in attacks:
            attack = make_attack(attack_name, cfg, epsilon)
            export_attack(
                clf, attack, model_key, attack_name, dataset,
                attack_indices[attack_name], out_root,
                batch_size=args.batch_size,
                epsilon=epsilon, seed=seed, img_format=args.format,
            )
        del clf
        if device == "cuda":
            torch.cuda.empty_cache()

    if args.zip:
        print("\nzipping...")
        archive = shutil.make_archive(out_root, "zip", out_root)
        print(f"wrote {archive}")

    print(f"\nDone. Inspect the images at: {out_root}")


if __name__ == "__main__":
    main()
