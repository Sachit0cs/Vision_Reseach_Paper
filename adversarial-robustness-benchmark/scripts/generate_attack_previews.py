"""CLI: export adversarial ("poisoned") images to disk for inspection.

WHAT THIS FILE IS (read first):
  * It is a PROOF-OF-CONCEPT visualizer. The real benchmark regenerates
    adversarial images in memory and never writes them to disk; this script's
    sole purpose is to dump a SMALL, representative sample of what the poisoned
    images look like for each (model, attack) pair, so they can be browsed or
    shown to others.
  * Default sample size: the first 10 clean images per (model, attack). That's
    7 models x 4 attacks x 10 = 280 files total — tiny, fast to regenerate.
    Override with --limit N if you need more (e.g. --limit 1000 to dump every
    image FGSM/PGD attack).
  * It stores the poisoned images in a gitignored folder (`attack_previews/`)
    with a fixed, per-attack directory structure (see the layout below).
  * It is NOT part of the benchmark. `scripts/run_benchmark.py` /
    `evaluation/pipeline.py` regenerate adversarial images in memory at run
    time and never read this folder — it is fully regenerable and safe to
    delete.

Output layout (gitignored ``attack_previews/``):

    attack_previews/
      README.txt
      gradient/<model>/<attack>/<label>_<id>.png   (+ manifest.json per folder)
      # shared/ (patch, typographic, corruptions) and transfer/ are added later.

Usage:
    python scripts/generate_attack_previews.py                       # 10 imgs / (model, attack)
    python scripts/generate_attack_previews.py --attacks fgsm,pgd     # subset of attacks
    python scripts/generate_attack_previews.py --models resnet50 --limit 50
    python scripts/generate_attack_previews.py --limit 1000           # FGSM/PGD full set
    python scripts/generate_attack_previews.py --zip                  # + attack_previews.zip

A completed (model, attack) folder is skipped on re-run, so a crashed or
interrupted run resumes where it stopped. If you raise --limit after a previous
run, delete the relevant folder(s) under attack_previews/ to force regeneration.
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
# Only relevant when --limit is unset AND the user opts into the full subset
# via --limit >= autoattack_eval_subset / square_eval_subset.
_SUBSET_ATTACKS = {"autoattack", "square"}
# Default sample size when --limit is not given. This is a preview/PoC tool,
# not the benchmark — 10 images per (model, attack) is enough to eyeball the
# perturbations. Override with --limit N to dump more.
_DEFAULT_LIMIT = 10

_README = """\
attack_previews/
====================================================================
Adversarial images exported for inspection (proof of concept).

This folder is a small visual sample of what poisoned images look
like — it is NOT the dataset used by the benchmark. The benchmark
regenerates adversarial images in memory and never reads this folder.
Safe to delete; regenerable from scratch.

Default: the first 10 images per (model, attack). Pass --limit N to
the generator to dump more. Folder is gitignored.

Layout
------
gradient/<model>/<attack>/<label>_<id>.png
    Per-model white-box / black-box attacks. An adversarial image
    depends on (clean image, model, attack), so every model has its
    own copy. <attack> is fgsm | pgd | autoattack | square.
    Each folder also holds manifest.json (labels, epsilon, seed).

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

    ``indices`` is the list of dataset indices to evaluate. The caller decides
    the per-attack sampling (PoC default: first 10; full set: 1000 for FGSM/PGD
    and the seeded class-balanced subset for AutoAttack/Square), so this
    function is sampling-agnostic.
    """
    n = len(indices)
    out_dir = os.path.join(out_root, "gradient", model_key, attack_name)
    manifest_path = os.path.join(out_dir, "manifest.json")

    # Resume: skip a (model, attack) that already finished, but only if the
    # PREVIOUS run used the same epsilon and seed. Otherwise the on-disk images
    # were generated under a different config and silently reusing them would
    # mislead the user. In that case we error out and ask them to delete the
    # folder explicitly — overwriting risks orphan stale PNGs from the old run.
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
        prev_eps = prev.get("epsilon")
        prev_seed = prev.get("seed")
        eps_match = prev_eps is not None and abs(float(prev_eps) - float(epsilon)) < 1e-9
        seed_match = prev_seed == seed
        if not (eps_match and seed_match):
            raise RuntimeError(
                f"{out_dir}: existing manifest was generated with "
                f"epsilon={prev_eps}, seed={prev_seed}; this run requests "
                f"epsilon={epsilon}, seed={seed}. Delete the folder to regenerate."
            )
        done = len([f for f in os.listdir(out_dir) if f.endswith("." + img_format)])
        if done >= n:
            print(f"  skip  {model_key}/{attack_name}  (already {done} images)")
            return
        # Same config but fewer images than requested — wipe and regenerate so
        # the new manifest never lists fewer files than exist on disk.
        for f in os.listdir(out_dir):
            if f.endswith("." + img_format) or f == "manifest.json":
                os.remove(os.path.join(out_dir, f))
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
    parser.add_argument("--limit", type=int, default=_DEFAULT_LIMIT,
                        help=f"Cap images per (model, attack). Default: "
                             f"{_DEFAULT_LIMIT} (PoC sample). Set to 0 for the "
                             f"full set (1000 for FGSM/PGD, 200 for AA/Square).")
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

    # Per-attack indices. Default is _DEFAULT_LIMIT (10) for every attack —
    # this is a preview/PoC tool. --limit 0 unlocks the full set: 1000 for
    # FGSM/PGD, the seeded class-balanced subset for AutoAttack/Square.
    full_set = (args.limit is not None and args.limit <= 0)
    cap = None if full_set else args.limit

    def indices_for(attack_name: str) -> list[int]:
        if attack_name in _SUBSET_ATTACKS:
            subset_n = cfg["attack"].get(f"{attack_name}_eval_subset") or len(dataset)
            if cap is not None:
                subset_n = min(cap, subset_n)
            _, idx = load_eval_subset(subset_n, seed=seed)
            return idx
        n = min(cap, len(dataset)) if cap is not None else len(dataset)
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
