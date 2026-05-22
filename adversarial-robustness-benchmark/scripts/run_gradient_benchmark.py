"""CLI: run the gradient-attack benchmark and write per-(model, attack) JSONs.

Phase 2 minimal runner. Iterates over every (model, attack) in config.yaml,
generates adversarial images in memory, and writes one result file per pair to
results/gradient/<model>__<attack>.json containing clean and robust accuracy.

Hyperparameters are read from config.yaml — nothing is invented here:
  * epsilon, pgd_step, pgd_iters live under config['attack'].
  * AutoAttack/Square run on the seeded 200-image class-balanced subset
    (autoattack_eval_subset / square_eval_subset). FGSM/PGD run on the full
    1000-image clean set.

Per-(model, attack) JSONs are the unit of resumption: existing files are
skipped on re-run unless --force is passed, so a crashed run picks up where it
left off without recomputing anything expensive.

Usage:
    python scripts/run_gradient_benchmark.py
    python scripts/run_gradient_benchmark.py --attacks fgsm,pgd
    python scripts/run_gradient_benchmark.py --models resnet50 --force
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

from datasets.loader import load_clean_dataset, load_eval_subset  # noqa: E402
from models.classifiers import build_classifier  # noqa: E402
from attacks.gradient import build_attack  # noqa: E402

_GRADIENT_ATTACKS = ["fgsm", "pgd", "autoattack", "square"]
_SUBSET_ATTACKS = {"autoattack", "square"}

# Per-model AutoAttack batch sizes. Larger transformers OOM on a T4 at bs=32.
_AA_BATCH = {
    "resnet50": 32,
    "vgg16": 16,
    "convnext_tiny": 16,
    "vit_b_16": 16,
    "swin_t": 16,
    "efficientnet_b0": 32,
    "clip_vit_b16": 16,
}

# Per-model FGSM/PGD batch sizes for adversarial generation. PGD does 20
# backward passes per batch; we cap a bit below inference-only batch sizes.
_PGD_BATCH = {
    "resnet50": 32,
    "vgg16": 16,
    "convnext_tiny": 16,
    "vit_b_16": 16,
    "swin_t": 16,
    "efficientnet_b0": 32,
    "clip_vit_b16": 16,
}


def load_config() -> dict:
    with open(os.path.join(_REPO_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_batched_inputs(clf, dataset, indices):
    """Preprocess dataset[indices] into a single [0,1] tensor + label tensor."""
    tensors, labels = [], []
    for i in indices:
        image, label = dataset[i]
        tensors.append(clf.preprocess(image))
        labels.append(int(label))
    return torch.stack(tensors), torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def evaluate_accuracy(clf, batch_0_1, labels, batch_size=64) -> tuple[int, int]:
    """Return (correct, total) of argmax(clf.logits(batch)) == labels."""
    correct = 0
    total = batch_0_1.shape[0]
    for start in range(0, total, batch_size):
        x = batch_0_1[start:start + batch_size].to(clf.device)
        y = labels[start:start + batch_size].to(clf.device)
        preds = clf.logits(x).argmax(dim=1)
        correct += int((preds == y).sum().item())
    return correct, total


def make_attack(name: str, cfg: dict, model_key: str):
    a = cfg["attack"]
    seed = int(cfg["seed"])
    eps = float(a["epsilon"])
    if name == "fgsm":
        return build_attack("fgsm", epsilon=eps)
    if name == "pgd":
        return build_attack(
            "pgd",
            epsilon=eps,
            step_size=float(a["pgd_step"]),
            num_steps=int(a["pgd_iters"]),
            seed=seed,
        )
    if name == "autoattack":
        return build_attack(
            "autoattack",
            epsilon=eps,
            seed=seed,
            batch_size=_AA_BATCH.get(model_key, 16),
        )
    if name == "square":
        return build_attack("square", epsilon=eps, seed=seed)
    raise ValueError(f"Unknown attack: {name}")


def generate_adversarial(clf, attack, model_key, attack_name, clean_batch, labels):
    """Run the attack — batched for FGSM/PGD, single-shot for AA/Square."""
    if attack_name in {"fgsm", "pgd"}:
        bs = _PGD_BATCH.get(model_key, 16)
        out_chunks = []
        for start in tqdm(
            range(0, clean_batch.shape[0], bs),
            desc=f"  {model_key}/{attack_name}",
            leave=False,
        ):
            x = clean_batch[start:start + bs]
            y = labels[start:start + bs]
            adv = attack.apply(clf, x, y)
            out_chunks.append(adv.detach().cpu())
        return torch.cat(out_chunks, dim=0)
    # AutoAttack/Square handle their own batching internally.
    return attack.apply(clf, clean_batch, labels).detach().cpu()


def indices_for_attack(attack_name: str, cfg: dict, dataset_len: int) -> list[int]:
    if attack_name in _SUBSET_ATTACKS:
        n = cfg["attack"].get(f"{attack_name}_eval_subset") or dataset_len
        _, idx = load_eval_subset(int(n), seed=int(cfg["seed"]))
        return idx
    return list(range(dataset_len))


def nan_diagnostic(tensor_adv) -> dict:
    """Spot-check adversarial outputs for NaN — the PGD attack no-ops on NaN
    gradients, so a model that emits NaN logits at a perturbed point will look
    falsely robust."""
    n = int(tensor_adv.shape[0])
    nan_count = int(torch.isnan(tensor_adv).any(dim=tuple(range(1, tensor_adv.dim()))).sum().item())
    return {"num_samples": n, "num_with_nan": nan_count}


def run_one(model_key: str, attack_name: str, cfg: dict, out_dir: str, force: bool):
    """Run a single (model, attack) cell and write its JSON to out_dir."""
    out_path = os.path.join(out_dir, f"{model_key}__{attack_name}.json")
    if not force and os.path.exists(out_path):
        print(f"  skip  {model_key}/{attack_name}  (cached at {out_path})")
        return out_path

    set_seed(int(cfg["seed"]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{model_key} | {attack_name}] loading model on {device} ...")
    clf = build_classifier(model_key, device)

    dataset = load_clean_dataset()
    indices = indices_for_attack(attack_name, cfg, len(dataset))
    print(f"  dataset: {len(indices)} images")

    clean_batch, labels = build_batched_inputs(clf, dataset, indices)

    t0 = time.perf_counter()
    clean_correct, clean_total = evaluate_accuracy(clf, clean_batch, labels)
    t_clean = time.perf_counter() - t0
    print(f"  clean: {clean_correct}/{clean_total} = {clean_correct/clean_total:.3f}  ({t_clean:.1f}s)")

    attack = make_attack(attack_name, cfg, model_key)

    t0 = time.perf_counter()
    adv_batch = generate_adversarial(clf, attack, model_key, attack_name, clean_batch, labels)
    t_attack = time.perf_counter() - t0

    diag = nan_diagnostic(adv_batch)
    if diag["num_with_nan"]:
        print(f"  WARNING: {diag['num_with_nan']}/{diag['num_samples']} adversarial samples contain NaN")

    t0 = time.perf_counter()
    robust_correct, robust_total = evaluate_accuracy(clf, adv_batch, labels)
    t_eval = time.perf_counter() - t0
    print(f"  robust ({attack_name}): {robust_correct}/{robust_total} = "
          f"{robust_correct/robust_total:.3f}  ({t_attack:.1f}s attack, {t_eval:.1f}s eval)")

    # L-infinity perturbation magnitude — a quick sanity check that the attack
    # respected its epsilon budget.
    with torch.no_grad():
        delta = (adv_batch - clean_batch).abs()
        linf_max = float(delta.amax().item())
        linf_mean = float(delta.amax(dim=tuple(range(1, delta.dim()))).mean().item())

    record = {
        "model": model_key,
        "attack": attack_name,
        "epsilon": float(cfg["attack"]["epsilon"]),
        "seed": int(cfg["seed"]),
        "dataset_size": clean_total,
        "indices": indices,
        "clean_correct": clean_correct,
        "robust_correct": robust_correct,
        "clean_accuracy": clean_correct / clean_total,
        "robust_accuracy": robust_correct / robust_total,
        "linf_max": linf_max,
        "linf_mean_per_sample": linf_mean,
        "nan_diagnostic": diag,
        "wall_clock_s": {
            "clean_eval": t_clean,
            "attack": t_attack,
            "adv_eval": t_eval,
        },
    }
    if attack_name == "pgd":
        record["pgd_iters"] = int(cfg["attack"]["pgd_iters"])
        record["pgd_step"] = float(cfg["attack"]["pgd_step"])
    if attack_name == "square":
        record["n_queries"] = 5000
    if attack_name == "autoattack":
        record["autoattack_version"] = "standard"
        record["autoattack_batch_size"] = _AA_BATCH.get(model_key, 16)

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"  wrote {out_path}")

    # Free GPU memory before the next attack on this model.
    del clf, clean_batch, adv_batch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--attacks", type=str, default=None)
    parser.add_argument("--force", action="store_true",
                        help="Re-run pairs whose JSON already exists.")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    models = [m.strip() for m in args.models.split(",")] if args.models else list(cfg["models"])
    attacks = [a.strip() for a in args.attacks.split(",")] if args.attacks else list(_GRADIENT_ATTACKS)
    out_dir = args.output_dir or os.path.join(_REPO_ROOT, "results", "gradient")
    os.makedirs(out_dir, exist_ok=True)

    print(f"models : {models}")
    print(f"attacks: {attacks}")
    print(f"output : {out_dir}")
    print(f"epsilon: {cfg['attack']['epsilon']:.6f}  pgd_iters: {cfg['attack']['pgd_iters']}")

    failures = []
    t0 = time.perf_counter()
    for model_key in models:
        for attack_name in attacks:
            try:
                run_one(model_key, attack_name, cfg, out_dir, args.force)
            except Exception as exc:
                print(f"  FAIL  {model_key}/{attack_name}: {exc}")
                failures.append((model_key, attack_name, repr(exc)))

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed/60:.1f} min  failures: {len(failures)}")
    for m, a, e in failures:
        print(f"  - {m}/{a}: {e}")


if __name__ == "__main__":
    main()
