"""CLI: PGD-iteration convergence sweep for the report.

Runs PGD on ResNet-50 across iters ∈ {0, 1, 5, 10, 20, 40} on the 200-image
class-balanced eval subset (the AutoAttack subset — same indices, kept cheap).
Writes results/gradient/pgd_iter_sweep.json. Iter 0 means "no attack", which
yields the clean accuracy and anchors the curve.
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml  # noqa: E402

from attacks.gradient import PGD  # noqa: E402
from datasets.loader import load_eval_subset  # noqa: E402
from models.classifiers import build_classifier  # noqa: E402

ITERS = [0, 1, 5, 10, 20, 40]
MODEL = "resnet50"


def main() -> None:
    with open(os.path.join(_REPO_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["seed"])
    eps = float(cfg["attack"]["epsilon"])
    step = float(cfg["attack"]["pgd_step"])
    n = int(cfg["attack"]["autoattack_eval_subset"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = build_classifier(MODEL, device)
    dataset, indices = load_eval_subset(n, seed=seed)

    tensors, labels = [], []
    for i in indices:
        img, lab = dataset[i]
        tensors.append(clf.preprocess(img))
        labels.append(int(lab))
    clean_batch = torch.stack(tensors)
    label_t = torch.tensor(labels, dtype=torch.long)

    bs = 16
    accs, times = [], []
    for iters in ITERS:
        t0 = time.perf_counter()
        correct = 0
        for start_idx in range(0, clean_batch.shape[0], bs):
            x = clean_batch[start_idx:start_idx + bs]
            y = label_t[start_idx:start_idx + bs]
            if iters == 0:
                adv = x
            else:
                attack = PGD(epsilon=eps, step_size=step, num_steps=iters, seed=seed)
                adv = attack.apply(clf, x, y).cpu()
            with torch.no_grad():
                preds = clf.logits(adv.to(device)).argmax(dim=1).cpu()
            correct += int((preds == y).sum().item())
        acc = correct / clean_batch.shape[0]
        elapsed = time.perf_counter() - t0
        accs.append(acc)
        times.append(elapsed)
        print(f"  iters={iters:>3}  acc={acc:.3f}  ({elapsed:.1f}s)")

    out = {
        "model": MODEL,
        "subset_size": clean_batch.shape[0],
        "indices": indices,
        "epsilon": eps,
        "step_size": step,
        "seed": seed,
        "iters": ITERS,
        "robust_accuracy": accs,
        "wall_clock_s": times,
    }
    out_path = os.path.join(_REPO_ROOT, "results", "gradient", "pgd_iter_sweep.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
