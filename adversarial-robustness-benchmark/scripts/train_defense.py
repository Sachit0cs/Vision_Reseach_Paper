"""CLI: Madry-style PGD adversarial fine-tuning of ResNet-50 (Phase C).

Reuses the existing PGD attack from ``attacks/gradient.py`` as the inner
max-step of the min-max adversarial-training objective:

    min_theta  E_(x,y) [ max_{||delta||_inf <= eps} L(f_theta(x+delta), y) ]

For each batch:
  1. INNER MAX — eval-mode PGD-5 against the CURRENT model state, producing
     adversarial copies of every training image.
  2. OUTER MIN — train-mode CE loss on the adversarial images only (plain
     Madry recipe; no clean-mix, no TRADES KL-regularizer).

The classifier wrapper from ``models/classifiers.py`` is reused as-is so the
training loop sees the same preprocessing + normalization the benchmark uses
at eval time. The model's parameters are unfrozen for training (the wrapper
freezes them by default for inference) and refrozen on exit so the saved
checkpoint loads cleanly back through ``DefenseModel``.

Outputs:
  * models/checkpoints/defense_epoch_{1,2,3}.pt
  * models/checkpoints/defense_final.pt   (== last epoch, easier to load)
  * results/defense/training_log.json     (per-epoch loss + adv-acc + timing)

Usage:
    python scripts/train_defense.py
    python scripts/train_defense.py --train-dir /kaggle/input/imagenet100/train
    python scripts/train_defense.py --resume models/checkpoints/defense_epoch_1.pt
    python scripts/train_defense.py --epochs 2 --batch-size 32   # OOM fallback
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
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from datasets.loader import load_imagenet_train_subset  # noqa: E402
from models.classifiers import build_classifier  # noqa: E402
from attacks.gradient import PGD  # noqa: E402


def load_config() -> dict:
    with open(os.path.join(_REPO_ROOT, "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def train_one_epoch(classifier, loader, pgd, optimizer, device, epoch_idx):
    """One full pass over the loader; return (mean_loss, adv_top1, num_batches)."""
    model = classifier.model
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batches = 0
    t0 = time.perf_counter()

    pbar = tqdm(loader, desc=f"epoch {epoch_idx}", leave=False)
    for clean_images, labels in pbar:
        clean_images = clean_images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # --- INNER MAX: generate adversarial batch against current weights ---
        model.eval()
        adv_images = pgd.apply(classifier, clean_images, labels)
        # Cut the autograd graph from the attack — gradients should flow
        # ONLY through the training-step forward pass below, not back through
        # the PGD unroll (PGD unroll uses model params with frozen-effect
        # in eval mode, but the safe thing is an explicit detach).
        adv_images = adv_images.detach()

        # --- OUTER MIN: one SGD step on the adversarial images ---
        model.train()
        optimizer.zero_grad()
        logits = classifier.logits(adv_images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            batch_correct = int((preds == labels).sum().item())

        total_loss += float(loss.item()) * labels.size(0)
        total_correct += batch_correct
        total_samples += labels.size(0)
        n_batches += 1
        pbar.set_postfix(
            loss=f"{total_loss / total_samples:.3f}",
            adv_acc=f"{total_correct / total_samples:.3f}",
        )

    elapsed = time.perf_counter() - t0
    return {
        "epoch": epoch_idx,
        "mean_adv_loss": total_loss / max(total_samples, 1),
        "mean_adv_acc": total_correct / max(total_samples, 1),
        "num_batches": n_batches,
        "num_samples": total_samples,
        "wall_clock_s": elapsed,
    }


def save_checkpoint(path: str, model, epoch: int, optimizer=None, extra: dict | None = None):
    payload: dict = {
        "model": model.state_dict(),
        "epoch": int(epoch),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, default=None,
                        help="Path to WNID-folder ImageNet train tree "
                             "(default: config['defense']['train_dir']).")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--images-per-class", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--pgd-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint to resume from.")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-path", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    d = cfg.get("defense") or {}
    seed = int(cfg.get("seed", 42))

    train_dir = args.train_dir or d.get("train_dir")
    num_classes = args.num_classes or int(d.get("num_classes", 100))
    images_per_class = args.images_per_class or int(d.get("images_per_class", 200))
    epochs = args.epochs or int(d.get("epochs", 3))
    batch_size = args.batch_size or int(d.get("batch_size", 64))
    lr = args.lr if args.lr is not None else float(d.get("lr", 0.001))
    momentum = float(d.get("momentum", 0.9))
    weight_decay = float(d.get("weight_decay", 5e-4))
    pgd_eps = float(d.get("pgd_epsilon", 8 / 255))
    pgd_alpha = float(d.get("pgd_alpha", 2 / 255))
    pgd_steps = args.pgd_steps or int(d.get("pgd_steps", 5))
    num_workers = args.num_workers if args.num_workers is not None else int(d.get("num_workers", 2))
    checkpoint_dir = _abs(args.checkpoint_dir or d.get("checkpoint_dir", "models/checkpoints"))
    log_path = _abs(args.log_path or "results/defense/training_log.json")

    if not train_dir:
        raise SystemExit(
            "No --train-dir given and config['defense']['train_dir'] is empty. "
            "Pass --train-dir /path/to/imagenet/train (WNID folders)."
        )

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[defense] device={device}  train_dir={train_dir}")
    print(f"[defense] {num_classes} classes x {images_per_class} imgs/class  "
          f"batch={batch_size}  epochs={epochs}  lr={lr}  pgd-{pgd_steps}@eps={pgd_eps:.4f}")

    # 1. Build classifier — TorchVisionClassifier loads pretrained ResNet-50.
    classifier = build_classifier("resnet50", device=device)
    model = classifier.model
    for p in model.parameters():  # wrapper freezes by default; unfreeze for training.
        p.requires_grad_(True)

    # 2. Dataset + loader.
    dataset = load_imagenet_train_subset(
        train_dir=train_dir,
        num_classes=num_classes,
        images_per_class=images_per_class,
        seed=seed,
        augment=True,
    )
    print(f"[defense] dataset size: {len(dataset)} images "
          f"({len(dataset.selected_wnids)} WNIDs)")

    pin_memory = device == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    # 3. Attack (inner max). Use a seedless PGD so each batch sees a different
    # random start — matches Madry's recipe; deterministic-RNG seeding is
    # reserved for evaluation.
    pgd = PGD(epsilon=pgd_eps, step_size=pgd_alpha, num_steps=pgd_steps, random_start=True)

    # 4. Optimizer (SGD with momentum, standard Madry hyperparameters).
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=momentum, weight_decay=weight_decay,
    )

    # 5. Optional resume.
    start_epoch = 1
    log_entries: list[dict] = []
    if args.resume:
        if not os.path.exists(args.resume):
            raise SystemExit(f"--resume path does not exist: {args.resume}")
        try:
            state = torch.load(args.resume, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state.get("epoch", 0)) + 1
        print(f"[defense] resumed from {args.resume} at epoch {start_epoch}")
        # Preserve prior log entries if they exist.
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                prior = json.load(f)
            log_entries = prior.get("epochs", [])

    # 6. Training loop.
    run_started_at = time.time()
    for epoch in range(start_epoch, epochs + 1):
        stats = train_one_epoch(classifier, loader, pgd, optimizer, device, epoch)
        log_entries.append(stats)
        print(f"[defense] epoch {epoch}/{epochs}: "
              f"loss={stats['mean_adv_loss']:.3f}  "
              f"adv_acc={stats['mean_adv_acc']:.3f}  "
              f"time={stats['wall_clock_s']/60:.1f} min")

        ckpt_path = os.path.join(checkpoint_dir, f"defense_epoch_{epoch}.pt")
        save_checkpoint(ckpt_path, model, epoch, optimizer,
                        extra={"config": d, "seed": seed})
        print(f"[defense] wrote {ckpt_path}")

        log = {
            "config": d,
            "seed": seed,
            "train_dir": train_dir,
            "epochs_planned": epochs,
            "device": device,
            "epochs": log_entries,
            "run_started_at": run_started_at,
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)

    # 7. Final checkpoint — model-only, smaller file for distribution.
    final_path = os.path.join(checkpoint_dir, "defense_final.pt")
    # Re-freeze and switch to eval before saving so the checkpoint loads
    # back through DefenseModel with the same invariants the wrapper expects.
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    save_checkpoint(final_path, model, epochs, extra={"config": d, "seed": seed})
    print(f"[defense] wrote {final_path}")
    print(f"[defense] done in {(time.time() - run_started_at)/60:.1f} min")


if __name__ == "__main__":
    main()
