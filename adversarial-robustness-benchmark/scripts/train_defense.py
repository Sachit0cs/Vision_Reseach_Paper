"""CLI: Madry-style PGD adversarial fine-tuning of ResNet-50 (Phase C).

Reuses the existing PGD attack from ``attacks/gradient.py`` as the inner
max-step of the min-max adversarial-training objective:

    min_theta  E_(x,y) [ max_{||delta||_inf <= eps} L(f_theta(x+delta), y) ]

Recipe (results-oriented — see config.yaml ``defense`` block):
  * FINE-TUNE the ImageNet-1k-pretrained ResNet-50 (loaded below), NOT from
    scratch. Fine-tuning keeps clean accuracy high and converges in a handful
    of epochs; from-scratch AT on ImageNet needs ~90+ epochs.
  * Train on ALL ``num_classes`` (default 1000) classes so the training label
    space matches the 1000-class evaluation benchmark. Training on only 100
    classes while scoring on 1000 was the bug that produced the 6% clean
    "collapse" of the first run.
  * Cosine LR decay over the run; per-epoch CLEAN accuracy is tracked alongside
    adversarial accuracy so a clean-accuracy collapse is caught immediately.

For each batch:
  1. INNER MAX — PGD-5 against the CURRENT model state with BN in eval mode
     (running stats frozen during the attack — deliberate, avoids polluting BN
     statistics with the unrolled adversarial forward passes).
  2. OUTER MIN — CLEAN+ADVERSARIAL mixed CE loss:
       loss = clean_weight * CE(clean) + (1 - clean_weight) * CE(adv)
     Pure-adversarial loss (clean_weight=0) collapses the pretrained net to a
     uniform predictor: its PGD adversarials start at loss ABOVE ln(num_classes),
     so the easiest way to cut loss is to output uniform (loss -> ln(num_classes))
     and the model craters on clean AND adv. The clean term is fittable (loss can
     fall well below that), which removes the attractor and anchors clean accuracy
     — the actual collapse fix. (Freezing BN was tried and made it WORSE; default
     ``freeze_bn`` is now False.) Optional ``grad_clip_norm`` caps gradient spikes.

The classifier wrapper from ``models/classifiers.py`` is reused as-is so the
training loop sees the same preprocessing + normalization the benchmark uses
at eval time. The model's parameters are unfrozen for training (the wrapper
freezes them by default for inference) and refrozen on exit so the saved
checkpoint loads cleanly back through ``DefenseModel``.

Outputs:
  * models/checkpoints/defense_epoch_{1..N}.pt
  * models/checkpoints/defense_final.pt   (== last epoch, easier to load)
  * results/defense/training_log.json     (per-epoch loss + clean/adv acc + timing)

Usage:
    python scripts/train_defense.py
    python scripts/train_defense.py --train-dir /path/to/imagenet/train   # 1000 WNID folders
    python scripts/train_defense.py --resume models/checkpoints/defense_epoch_1.pt
    python scripts/train_defense.py --epochs 5 --batch-size 32   # OOM fallback
    # cheap collapse-diagnostic: auto-abort if instantaneous clean acc craters
    python scripts/train_defense.py --images-per-class 50 --abort-clean-below 0.3
    # A/B the BN fix (reproduce the old collapse):
    python scripts/train_defense.py --no-freeze-bn --abort-clean-below 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import deque

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


def set_bn_eval(model) -> None:
    """Put every BatchNorm module in eval mode (freeze running_mean/var).

    The affine params (weight/bias) still receive gradients and train normally —
    ONLY the running statistics are frozen.

    History: this was hypothesised as the clean-collapse fix (the idea being that
    train-mode BN drags running_mean/var toward the adversarial distribution).
    An A/B test DISPROVED it — freezing BN made the collapse FASTER, because with
    BN pinned to clean stats the un-renormalised adversarial activations produce a
    higher starting loss (~9.85 vs ~7.19), which makes the uniform-predictor basin
    even more attractive. The real cause was the loss objective, not BN — see the
    clean+adv mix in ``train_one_epoch``. Kept as an opt-in toggle (default off).
    """
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def eps_for_epoch(epoch, eps_max, eps_start, warmup_epochs):
    """Linear eps warmup, 1-indexed epoch. Ramps eps_start -> eps_max over the
    first `warmup_epochs` epochs, then holds eps_max. 0/1 disables warmup."""
    if not warmup_epochs or warmup_epochs <= 1 or epoch >= warmup_epochs:
        return eps_max
    return eps_start + (eps_max - eps_start) * (epoch - 1) / (warmup_epochs - 1)


class CleanCollapse(RuntimeError):
    """Raised to abort early when instantaneous clean accuracy craters."""


def train_one_epoch(classifier, loader, pgd, optimizer, device, epoch_idx,
                    clean_weight=0.5, freeze_bn=False, grad_clip_norm=None,
                    abort_clean_below=None, abort_after_batch=200,
                    inst_window=20):
    """One full pass over the loader; returns per-epoch loss + clean/adv top-1.

    clean_weight       weight on the CLEAN cross-entropy term in the clean+adv
                       mixed loss (0 = pure-adversarial Madry). The clean term is
                       fittable, so it removes the uniform-predictor attractor that
                       pure-adversarial loss collapses into — the actual collapse fix.
    freeze_bn          pin BN running stats during the training forward. Tried as a
                       collapse fix; it made the collapse WORSE (default now False),
                       kept only as a toggle.
    grad_clip_norm     max global grad norm (None disables clipping).
    abort_clean_below  if set, raise ``CleanCollapse`` once the windowed
                       *instantaneous* clean acc falls below this AFTER
                       ``abort_after_batch`` batches (saves GPU on a collapse).
    inst_window        number of recent batches for the instantaneous metrics.
    """
    model = classifier.model
    params = [p for p in model.parameters() if p.requires_grad]
    total_loss = 0.0
    total_adv_correct = 0
    total_clean_correct = 0
    total_samples = 0
    n_batches = 0
    # Sliding windows of (correct, n) per batch -> true INSTANTANEOUS accuracy.
    # The cumulative epoch means lag badly and hide an in-progress collapse, so
    # we surface a windowed value that reacts within ~inst_window batches.
    win_clean = deque(maxlen=inst_window)
    win_adv = deque(maxlen=inst_window)
    t0 = time.perf_counter()

    pbar = tqdm(loader, desc=f"epoch {epoch_idx}", leave=False)
    for clean_images, labels in pbar:
        clean_images = clean_images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # --- INNER MAX: generate adversarial batch against current weights ---
        # BN in eval mode during the attack freezes the running statistics so the
        # unrolled adversarial forward passes do not pollute them.
        model.eval()
        adv_images = pgd.apply(classifier, clean_images, labels)
        # Cut the autograd graph from the attack — gradients flow ONLY through the
        # training-step forward pass below, not the PGD unroll.
        adv_images = adv_images.detach()

        # --- CLEAN-ACCURACY DIAGNOSTIC (eval mode, no grad) ---
        with torch.no_grad():
            clean_logits = classifier.logits(clean_images)
            batch_clean = int((clean_logits.argmax(dim=1) == labels).sum().item())
            total_clean_correct += batch_clean
            win_clean.append((batch_clean, labels.size(0)))

        # --- OUTER MIN: clean+adversarial mixed-loss SGD step ---
        model.train()
        if freeze_bn:
            set_bn_eval(model)
        optimizer.zero_grad()

        logits_adv = classifier.logits(adv_images)
        loss_adv = F.cross_entropy(logits_adv, labels)

        # CLEAN+ADV MIX (the collapse fix): a clean CE term is *fittable* — its loss
        # can fall well below ln(num_classes), whereas the pretrained model's PGD
        # adversarials start ABOVE it, so a pure-adversarial loss is minimized by the
        # trivial uniform predictor (loss -> ln(num_classes)) and the net collapses to
        # a constant. The clean term keeps a learnable signal every step and anchors
        # clean accuracy. clean_weight=0 recovers the old pure-Madry objective.
        if clean_weight > 0.0:
            logits_clean = classifier.logits(clean_images)
            loss = clean_weight * F.cross_entropy(logits_clean, labels) \
                + (1.0 - clean_weight) * loss_adv
        else:
            loss = loss_adv

        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)
        optimizer.step()

        with torch.no_grad():
            batch_adv = int((logits_adv.argmax(dim=1) == labels).sum().item())
            total_adv_correct += batch_adv
            win_adv.append((batch_adv, labels.size(0)))

        total_loss += float(loss.item()) * labels.size(0)
        total_samples += labels.size(0)
        n_batches += 1

        inst_clean = sum(c for c, _ in win_clean) / max(sum(n for _, n in win_clean), 1)
        inst_adv = sum(c for c, _ in win_adv) / max(sum(n for _, n in win_adv), 1)
        pbar.set_postfix(
            loss=f"{total_loss / total_samples:.3f}",
            clean=f"{total_clean_correct / total_samples:.3f}",
            adv=f"{total_adv_correct / total_samples:.3f}",
            i_clean=f"{inst_clean:.3f}",   # instantaneous (last inst_window batches)
            i_adv=f"{inst_adv:.3f}",
        )

        if (abort_clean_below is not None and n_batches >= abort_after_batch
                and inst_clean < abort_clean_below):
            raise CleanCollapse(
                f"instantaneous clean acc {inst_clean:.3f} < {abort_clean_below} "
                f"after {n_batches} batches — aborting to save GPU. Confirmed dead ends: "
                f"lowering LR, freezing BN. If this fires with clean_weight>0, raise it "
                f"(e.g. 0.7) or add an eps/PGD warmup."
            )

    elapsed = time.perf_counter() - t0
    return {
        "epoch": epoch_idx,
        "mean_adv_loss": total_loss / max(total_samples, 1),
        "mean_clean_acc": total_clean_correct / max(total_samples, 1),
        "mean_adv_acc": total_adv_correct / max(total_samples, 1),
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
    parser.add_argument("--clean-weight", type=float, default=None,
                        help="Weight on the clean CE term in the clean+adv mixed loss "
                             "(0 = pure-adversarial Madry; default: config, 0.5).")
    parser.add_argument("--freeze-bn", dest="freeze_bn", action="store_true",
                        help="Freeze BN running stats during the training fwd (tried; made collapse worse).")
    parser.add_argument("--no-freeze-bn", dest="freeze_bn", action="store_false",
                        help="Let BN running stats update during training (standard, default).")
    parser.set_defaults(freeze_bn=None)   # None -> fall back to config['defense']['freeze_bn']
    parser.add_argument("--grad-clip-norm", type=float, default=None,
                        help="Max global grad norm; 0 disables (default: config).")
    parser.add_argument("--abort-clean-below", type=float, default=None,
                        help="Abort if instantaneous clean acc drops below this after "
                             "--abort-after-batch batches (saves GPU on a collapse).")
    parser.add_argument("--abort-after-batch", type=int, default=200)
    parser.add_argument("--eps-warmup-epochs", type=int, default=None,
                        help="Ramp training eps from --eps-start to full over first N epochs "
                             "(0/1 = off; default: config).")
    parser.add_argument("--eps-start", type=float, default=None,
                        help="Starting (weak) eps for the warmup ramp (default: config, 2/255).")
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
    eps_warmup_epochs = (args.eps_warmup_epochs if args.eps_warmup_epochs is not None
                         else int(d.get("eps_warmup_epochs", 0)))
    pgd_eps_start = (args.eps_start if args.eps_start is not None
                     else float(d.get("pgd_eps_start", pgd_eps)))
    num_workers = args.num_workers if args.num_workers is not None else int(d.get("num_workers", 2))
    checkpoint_dir = _abs(args.checkpoint_dir or d.get("checkpoint_dir", "models/checkpoints"))
    log_path = _abs(args.log_path or "results/defense/training_log.json")

    # Recipe-stability knobs. clean_weight is the COLLAPSE FIX: the clean+adv mixed
    # loss removes the uniform-predictor attractor that pure-adversarial loss falls
    # into. grad_clip_norm is cheap insurance against gradient spikes. freeze_bn was
    # tried and made the collapse worse (default False now). CLI overrides config;
    # grad_clip_norm of 0/null disables clipping; clean_weight=0 = pure-Madry.
    clean_weight = args.clean_weight if args.clean_weight is not None else float(d.get("clean_weight", 0.5))
    freeze_bn = args.freeze_bn if args.freeze_bn is not None else bool(d.get("freeze_bn", False))
    _gc = args.grad_clip_norm if args.grad_clip_norm is not None else d.get("grad_clip_norm", 1.0)
    grad_clip_norm = None if _gc in (0, 0.0, None) else float(_gc)
    abort_clean_below = args.abort_clean_below   # None unless explicitly requested

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
    print(f"[defense] clean_weight={clean_weight}  freeze_bn={freeze_bn}  "
          f"grad_clip_norm={grad_clip_norm}  abort_clean_below={abort_clean_below}")
    print(f"[defense] eps_warmup_epochs={eps_warmup_epochs}  "
          f"eps_start={pgd_eps_start:.5f} ({pgd_eps_start*255:.2f}/255) -> "
          f"eps_max={pgd_eps:.5f} ({pgd_eps*255:.2f}/255)")

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

    # 4. Optimizer (SGD with momentum, standard Madry hyperparameters) +
    #    cosine LR decay over the planned epochs. Cosine decay is the standard
    #    schedule for adversarial fine-tuning and helps the model settle into a
    #    robust minimum without destroying the pretrained clean features early.
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=momentum, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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
        # Advance the cosine schedule to where the resumed run left off.
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"[defense] resumed from {args.resume} at epoch {start_epoch}")
        # Preserve prior log entries if they exist.
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                prior = json.load(f)
            log_entries = prior.get("epochs", [])

    # 6. Training loop.
    run_started_at = time.time()
    for epoch in range(start_epoch, epochs + 1):
        eps_e = eps_for_epoch(epoch, pgd_eps, pgd_eps_start, eps_warmup_epochs)
        pgd.epsilon = eps_e
        pgd.step_size = eps_e * (pgd_alpha / pgd_eps) if pgd_eps > 0 else pgd_alpha
        print(f"[defense] epoch {epoch}: train eps = {eps_e:.5f} ({eps_e*255:.2f}/255)")
        try:
            stats = train_one_epoch(
                classifier, loader, pgd, optimizer, device, epoch,
                clean_weight=clean_weight, freeze_bn=freeze_bn,
                grad_clip_norm=grad_clip_norm,
                abort_clean_below=abort_clean_below,
                abort_after_batch=args.abort_after_batch,
            )
        except CleanCollapse as exc:
            print(f"[defense] ABORTED epoch {epoch}: {exc}")
            log = {
                "config": d, "seed": seed, "train_dir": train_dir,
                "epochs_planned": epochs, "device": device,
                "epochs": log_entries, "run_started_at": run_started_at,
                "aborted": {"epoch": epoch, "reason": str(exc)},
            }
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log, f, indent=2)
            raise SystemExit(1)
        stats["lr"] = optimizer.param_groups[0]["lr"]
        stats["train_eps"] = eps_e
        scheduler.step()
        log_entries.append(stats)
        print(f"[defense] epoch {epoch}/{epochs}: "
              f"loss={stats['mean_adv_loss']:.3f}  "
              f"clean_acc={stats['mean_clean_acc']:.3f}  "
              f"adv_acc={stats['mean_adv_acc']:.3f}  "
              f"lr={stats['lr']:.4f}  "
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
