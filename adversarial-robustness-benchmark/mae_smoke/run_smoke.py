"""Smoke test: does pretrained-MAE purification preserve signal and blunt PGD?

Standalone and DELETABLE — everything lives under ``mae_smoke/``. This script
imports the existing repo packages READ-ONLY (datasets/models/attacks) and edits
nothing. NO training.

What it measures
----------------
1. Clean-accuracy COST of purification:  bare  vs  purify(K=1)  vs  purify(K).
2. A NON-ADAPTIVE robustness INDICATOR:  PGD-20 on the BARE classifier, then
   purify, then classify. This is only an indicator — a real robustness verdict
   needs an adaptive (BPDA + EOT) attack on the full purify->classify pipeline,
   OR (in-scope here) running the FULL gradient benchmark (which includes the
   gradient-free Square + AutoAttack) on a purifier-wrapped classifier. A
   purifier looking good against this PGD probe is necessary, NOT sufficient.
3. Saves an image grid (original | masked | reconstruction | purified) so you can
   *see* whether the MAE reconstruction is sane.

Run
---
Local (CPU is fine, a few minutes)::

    python mae_smoke/run_smoke.py --n 16 --k 2

Kaggle GPU (bigger, sharper)::

    python mae_smoke/run_smoke.py --n 200 --k 8 --tag mr075
    python mae_smoke/run_smoke.py --n 200 --k 8 --mask-ratio 0.5 --tag mr050

Outputs (all under ``mae_smoke/``): ``smoke_results[_tag].json`` and
``figures/reconstruction_grid[_tag].png``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

# Put the benchmark root (parent of mae_smoke/) on the path so the existing
# packages import, exactly like scripts/train_defense.py does.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_ROOT = os.path.dirname(_HERE)
for _p in (_BENCH_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datasets.loader import load_eval_subset      # noqa: E402
from models.classifiers import build_classifier   # noqa: E402
from attacks.gradient import PGD                   # noqa: E402

from mae_purifier import MAEPurifier               # noqa: E402  (sibling module)

_EPS = 8 / 255
_STEP = 2 / 255


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def _logits_in_chunks(classifier, x: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    """Batched, detached, on-CPU logits — keeps memory flat on a T4 or CPU."""
    outs = []
    for i in range(0, x.size(0), chunk):
        outs.append(classifier.logits(x[i : i + chunk]).detach().cpu())
    return torch.cat(outs, dim=0)


def _pgd_in_chunks(pgd_attack, classifier, imgs: torch.Tensor,
                   labels: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    """Run PGD sub-batch by sub-batch to avoid OOM on large datasets."""
    adv_chunks = []
    for i in range(0, imgs.size(0), chunk):
        adv_chunks.append(
            pgd_attack.apply(classifier, imgs[i : i + chunk], labels[i : i + chunk]).cpu()
        )
    return torch.cat(adv_chunks, dim=0)


def _save_grid(orig, masked, recon, purified, out_dir, num, tag):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    num = min(num, orig.size(0))
    titles = ["original", "masked", "MAE reconstruction", "purified"]
    cols = [orig, masked, recon, purified]

    fig, axes = plt.subplots(num, 4, figsize=(4 * 2.1, num * 2.1))
    if num == 1:
        axes = axes[None, :]
    for r in range(num):
        for c, batch in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(batch[r].permute(1, 2, 0).cpu().clamp(0, 1).numpy())
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, f"reconstruction_grid{tag}.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16, help="number of eval images")
    ap.add_argument("--k", type=int, default=2, help="MAE mask-ensemble size")
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--pgd-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--mae-id", type=str, default="facebook/vit-mae-base")
    ap.add_argument("--num-grids", type=int, default=8)
    ap.add_argument("--tag", type=str, default="",
                    help="suffix for output files so multiple runs do not overwrite")
    ap.add_argument("--chunk", type=int, default=32,
                    help="MAE sub-batch size (lower if you OOM on the GPU)")
    ap.add_argument("--pgd-chunk", type=int, default=16,
                    help="PGD sub-batch size (lower if you OOM during the attack)")
    args = ap.parse_args()

    tag = f"_{args.tag}" if args.tag else ""
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    print(f"[smoke] device={device}  n={args.n}  k={args.k}  mask_ratio={args.mask_ratio}  "
          f"pgd_steps={args.pgd_steps}  tag='{args.tag}'")

    # 1. Data — reuse the benchmark's class-balanced eval subset (+ labels).
    dataset, indices = load_eval_subset(args.n, seed=args.seed)
    clf = build_classifier("resnet50", device=device)
    imgs = torch.stack([clf.preprocess(dataset[i][0]) for i in indices]).to(device)
    labels = torch.tensor([dataset[i][1] for i in indices], device=device)
    labels_cpu = labels.cpu()
    print(f"[smoke] loaded {imgs.size(0)} images @ {tuple(imgs.shape[-2:])}")

    # 2. NON-ADAPTIVE robustness indicator — run BEFORE loading MAE so only
    #    ResNet-50 + gradients occupy GPU RAM (avoids OOM on 16 GB cards).
    #    Also chunk the PGD attack itself to keep per-batch memory bounded.
    print("[smoke] running PGD (bare classifier only, MAE not yet loaded) ...")
    clean_bare = _accuracy(_logits_in_chunks(clf, imgs), labels_cpu)
    pgd = PGD(epsilon=_EPS, step_size=_STEP, num_steps=args.pgd_steps, random_start=True)
    adv_cpu = _pgd_in_chunks(pgd, clf, imgs, labels, chunk=args.pgd_chunk)
    adv_bare = _accuracy(_logits_in_chunks(clf, adv_cpu.to(device)), labels_cpu)

    # 3. MAE purifier (downloads ~440 MB on first run).
    t0 = time.perf_counter()
    try:
        purifier = MAEPurifier(model_id=args.mae_id, device=device)
    except Exception as exc:  # noqa: BLE001 — surface a clear, actionable message
        raise SystemExit(
            f"[smoke] Could not load the MAE ('{args.mae_id}'): {exc}\n"
            "        Needs `transformers` and internet for the first download. "
            "On Kaggle, enable internet or add the model as a dataset."
        )
    print(f"[smoke] MAE loaded (norm_pix_loss={purifier.norm_pix_loss}) "
          f"in {time.perf_counter() - t0:.1f}s")

    pure = lambda x, k: purifier.purify(x, n_masks=k, mask_ratio=args.mask_ratio, chunk=args.chunk)

    # 4. Clean accuracy: bare vs purified(K=1) vs purified(K). => cost of masking.
    clean_pur1 = _accuracy(_logits_in_chunks(clf, pure(imgs, 1).to(device)), labels_cpu)
    clean_purK = _accuracy(_logits_in_chunks(clf, pure(imgs, args.k).to(device)), labels_cpu)

    # 4b. Purify the adversarial examples that were generated above.
    adv_purK = _accuracy(_logits_in_chunks(clf, pure(adv_cpu, args.k).to(device)), labels_cpu)

    # 5. Visual sanity grid.
    masked_v, recon_v, purified_v = purifier.visualize(imgs[: args.num_grids], mask_ratio=args.mask_ratio)
    grid_path = _save_grid(imgs[: args.num_grids].cpu(), masked_v.cpu(), recon_v.cpu(),
                           purified_v.cpu(), os.path.join(_HERE, "figures"), args.num_grids, tag)

    # 6. Persist + summarise.
    results = {
        "config": {
            "n": args.n, "k": args.k, "mask_ratio": args.mask_ratio,
            "pgd_steps": args.pgd_steps, "seed": args.seed, "device": device,
            "mae_id": args.mae_id, "norm_pix_loss": purifier.norm_pix_loss,
            "epsilon": _EPS, "classifier": "resnet50", "tag": args.tag,
        },
        "clean_accuracy": {
            "bare": clean_bare,
            "purified_k1": clean_pur1,
            "purified_kK": clean_purK,
        },
        "robustness_indicator_NON_adaptive": {
            "pgd_bare": adv_bare,
            "pgd_then_purify_kK": adv_purK,
            "_warning": "NON-adaptive. Attack was crafted on the bare classifier, "
                        "not through the purifier. Real verdict needs BPDA+EOT or the "
                        "full gradient benchmark (Square/AutoAttack) on the wrapped pipeline.",
        },
        "grid_png": os.path.relpath(grid_path, _HERE),
    }
    json_path = os.path.join(_HERE, f"smoke_results{tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    def pct(x):
        return f"{100 * x:5.1f}%"

    print("\n=========================  MAE PURIFIER SMOKE TEST  =========================")
    print(f"  images={args.n}  ensemble K={args.k}  mask_ratio={args.mask_ratio}  "
          f"classifier=resnet50")
    print("  --- clean accuracy (cost of purification) ---")
    print(f"    bare (no purify)        : {pct(clean_bare)}")
    print(f"    purify K=1              : {pct(clean_pur1)}")
    print(f"    purify K={args.k:<2d}             : {pct(clean_purK)}   <- want this near bare")
    print("  --- robustness INDICATOR (NON-adaptive — indicator only!) ---")
    print(f"    PGD-{args.pgd_steps} on bare clf     : {pct(adv_bare)}   (expect ~0% for resnet50)")
    print(f"    PGD then purify K={args.k:<2d}      : {pct(adv_purK)}   <- >0 = purify moves the needle")
    print("  ---------------------------------------------------------------------------")
    print("  NOTE: the PGD probe attacks the BARE classifier, then purifies. It does NOT")
    print("        prove robustness — a purifier can look great here yet collapse under an")
    print("        adaptive attack. The in-scope adaptive check is Square/AutoAttack from")
    print("        your existing gradient benchmark run on the purifier-wrapped classifier.")
    print(f"  grid : {grid_path}")
    print(f"  json : {json_path}")
    print("============================================================================\n")


if __name__ == "__main__":
    main()
