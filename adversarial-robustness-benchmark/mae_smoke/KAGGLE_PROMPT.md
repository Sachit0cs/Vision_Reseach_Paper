# Kaggle run prompt — MAE purifier smoke test (bigger run)

Paste everything between the lines below to your Kaggle-side agent (or follow it
yourself). It is self-contained: project context, hard isolation rules, how to
run, how to read the result, and an appendix with the exact file contents in case
the `mae_smoke/` folder is missing.

---------------------------------------------------------------------------------

You are running on **Kaggle (T4 GPU, internet ON)**. You are helping with an
adversarial-robustness research project. Your job: run an **MAE input-purifier
smoke test** at a proper sample size and report the numbers. Read the rules first.

## HARD CONSTRAINTS (do not violate)
1. **Write ONLY inside `adversarial-robustness-benchmark/mae_smoke/`.** Do NOT
   create, edit, move, or delete any file outside that folder. Read-only imports
   from the rest of the repo are fine.
2. **Do NOT modify** `config.yaml`, `models/`, `attacks/`, `datasets/`,
   `scripts/`, or anything under `results/`. The `mae_smoke/` folder must stay
   fully self-contained and **deletable** — deleting it must return the project to
   its Madry-adversarial-training state with zero side effects.
3. **Do NOT add new attacks or new dependencies.** Everything needed is already
   present (`transformers` is a repo dependency).
4. The `mae_smoke/` files are **committed on the `defense-mae` branch**, so they
   already exist after checkout — you should not need to create anything. ONLY if a
   file is somehow missing, recreate it in that folder verbatim from the
   **Appendix** below — nowhere else.

## PROJECT CONTEXT (so you understand what you're running)
- A benchmark of **7 ImageNet classifiers** (ResNet-50, VGG-16, ConvNeXt-Tiny,
  ViT-B/16, Swin-T, EfficientNet-B0, CLIP ViT-B/16) under L∞ ε=8/255, across 4
  attack axes (gradient = FGSM/PGD/AutoAttack/Square, transfer, typographic,
  common-corruptions). All 4 axes are already done.
- **Pixel contract:** classifiers take `[0,1]` tensors; normalization happens
  *inside* `.logits()`. Attacks operate in `[0,1]`. Stay in `[0,1]` everywhere.
- **Defense:** the existing approach is Madry PGD adversarial training
  (`defense_resnet50`). We are evaluating an **alternative**: an **MAE input
  purifier**.
- **The purifier is a WRAPPER, not an architecture change:** the prediction is
  `classifier( purify(image) )`. The classifiers are **never modified or retrained**.
  The purifier uses a frozen pretrained `facebook/vit-mae-base` (a ViT) to mask
  ~75% of the 16×16 patches and reconstruct them, cleansing perturbation. **No
  training happens.**
- **The architecture being benchmarked is ResNet-50** (so it compares apples-to-
  apples with the clean ResNet-50 baseline and the Madry `defense_resnet50`). The
  MAE (a ViT) is just the front-end cleaner — two networks: MAE purifies, ResNet-50
  classifies and is the thing measured.

## SETUP
All `mae_smoke/` code lives on the **`defense-mae` branch** (it is NOT on `main`).
After checking out that branch the folder is fully present — do not recreate files.
```bash
git clone <repo-url> && cd adversarial-robustness-benchmark
git fetch origin && git checkout defense-mae       # the branch that holds mae_smoke/
ls mae_smoke                                        # expect mae_purifier.py, run_smoke.py, README.md, ...

python -c "import torch; print('cuda', torch.cuda.is_available())"   # expect True
pip -q install transformers    # already a repo dep; installs only if the image lacks it
```

## RUN (the bigger run + a 2-point mask-ratio sweep)
Each run downloads `facebook/vit-mae-base` (~440 MB) once, then takes a few minutes
on a T4. `--tag` keeps runs from overwriting each other.
```bash
# default mask ratio 0.75 (aggressive cleansing, higher clean-accuracy cost)
python mae_smoke/run_smoke.py --n 200 --k 8 --tag mr075

# lower mask ratio 0.5 (keeps more of the image -> less clean-acc cost)
python mae_smoke/run_smoke.py --n 200 --k 8 --mask-ratio 0.5 --tag mr050
```
If you hit GPU OOM, lower `--chunk` (e.g. `--chunk 16`) or `--n`.

## REPORT BACK
For **each** run, report from `mae_smoke/smoke_results_<tag>.json`:
- `clean_accuracy`: `bare`, `purified_k1`, `purified_kK`  → the clean-accuracy COST.
- `robustness_indicator_NON_adaptive`: `pgd_bare`, `pgd_then_purify_kK`.
Also attach the grids `mae_smoke/figures/reconstruction_grid_<tag>.png`. Then give a
2-line plain summary: how much clean accuracy purification costs at each mask ratio,
and whether the (non-adaptive) PGD number is meaningfully above 0.

## HOW TO READ IT — AND WHAT IT DOES NOT PROVE
- **Clean cost:** `purified_kK` vs `bare`. A lower mask ratio should cost less.
- **PGD recovery:** `pgd_then_purify_kK` > 0 means purification does *something*.
- **CRITICAL CAVEAT:** the PGD probe attacks the **bare** classifier and *then*
  purifies — it is **NON-adaptive**. A purifier can look strong here and still
  collapse under an adaptive attack (this is the classic "obfuscated gradients"
  trap). **Do NOT claim robustness from this number.** The proper in-scope adaptive
  check is to run the existing gradient benchmark (which includes the gradient-free
  **Square** and **AutoAttack**) against the purifier-wrapped classifier — that is a
  *separate* step that touches existing files and is NOT part of this smoke test.

---------------------------------------------------------------------------------

## Appendix — exact file contents (recreate in `mae_smoke/` ONLY if missing)

### `mae_smoke/mae_purifier.py`
```python
"""MAE-based adversarial input purifier (smoke-test scaffold).

Standalone and DELETABLE. A pretrained Masked Autoencoder
(`facebook/vit-mae-base`) used as a TEST-TIME purifier placed IN FRONT of an
unmodified classifier:  prediction = classifier( purify(image) ).
NOT an architecture change. NO training. All tensors are [0,1] (B,3,224,224).
"""

from __future__ import annotations

import torch

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class MAEPurifier:
    def __init__(self, model_id: str = "facebook/vit-mae-base", device: str = "cpu"):
        from transformers import ViTMAEForPreTraining

        self.device = device
        self.model = ViTMAEForPreTraining.from_pretrained(model_id).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.image_size = int(self.model.config.image_size)
        self.patch_size = int(self.model.config.patch_size)
        self.norm_pix_loss = bool(getattr(self.model.config, "norm_pix_loss", False))

        self._mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)

    def _norm(self, x01):
        return (x01 - self._mean) / self._std

    def _denorm(self, xn):
        return xn * self._std + self._mean

    @torch.no_grad()
    def _reconstruct_once(self, pix_norm, mask_ratio):
        self.model.config.mask_ratio = float(mask_ratio)
        out = self.model(pixel_values=pix_norm)

        pred = out.logits
        mask = out.mask
        orig_patches = self.model.patchify(pix_norm)

        if self.norm_pix_loss:
            pm = orig_patches.mean(dim=-1, keepdim=True)
            pv = orig_patches.var(dim=-1, keepdim=True)
            pred = pred * (pv + 1e-6) ** 0.5 + pm

        m = mask.unsqueeze(-1)
        combined = orig_patches * (1 - m) + pred * m
        purified_norm = self.model.unpatchify(combined)
        recon_full_norm = self.model.unpatchify(pred)
        return purified_norm, mask, recon_full_norm

    @torch.no_grad()
    def purify(self, x01, n_masks: int = 4, mask_ratio: float = 0.75, chunk: int = 32):
        x01 = x01.to(self.device).float()
        n = max(1, int(n_masks))
        outs = []
        for s in range(0, x01.size(0), chunk):
            pix = self._norm(x01[s : s + chunk])
            acc = torch.zeros_like(pix)
            for _ in range(n):
                purified_norm, _, _ = self._reconstruct_once(pix, mask_ratio)
                acc = acc + purified_norm
            outs.append(self._denorm(acc / n).clamp(0.0, 1.0))
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def visualize(self, x01, mask_ratio: float = 0.75):
        x01 = x01.to(self.device).float()
        pix = self._norm(x01)
        purified_norm, mask, recon_full_norm = self._reconstruct_once(pix, mask_ratio)

        orig_patches = self.model.patchify(pix)
        m = mask.unsqueeze(-1)
        masked_patches = orig_patches * (1 - m)
        masked_view = self.model.unpatchify(masked_patches)

        return (
            self._denorm(masked_view).clamp(0.0, 1.0),
            self._denorm(recon_full_norm).clamp(0.0, 1.0),
            self._denorm(purified_norm).clamp(0.0, 1.0),
        )
```

### `mae_smoke/run_smoke.py`
> Identical to the committed `mae_smoke/run_smoke.py` on the `defense-mae` branch.
> If recreating by hand, copy that file. Run with:
> `python mae_smoke/run_smoke.py --n 200 --k 8 --tag mr075`

(End of prompt.)
