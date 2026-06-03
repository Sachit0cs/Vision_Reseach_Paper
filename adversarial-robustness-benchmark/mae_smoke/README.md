# `mae_smoke/` — MAE purifier smoke test (throwaway)

A **self-contained, deletable** probe of whether a pretrained Masked Autoencoder
(MAE) can work as an adversarial **input purifier** for this benchmark. If it
looks unpromising, **delete this one folder** and your repo is exactly as before —
nothing outside `mae_smoke/` is touched, and no existing file is modified.

## What a "purifier" is (and is not)

- **Is:** a separate, frozen, pretrained network placed *in front of* a classifier —
  `prediction = classifier( purify(image) )`. Your 7 classifiers are **unmodified**.
- **Is not:** an architecture change. We do **not** add layers inside ResNet-50 etc.,
  and we do **not** train anything here.

The MAE masks ~75% of the 16×16 patches and reconstructs them from the visible
context, so perturbation hiding in masked patches is thrown away and re-drawn from a
clean prior. Averaging over `K` random maskings spreads that cleansing over the image.

## Files

| File | Role |
|---|---|
| `mae_purifier.py` | `MAEPurifier` — wraps `facebook/vit-mae-base`; `purify(x01, n_masks, mask_ratio) -> x01`. No grad, no training. |
| `run_smoke.py` | Loads a few benchmark images (reusing `datasets.loader` + `models.classifiers`), measures the clean-accuracy cost + a non-adaptive PGD indicator, saves a visual grid. |
| `figures/`, `smoke_results.json` | Outputs (created on run). |

No new dependencies — `transformers` is already required by the repo (for CLIP).
The MAE checkpoint (~440 MB) downloads on first run.

## Run

```bash
# Local (CPU is fine; a few minutes)
python mae_smoke/run_smoke.py --n 16 --k 2

# Kaggle GPU (bigger, sharper picture)
python mae_smoke/run_smoke.py --n 200 --k 8
```

## How to read the result

- **Clean accuracy block** — `bare` vs `purify K=1` vs `purify K`. Purification
  *costs* clean accuracy (masking discards most of the image); a higher `K` should
  recover much of it. If `purify K` is far below `bare`, the purifier is too lossy.
- **Robustness INDICATOR block** — `PGD on bare` (≈0% expected for ResNet-50) vs
  `PGD then purify`. If the purified number is meaningfully above 0, purification is
  doing *something*.
- **`figures/reconstruction_grid.png`** — original | masked | reconstruction |
  purified. The reconstruction should look like a plausible (blurry) version of the
  original. Washed-out / wrong-colour output means the `norm_pix_loss` de-normalisation
  needs fixing.

## The big caveat (read this)

The PGD probe here is **NON-adaptive**: it attacks the *bare* classifier and then
purifies. That is an **indicator, not proof**. Purifiers are the classic source of
*"obfuscated gradients — a false sense of security"*: they can look strong against a
non-adaptive attack and collapse under an **adaptive BPDA + EOT** attack that targets
the full `purify → classify` pipeline. Treat a good result here as "worth continuing,"
not "it works." The adaptive evaluation is the next step in the project plan.

## To discard

```bash
rm -rf mae_smoke/        # PowerShell: Remove-Item -Recurse -Force mae_smoke
```
