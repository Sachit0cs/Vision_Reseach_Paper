# Defense ResNet-50 vs Baseline ResNet-50 — Results Summary

Numbers-and-results doc. Not a paper draft.

## Setup

- **Baseline:** torchvision ResNet-50, ImageNet-1k pretrained, evaluated on ImageNet-100 val subset.
- **Defense:** ResNet-50 trained **from scratch** with adversarial training (PGD-5, ε = 8/255) on ImageNet-100, 8 epochs on a single Kaggle T4. Final training adv-acc = **27.2 %** (see [training_log.json](training_log.json)).
- All evals: ε = 8/255 L∞, seed = 42, same 1000-image (or 200-image for AutoAttack / Square) subset.

## Headline number

The defense model **never converged on clean data**. Clean test accuracy = **6 %**. Every "robustness gain" downstream has to be read against that.

## Axis 1 — Gradient attacks (white-box)

| Attack | Baseline R50 | Defense R50 | Δ |
|---|---|---|---|
| Clean | 0.781 | **0.060** | −0.721 |
| FGSM (ε=8/255) | 0.408 | 0.024 | −0.384 |
| PGD-20 | 0.004 | 0.015 | +0.011 |
| AutoAttack (standard) | 0.000 | 0.005 | +0.005 |
| Square (5000 q) | 0.010 | 0.015 | +0.005 |

**Read:** the +1 pp "wins" on PGD/AA/Square are noise — both models are sitting at floor. Baseline was already broken by strong attacks; defense is broken *and* can't classify clean images. Net regression.

## Axis 2 — Transfer attacks (surrogate = baseline ResNet-50, PGD-20)

| Model | Clean | Transfer-robust | Drop | Fooling rate |
|---|---|---|---|---|
| Baseline R50 (white-box on itself) | 0.781 | 0.003 | 0.778 | 0.996 |
| Defense R50 | 0.060 | 0.059 | **0.001** | **0.017** |
| For reference — ConvNeXt-T | 0.783 | 0.638 | 0.145 | 0.198 |
| For reference — ViT-B/16 | 0.796 | 0.747 | 0.049 | 0.079 |

**Read:** the "0.1 pp drop" looks like total transfer immunity, but it's an artifact — the defense model classifies almost nothing correctly to begin with, so adversarial perturbations have nothing to flip. The 1 / 60 fooled samples is consistent with random label noise. Cannot claim transfer robustness from this.

## Axis 3 — Typographic attack (targeted, lowercase first-synonym)

| Model | Clean | Robust | TASR | Fooling rate |
|---|---|---|---|---|
| Baseline R50 | 0.781 | 0.768 | 0.001 | 0.036 |
| Defense R50 | 0.060 | 0.058 | 0.002 | **0.117** |
| (Hook finding) CLIP ViT-B/16 | 0.621 | 0.368 | **0.342** | 0.452 |

**Read:** same story. TASR ≈ 0 for both ResNet-50 variants — neither reads the printed word. Defense model has higher fooling rate (11.7 % vs 3.6 %) on originally-correct samples, i.e. when it *does* get one right, a sticker is more likely to flip it. Tiny denominator (60 originally-correct), so the bar is wide. CLIP row kept for context — that's the paper's hook, untouched by this change.

## Axis 4 — ImageNet-C corruptions (severity 3, 15 types)

Mean robust accuracy across 15 corruptions:

| Model | Mean robust | Range |
|---|---|---|
| Baseline R50 | **0.498** | 0.088 (glass blur) … 0.771 (elastic) |
| Defense R50 | 0.035 | 0.004 (fog/contrast) … 0.061 (jpeg) |

Per-corruption fooling rates for the defense model exceed the baseline on **13 of 15** corruptions (contrast, fog, frost, motion blur, defocus blur all > 65 % fooling). Only `jpeg_compression` and `elastic_transform` are stable, and only because the model is already at ~6 % everywhere.

**Read:** defense model is *more* corruption-fragile in relative terms, not less. No corruption-robustness benefit.

## Where defense "wins" vs where it actually fails

**Apparent wins (all artifacts of low clean baseline):**
- PGD / AutoAttack / Square: +0.5–1.1 pp — noise at floor.
- Transfer attack drop: only 0.1 pp — but starting from 6 %.
- Typographic robust accuracy: 5.8 % vs 76.8 % is not a win; the "small drop from clean" is meaningless.

**Real failures:**
- Clean accuracy collapse: **78.1 % → 6.0 %** (−72 pp). Dominates everything else.
- FGSM: 40.8 % → 2.4 %. Even the weakest attack is now ~floor.
- Mean corruption accuracy: 49.8 % → 3.5 % (−46 pp).
- Higher fooling rates on the few clean-correct samples (typographic 11.7 %, several corruptions > 60 %), suggesting what little the model learned is *more* brittle, not less.

## Why this happened

8 epochs of adversarial training from scratch on ImageNet-100 with PGD-5 inner loop is far below convergence (training adv-acc still climbing linearly: 4.5 % → 27.2 % across the run). Standard TRADES / Madry recipes use ≥100 epochs from a clean-pretrained init. We have neither the epoch budget nor a clean-pretrained ImageNet-100 init on free Kaggle T4.

## What to do next

Two viable paths given the compute budget:

1. **Initialize from ImageNet-1k pretrained weights** (already have them — that's the baseline). Adversarial-fine-tune for the same 8 epochs. Clean accuracy should stay > 60 %, and PGD robustness should land in the 20–35 % range, which is the regime where the gradient/transfer/typographic/corruption numbers become *comparable* rather than dominated by the clean collapse.
2. If we keep training from scratch, need ≥40–60 epochs and a clean-eval checkpoint to confirm the model is learning the task at all before any robustness claim is meaningful.

Option 1 is the only one that fits the workshop deadline.
