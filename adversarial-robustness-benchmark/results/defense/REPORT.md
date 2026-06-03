# MAE Purifier Defense vs Baseline ResNet-50 — Results

_Numbers-and-results doc (not a paper draft). Defense = `MAE_defense_model`._

## Setup

- **Defense (`MAE_defense_model`):** a **test-time input purifier** in front of an
  **unmodified, pretrained ResNet-50** — `prediction = resnet50( purify(x) )`. The
  purifier is a **frozen pretrained `facebook/vit-mae-base`** that masks a fraction
  of the 16×16 patches and reconstructs them, averaged over `K` random masks.
  **No training**; the ResNet-50 weights are never changed.
  ([models/mae_defense.py](../../models/mae_defense.py))
  - `mask_ratio = 0.5`, `K = 8`, MAE = `facebook/vit-mae-base`.
- **Baseline:** the same pretrained torchvision ResNet-50, no purifier.
- **Adaptive evaluation (honest gradients).** The purifier is non-differentiable
  (runs under `no_grad`) and stochastic. To avoid a false "obfuscated gradients"
  result, `logits` uses a **BPDA straight-through estimator** (forward = purified
  image; backward = identity through the purifier). So **FGSM/PGD here are adaptive
  attacks against the defense**, not naive attacks on the bare classifier.
- **Protocol:** ε = 8/255 L∞, seed = 42, 1000-image class-balanced ImageNet-val
  subset; PGD-20; Tesla T4. Same images/ε/steps as the baselines.
- **⚠ Incomplete evaluation:** **AutoAttack and Square were skipped this session
  (compute).** They are the gradient-masking cross-checks and are pending a later
  Kaggle session. Read every robustness number below as **provisional** until they
  are run — see *Pending attacks*.

## Headline

The purifier is **cheap** (clean accuracy 78.1% → **74.2%**, only −3.9 pp) but it
**does not provide white-box adversarial robustness.** Under the **adaptive
BPDA-PGD-20** attack it scores **1.3%**, essentially tied with the baseline's
**0.4%** — both at floor. The earlier *non-adaptive* smoke test (attack the bare
classifier, then purify) recovered ~31%; that signal **disappears once the attack
is made adaptive** — a textbook obfuscated-gradients result.

## Axis 1 — Gradient attacks (white-box, adaptive via BPDA)

| Attack | Baseline R50 | MAE defense | Δ |
|---|---|---|---|
| Clean | 0.781 | 0.742 | −0.039 |
| FGSM (ε=8/255) | 0.408 | 0.414 | +0.006 |
| PGD-20 (BPDA-adaptive) | 0.004 | 0.013 | +0.009 |
| AutoAttack | — | — | _pending_ |
| Square | — | — | _pending_ |

**Read:** FGSM accuracy (~41%) is **the same with or without the purifier** — that
is ResNet-50's own single-step resistance, not a defense effect. Under the
multi-step adaptive PGD both models collapse to the floor (1.3% = 13/1000 vs 0.4%
= 4/1000); the +0.9 pp is 9 images at floor, **not** a robustness gain. **No
white-box robustness.**

## Axis 2 — Transfer attack (surrogate = bare ResNet-50, PGD-20)

| Model | Clean | Transfer-robust | Drop | Fooling rate |
|---|---|---|---|---|
| Baseline R50 (white-box on self) | 0.781 | 0.003 | 0.778 | 99.6% |
| MAE defense | 0.742 | **0.053** | 0.689 | 92.9% |

**Read:** the **one place the purifier measurably helps.** Adversarials crafted on
the *bare* ResNet-50 partly fail to transfer through the purifier (5.3% survive vs
0.3%). Because the defense shares the ResNet-50 backbone with the surrogate, this is
effectively a *same-backbone* transfer, so the +5 pp is a genuine — if small —
disruption effect. Still mostly broken (92.9% fooling).

## Axis 3 — Typographic attack (targeted, lowercase first-synonym)

| Model | Clean | Robust | TASR | Fooling rate |
|---|---|---|---|---|
| Baseline R50 | 0.781 | 0.768 | 0.001 | 3.6% |
| MAE defense | 0.742 | 0.724 | 0.000 | 5.8% |

**Read:** no effect either way — ResNet-50 doesn't read printed words (TASR ≈ 0 for
both). The defense is marginally **worse** (robust 72.4% vs 76.8%), tracking its
lower clean accuracy. No win here (expected; typographic is a CLIP phenomenon).

## Axis 4 — Common corruptions (severity 3, 15 types)

| Model | Mean robust | 
|---|---|
| Baseline R50 | **0.498** |
| MAE defense | 0.480 |

Mean is **−1.8 pp worse**, but the per-type breakdown shows a real, interpretable
pattern — the purifier (a reconstruction model) **helps on noise, hurts elsewhere:**

| Corruption | Baseline | MAE def | Δ | | Corruption | Baseline | MAE def | Δ |
|---|---|---|---|---|---|---|---|---|
| gaussian_noise | 0.487 | **0.506** | +0.019 | | brightness | 0.721 | 0.686 | −0.035 |
| shot_noise | 0.457 | **0.478** | +0.021 | | contrast | 0.700 | 0.656 | −0.044 |
| glass_blur | 0.088 | **0.130** | +0.042 | | defocus_blur | 0.447 | 0.413 | −0.034 |
| impulse_noise | 0.358 | 0.334 | −0.024 | | elastic_transform | 0.771 | 0.728 | −0.043 |
| | | | | | fog | 0.384 | 0.337 | −0.047 |
| | | | | | frost | 0.618 | 0.583 | −0.035 |
| | | | | | jpeg_compression | 0.660 | 0.649 | −0.011 |
| | | | | | motion_blur | 0.129 | 0.118 | −0.011 |
| | | | | | pixelate | 0.594 | 0.576 | −0.018 |
| | | | | | snow | 0.664 | 0.652 | −0.012 |
| | | | | | zoom_blur | 0.393 | 0.359 | −0.034 |

**Read:** the purifier improves the three **noise** corruptions (gaussian/shot/glass)
— consistent with it being a denoiser — but slightly degrades the blur/weather/clean
types, so the mean nets out negative. A nice mechanism check, not a robustness win.

## Where it wins / where it fails

**Real (modest) effects:**
- **Transfer:** 5.3% vs 0.3% — partial disruption of transferred adversarials.
- **Noise corruptions:** small gains on gaussian/shot/glass noise (it denoises).

**Failures / costs:**
- **White-box PGD (adaptive/BPDA): no robustness** — 1.3%, floor, ≈ baseline.
- **FGSM:** identical to baseline (~41%); purifier adds nothing.
- **Typographic / corruptions mean:** slightly worse (clean-accuracy carryover).
- **Clean accuracy:** −3.9 pp.

## The obfuscated-gradients lesson (the actual finding)

This run is most useful as a **clean, adaptively-evaluated negative result**:

| PGD-20 vs the MAE purifier | robust acc |
|---|---|
| **Non-adaptive** (attack bare R50, then purify) — `mae_smoke` | ~0.31 |
| **Adaptive** (BPDA straight-through) — this run | **0.013** |

The 30-point gap between the two **is** the obfuscated-gradients pitfall, measured
directly: a purifier that looks robust under a naive attack collapses to floor once
the attacker differentiates through it. That contrast is an honest, teachable result
on its own.

## Pending attacks (and the expectation)

AutoAttack and Square were skipped (compute) and will run in a later session.
Expectation, stated honestly up front:
- **AutoAttack** (APGD ≥ PGD strength) attacks via the same BPDA path → it will
  **not rescue** the defense; expect ≈ floor, like PGD.
- **Square** (gradient-free, queries the true stochastic pipeline) is the remaining
  check on whether the residual 1.3% is real robustness or just stochastic-masking
  noise. If Square also reads ~floor, the null-robustness verdict is confirmed.

The provisional verdict is therefore a **null adversarial-robustness result**; the
pending attacks are expected to **confirm**, not overturn it.

## Honest verdict

The MAE purifier is a **low-cost, model-agnostic front-end** that gives a small
transfer-robustness and noise-corruption benefit, **but it does not beat the
baseline on white-box adversarial robustness** — under an adaptive (BPDA) attack it
is at floor, like the undefended model. As an adversarial *defense* it is, on the
evidence so far, a **null result** — but a scientifically clean one, and the
non-adaptive-vs-adaptive gap is a genuine finding worth reporting.

---

### Reproducibility

- Seed 42; ε = 8/255 L∞; PGD-20 (step 2/255); 1000-image class-balanced subset.
- Defense: `facebook/vit-mae-base` (frozen), `mask_ratio=0.5`, `K=8`, BPDA
  straight-through; ResNet-50 = torchvision DEFAULT weights (unmodified).
- Host: Tesla T4, torch 2.10.0+cu128, torchvision 0.25.0+cu128, transformers 5.0.0.
- Per-cell JSONs: `results/{gradient,transfer,typographic,corruptions}/MAE_defense_model*`.
- **Not yet run:** AutoAttack, Square (gradient-masking cross-checks).
