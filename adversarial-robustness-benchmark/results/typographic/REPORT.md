# Typographic-Attack Benchmark — Report

_Run started: 2026-05-25 12:43 UTC.  Author: Sachit Jain._

This report covers Phase 1, Axis C of the project — a single semantic, model-agnostic attack (the typographic overlay) evaluated against the 7 ImageNet baselines listed in `config.yaml`. The poisoned dataset is generated once by `scripts/generate_datasets.py --typographic` and shared across all models; this report aggregates the per-model JSONs in this directory.

## 1. Setup

- **Hardware**: Tesla T4.
- **OS**: Linux-6.6.122+-x86_64-with-glibc2.35.
- **Libraries**: torch 2.10.0+cu128, torchvision 0.25.0+cu128, transformers 5.0.0, pillow 11.3.0.
- **Attack**: typographic overlay (`attacks/typographic.py`), white sticker with bold black text drawn near the top of each clean image. Config: `font_size_frac=0.12`, `position='top'`, `padding_frac=0.04`, `opacity=1.0`, `jpeg_quality=90`, `text_form='lowercase_first_synonym'`.
- **Dataset**: the full 1000-image clean benchmark, one poisoned variant per clean image (`data/poisoned/typographic/`). Each poisoned image is stamped with the lowercase first synonym of a random *wrong* ImageNet class, picked deterministically with the global seed.
- **Global seed**: 42 (`config.yaml`). Set on `random`, `numpy`, `torch` at the start of every model.

## 2. Headline accuracy table

| Model | Clean | Robust | Drop | Fooling rate | TASR |
| --- | --- | --- | --- | --- | --- |
| resnet50 | 0.781 | 0.768 | 0.013 | 0.036 | 0.001 |
| vgg16 | 0.687 | 0.663 | 0.024 | 0.076 | 0.000 |
| convnext_tiny | 0.787 | 0.777 | 0.010 | 0.032 | 0.000 |
| vit_b_16 | 0.793 | 0.773 | 0.020 | 0.039 | 0.001 |
| swin_t | 0.775 | 0.770 | 0.005 | 0.032 | 0.002 |
| efficientnet_b0 | 0.743 | 0.722 | 0.021 | 0.055 | 0.001 |
| clip_vit_b16 | 0.621 | 0.368 | 0.253 | 0.452 | 0.342 |

Columns: clean accuracy, robust accuracy (untargeted — did the model flip away from the true label?), accuracy drop (clean − robust), fooling rate (originally-correct images that flipped), and TASR — the **targeted-attack success rate**, the fraction of images where the model predicted the *exact* class named by the overlay text.

Machine-readable copy: `accuracy_table.csv`.

## 3. Sanity checks

- **Typographic overlay causes a non-zero accuracy drop on every model.**
  - `resnet50`: clean **0.781** → robust **0.768** (drop **0.013**) → **PASS**.
  - `vgg16`: clean **0.687** → robust **0.663** (drop **0.024**) → **PASS**.
  - `convnext_tiny`: clean **0.787** → robust **0.777** (drop **0.010**) → **PASS**.
  - `vit_b_16`: clean **0.793** → robust **0.773** (drop **0.020**) → **PASS**.
  - `swin_t`: clean **0.775** → robust **0.770** (drop **0.005**) → **PASS**.
  - `efficientnet_b0`: clean **0.743** → robust **0.722** (drop **0.021**) → **PASS**.
  - `clip_vit_b16`: clean **0.621** → robust **0.368** (drop **0.253**) → **PASS**.
- **CLIP resists typographic more than the pure-classifier mean.**  pure mean drop **0.016** vs CLIP drop **0.253** → **FAIL**.
- **Coverage**: 7 / 7 models reported. **PASS**.

## 4. Per-model analysis

Across the 7 evaluated models the typographic overlay caused an average accuracy drop of **0.05** absolute. The most affected model was `clip_vit_b16`, dropping from **0.62** clean to **0.37** robust (drop **0.25**, TASR **0.34**). `clip_vit_b16` (zero-shot, language-grounded) recorded clean **0.62** → robust **0.37** (drop **0.25**, TASR **0.34**). Compare this directly with the pure-classifier average — that gap is the paper's language-grounding finding.

## 5. Figures

![Figure 1 — Clean vs typographic-robust accuracy per model (sorted by clean acc).](figures/01_clean_vs_robust.png)  
*Figure 1 — Clean vs typographic-robust accuracy per model (sorted by clean acc).*
![Figure 2 — Accuracy drop and targeted-attack success rate per model.](figures/02_drop_and_tasr.png)  
*Figure 2 — Accuracy drop and targeted-attack success rate per model.*
![Figure 3 — Language-grounded (CLIP) vs pure-classifier mean drop.](figures/03_clip_vs_others.png)  
*Figure 3 — Language-grounded (CLIP) vs pure-classifier mean drop.*
![Figure 4 — Clean | poisoned examples (typographic overlay).](figures/04_examples.png)  
*Figure 4 — Clean | poisoned examples (typographic overlay).*

## 6. Interpretation

The typographic attack is the paper's headline semantic attack. Pure vision classifiers are expected to latch onto the rendered text token and predict the wrong class — the TASR column quantifies how often the model is fooled into the *exact* class named on the sticker, not just into any wrong class. A language-grounded model like CLIP classifies by matching against text descriptions of every class; the hypothesis (project brief §3.6, hook #1) is that this makes it resist text overlays compared to pure classifiers. Compare CLIP's drop to the pure-classifier average in the sanity-check block above.

## 7. Reproducibility footer

- **Wall-clock (this report build session)**: 0.1 min.
- **Per-model cumulative compute (sum of per-cell `wall_clock_s`)**: clip_vit_b16 0.5 min, convnext_tiny 0.3 min, efficientnet_b0 0.1 min, resnet50 0.2 min, swin_t 0.3 min, vgg16 0.2 min, vit_b_16 0.4 min.
- **Total compute (sum across models)**: 1.9 min.
- **Library versions**: torch 2.10.0+cu128, torchvision 0.25.0+cu128, transformers 5.0.0, pillow 11.3.0.
- **GPU**: Tesla T4.
- **Seed**: 42 (set on `random`, `numpy`, `torch`).
- **Re-run**: `python scripts/run_typographic_benchmark.py` then `python scripts/build_typographic_report.py` then `python scripts/build_gradient_report_pdf.py --input results/typographic/REPORT.md --output results/typographic/REPORT.pdf`. Per-model JSONs are the resumption unit — delete one to force its recomputation.
