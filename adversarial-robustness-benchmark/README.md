# Adversarial Robustness Benchmark for Vision Classifiers

A platform that measures and ranks how robust modern image classifiers (CNNs,
Vision Transformers, and a language-grounded model) are when their inputs are
adversarially or semantically corrupted — then designs and evaluates an
improved defensive architecture.

See `Adversarial_Robustness_Benchmark_PROJECT_BRIEF.pdf` (one directory up) for
the full scientific design. This README covers setup and usage only.

## Status

| Phase | Description | State |
|-------|-------------|-------|
| 0 | Repo skeleton, config, model wrappers | structure created |
| 1 | Datasets & attacks | **clean dataset done**; attacks pending |
| 2 | Benchmark (metrics, pipeline, reporting) | pending |
| 3 | Analysis | pending |
| 4 | Novel defense architecture | pending |
| 5 | Paper | pending |

Only the repository structure and the **clean (original) benchmark dataset**
have been built so far. The poisoned datasets and all model/attack code are
stubbed (`raise NotImplementedError`) and will be filled in phase by phase.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# AutoAttack + RobustBench — needed from Phase 1 onward only.
# Installed with --no-deps; their pinned deps otherwise break pip's resolver.
pip install --no-deps -r requirements-attacks.txt
```

## Repository layout

```
adversarial-robustness-benchmark/
  config.yaml              global config: seed, paths, epsilon, model list
  datasets/                ImageNet-val subset loader + class-balanced sampler
  models/                  7 classifier wrappers + RobustBench + defense module
  attacks/                 FGSM/PGD/AutoAttack/Square/patch/typographic/corruptions
  metrics/                 clean acc, robust acc, flip rate, CDS, softmax-KL
  evaluation/              models x attacks orchestrator + protocol logic
  reporting/               ranking tables + plots
  scripts/                 generate_datasets / run_benchmark / train_defense CLIs
  results/                 generated json, csv, png, reports
  tests/                   pytest suites
  notebooks/               thin Kaggle GPU runner
  data/                    datasets (gitignored)
```

## Generating the clean dataset

```bash
python scripts/generate_datasets.py --clean
```

This downloads the ImageNet-1k validation split (~870 MB, parquet) from the
public mirror `evanarlian/imagenet_1k_resized_256` and writes a class-balanced
1,000-image clean set to `data/clean/`:

```
data/clean/
  images/0000_00000.jpg ... 0999_00999.jpg   (1 image per ImageNet class)
  manifest.json                              (id, filename, label, class_name)
  imagenet_classes.json                      (1000 class names, index-aligned)
```

The clean set (`data/clean/`, ~17 MB) is **committed to this repository**, so
after cloning it is already present — you only need to run the generator above
to rebuild or expand it. The 870 MB parquet download cache (`data/_hf_cache/`)
is *not* committed; it is regenerated on demand.

This repository is **private**: the benchmark images derive from ImageNet,
whose terms restrict public redistribution. Keep it to internal research use.

The clean-set build is CPU-only and needs no GPU. Subsequent phases
(model inference, attack generation) require a GPU — see `notebooks/kaggle_runner.ipynb`.

## Running the benchmark (later phases)

```bash
python scripts/run_benchmark.py     # not yet implemented
python scripts/train_defense.py     # not yet implemented
```
