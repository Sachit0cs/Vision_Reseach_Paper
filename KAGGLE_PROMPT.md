# Kaggle-side prompt — train the defense model and run the full benchmark

> Paste everything below the line into the Claude session running on Kaggle (the GPU side).
> Before you do, make sure the **local side has pushed** the fixed `config.yaml` and
> `scripts/train_defense.py` to GitHub — otherwise Kaggle will pull the old, broken code.

---

You are Claude Code running on a **Kaggle notebook with a single Tesla T4 GPU**. You are the
**compute window** of a two-window workflow. Read this whole brief before running anything.

## The two-window architecture

This project runs across two environments that talk to each other **only through GitHub**:

- **Window A — Local (Windows laptop, NO GPU).** Where the human + a local Claude author code,
  generate the human-facing PDF/Markdown report, and commit changes. It cannot train or run the
  benchmark (no CUDA, no ImageNet).
- **Window B — Kaggle (Tesla T4 GPU). ← YOU ARE HERE.** The only place with a GPU and the
  ImageNet data. You pull the latest code, **train the defense model**, **run the full benchmark
  on it**, write the result JSONs, and **push those JSONs back to GitHub**.

Sync protocol:
```
Local edits code ──push──► GitHub ──pull──► Kaggle (you)
Kaggle runs + writes results ──push──► GitHub ──pull──► Local regenerates the report
```
So your job ends when the **defense result JSONs are committed and pushed**. The human will
(a) save your trained checkpoint as a Kaggle Dataset for reuse, and (b) pull your JSONs locally.

## Progress so far (what already exists)

- A **7-model × 4-axis** adversarial-robustness benchmark is **complete**. The 7 baselines
  (ResNet-50, VGG-16, ConvNeXt-T, ViT-B/16, Swin-T, EfficientNet-B0, CLIP ViT-B/16) each have
  per-cell JSONs under `results/{gradient,transfer,typographic,corruptions}/`.
- The one headline finding: **CLIP typographic TASR ≈ 34%** (language grounding is a liability).
- An **8th model — `defense_resnet50`** (Madry PGD adversarial training) was attempted earlier and
  **FAILED**: 6% clean accuracy. Root cause (now fixed in code): it was **trained on only 100
  classes but evaluated on the full 1000-class benchmark**, capping clean accuracy near 10%.
- **The old defense checkpoint and its Kaggle Dataset have been deleted** — you start the defense
  from a clean slate. The fix is already committed: `config.yaml` now sets `num_classes: 1000`,
  `epochs: 10`, `lr: 0.01` (cosine), pretrained init, and `train_dir` → full ImageNet-1k;
  `train_defense.py` adds a cosine LR schedule and **per-epoch clean-accuracy tracking**.

## Your mission

1. Train a **new** `defense_resnet50` by **adversarially fine-tuning the pretrained ResNet-50 on
   all 1000 ImageNet classes** (this is what the fixed code does — do not change to from-scratch).
2. Run **all four benchmark axes** on the new defense model, producing per-cell JSONs in
   `results/…` in the **exact same schema and filenames** as the other models.
3. Push the result JSONs to GitHub.

---

## Step 0 — Setup

```bash
# Clone the repo (use the GitHub PAT stored as a Kaggle Secret; do NOT hard-code it in the notebook).
git clone https://<PAT>@github.com/<user>/<repo>.git
cd <repo>/adversarial-robustness-benchmark
pip install -q -r requirements.txt    # or: torch torchvision transformers timm autoattack torchattacks pyyaml tqdm pillow scipy

git pull --rebase    # make sure you have the FIXED train_defense.py + config.yaml
python -c "import yaml; d=yaml.safe_load(open('config.yaml'))['defense']; print('num_classes=',d['num_classes'])"
# MUST print 1000. If it prints 100, the local side has not pushed the fix — STOP and tell the human.
```

**Attach the datasets** to the Kaggle notebook:
- Full **ImageNet-1k training set** — the `imagenet-object-localization-challenge` dataset. Its train
  tree (1000 WNID folders) is at:
  `/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train`
- (The 1000-image **eval** benchmark and the poisoned sets are regenerated below — not committed.)

## Step 1 — Smoke test (5 min, do this first)

Confirm clean accuracy does **not** collapse before committing to the full run:

```bash
python scripts/train_defense.py \
  --train-dir /kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train \
  --epochs 1 --images-per-class 50 --num-workers 4
```
Watch the per-epoch line. **Expected:** `clean_acc` in the **0.50–0.75** range and climbing
(a pretrained net starts high). **FAILURE signature:** `clean_acc ≈ 0.06–0.10` → the label-space
bug is back; verify `num_classes=1000` and that the train dir really has ~1000 folders, then stop
and report. Delete the smoke checkpoints before the real run (`rm -f models/checkpoints/defense_*.pt`).

## Step 2 — Train the real defense model

```bash
python scripts/train_defense.py \
  --train-dir /kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train
# Uses config defaults: 1000 classes, 300 imgs/class, 10 epochs, lr 0.01 cosine, PGD-5 @ 8/255.
```
- Rough budget: ~30–40 min/epoch on a T4 → ~5–7 h total. Kaggle sessions time out (~9–12 h).
- It **checkpoints every epoch** to `models/checkpoints/defense_epoch_{N}.pt`. If a session ends,
  resume with `--resume models/checkpoints/defense_epoch_{N}.pt` (save the checkpoints as a Kaggle
  Dataset between sessions). If time is tight, lower `--epochs` or `--images-per-class`.
- Final artifact: **`models/checkpoints/defense_final.pt`** (this is what the benchmark loads).
- **Target:** clean acc ~60–70%, and you should expect real PGD robustness later (not 0%).
  If clean acc is still single digits, do **not** proceed — report back.

The human will store `defense_final.pt` as a Kaggle Dataset. **Do NOT git-commit the `.pt`**
(it is large and gitignored).

## Step 3 — Build the evaluation datasets (regenerated per session)

```bash
python scripts/generate_datasets.py --clean          # 1000-image ImageNet-val benchmark
python scripts/generate_datasets.py --typographic     # wrong-class sticker overlays (seconds)
python scripts/generate_datasets.py --corruptions     # 15 corruptions @ severity 3 (~1 h, CPU)
python scripts/generate_datasets.py --transfer        # PGD adversarials on ResNet-50 surrogate (GPU, minutes)
```

## Step 4 — Run all 4 axes on the defense model

⚠️ **Use `--force`.** The old defense JSONs are committed in the repo, and the runners skip any cell
that already has a JSON. `--force` makes them recompute with your NEW checkpoint. The defense
checkpoint is auto-loaded from `models/checkpoints/defense_final.pt`
(or set `export DEFENSE_CHECKPOINT_PATH=/abs/path/to/defense_final.pt`).

```bash
python scripts/run_gradient_benchmark.py    --models defense_resnet50 --force
python scripts/run_transfer_benchmark.py    --models defense_resnet50 --force
python scripts/run_typographic_benchmark.py --models defense_resnet50 --force
python scripts/run_corruptions_benchmark.py --models defense_resnet50 --force
```

These write, in the **identical schema/location** as every other model:
- `results/gradient/defense_resnet50__{fgsm,pgd,autoattack,square}.json`   (4)
- `results/transfer/defense_resnet50.json`                                  (1)
- `results/typographic/defense_resnet50.json`                               (1)
- `results/corruptions/defense_resnet50__{15 corruption types}.json`        (15)

→ **21 JSON files total.**

## Step 5 — Verify before pushing

```bash
ls results/gradient/defense_resnet50__*.json results/transfer/defense_resnet50.json \
   results/typographic/defense_resnet50.json results/corruptions/defense_resnet50__*.json | wc -l
# Expect 21.

python - <<'PY'
import json
g = json.load(open("results/gradient/defense_resnet50__pgd.json"))
print("clean_acc=", g["clean_accuracy"], " pgd_robust=", g["robust_accuracy"], " dataset_size=", g["dataset_size"])
assert g["dataset_size"] == 1000, "eval must be the 1000-class benchmark"
PY
```
Sanity: `clean_accuracy` should be ~0.6–0.7 (NOT 0.06), `dataset_size` 1000, and every JSON should
carry the same keys as the corresponding baseline file (e.g. compare against
`results/gradient/resnet50__pgd.json`). Then refresh the per-axis CSV/REPORT aggregates:

```bash
python scripts/build_gradient_report.py
python scripts/build_transfer_report.py
python scripts/build_typographic_report.py
python scripts/build_corruptions_report.py
```

## Step 6 — Push the results back to GitHub

```bash
git pull --rebase                      # avoid clobbering anything the local side pushed
git add results/                       # JSONs + regenerated CSVs/REPORTs only
git status                             # confirm NO *.pt and NO large dataset files are staged
git commit -m "defense_resnet50: re-trained on 1000 classes, full benchmark results"
git push
```

**Do not commit:** model checkpoints (`*.pt`), the regenerated datasets under `data/`, or the PAT.
If any are staged, unstage them.

## Success criteria

- [ ] `defense_final.pt` trained with `clean_acc` in the tens of percent (no 6% collapse).
- [ ] 21 `defense_resnet50` JSONs present, schema-identical to the baselines, `dataset_size=1000`.
- [ ] JSONs (and refreshed CSVs/REPORTs) committed and pushed to GitHub.
- [ ] Report back the headline defense numbers (clean acc, PGD-20 robust acc, AutoAttack robust acc)
      so the human can confirm it finally beats the baselines on robustness.

After you push, the human pulls locally and regenerates the combined report
(`generate_benchmark_report.py` with `INCLUDE_DEFENSE = True`) to fold the defense back in as the
8th model.
