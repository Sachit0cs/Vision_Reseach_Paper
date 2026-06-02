# Research & Codebase Audit — Adversarial-Robustness Benchmark

_Prepared 2026-05-29 for Sachit Jain. Method: a `/council` of 4 independent Opus critics (code-correctness, experimental-methodology, results-reality-check, internal-consistency), each read-only over the full repo, plus direct verification of every load-bearing claim against `file:line` and the raw JSONs by the coordinating agent. Where a claim could not be confirmed from the artifacts, it is marked **[UNVERIFIED]** rather than asserted._

> **One-line bottom line:** You have built a genuinely solid, reproducible, and — at its sharp end — _honest_ benchmark, and you have **one real, paper-worthy finding (the CLIP typographic inversion, TASR 34.2 %)**. But the headline contribution you were chasing — a defense that beats the baselines — **does not exist yet**: the defense model is a failed training run scoring 6 % clean accuracy, and its failure is **mis-diagnosed** in your own write-up. Three of the four other axes are correct-but-confirmatory, and the corruptions axis fails its own validity check.

---

## 0. Did the research actually accomplish anything? (Honest answer)

**Yes, but less than the project framing implies. In plain terms:**

- ✅ **Real infrastructure.** A correctly-engineered 7-model × 4-axis benchmark whose every reported number traces back to a real per-cell JSON. The critics spot-checked numbers across all five reports and found **no evidence of hand-editing or fabrication.** This is real, reusable, trustworthy scientific plumbing.
- ✅ **One real finding** worth a paper hook: the **CLIP typographic inversion** (CLIP drops 25.3 %, TASR 34.2 %, vs ~1.6 % for the six pure classifiers). It is backed by the JSONs and it inverts your own stated hypothesis. _Caveat:_ it is an **extension** of a known phenomenon (Goh et al. 2021), not a discovery, and the protocol is accidentally favorable to the finding (see Flaw C-1).
- 🟡 **Three confirmatory axes** (gradient collapse, transfer CNN→Transformer pattern, corruption fragility): correct and well-measured, but they re-demonstrate things the field already knows. Publishable as *supporting* tables, not as contributions.
- ❌ **The defense did not work.** Clean accuracy collapsed to 6 %. Every "win" is a floor-effect artifact, and your own RESULTS.md says so. This was meant to be _the_ contribution ("beat the baselines"); right now there is nothing to claim.

**So:** real, honest infrastructure + one real (known-phenomenon) finding + one failed defense. That is **enough for a university-scope / workshop paper built around the CLIP hook** — it is **not** enough to claim a working defense, and not enough to claim a novel discovery.

---

## 1. Where the research has done well (genuine strengths)

These are real and you should not second-guess them:

1. **The `[0,1]` pixel-space contract is honored end-to-end.** Every classifier's `preprocess` ends at `ToTensor()` (no normalization), and normalization happens inside `logits()`; attacks operate and clamp in `[0,1]`. The critics traced this through all 7 models and all attacks and found **no double-normalization anywhere** — the single most common silent bug in adversarial-robustness code, and you avoided it. (`models/classifiers.py:80-86,144-154`, `attacks/gradient.py:54-55,135-136`)
2. **ε = 8/255 means the same physical perturbation for every model** because of the above. This is done right.
3. **CLIP logit handling is correct** — L2-normalized image/text features, `logit_scale.exp()` scaling, differentiable path preserved; the clean class-name string used by CLIP matches the typographic overlay source, so TASR alignment is sound. (`models/classifiers.py:132,147-154`)
4. **Seeded, class-balanced subset selection** avoids the naive "classes 0..n-1" truncation bias — a correctness detail most students miss. (`datasets/loader.py`, `datasets/sampler.py`)
5. **You used the real tools, not hand-rolled substitutes:** the official `autoattack` standard ensemble (APGD-CE/APGD-T/FAB-T/Square) and a genuine black-box `Square` probe. The **AutoAttack ≤ PGD** sanity check passes for all 7 models. (`attacks/gradient.py:141,164-179`)
6. **The transfer harness is validated:** ResNet-50 self-transfer (0.003) matches white-box PGD (0.004) within tolerance, which proves the plumbing is correct. (`results/transfer/REPORT.md:33`)
7. **A NaN-output diagnostic exists** — you explicitly anticipated the "NaN looks robust" failure mode (even though the current check can't catch the *sign* variant — see Flaw B-1).
8. **Reproducibility scaffolding is strong:** global seed threaded through every runner, seeded PGD random start, per-cell JSON resumption units, host-env capture in result files, library versions recorded.
9. **Intellectual honesty at the sharp end.** `results/defense/RESULTS.md` is the most honest document in the repo: it leads with "the defense model never converged… clean accuracy = 6 %," labels every apparent win an artifact, reports the losses bluntly, and refuses to cherry-pick. The corruptions report prints its own sanity check as **"FAIL"** rather than hiding it. This discipline is exactly what separates a paper from a project — keep it.

---

## 2. Critical flaws (research-integrity level — fix before any claim)

### C-1. The defense is evaluated on a 1000-class set but was trained on only 100 classes → the "6 % clean collapse" is largely a label-space artifact, and your own diagnosis of *why* is wrong.

This is the single most important finding of the whole audit, and **all of it is verified**:

- The defense was fine-tuned on the first **100** ImageNet WNIDs (`datasets/loader.py:404`, `selected = available[:num_classes]`), but each image keeps its **full 1000-class index** as its label (`loader.py:409`, `label = wnid_to_idx[wnid]`), and the loss is cross-entropy over the full **1000-way** head (`scripts/train_defense.py:99-100`).
- Every defense eval JSON has **`"dataset_size": 1000`** with indices `0…999` — the full 1000-class, 1-image-per-class benchmark (verified directly in `results/gradient/defense_resnet50__pgd.json:6`).
- **Consequence:** ~900 of the 1000 eval images belong to classes the model was never trained to output. Clean accuracy is **capped near 100/1000 = 10 %** *before any adversarial degradation.* The observed 6 % (60/1000) is consistent with the model doing real work on its own ~100 classes (≈60 % within-distribution) while being graded against a 10× larger label space.
- **Your RESULTS.md (`results/defense/RESULTS.md:76`) blames the collapse entirely on "8 epochs from scratch… need ≥100 epochs." That is a mis-diagnosis.** The dominant cause is the train/eval **label-space mismatch**, not the epoch budget. The "fix" RESULTS.md proposes (`:82`, "initialize from pretrained") will **not** fix this on its own.
- This also **violates your own non-negotiable rule #4** ("use the same eval protocol… otherwise the comparison is unfair," `plan.md:59`) and means **Gate C was never actually passed** (it required clean ≥ 55 % / PGD ≥ 5 % on the *trained* classes — `plan.md:611-612`), yet Phase D ran anyway.

**Fix:** evaluate the defense on the **100 trained classes only** (or report within-distribution accuracy), and re-run. Until then, no defense number means anything.

### C-2. The defense training run contradicts the committed code about its own most basic fact (scratch vs. pretrained).

- `results/defense/RESULTS.md:8` says the model was "trained **from scratch**."
- But the committed training script **loads pretrained ImageNet-1k weights**: `build_classifier("resnet50")` with the comment "TorchVisionClassifier loads pretrained ResNet-50" (`scripts/train_defense.py:194-195`, verified).
- So either RESULTS.md is mislabeled, or the actual Kaggle run used an uncommitted edit. **[UNVERIFIED]** — I cannot tell which from the artifacts, and the checkpoint (`defense_final.pt`) is gitignored / Kaggle-local so I can't inspect the loaded weights.
- Note: the epoch-1 *adversarial* accuracy of ~4.5 % is **not** by itself evidence of scratch-init — a pretrained model already has ~0 % accuracy on its own PGD adversarials, so a low early adv-acc is expected either way. This is exactly why the scratch-vs-pretrained question can't be settled from the loss curve alone.

**Why it matters:** the paper cannot describe its own method correctly until this is resolved. Decide it, then make the code, config, and write-up agree.

### C-3. There is no salvageable robustness claim from the defense — it is a null result, and it must be labeled as one everywhere, not just in RESULTS.md.

- The only positive deltas (PGD 0.015 vs 0.004; AA 0.005 vs 0.000; Square 0.015 vs 0.010) are **≤ 1.1 pp = 1–11 images** on a model sitting at floor — statistically indistinguishable from zero. Your RESULTS.md correctly calls these "noise at floor."
- **But** the defense rows have been merged into the *frozen baseline* accuracy tables (`results/gradient/accuracy_table.csv`, `results/transfer/…`, `results/typographic/…`, `results/corruptions/…`) **without any inline "INVALID — clean collapse" flag**, and commit `06835c7` is titled *"Add defense_resnet50 results across all 4 benchmark axes."* A reader consuming the CSVs or the commit log — without opening RESULTS.md — would mistake a failed run for a completed, valid result.

**Fix:** either pull the defense rows out of the frozen CSVs until the re-run, or add an explicit invalidity flag column. The honesty in RESULTS.md needs to travel with the numbers.

---

## 3. Major flaws (will draw referee/examiner fire)

### B-1. Silent NaN no-op in FGSM/PGD can fabricate robustness for any future unstable model.

`attacks/gradient.py:133` (and FGSM) step by `grad.sign()`. Because `torch.sign(NaN) == 0`, if `logits` ever produces a NaN at a perturbed point, the step is zero, the image is unchanged, and the model looks **perfectly robust**. The existing diagnostic only checks `isnan` on the *output image* (which stays finite: `image + 0`), so it **cannot detect this**. Not currently triggered (all baselines are at ~0 %), but it is a latent landmine — e.g. any fp16 or unstable defense model could silently report false robustness. **Fix:** assert finite gradients/logits inside the attack loop.

### B-2. The CLIP typographic comparison is confounded by an exact-string-match channel built into the protocol.

- The overlay text uses `text_form: "lowercase_first_synonym"`; CLIP's class prototypes are built from `c.split(",")[0].strip().lower()` (`models/classifiers.py:132`) — i.e. the sticker text and CLIP's label string are **token-identical by construction.**
- TASR rewards predicting the *exact* class named on the sticker, so CLIP has a direct string-collision path the 6 supervised models structurally cannot have. This is *partly the point* of a typographic attack, but as a controlled experiment it is **maximally favorable** to the finding.
- The report gives only **absolute** drops; CLIP's clean baseline (0.621) is lower than the supervised models' (~0.69–0.79), so the "16.3×" framing is inflated relative to a **relative-drop** metric — which COUNCIL_PLAN.md's own rule (`:97`) demanded.

**Fix (cheap, high-value):** add control overlay phrasings (full synset string, a different synonym, a paraphrase) and confirm the effect survives; report relative drop alongside absolute. This converts the finding from "looks engineered" to "robust."

### B-3. The corruptions axis fails its own pre-registered validity window and is not canonical ImageNet-C.

- Severity-3 mean drop is **0.263**, outside the literature window `[0.05, 0.25]` you set — `results/corruptions/REPORT.md:31` prints **"FAIL"** in plain text.
- 4 of 15 corruptions (`motion_blur`, `snow`, `frost`, `fog`) are home-rolled NumPy approximations, not the canonical Wand/ImageMagick versions (`attacks/corruptions.py:21-28`), so **absolute numbers are not comparable to any published ImageNet-C result.** Additionally, `glass_blur` is an O(H·W) Python loop doing in-place swaps during traversal, so it is not the canonical Hendrycks glass-blur either (`attacks/corruptions.py:112-124`).
- `plan.md:602` re-labels this self-printed FAIL as "[~]… not a correctness fail." The justification is disclosed and partly reasonable, but **calling your own printed "FAIL" a non-fail is exactly what an examiner notices.**

**Fix:** frame this axis as "ImageNet-C-**style**, not ImageNet-C"; cut any sentence comparing absolute numbers to literature; report **model rankings** (which are usable) rather than absolute severities. Cheap win: report the 11 faithful corruptions separately and check whether *that* subset lands in-window.

### B-4. Single seed, no confidence intervals, strongest attacks on only 200 images.

- AutoAttack and Square — the two attacks that actually certify "all models → 0 %" — run on only **200 images** (`config.yaml:42-43`). The defense's per-axis "wins" are 1–11 images. No error bars, no multi-seed anywhere; CIs are a stretch goal (`plan.md:658`).
- **Nuance:** for the *baseline* gradient collapse this is fine — the effect is a saturated floor (→ 0.000) no CI would lift. For the **defense** and any close call, single-seed numbers carry **no statistical weight.** Be explicit about which claims rest on floor effects (safe) vs. small margins (not safe).

### B-5. AutoAttack/Square clean accuracy is measured on a *different* sample than FGSM/PGD.

FGSM/PGD use the 1000-image set; AutoAttack/Square use a 200-image subset (clean acc 0.755 vs 0.781 for ResNet-50, verified). Each per-cell number is internally valid, but the **gradient-masking gate subtracts a 200-image Square robust-acc from a 1000-image PGD robust-acc** — not apples-to-apples, and not disclosed in the gate. Harmless at floor, but state it.

### B-6. No adaptive attack, no RobustBench baseline — and the configured AT reference model was never run.

`Salman2020Do_R50` is configured as a "robustness-ceiling reference" (`config.yaml:55-59`) but **never evaluated**, so there is no calibration of what a *real* AT ResNet-50 scores under your exact harness — meaning even a working defense couldn't be contextualized. For the explicitly university-scoped paper this is **defensible only if the paper makes no defense-effectiveness claim** (which, per §2, it currently can't anyway). For any workshop/SaTML venue it is an auto-reject (COUNCIL_PLAN.md:94). Decide your venue, then decide if this is in scope.

---

## 4. Minor flaws / cleanups

- **M-1. Dead, mis-advertised metrics.** `metrics/classification.py` advertises CDS / softmax-KL / label-flip-rate in its docstring but they are `NotImplementedError` stubs (`:58-67`); the runners compute their metrics inline instead. Dead code that misrepresents capability. (The metrics that *are* used — clean/robust accuracy, fooling rate conditioned on originally-correct samples, TASR — are correct.)
- **M-2. The Phase-2 abstraction layer was bypassed.** `evaluation/pipeline.py`, `evaluation/protocols.py`, `reporting/tables.py`, `reporting/visualizer.py`, `attacks/patch.py`, `scripts/run_benchmark.py` are still stubs; the real work runs through standalone per-axis scripts. Fine pragmatically, but the repo structure advertises an architecture that isn't wired up.
- **M-3. Inner-loop BatchNorm mismatch in defense training.** PGD inner-max runs in `model.eval()` (frozen BN stats) while the outer step runs in `model.train()` (`train_defense.py:88,97`). Standard Madry keeps `train()` throughout for consistent BN. A secondary contributor to the collapse.
- **M-4. Two divergent training logs, both mislabeled.** `training_log.json` (8 epochs) and `training_log_v1.json` (3 epochs) both carry `config.epochs: 3` while the active run did 8; `train_dir` in the log doesn't match `config.yaml`. Reproducibility ambiguity.
- **M-5. Determinism not fully pinned.** Seeds are set, but `cudnn.deterministic`/`benchmark` are never configured and DataLoader uses `num_workers=2`, so exact bit-reproducibility on GPU is not guaranteed. **[UNVERIFIED run-to-run variance]** — I could not run the GPU pipeline.
- **M-6. Tests are mostly smoke tests** — they exercise shapes/plumbing more than asserting numerical correctness. **[Partially verified]** — flagged by the code critic; not exhaustively re-read.

---

## 5. Inconsistencies (cross-document — these make the project tell more than one story)

| # | Inconsistency | Sources | Severity | Which is true |
|---|---|---|---|---|
| I-1 | **Scope: "university-tier, NOT workshop"** vs **"workshop-tier; TRADES/adaptive/RobustBench non-negotiable."** | `plan.md:5,24-34` vs `COUNCIL_PLAN.md:4,94-95` | LOW (deliberate) | `plan.md` governs — it calls COUNCIL_PLAN "the strategic backstory" (`:3`). But COUNCIL_PLAN has **no "SUPERSEDED" banner**, so a reader landing there inherits the wrong bar. Add the banner. |
| I-2 | **Defense method:** plan says "take **pretrained** ResNet-50, **fine-tune**, **3 epochs**, expect **55–70 %** clean." Execution: "**from scratch**, **8 epochs**, **6 %** clean." | `plan.md:258-262,272,413` vs `RESULTS.md:8,13` | HIGH (as tracking doc) | RESULTS.md describes what happened; plan was never updated. plan.md still marks Phase C **"← NEXT"** (`:42`) and Gates C/D unchecked, though the defense is already trained & committed. Also conflicts with the committed code, which loads pretrained (see C-2). |
| I-3 | **Gates C & D were failed but treated as shipped.** Gate C needs clean ≥ 55 % / PGD ≥ 5 %; actual 6 % / 1.5 %. Gate D needs ≥ 3 real axis wins; actual ≈ 0. | `plan.md:610-617` vs `RESULTS.md:13,62-67`, `defense_resnet50__pgd.json` | MEDIUM | Gates failed. The commit message + merged CSV rows package it as a completed result (see C-3). |
| I-4 | **Self-printed "FAIL" re-labeled "not a correctness fail."** | `corruptions/REPORT.md:31` vs `plan.md:602` | LOW-MED | Borderline-defensible spin; the raw FAIL + cause are left visible, so not *hidden*, but the re-interpretation lives in the plan. |
| I-5 | **READMEs are badly stale.** Sub-README says attacks are "pending / `raise NotImplementedError` stubs" and `train_defense.py` is "not yet implemented" — all false now. Top-level `README.md` is a garbled BOM-corrupted placeholder ("Vision_Rese**a**ch_Paper", misspelled). | `adversarial-robustness-benchmark/README.md:13-24,87-88`; `README.md` | MEDIUM | Repo state (implemented) is correct; both READMEs misinform. Rewrite both. |
| I-6 | **Eval-set naming:** plan says defense evaluated on "the existing **1000-image set** / same images." RESULTS.md calls it an **"ImageNet-100 val subset"** (and two lines later "1000-image subset" — internally inconsistent). Truth: the JSONs show the **1000-class** benchmark. | `plan.md:34,59` vs `RESULTS.md:7,9` vs `defense_*.json` | HIGH | The JSONs are ground truth: 1000-class. RESULTS.md's "ImageNet-100 val subset" label is simply **wrong per its own data** — and this wrong label hides the C-1 mismatch. |
| I-7 | **Stale auto-memory.** My stored project memory says "workshop-tier… defense path = TRADES ResNet-50." Reality: university-tier, **plain Madry PGD, no TRADES**. | (assistant memory) vs `plan.md:5,33` | LOW | I will correct my memory (below). The *code* is consistently Madry — no stale TRADES code path exists. |

**Note on what is _consistent_:** the Madry-vs-TRADES story is clean — docstrings, config, and implementation all agree it's plain PGD adversarial training, with no orphaned TRADES code. And the cross-report clean-accuracy differences (e.g. vgg16 0.687 in gradient vs 0.695 in transfer) are **real but explained**: transfer evaluates all models on ResNet-50's preprocessing pipeline, which the transfer report discloses (`transfer/REPORT.md:12`). Not a fabrication.

---

## 6. Per-axis results verdict

| Axis | Verdict | Evidence | Usable for the paper? |
|---|---|---|---|
| **Gradient** (FGSM/PGD/AA/Square) | **REAL, unsurprising** | All 7 → ~0 % under PGD/AA/Square; AA ≤ PGD holds; no gradient masking; numbers trace to JSONs | Yes — as a *confirmatory* baseline table |
| **Typographic** | **REAL — the one genuine hook** | CLIP TASR 0.342 vs pure mean ~0.001; fully backed by JSONs; inverts the stated hypothesis | Yes — **lead with it**, after fixing the protocol confound (B-2) |
| **Corruptions** | **WEAK / off-calibration** | 105 cells internally consistent, but sanity **FAIL** (0.263 > 0.25) and 4/15 are NumPy approximations | Rankings only; **cut absolute-number literature comparisons** |
| **Transfer** | **REAL, modest** | Self-transfer 0.003 ≈ PGD 0.004 validates harness; CNN→Transformer pattern holds | Yes — *confirmatory* |
| **Defense** | **NULL / ARTIFACT** | Clean 6 %; all "wins" are floor noise; trained on 100 classes, scored on 1000 (C-1) | **No** — must re-run before any claim |

---

## 7. What I could NOT verify (per your "skip, don't guess" instruction)

- **Whether the actual Kaggle run loaded pretrained or scratch weights** (C-2). The committed code loads pretrained; RESULTS.md says scratch; the checkpoint is gitignored, so the loaded `state_dict` can't be inspected here.
- **Run-to-run / multi-seed variance** — I did not execute the GPU pipeline; `cudnn` determinism is unset (M-5).
- **The exact cause of the corruptions window miss** — no ablation isolates the 4 approximated corruptions from the 11 faithful ones, so the "sampling + approximation" attribution is plausible but unproven.
- **CLIP relative-drop magnitude** — I did not recompute it; structurally I expect "16.3×" to shrink under a relative metric (B-2).
- **Whether the binary figures (PNG/PDF) plot the same numbers as the JSONs** — only that the files exist.
- **Test-suite rigor** — sampled, not exhaustively re-read (M-6).
- **Whether a paper draft preserves the honest framing** — no draft exists in the repo yet (Phase E pending), so I can only confirm the *supporting docs* intend honesty.

---

## 8. Prioritized fix list

**Before any defense claim (blocking):**
1. **Re-evaluate the defense on its 100 trained classes** (fix C-1), and **decide scratch-vs-pretrained** and make code/config/write-up agree (C-2). Per the math, a pretrained fine-tune evaluated in-distribution should land clean > 60 % / PGD 15–30 % — the regime where the other axes become comparable instead of floor-dominated.
2. **Flag or remove the invalid defense rows** from the frozen CSVs (C-3).

**Before submission (high-value, cheap):**
3. Add **control overlay phrasings + relative-drop reporting** to harden the CLIP finding (B-2).
4. Re-frame corruptions as "ImageNet-C-**style**," cut literature-comparison sentences, report the 11 faithful corruptions separately (B-3).
5. Add a **finite-gradient assert** in the attack loop (B-1).

**Hygiene:**
6. Rewrite both READMEs; add a "SUPERSEDED" banner to COUNCIL_PLAN.md; sync `plan.md` Phase-C/D status and gates to reality; reconcile the two training logs and `config.yaml` (I-1…I-6, M-4).

---

### Council provenance

<details>
<summary>4 Opus critics + coordinator verification (click to expand)</summary>

- **Code-correctness auditor** → "Non-defense pipeline is contract-clean and trustworthy; the defense rests on a real training bug (1000-way head trained on 100-class labels, no clean loss, eval-mode BN), which RESULTS.md misdiagnoses as an epoch-budget problem; latent NaN-sign no-op could fabricate robustness for future unstable models."
- **Experimental-methodology referee** → "Reject (as a defense paper) / major-revision (as a benchmark paper). Biggest objection: defense trained on 100 classes, evaluated on 1000 → headline numbers are label-space artifacts; combined with from-scratch/unconverged/single-seed, no salvageable defense claim. Baseline benchmark + controlled CLIP finding is publishable at university scope."
- **Results reality-check auditor** → "Trustworthy benchmark with exactly one real finding (CLIP TASR 34.2 %); the rest is confirmatory or, for the defense, an honestly-reported failed run — real and unfabricated, but thin. Every spot-checked report number traces to a real JSON; no hand-editing."
- **Internal-consistency auditor** → "Honest at its sharp end (RESULTS.md), but does not yet tell one coherent story: plan still marks the failed defense 'NEXT/pending' with unchecked gates it actually failed; commit message + merged CSVs package the collapse as 'results across all 4 axes'; both READMEs falsely claim attacks are unimplemented stubs; real root cause (100-train-class scored on 1000-class eval, mislabeled 'ImageNet-100 val subset') misdiagnosed everywhere."
- **Coordinator** independently verified: `train_defense.py:194-195` (pretrained load), `loader.py:404-417` (100 WNIDs, 1000-class labels), `train_defense.py:88-100` (eval-mode BN inner loop, no clean loss), `gradient.py:133` (sign no-op), and `defense_resnet50__pgd.json:6` (`dataset_size: 1000`).

</details>
