"""CLI: build the accuracy table, figures, and REPORT.md for the typographic run.

Inputs : results/typographic/<model>.json  (written by run_typographic_benchmark.py)
         data/poisoned/typographic/manifest.json
         data/poisoned/typographic/images/*.jpg
         data/clean/images/*.jpg                 (for clean/poisoned side-by-side)
Outputs:
  * results/typographic/accuracy_table.csv
  * results/typographic/figures/*.png
  * results/typographic/REPORT.md
  * results/typographic/sanity_checks.json
"""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torchvision

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

MODELS = [
    "resnet50",
    "vgg16",
    "convnext_tiny",
    "vit_b_16",
    "swin_t",
    "efficientnet_b0",
    "clip_vit_b16",
]

TYPO_DIR = os.path.join(_REPO_ROOT, "results", "typographic")
FIG_DIR = os.path.join(TYPO_DIR, "figures")
POISONED_ROOT = os.path.join(_REPO_ROOT, "data", "poisoned", "typographic")
CLEAN_IMG_DIR = os.path.join(_REPO_ROOT, "data", "clean", "images")


def load_results() -> dict:
    out: dict[str, dict] = {}
    for m in MODELS:
        p = os.path.join(TYPO_DIR, f"{m}.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                out[m] = json.load(f)
    return out


def fmt(x, places=3):
    return "—" if x is None else f"{x:.{places}f}"


def build_accuracy_table(results: dict) -> list[list]:
    header = ["model", "clean_accuracy", "typographic_robust_accuracy",
              "accuracy_drop", "fooling_rate",
              "targeted_attack_success_rate"]
    rows = [header]
    for m in MODELS:
        r = results.get(m)
        if not r:
            rows.append([m, None, None, None, None, None])
            continue
        drop = r["clean_accuracy"] - r["robust_accuracy"]
        rows.append([
            m,
            r["clean_accuracy"],
            r["robust_accuracy"],
            drop,
            r["fooling_rate"],
            r["targeted_attack_success_rate"],
        ])
    return rows


def write_csv(rows: list[list], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow([f"{v:.4f}" if isinstance(v, float) else ("" if v is None else v)
                        for v in row])


def render_markdown_table(rows: list[list]) -> str:
    pretty = ["Model", "Clean", "Robust", "Drop", "Fooling rate", "TASR"]
    out = ["| " + " | ".join(pretty) + " |",
           "| " + " | ".join(["---"] * len(pretty)) + " |"]
    for row in rows[1:]:
        out.append("| " + " | ".join([row[0]] + [fmt(v) for v in row[1:]]) + " |")
    return "\n".join(out)


def chart_clean_vs_robust(results: dict, path: str) -> None:
    """Bar chart: per-model clean vs typographic-robust accuracy."""
    have = [m for m in MODELS if m in results]
    order = sorted(have, key=lambda m: -results[m]["clean_accuracy"])

    clean = [results[m]["clean_accuracy"] for m in order]
    robust = [results[m]["robust_accuracy"] for m in order]
    x = np.arange(len(order))
    w = 0.4

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, clean, width=w, label="clean", color="#2a9d8f")
    ax.bar(x + w / 2, robust, width=w, label="typographic (robust)", color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title("Clean vs typographic-robust accuracy per model")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart_drop_and_tasr(results: dict, path: str) -> None:
    """Two-panel chart: accuracy drop and TASR per model."""
    have = [m for m in MODELS if m in results]
    order = sorted(have, key=lambda m: -(results[m]["clean_accuracy"] - results[m]["robust_accuracy"]))
    drop = [results[m]["clean_accuracy"] - results[m]["robust_accuracy"] for m in order]
    tasr = [results[m]["targeted_attack_success_rate"] for m in order]
    x = np.arange(len(order))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(x, drop, color="#f4a261")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order, rotation=20, ha="right")
    axes[0].set_ylim(0, max(drop) * 1.15 if drop else 1.0)
    axes[0].set_ylabel("clean − robust")
    axes[0].set_title("Accuracy drop  (sorted desc)")
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)
    for xi, v in zip(x, drop):
        axes[0].annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                         xytext=(0, 4), ha="center", fontsize=9)

    axes[1].bar(x, tasr, color="#e76f51")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order, rotation=20, ha="right")
    axes[1].set_ylim(0, max(tasr) * 1.15 if tasr and max(tasr) > 0 else 1.0)
    axes[1].set_ylabel("P(predict overlay class)")
    axes[1].set_title("Targeted-attack success rate (TASR)")
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)
    for xi, v in zip(x, tasr):
        axes[1].annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                         xytext=(0, 4), ha="center", fontsize=9)

    fig.suptitle("Typographic attack — per-model accuracy drop and TASR")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart_clip_vs_others(results: dict, path: str) -> None:
    """Highlight CLIP vs the rest — the central paper finding."""
    pure = [m for m in MODELS if m in results and m != "clip_vit_b16"]
    clip_present = "clip_vit_b16" in results
    if not pure and not clip_present:
        return

    pure_drop = [results[m]["clean_accuracy"] - results[m]["robust_accuracy"] for m in pure]
    pure_avg = float(np.mean(pure_drop)) if pure_drop else 0.0
    clip_drop = (results["clip_vit_b16"]["clean_accuracy"]
                 - results["clip_vit_b16"]["robust_accuracy"]) if clip_present else None

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Pure classifiers (mean)"] + (["CLIP zero-shot"] if clip_present else [])
    values = [pure_avg] + ([clip_drop] if clip_present else [])
    colors = ["#264653"] + (["#e9c46a"] if clip_present else [])
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("clean − robust  (accuracy drop)")
    ax.set_title("Typographic accuracy drop: language-grounded vs pure classifiers")
    for i, v in enumerate(values):
        ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=11)
    ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart_examples(path: str) -> None:
    """3x2 grid: 3 (clean, poisoned) pairs sampled from the manifest."""
    manifest_path = os.path.join(POISONED_ROOT, "manifest.json")
    if not os.path.exists(manifest_path):
        return
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Pick 3 distinct true classes.
    seen = set()
    chosen = []
    for rec in manifest["images"]:
        if rec["label"] in seen:
            continue
        chosen.append(rec)
        seen.add(rec["label"])
        if len(chosen) == 3:
            break

    fig, axes = plt.subplots(3, 2, figsize=(6, 8))
    for row, rec in enumerate(chosen):
        clean_path = os.path.join(CLEAN_IMG_DIR, os.path.basename(rec["filename"]))
        poison_path = os.path.join(POISONED_ROOT, rec["filename"])
        if os.path.exists(clean_path):
            axes[row][0].imshow(Image.open(clean_path).convert("RGB"))
        axes[row][0].set_title(
            f"clean: {rec['true_class_name'].split(',')[0][:20]}", fontsize=9
        )
        if os.path.exists(poison_path):
            axes[row][1].imshow(Image.open(poison_path).convert("RGB"))
        axes[row][1].set_title(
            f"typo overlay: \"{rec['overlay_text'][:18]}\"", fontsize=9
        )
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle("Typographic examples (clean | poisoned)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def sanity_checks(results: dict) -> dict:
    """Surface a few cross-model invariants the report can cite."""
    out: dict = {}

    # 1. Per-model: typographic ≠ identity transform (some accuracy loss expected).
    per_model = {}
    for m, r in results.items():
        drop = r["clean_accuracy"] - r["robust_accuracy"]
        per_model[m] = {
            "clean": r["clean_accuracy"],
            "robust": r["robust_accuracy"],
            "drop": drop,
            "passed": drop > 0.0,
        }
    out["nonzero_drop_per_model"] = per_model

    # 2. Pure-vs-language-grounded gap.
    pure = [m for m in results if m != "clip_vit_b16"]
    if pure and "clip_vit_b16" in results:
        pure_drop = float(np.mean([results[m]["clean_accuracy"] - results[m]["robust_accuracy"]
                                   for m in pure]))
        clip_drop = (results["clip_vit_b16"]["clean_accuracy"]
                     - results["clip_vit_b16"]["robust_accuracy"])
        out["clip_vs_pure"] = {
            "pure_mean_drop": pure_drop,
            "clip_drop": clip_drop,
            "clip_resists_more": bool(clip_drop < pure_drop),
        }

    # 3. Coverage: all 7 models reported.
    out["models_reported"] = sorted(results.keys())
    out["all_models_present"] = len(results) == len(MODELS)
    return out


def wall_clock_summary(results: dict) -> dict:
    total = 0.0
    per_model = {}
    for m, r in results.items():
        s = sum(r["wall_clock_s"].values())
        per_model[m] = s
        total += s
    return {"total_s": total, "per_model_s": per_model}


def library_versions() -> dict:
    versions = {"torch": torch.__version__, "torchvision": torchvision.__version__}
    for name in ["transformers", "PIL"]:
        try:
            if name == "PIL":
                import PIL
                versions["pillow"] = PIL.__version__
            else:
                mod = __import__(name)
                versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = "unavailable"
    return versions


def per_model_paragraph(results: dict) -> str:
    if not results:
        return "_no per-model JSONs found_"
    have = sorted(results, key=lambda m: -(results[m]["clean_accuracy"] - results[m]["robust_accuracy"]))
    worst = have[0]
    rw = results[worst]
    lines = [
        f"Across the {len(have)} evaluated models the typographic overlay "
        f"caused an average accuracy drop of "
        f"**{np.mean([results[m]['clean_accuracy'] - results[m]['robust_accuracy'] for m in have]):.2f}** "
        f"absolute. The most affected model was `{worst}`, dropping from "
        f"**{rw['clean_accuracy']:.2f}** clean to **{rw['robust_accuracy']:.2f}** robust "
        f"(drop **{rw['clean_accuracy'] - rw['robust_accuracy']:.2f}**, "
        f"TASR **{rw['targeted_attack_success_rate']:.2f}**)."
    ]
    if "clip_vit_b16" in results:
        c = results["clip_vit_b16"]
        lines.append(
            f"`clip_vit_b16` (zero-shot, language-grounded) recorded clean "
            f"**{c['clean_accuracy']:.2f}** → robust **{c['robust_accuracy']:.2f}** "
            f"(drop **{c['clean_accuracy'] - c['robust_accuracy']:.2f}**, "
            f"TASR **{c['targeted_attack_success_rate']:.2f}**). Compare this "
            f"directly with the pure-classifier average — that gap is the paper's "
            f"language-grounding finding."
        )
    return " ".join(lines)


def make_report(results, csv_path, fig_paths, sanity, wall, run_started, start_time) -> str:
    rows = build_accuracy_table(results)
    table_md = render_markdown_table(rows)
    versions = library_versions()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    elapsed = (time.time() - start_time) / 60.0

    # Sanity-check markdown.
    sc_lines = []
    if "nonzero_drop_per_model" in sanity:
        sc_lines.append("- **Typographic overlay causes a non-zero accuracy drop on every model.**")
        for m, v in sanity["nonzero_drop_per_model"].items():
            sc_lines.append(
                f"  - `{m}`: clean **{v['clean']:.3f}** → robust **{v['robust']:.3f}** "
                f"(drop **{v['drop']:.3f}**) → "
                f"**{'PASS' if v['passed'] else 'FAIL'}**."
            )
    if "clip_vs_pure" in sanity:
        v = sanity["clip_vs_pure"]
        sc_lines.append(
            f"- **CLIP resists typographic more than the pure-classifier mean.**  "
            f"pure mean drop **{v['pure_mean_drop']:.3f}** vs CLIP drop "
            f"**{v['clip_drop']:.3f}** → "
            f"**{'PASS' if v['clip_resists_more'] else 'FAIL'}**."
        )
    sc_lines.append(
        f"- **Coverage**: {len(sanity['models_reported'])} / {len(MODELS)} models reported. "
        f"**{'PASS' if sanity['all_models_present'] else 'FAIL'}**."
    )

    figures_md = "\n".join(
        f"![{caption}]({os.path.relpath(p, TYPO_DIR)})  \n*{caption}*"
        for caption, p in fig_paths.items() if p and os.path.exists(p)
    )

    typo_cfg = {}
    manifest_path = os.path.join(POISONED_ROOT, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            typo_cfg = json.load(f).get("config", {})

    lines = [
        "# Typographic-Attack Benchmark — Report",
        "",
        f"_Run started: {run_started}.  Author: Sachit Jain._",
        "",
        "This report covers Phase 1, Axis C of the project — a single semantic, "
        "model-agnostic attack (the typographic overlay) evaluated against the 7 "
        "ImageNet baselines listed in `config.yaml`. The poisoned dataset is "
        "generated once by `scripts/generate_datasets.py --typographic` and "
        "shared across all models; this report aggregates the per-model JSONs "
        "in this directory.",
        "",
        "## 1. Setup",
        "",
        f"- **Hardware**: {gpu}.",
        f"- **OS**: {platform.platform()}.",
        f"- **Libraries**: torch {versions['torch']}, torchvision {versions['torchvision']}, "
        f"transformers {versions.get('transformers','?')}, pillow {versions.get('pillow','?')}.",
        f"- **Attack**: typographic overlay (`attacks/typographic.py`), white sticker "
        f"with bold black text drawn near the top of each clean image. Config: "
        f"`font_size_frac={typo_cfg.get('font_size_frac')}`, "
        f"`position={typo_cfg.get('position')!r}`, "
        f"`padding_frac={typo_cfg.get('padding_frac')}`, "
        f"`opacity={typo_cfg.get('opacity')}`, "
        f"`jpeg_quality={typo_cfg.get('jpeg_quality')}`, "
        f"`text_form={typo_cfg.get('text_form')!r}`.",
        f"- **Dataset**: the full 1000-image clean benchmark, one poisoned variant "
        f"per clean image (`data/poisoned/typographic/`). Each poisoned image is "
        f"stamped with the lowercase first synonym of a random *wrong* ImageNet "
        f"class, picked deterministically with the global seed.",
        f"- **Global seed**: 42 (`config.yaml`). Set on `random`, `numpy`, `torch` "
        f"at the start of every model.",
        "",
        "## 2. Headline accuracy table",
        "",
        table_md,
        "",
        f"Columns: clean accuracy, robust accuracy (untargeted — did the model "
        f"flip away from the true label?), accuracy drop (clean − robust), "
        f"fooling rate (originally-correct images that flipped), and TASR — the "
        f"**targeted-attack success rate**, the fraction of images where the "
        f"model predicted the *exact* class named by the overlay text.",
        "",
        f"Machine-readable copy: `{os.path.relpath(csv_path, TYPO_DIR)}`.",
        "",
        "## 3. Sanity checks",
        "",
        *sc_lines,
        "",
        "## 4. Per-model analysis",
        "",
        per_model_paragraph(results),
        "",
        "## 5. Figures",
        "",
        figures_md,
        "",
        "## 6. Interpretation",
        "",
        "The typographic attack is the paper's headline semantic attack. Pure "
        "vision classifiers are expected to latch onto the rendered text token "
        "and predict the wrong class — the TASR column quantifies how often the "
        "model is fooled into the *exact* class named on the sticker, not just "
        "into any wrong class. A language-grounded model like CLIP classifies "
        "by matching against text descriptions of every class; the hypothesis "
        "(project brief §3.6, hook #1) is that this makes it resist text "
        "overlays compared to pure classifiers. Compare CLIP's drop to the "
        "pure-classifier average in the sanity-check block above.",
        "",
        "## 7. Reproducibility footer",
        "",
        f"- **Wall-clock (this report build session)**: {elapsed:.1f} min.",
        f"- **Per-model cumulative compute (sum of per-cell `wall_clock_s`)**: "
        + ", ".join(f"{m} {wall['per_model_s'].get(m, 0)/60:.1f} min" for m in sorted(wall['per_model_s']))
        + ".",
        f"- **Total compute (sum across models)**: {wall['total_s']/60:.1f} min.",
        f"- **Library versions**: torch {versions['torch']}, "
        f"torchvision {versions['torchvision']}, "
        f"transformers {versions.get('transformers','?')}, "
        f"pillow {versions.get('pillow','?')}.",
        f"- **GPU**: {gpu}.",
        f"- **Seed**: 42 (set on `random`, `numpy`, `torch`).",
        f"- **Re-run**: `python scripts/run_typographic_benchmark.py` then "
        f"`python scripts/build_typographic_report.py` then "
        f"`python scripts/build_gradient_report_pdf.py "
        f"--input results/typographic/REPORT.md "
        f"--output results/typographic/REPORT.pdf`. Per-model JSONs are the "
        f"resumption unit — delete one to force its recomputation.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    start = time.time()
    run_started = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    os.makedirs(FIG_DIR, exist_ok=True)
    results = load_results()
    if not results:
        print("No per-model JSONs found in", TYPO_DIR)
        print("Run: python scripts/run_typographic_benchmark.py")
        sys.exit(1)

    rows = build_accuracy_table(results)
    csv_path = os.path.join(TYPO_DIR, "accuracy_table.csv")
    write_csv(rows, csv_path)
    print(f"  wrote {csv_path}")

    fig_paths = {}
    p = os.path.join(FIG_DIR, "01_clean_vs_robust.png")
    chart_clean_vs_robust(results, p)
    fig_paths["Figure 1 — Clean vs typographic-robust accuracy per model (sorted by clean acc)."] = p
    print(f"  wrote {p}")

    p = os.path.join(FIG_DIR, "02_drop_and_tasr.png")
    chart_drop_and_tasr(results, p)
    fig_paths["Figure 2 — Accuracy drop and targeted-attack success rate per model."] = p
    print(f"  wrote {p}")

    p = os.path.join(FIG_DIR, "03_clip_vs_others.png")
    chart_clip_vs_others(results, p)
    if os.path.exists(p):
        fig_paths["Figure 3 — Language-grounded (CLIP) vs pure-classifier mean drop."] = p
        print(f"  wrote {p}")

    p = os.path.join(FIG_DIR, "04_examples.png")
    chart_examples(p)
    if os.path.exists(p):
        fig_paths["Figure 4 — Clean | poisoned examples (typographic overlay)."] = p
        print(f"  wrote {p}")

    sanity = sanity_checks(results)
    sanity_path = os.path.join(TYPO_DIR, "sanity_checks.json")
    with open(sanity_path, "w", encoding="utf-8") as f:
        json.dump(sanity, f, indent=2)
    print(f"  wrote {sanity_path}")

    wall = wall_clock_summary(results)
    report_md = make_report(results, csv_path, fig_paths, sanity, wall, run_started, start)
    report_path = os.path.join(TYPO_DIR, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"  wrote {report_path}")


if __name__ == "__main__":
    main()
