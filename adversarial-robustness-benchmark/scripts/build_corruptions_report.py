"""CLI: build the accuracy table, figures, and REPORT.md for the corruptions run.

Inputs : results/corruptions/<model>__<corruption>.json
         data/poisoned/corruptions/manifest.json
         data/poisoned/corruptions/<corruption>/*.jpg  (for the example grid)
         data/clean/images/*.jpg                       (for the example grid)
Outputs:
  * results/corruptions/accuracy_table.csv             (long: model x corruption)
  * results/corruptions/accuracy_matrix.csv            (wide: 7 x 15)
  * results/corruptions/figures/*.png
  * results/corruptions/REPORT.md
  * results/corruptions/sanity_checks.json
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

from attacks.corruptions import CORRUPTION_TYPES  # noqa: E402

MODELS = [
    "resnet50",
    "vgg16",
    "convnext_tiny",
    "vit_b_16",
    "swin_t",
    "efficientnet_b0",
    "clip_vit_b16",
]

CORR_DIR = os.path.join(_REPO_ROOT, "results", "corruptions")
FIG_DIR = os.path.join(CORR_DIR, "figures")
POISONED_ROOT = os.path.join(_REPO_ROOT, "data", "poisoned", "corruptions")
CLEAN_IMG_DIR = os.path.join(_REPO_ROOT, "data", "clean", "images")

# Canonical 4-group taxonomy from Hendrycks & Dietterich (2019).
GROUPS = {
    "noise": ("gaussian_noise", "shot_noise", "impulse_noise"),
    "blur": ("defocus_blur", "glass_blur", "motion_blur", "zoom_blur"),
    "weather": ("snow", "frost", "fog", "brightness"),
    "digital": ("contrast", "elastic_transform", "pixelate", "jpeg_compression"),
}


def load_results() -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    if not os.path.isdir(CORR_DIR):
        return out
    for fname in os.listdir(CORR_DIR):
        if not fname.endswith(".json") or "__" not in fname:
            continue
        path = os.path.join(CORR_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                rec = json.load(f)
        except Exception:
            continue
        model = rec.get("model")
        corruption = rec.get("corruption_type")
        if model and corruption:
            out[(model, corruption)] = rec
    return out


def fmt(x, places=3):
    return "—" if x is None else f"{x:.{places}f}"


def build_long_table(results: dict) -> list[list]:
    header = ["model", "corruption_type", "clean_accuracy",
              "robust_accuracy", "accuracy_drop", "fooling_rate"]
    rows = [header]
    for m in MODELS:
        for ct in CORRUPTION_TYPES:
            r = results.get((m, ct))
            if not r:
                rows.append([m, ct, None, None, None, None])
                continue
            rows.append([
                m, ct,
                r["clean_accuracy"],
                r["robust_accuracy"],
                r["accuracy_drop"],
                r["fooling_rate"],
            ])
    return rows


def build_wide_matrix(results: dict) -> tuple[list[list], np.ndarray]:
    """Wide matrix: rows = models, cols = corruption types, cells = robust acc."""
    header = ["model"] + list(CORRUPTION_TYPES) + ["mean_robust", "mean_drop"]
    rows = [header]
    mat = np.full((len(MODELS), len(CORRUPTION_TYPES)), np.nan, dtype=np.float64)
    for i, m in enumerate(MODELS):
        for j, ct in enumerate(CORRUPTION_TYPES):
            r = results.get((m, ct))
            if r is not None:
                mat[i, j] = r["robust_accuracy"]
        row_vals = mat[i]
        mean_robust = float(np.nanmean(row_vals)) if np.any(~np.isnan(row_vals)) else None
        # mean drop per row needs clean acc per (m,ct) — they should all share the
        # same clean acc within a model. Use any non-null entry as the clean acc.
        clean = None
        for ct in CORRUPTION_TYPES:
            r = results.get((m, ct))
            if r is not None:
                clean = r["clean_accuracy"]
                break
        mean_drop = (clean - mean_robust) if (clean is not None and mean_robust is not None) else None
        rows.append([m] + [(None if np.isnan(v) else float(v)) for v in row_vals]
                    + [mean_robust, mean_drop])
    return rows, mat


def write_csv(rows: list[list], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow([f"{v:.4f}" if isinstance(v, float) else ("" if v is None else v)
                        for v in row])


def render_markdown_wide(rows: list[list]) -> str:
    header = rows[0]
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows[1:]:
        cells = [row[0]] + [fmt(v) for v in row[1:]]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def chart_heatmap(mat: np.ndarray, path: str) -> None:
    """7×15 heatmap of robust accuracy."""
    fig, ax = plt.subplots(figsize=(13, 5))
    cax = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(CORRUPTION_TYPES)))
    ax.set_xticklabels(CORRUPTION_TYPES, rotation=45, ha="right")
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(MODELS)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=("white" if v < 0.4 else "black"), fontsize=8)
    ax.set_title("Robust accuracy per (model, corruption type)")
    fig.colorbar(cax, ax=ax, label="accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart_mean_drop_per_model(results: dict, path: str) -> None:
    """Bar chart: mean accuracy drop per model, averaged across 15 corruptions."""
    drops = []
    have = []
    for m in MODELS:
        per_corr = [results[(m, ct)]["accuracy_drop"]
                    for ct in CORRUPTION_TYPES if (m, ct) in results]
        if per_corr:
            have.append(m)
            drops.append(float(np.mean(per_corr)))
    if not have:
        return
    order = np.argsort(drops)[::-1]
    have_sorted = [have[i] for i in order]
    drops_sorted = [drops[i] for i in order]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(have_sorted, drops_sorted, color="#e76f51")
    for x, v in enumerate(drops_sorted):
        ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9)
    ax.set_ylabel("mean (clean − robust) across 15 corruptions")
    ax.set_title("Mean accuracy drop per model (lower = more robust)")
    ax.set_xticklabels(have_sorted, rotation=20, ha="right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart_mean_drop_per_corruption(results: dict, path: str) -> None:
    """Bar chart: mean accuracy drop per corruption type, averaged across 7 models."""
    drops = []
    have = []
    for ct in CORRUPTION_TYPES:
        per_model = [results[(m, ct)]["accuracy_drop"]
                     for m in MODELS if (m, ct) in results]
        if per_model:
            have.append(ct)
            drops.append(float(np.mean(per_model)))
    if not have:
        return
    order = np.argsort(drops)[::-1]
    have_sorted = [have[i] for i in order]
    drops_sorted = [drops[i] for i in order]
    fig, ax = plt.subplots(figsize=(12, 5))
    # Colour bars by canonical group.
    group_colour = {"noise": "#264653", "blur": "#2a9d8f",
                    "weather": "#e9c46a", "digital": "#f4a261"}
    colours = []
    for ct in have_sorted:
        grp = next((g for g, items in GROUPS.items() if ct in items), "digital")
        colours.append(group_colour[grp])
    ax.bar(have_sorted, drops_sorted, color=colours)
    for x, v in enumerate(drops_sorted):
        ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9)
    ax.set_ylabel("mean (clean − robust) across 7 models")
    ax.set_title("Mean accuracy drop per corruption type (sorted desc)")
    ax.set_xticklabels(have_sorted, rotation=45, ha="right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    # Legend
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=c, label=g) for g, c in group_colour.items()]
    ax.legend(handles=legend, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart_group_bars(results: dict, path: str) -> None:
    """Per-model bars grouped by canonical taxonomy (noise/blur/weather/digital)."""
    rows = []
    for m in MODELS:
        per_group = {}
        for g, items in GROUPS.items():
            drops = [results[(m, ct)]["accuracy_drop"]
                     for ct in items if (m, ct) in results]
            per_group[g] = float(np.mean(drops)) if drops else np.nan
        rows.append((m, per_group))
    have = [(m, g) for m, g in rows if not all(np.isnan(v) for v in g.values())]
    if not have:
        return
    width = 0.2
    groups = list(GROUPS.keys())
    x = np.arange(len(have))
    fig, ax = plt.subplots(figsize=(11, 5))
    for gi, g in enumerate(groups):
        vals = [pg.get(g, np.nan) for _, pg in have]
        ax.bar(x + (gi - 1.5) * width, vals, width=width, label=g)
    ax.set_xticks(x)
    ax.set_xticklabels([m for m, _ in have], rotation=20, ha="right")
    ax.set_ylabel("mean drop within group")
    ax.set_title("Per-model mean drop, grouped by Hendrycks taxonomy")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart_examples(path: str) -> None:
    """Grid showing one image rendered under every corruption type."""
    manifest_path = os.path.join(POISONED_ROOT, "manifest.json")
    if not os.path.exists(manifest_path):
        return
    with open(manifest_path, "r", encoding="utf-8") as f:
        top = json.load(f)
    types = top.get("corruption_types", list(CORRUPTION_TYPES))
    # Pick one sample image from the first corruption type's per-sub manifest.
    sample_sub = os.path.join(POISONED_ROOT, types[0], "manifest.json")
    if not os.path.exists(sample_sub):
        return
    with open(sample_sub, "r", encoding="utf-8") as f:
        sub = json.load(f)
    if not sub["images"]:
        return
    sample_rec = sub["images"][0]
    panels: list[tuple[str, str]] = [("clean", os.path.join(CLEAN_IMG_DIR, os.path.basename(sample_rec["filename"])))]
    for ct in types:
        panels.append((ct, os.path.join(POISONED_ROOT, ct, sample_rec["filename"])))

    cols = 4
    rows = (len(panels) + cols - 1) // cols
    cell = 200
    cap_h = 22
    gap = 6
    grid_w = cols * cell + (cols + 1) * gap
    grid_h = rows * (cell + cap_h) + (rows + 1) * gap
    from PIL import ImageDraw
    grid = Image.new("RGB", (grid_w, grid_h), (32, 32, 32))
    d = ImageDraw.Draw(grid)
    for i, (caption, p) in enumerate(panels):
        row, col = divmod(i, cols)
        x = gap + col * (cell + gap)
        y = gap + row * (cell + cap_h + gap)
        if os.path.exists(p):
            im = Image.open(p).convert("RGB").resize((cell, cell))
            grid.paste(im, (x, y))
        d.text((x + 4, y + cell + 4), caption[:24], fill=(220, 220, 220))
    grid.save(path)


def sanity_checks(results: dict) -> dict:
    out: dict = {}

    # 1. Each corruption causes >1% drop on at least 5/7 models.
    per_corruption = {}
    for ct in CORRUPTION_TYPES:
        drops = [results[(m, ct)]["accuracy_drop"]
                 for m in MODELS if (m, ct) in results]
        n_signif = sum(1 for d in drops if d > 0.01)
        per_corruption[ct] = {
            "n_models_reported": len(drops),
            "n_with_drop_gt_1pct": n_signif,
            "mean_drop": float(np.mean(drops)) if drops else None,
            "passed": n_signif >= 5,
        }
    out["per_corruption_drop_threshold"] = per_corruption

    # 2. Overall severity-3 mean drop sits in literature window 5–25%.
    all_drops = [r["accuracy_drop"] for r in results.values()]
    if all_drops:
        mean_drop = float(np.mean(all_drops))
        out["overall_mean_drop"] = {
            "value": mean_drop,
            "expected_window": [0.05, 0.25],
            "passed": 0.05 <= mean_drop <= 0.25,
        }

    # 3. Coverage: every (model, corruption) pair reported.
    expected = len(MODELS) * len(CORRUPTION_TYPES)
    have = sum(1 for m in MODELS for ct in CORRUPTION_TYPES if (m, ct) in results)
    out["coverage"] = {
        "expected_pairs": expected,
        "reported_pairs": have,
        "passed": have == expected,
    }
    return out


def wall_clock_summary(results: dict) -> dict:
    total = 0.0
    per_model: dict[str, float] = {}
    for (m, _), r in results.items():
        s = sum(r["wall_clock_s"].values())
        per_model[m] = per_model.get(m, 0.0) + s
        total += s
    return {"total_s": total, "per_model_s": per_model}


def _eval_host_summary(results: dict) -> dict:
    seen_gpus, seen_torch, seen_tv = set(), set(), set()
    for r in results.values():
        env = r.get("host_env") or {}
        if env.get("gpu"):
            seen_gpus.add(env["gpu"])
        if env.get("torch"):
            seen_torch.add(env["torch"])
        if env.get("torchvision"):
            seen_tv.add(env["torchvision"])
    if not (seen_gpus or seen_torch):
        return {"recorded": False}

    def _one(s, fallback="?"):
        if not s:
            return fallback
        return next(iter(s)) if len(s) == 1 else " | ".join(sorted(s))
    return {
        "recorded": True,
        "gpu": _one(seen_gpus),
        "torch": _one(seen_torch),
        "torchvision": _one(seen_tv),
    }


def make_report(
    results: dict, csv_long_path: str, csv_wide_path: str,
    fig_paths: dict[str, str], sanity: dict, wall: dict,
    run_started: str, start_time: float,
) -> str:
    long_rows = build_long_table(results)
    wide_rows, _ = build_wide_matrix(results)
    wide_md = render_markdown_wide(wide_rows)
    elapsed = (time.time() - start_time) / 60.0
    eval_host = _eval_host_summary(results)
    build_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    top_manifest_path = os.path.join(POISONED_ROOT, "manifest.json")
    top_manifest = {}
    if os.path.exists(top_manifest_path):
        with open(top_manifest_path, "r", encoding="utf-8") as f:
            top_manifest = json.load(f)
    severity = top_manifest.get("severity", "?")
    config = top_manifest.get("config", {})

    sc_lines: list[str] = []
    if "overall_mean_drop" in sanity:
        v = sanity["overall_mean_drop"]
        lo, hi = v["expected_window"]
        sc_lines.append(
            f"- **Severity-{severity} mean drop in literature window [{lo:.2f}, {hi:.2f}]**: "
            f"observed **{v['value']:.3f}**  →  **{'PASS' if v['passed'] else 'FAIL'}**."
        )
    if "per_corruption_drop_threshold" in sanity:
        fails = [ct for ct, v in sanity["per_corruption_drop_threshold"].items()
                 if not v["passed"]]
        sc_lines.append(
            f"- **Every corruption causes >1% drop on ≥5 of 7 models**: "
            + (f"PASS — all {len(CORRUPTION_TYPES)} types meet the bar."
               if not fails else
               f"**FAIL** for {len(fails)} type(s): {', '.join(fails)}.")
        )
    if "coverage" in sanity:
        c = sanity["coverage"]
        sc_lines.append(
            f"- **Coverage**: {c['reported_pairs']} / {c['expected_pairs']} "
            f"(model × corruption) pairs reported.  "
            f"**{'PASS' if c['passed'] else 'FAIL'}**."
        )

    figures_md = "\n".join(
        f"![{caption}]({os.path.relpath(p, CORR_DIR).replace(os.sep, '/')})  \n*{caption}*"
        for caption, p in fig_paths.items() if p and os.path.exists(p)
    )

    # Headline paragraph: which model is most robust, which corruption is hardest.
    per_model_drop = {}
    for m in MODELS:
        drops = [results[(m, ct)]["accuracy_drop"] for ct in CORRUPTION_TYPES if (m, ct) in results]
        if drops:
            per_model_drop[m] = float(np.mean(drops))
    per_corr_drop = {}
    for ct in CORRUPTION_TYPES:
        drops = [results[(m, ct)]["accuracy_drop"] for m in MODELS if (m, ct) in results]
        if drops:
            per_corr_drop[ct] = float(np.mean(drops))

    headline_lines: list[str] = []
    if per_model_drop:
        best_model = min(per_model_drop, key=per_model_drop.get)
        worst_model = max(per_model_drop, key=per_model_drop.get)
        headline_lines.append(
            f"Across {len(per_model_drop)} of {len(MODELS)} models and "
            f"{len(per_corr_drop)} of {len(CORRUPTION_TYPES)} corruption types at "
            f"severity {severity}, the mean accuracy drop was "
            f"**{np.mean(list(per_model_drop.values())):.3f}**. The most "
            f"robust model on this axis was `{best_model}` "
            f"(mean drop **{per_model_drop[best_model]:.3f}**); the least "
            f"robust was `{worst_model}` "
            f"(mean drop **{per_model_drop[worst_model]:.3f}**)."
        )
    if per_corr_drop:
        hardest = max(per_corr_drop, key=per_corr_drop.get)
        easiest = min(per_corr_drop, key=per_corr_drop.get)
        headline_lines.append(
            f"The hardest corruption type (averaged across models) was "
            f"`{hardest}` with mean drop **{per_corr_drop[hardest]:.3f}**; "
            f"the easiest was `{easiest}` with mean drop "
            f"**{per_corr_drop[easiest]:.3f}**."
        )

    lines = [
        "# Common-Corruptions Benchmark — Report",
        "",
        f"_Run started: {run_started}.  Author: Sachit Jain._",
        "",
        f"This report covers Phase 1, Axis A — the 15 ImageNet-C-style common "
        f"corruptions at severity {severity}, evaluated on the 7 ImageNet "
        f"baselines listed in `config.yaml`. The corruption sets are generated "
        f"once by `scripts/generate_datasets.py --corruptions` and shared "
        f"across all models; this report aggregates the per-(model, corruption) "
        f"JSONs in this directory.",
        "",
        "## 1. Setup",
        "",
        (f"- **Eval host (where inference ran)**: {eval_host['gpu']}. "
         f"torch {eval_host['torch']}, torchvision {eval_host['torchvision']}.")
        if eval_host["recorded"] else
        ("- **Eval host (where inference ran)**: not recorded in the per-cell "
         "JSONs for this run."),
        f"- **Report rebuilt on**: {build_gpu}, {platform.platform()}.",
        f"- **Attack**: 15 corruption types (`attacks/corruptions.py`), severity "
        f"{severity}. Implementation: self-contained NumPy/SciPy/PIL "
        f"(no `imagenet_c` / ImageMagick dependency). Calibrations follow "
        f"Hendrycks & Dietterich (ICLR 2019); `motion_blur`/`snow`/`frost`/`fog` "
        f"use NumPy approximations of the canonical Wand versions and may not "
        f"be bit-equivalent to the published ImageNet-C numbers.",
        f"- **Dataset**: the full 1000-image clean benchmark, pre-cropped to "
        f"{config.get('crop_size', 224)}×{config.get('crop_size', 224)} "
        f"(torchvision eval transform) before corruption. One corrupted variant "
        f"per (clean image, corruption type) = "
        f"{top_manifest.get('num_images_total', '?')} images total.",
        f"- **Global seed**: 42 (`config.yaml`). Per-(image, corruption) RNG "
        f"seeded with `(seed + sha1(corruption_type) + image_index)` so a "
        f"single corruption type can be regenerated without touching the others.",
        "",
        "## 2. Headline (robust accuracy per (model, corruption))",
        "",
        wide_md,
        "",
        f"Machine-readable copies: `{os.path.relpath(csv_wide_path, CORR_DIR)}` "
        f"(wide), `{os.path.relpath(csv_long_path, CORR_DIR)}` (long).",
        "",
        "## 3. Sanity checks",
        "",
        *sc_lines,
        "",
        "## 4. Per-axis analysis",
        "",
        " ".join(headline_lines) if headline_lines else "_no per-cell JSONs found_",
        "",
        "## 5. Figures",
        "",
        figures_md,
        "",
        "## 6. Interpretation",
        "",
        ("Common corruptions are the non-adversarial, distribution-shift end "
         "of the robustness spectrum: each image is degraded by physical noise, "
         "blur, weather, or digital artifacts in a way that a human would still "
         "classify correctly. The drops here represent **realistic deployment "
         "failures** (a webcam in fog, a low-bandwidth JPEG upload, a shaky "
         "phone photo), not worst-case adversarial inputs. Unlike the gradient "
         "and typographic axes, this measure decouples natural fragility from "
         "language grounding — every model is graded on the same model-agnostic "
         "transformation. The Phase D defense will be re-evaluated on this axis "
         "to check whether adversarial fine-tuning sacrifices natural-corruption "
         "robustness as a side effect."),
        "",
        "## 7. Reproducibility footer",
        "",
        f"- **Wall-clock (this report build session)**: {elapsed:.1f} min.",
        f"- **Per-model cumulative compute (sum across 15 corruptions)**: "
        + ", ".join(f"{m} {wall['per_model_s'].get(m, 0) / 60:.1f} min"
                    for m in sorted(wall["per_model_s"]))
        + ".",
        f"- **Total compute (sum across model × corruption)**: {wall['total_s'] / 60:.1f} min.",
        f"- **Seed**: 42 (set on `random`, `numpy`, `torch`).",
        f"- **Re-run**: "
        f"`python scripts/generate_datasets.py --corruptions --severity {severity}` "
        f"then `python scripts/run_corruptions_benchmark.py` then "
        f"`python scripts/build_corruptions_report.py` then "
        f"`python scripts/build_gradient_report_pdf.py "
        f"--input results/corruptions/REPORT.md "
        f"--output results/corruptions/REPORT.pdf`. Per-(model, corruption) "
        f"JSONs are the resumption unit — delete one to force its recomputation.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    start = time.time()
    run_started = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    os.makedirs(FIG_DIR, exist_ok=True)
    results = load_results()
    if not results:
        print("No per-(model, corruption) JSONs found in", CORR_DIR)
        print("Run: python scripts/run_corruptions_benchmark.py")
        sys.exit(1)

    long_rows = build_long_table(results)
    long_path = os.path.join(CORR_DIR, "accuracy_table.csv")
    write_csv(long_rows, long_path)
    print(f"  wrote {long_path}")

    wide_rows, mat = build_wide_matrix(results)
    wide_path = os.path.join(CORR_DIR, "accuracy_matrix.csv")
    write_csv(wide_rows, wide_path)
    print(f"  wrote {wide_path}")

    fig_paths: dict[str, str] = {}
    p = os.path.join(FIG_DIR, "01_heatmap.png")
    chart_heatmap(mat, p)
    fig_paths["Figure 1 — Robust accuracy heatmap (7 models × 15 corruptions)."] = p
    print(f"  wrote {p}")

    p = os.path.join(FIG_DIR, "02_mean_drop_per_model.png")
    chart_mean_drop_per_model(results, p)
    fig_paths["Figure 2 — Mean accuracy drop per model (averaged across 15 corruptions)."] = p
    print(f"  wrote {p}")

    p = os.path.join(FIG_DIR, "03_mean_drop_per_corruption.png")
    chart_mean_drop_per_corruption(results, p)
    fig_paths["Figure 3 — Mean accuracy drop per corruption type (averaged across 7 models)."] = p
    print(f"  wrote {p}")

    p = os.path.join(FIG_DIR, "04_group_bars.png")
    chart_group_bars(results, p)
    fig_paths["Figure 4 — Per-model mean drop grouped by Hendrycks taxonomy."] = p
    print(f"  wrote {p}")

    p = os.path.join(FIG_DIR, "05_examples.png")
    chart_examples(p)
    if os.path.exists(p):
        fig_paths["Figure 5 — One image under each corruption type."] = p
        print(f"  wrote {p}")

    sanity = sanity_checks(results)
    sanity_path = os.path.join(CORR_DIR, "sanity_checks.json")
    with open(sanity_path, "w", encoding="utf-8") as f:
        json.dump(sanity, f, indent=2)
    print(f"  wrote {sanity_path}")

    wall = wall_clock_summary(results)
    report_md = make_report(results, long_path, wide_path, fig_paths,
                            sanity, wall, run_started, start)
    report_path = os.path.join(CORR_DIR, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"  wrote {report_path}")


if __name__ == "__main__":
    main()
