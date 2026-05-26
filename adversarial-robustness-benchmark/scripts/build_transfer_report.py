"""CLI: build the accuracy table, figures, and REPORT.md for the transfer run.

Inputs : results/transfer/<model>.json  (written by run_transfer_benchmark.py)
Outputs:
  * results/transfer/accuracy_table.csv
  * results/transfer/figures/01_clean_vs_robust.png
  * results/transfer/figures/02_accuracy_drop.png
  * results/transfer/REPORT.md
  * results/transfer/sanity_checks.json

Gate B sanity check: ResNet-50 self-transfer (white-box) is compared against
the gradient phase's PGD result for ResNet-50. They should match within ±0.005.
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

MODELS = [
    "resnet50",
    "vgg16",
    "convnext_tiny",
    "vit_b_16",
    "swin_t",
    "efficientnet_b0",
    "clip_vit_b16",
]

TRANSFER_DIR = os.path.join(_REPO_ROOT, "results", "transfer")
GRADIENT_DIR = os.path.join(_REPO_ROOT, "results", "gradient")
FIG_DIR = os.path.join(TRANSFER_DIR, "figures")


def load_results() -> dict:
    out: dict[str, dict] = {}
    for m in MODELS:
        p = os.path.join(TRANSFER_DIR, f"{m}.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                out[m] = json.load(f)
    return out


def fmt(x, places=3):
    return "—" if x is None else f"{x:.{places}f}"


def build_accuracy_table(results: dict) -> list[list]:
    header = ["model", "is_white_box", "clean_accuracy",
              "transfer_robust_accuracy", "accuracy_drop", "fooling_rate"]
    rows = [header]
    for m in MODELS:
        r = results.get(m)
        if not r:
            rows.append([m, None, None, None, None, None])
            continue
        rows.append([
            m,
            r.get("is_white_box", False),
            r["clean_accuracy"],
            r["transfer_robust_accuracy"],
            r["accuracy_drop"],
            r["fooling_rate"],
        ])
    return rows


def write_csv(rows: list[list], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in rows:
            out = []
            for v in row:
                if isinstance(v, float):
                    out.append(f"{v:.4f}")
                elif v is None:
                    out.append("")
                else:
                    out.append(str(v))
            w.writerow(out)


def render_markdown_table(rows: list[list]) -> str:
    pretty = ["Model", "White-box?", "Clean", "Transfer-robust", "Drop", "Fooling rate"]
    out = ["| " + " | ".join(pretty) + " |",
           "| " + " | ".join(["---"] * len(pretty)) + " |"]
    for row in rows[1:]:
        model = row[0]
        wb = "**yes**" if row[1] else "no"
        vals = [model, wb] + [fmt(v) for v in row[2:]]
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def chart_clean_vs_robust(results: dict, path: str) -> None:
    """Bar chart: per-model clean vs transfer-robust accuracy, surrogate highlighted."""
    have = [m for m in MODELS if m in results]
    order = sorted(have, key=lambda m: -results[m]["clean_accuracy"])

    clean = [results[m]["clean_accuracy"] for m in order]
    robust = [results[m]["transfer_robust_accuracy"] for m in order]
    x = np.arange(len(order))
    w = 0.4

    colors_robust = ["#e76f51" if results[m].get("is_white_box") else "#457b9d"
                     for m in order]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, clean, width=w, label="clean", color="#2a9d8f")
    bars = ax.bar(x + w / 2, robust, width=w, label="transfer-robust")
    for bar, c in zip(bars, colors_robust):
        bar.set_color(c)

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title("Clean vs transfer-robust accuracy per model\n"
                 "(red bar = white-box / surrogate, blue = black-box transfer)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def chart_accuracy_drop(results: dict, path: str) -> None:
    """Bar chart: accuracy drop per model, surrogate highlighted."""
    have = [m for m in MODELS if m in results]
    order = sorted(have, key=lambda m: -results[m]["accuracy_drop"])
    drops = [results[m]["accuracy_drop"] for m in order]
    colors = ["#e76f51" if results[m].get("is_white_box") else "#457b9d"
              for m in order]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(np.arange(len(order)), drops, color=colors)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("clean − transfer-robust")
    ax.set_title("Transfer accuracy drop per model\n"
                 "(red = white-box surrogate, blue = black-box targets, sorted desc)")
    ax.set_ylim(0, max(drops) * 1.18 if drops else 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for xi, v in enumerate(drops):
        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def sanity_checks(results: dict) -> dict:
    out: dict = {}

    # Gate B check 1: ResNet-50 self-transfer matches gradient PGD within ±0.005.
    wb_check: dict = {}
    if "resnet50" in results:
        wb = results["resnet50"]
        gradient_pgd_path = os.path.join(GRADIENT_DIR, "resnet50__pgd.json")
        if os.path.exists(gradient_pgd_path):
            with open(gradient_pgd_path, "r", encoding="utf-8") as f:
                gd = json.load(f)
            transfer_ra = wb["transfer_robust_accuracy"]
            gradient_ra = gd["robust_accuracy"]
            diff = abs(transfer_ra - gradient_ra)
            wb_check = {
                "transfer_robust_accuracy": transfer_ra,
                "gradient_pgd_robust_accuracy": gradient_ra,
                "absolute_diff": diff,
                "passed": diff <= 0.005,
                "note": ("Gate B: self-transfer should match gradient PGD within ±0.005. "
                         "Larger diff means different images are being sampled or "
                         "a different PGD random seed was used."),
            }
        else:
            wb_check = {
                "transfer_robust_accuracy": wb["transfer_robust_accuracy"],
                "gradient_pgd_robust_accuracy": None,
                "note": "gradient/resnet50__pgd.json not found — skip comparison.",
            }
    out["resnet50_self_transfer_vs_gradient_pgd"] = wb_check

    # Gate B check 2: at least 4 of 6 non-surrogate models show >10% drop.
    surrogate = next((r["surrogate"] for r in results.values()), "resnet50")
    non_surrogate = [m for m in results if not results[m].get("is_white_box")]
    high_drop = [m for m in non_surrogate if results[m]["accuracy_drop"] > 0.10]
    out["non_surrogate_transfer_drops"] = {
        m: results[m]["accuracy_drop"] for m in non_surrogate
    }
    out["gate_b_transfer_coverage"] = {
        "n_non_surrogate": len(non_surrogate),
        "n_with_gt10pct_drop": len(high_drop),
        "models_with_gt10pct_drop": high_drop,
        "passed": len(high_drop) >= 4,
        "note": "Gate B requires ≥4/6 non-surrogate models show >10% accuracy drop.",
    }

    # Coverage
    out["models_reported"] = sorted(results.keys())
    out["all_models_present"] = len(results) == len(MODELS)
    return out


def wall_clock_summary(results: dict) -> dict:
    total = 0.0
    per_model = {}
    for m, r in results.items():
        s = sum(r.get("wall_clock_s", {}).values())
        per_model[m] = s
        total += s
    return {"total_s": total, "per_model_s": per_model}


def library_versions() -> dict:
    versions = {"torch": torch.__version__, "torchvision": torchvision.__version__}
    for name in ["PIL"]:
        try:
            import PIL
            versions["pillow"] = PIL.__version__
        except ImportError:
            versions["pillow"] = "unavailable"
    return versions


def _eval_host_summary(results: dict) -> dict:
    seen_gpus, seen_torch, seen_tv = set(), set(), set()
    for r in results.values():
        env = r.get("host_env") or {}
        if env.get("gpu"): seen_gpus.add(env["gpu"])
        if env.get("torch"): seen_torch.add(env["torch"])
        if env.get("torchvision"): seen_tv.add(env["torchvision"])
    if not (seen_gpus or seen_torch):
        return {"recorded": False}
    def _one(s, fallback="?"):
        if not s: return fallback
        return next(iter(s)) if len(s) == 1 else " | ".join(sorted(s))
    return {
        "recorded": True,
        "gpu": _one(seen_gpus),
        "torch": _one(seen_torch),
        "torchvision": _one(seen_tv),
    }


def transferability_paragraph(results: dict) -> str:
    if not results:
        return "_no per-model JSONs found_"

    surrogate = next((r["surrogate"] for r in results.values()), "resnet50")
    wb = results.get(surrogate, {})
    wb_robust = wb.get("transfer_robust_accuracy")
    non_wb = [m for m in results if not results[m].get("is_white_box")]
    if not non_wb:
        return "_only surrogate model evaluated_"

    drops = [results[m]["accuracy_drop"] for m in non_wb]
    mean_drop = float(np.mean(drops))
    max_drop_model = max(non_wb, key=lambda m: results[m]["accuracy_drop"])
    max_drop = results[max_drop_model]["accuracy_drop"]
    min_drop_model = min(non_wb, key=lambda m: results[m]["accuracy_drop"])
    min_drop = results[min_drop_model]["accuracy_drop"]

    lines = [
        f"PGD-{results[list(results.keys())[0]].get('num_steps', 20)} adversarial "
        f"images crafted on `{surrogate}` (the white-box surrogate) were evaluated "
        f"against all {len(results)} architectures. "
    ]
    if wb_robust is not None:
        lines.append(
            f"On the surrogate itself (**white-box**), robust accuracy is "
            f"**{wb_robust:.3f}** — this should match the gradient phase's PGD result. "
        )
    lines.append(
        f"Across the {len(non_wb)} black-box target models, the mean accuracy drop "
        f"is **{mean_drop:.3f}**. "
        f"The most susceptible target is `{max_drop_model}` "
        f"(drop **{max_drop:.3f}**); the most resistant is `{min_drop_model}` "
        f"(drop **{min_drop:.3f}**). "
    )
    lines.append(
        "Architecture family matters for transferability: CNN → CNN transfers better "
        "than CNN → Transformer in general, so ViT/Swin/ConvNeXt are expected to show "
        "smaller drops than ResNet/VGG/EfficientNet. Check the table for this pattern."
    )
    return " ".join(lines)


def make_report(
    results: dict,
    csv_path: str,
    fig_paths: dict,
    sanity: dict,
    wall: dict,
    run_started: str,
    start_time: float,
) -> str:
    rows = build_accuracy_table(results)
    table_md = render_markdown_table(rows)
    versions = library_versions()
    elapsed = (time.time() - start_time) / 60.0
    eval_host = _eval_host_summary(results)
    build_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    surrogate = next((r["surrogate"] for r in results.values()), "resnet50")
    eps = next((r.get("epsilon", 8/255) for r in results.values()), 8/255)
    n_steps = next((r.get("num_steps", 20) for r in results.values()), 20)

    sc_lines = []
    wb_check = sanity.get("resnet50_self_transfer_vs_gradient_pgd", {})
    if wb_check.get("gradient_pgd_robust_accuracy") is not None:
        sc_lines.append(
            f"- **Gate B — self-transfer vs gradient PGD (resnet50):** "
            f"transfer = **{wb_check['transfer_robust_accuracy']:.3f}**, "
            f"gradient PGD = **{wb_check['gradient_pgd_robust_accuracy']:.3f}**, "
            f"diff = **{wb_check['absolute_diff']:.4f}** → "
            f"**{'PASS' if wb_check['passed'] else 'FAIL'}** (tolerance ±0.005)."
        )
    elif wb_check.get("transfer_robust_accuracy") is not None:
        sc_lines.append(
            f"- **Gate B — self-transfer (resnet50):** robust accuracy "
            f"**{wb_check['transfer_robust_accuracy']:.3f}** — gradient PGD JSON not "
            f"found for comparison; verify manually against `results/gradient/resnet50__pgd.json`."
        )

    gate_b = sanity.get("gate_b_transfer_coverage", {})
    if gate_b:
        sc_lines.append(
            f"- **Gate B — transferability coverage:** "
            f"{gate_b['n_with_gt10pct_drop']}/{gate_b['n_non_surrogate']} non-surrogate "
            f"models show >10% accuracy drop → "
            f"**{'PASS' if gate_b['passed'] else 'FAIL'}** (need ≥4). "
            + (f"Models: {gate_b['models_with_gt10pct_drop']}." if gate_b['models_with_gt10pct_drop'] else "")
        )

    sc_lines.append(
        f"- **Coverage**: {len(sanity['models_reported'])} / {len(MODELS)} models reported. "
        f"**{'PASS' if sanity['all_models_present'] else 'FAIL'}**."
    )

    figures_md = "\n".join(
        f"![{caption}]({os.path.relpath(p, TRANSFER_DIR).replace(os.sep, '/')})  \n*{caption}*"
        for caption, p in fig_paths.items() if p and os.path.exists(p)
    )

    lines = [
        "# Transfer-Attack Benchmark — Report",
        "",
        f"_Run started: {run_started}.  Author: Sachit Jain._",
        "",
        "This report covers Phase B of the project — surrogate-based transfer "
        "attack evaluation. PGD adversarial images were crafted once on the "
        f"`{surrogate}` surrogate (ε={eps:.4f}, {n_steps} steps) and evaluated "
        "on all 7 baseline architectures. The surrogate row is the white-box "
        "reference; the other six rows measure cross-architecture transferability.",
        "",
        "## 1. Setup",
        "",
        (f"- **Eval host**: {eval_host['gpu']}. "
         f"torch {eval_host['torch']}, torchvision {eval_host['torchvision']}.")
        if eval_host["recorded"] else
        "- **Eval host**: not recorded in per-model JSONs.",
        f"- **Report rebuilt on**: {build_gpu}, {platform.platform()}.",
        f"- **Attack**: PGD on `{surrogate}` surrogate. ε={eps:.6f} ({round(eps*255)}/255), "
        f"step={eps/4:.6f}, {n_steps} steps, random start, seed 42.",
        f"- **Dataset**: 1000-image clean benchmark, preprocessed through the "
        f"`{surrogate}` model's resize/crop pipeline. All 7 models evaluate on "
        f"the same pixel tensor via their own normalization inside `logits()`.",
        f"- **Global seed**: 42. Set on `random`, `numpy`, `torch`.",
        "",
        "## 2. Headline accuracy table",
        "",
        table_md,
        "",
        f"Columns: clean accuracy (on surrogate-preprocessed images), "
        f"transfer-robust accuracy (same adversarial images for all targets), "
        f"drop (clean − robust), fooling rate (originally-correct images that flipped). "
        f"**White-box = yes** marks the surrogate itself (ResNet-50).",
        "",
        f"Machine-readable copy: `{os.path.relpath(csv_path, TRANSFER_DIR)}`.",
        "",
        "## 3. Sanity checks (Gate B)",
        "",
        *sc_lines,
        "",
        "## 4. Transferability analysis",
        "",
        transferability_paragraph(results),
        "",
        "## 5. Figures",
        "",
        figures_md,
        "",
        "## 6. Reproducibility footer",
        "",
        f"- **Wall-clock (this report build)**: {elapsed:.1f} min.",
        f"- **Per-model eval time**: "
        + ", ".join(f"{m} {wall['per_model_s'].get(m, 0)/60:.1f} min" for m in sorted(wall['per_model_s']))
        + ".",
        f"- **Total eval compute**: {wall['total_s']/60:.1f} min.",
        f"- **Seed**: 42.",
        f"- **Re-run**: `python scripts/generate_datasets.py --transfer` then "
        f"`python scripts/run_transfer_benchmark.py` then "
        f"`python scripts/build_transfer_report.py` then "
        f"`python scripts/build_gradient_report_pdf.py "
        f"--input results/transfer/REPORT.md "
        f"--output results/transfer/REPORT.pdf`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    start = time.time()
    run_started = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    os.makedirs(FIG_DIR, exist_ok=True)
    results = load_results()
    if not results:
        print("No per-model JSONs found in", TRANSFER_DIR)
        print("Run: python scripts/generate_datasets.py --transfer")
        print("     python scripts/run_transfer_benchmark.py")
        sys.exit(1)

    rows = build_accuracy_table(results)
    csv_path = os.path.join(TRANSFER_DIR, "accuracy_table.csv")
    write_csv(rows, csv_path)
    print(f"  wrote {csv_path}")

    fig_paths = {}

    p = os.path.join(FIG_DIR, "01_clean_vs_robust.png")
    chart_clean_vs_robust(results, p)
    fig_paths["Figure 1 — Clean vs transfer-robust accuracy per model (red = surrogate/white-box)."] = p
    print(f"  wrote {p}")

    p = os.path.join(FIG_DIR, "02_accuracy_drop.png")
    chart_accuracy_drop(results, p)
    fig_paths["Figure 2 — Accuracy drop per model under transfer attack (sorted desc)."] = p
    print(f"  wrote {p}")

    sanity = sanity_checks(results)
    sanity_path = os.path.join(TRANSFER_DIR, "sanity_checks.json")
    with open(sanity_path, "w", encoding="utf-8") as f:
        json.dump(sanity, f, indent=2)
    print(f"  wrote {sanity_path}")

    # Print Gate B pass/fail to console.
    gate_b = sanity.get("gate_b_transfer_coverage", {})
    if gate_b:
        status = "PASS" if gate_b["passed"] else "FAIL"
        print(f"  Gate B coverage: {status} "
              f"({gate_b['n_with_gt10pct_drop']}/{gate_b['n_non_surrogate']} models >10% drop)")
    wb_check = sanity.get("resnet50_self_transfer_vs_gradient_pgd", {})
    if wb_check.get("gradient_pgd_robust_accuracy") is not None:
        status = "PASS" if wb_check["passed"] else "FAIL"
        print(f"  Gate B self-transfer: {status} "
              f"(diff={wb_check['absolute_diff']:.4f}, tolerance=0.005)")

    wall = wall_clock_summary(results)
    report_md = make_report(results, csv_path, fig_paths, sanity, wall, run_started, start)
    report_path = os.path.join(TRANSFER_DIR, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"  wrote {report_path}")


if __name__ == "__main__":
    main()
