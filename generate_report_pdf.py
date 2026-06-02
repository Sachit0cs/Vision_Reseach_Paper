"""
Adversarial Robustness Benchmark — Full Results PDF
Generates a comprehensive PDF report with all 4 axes + defense comparison.
Page size: 17x11 (landscape super-A3) so tables never clip.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────────────────────────────

MODELS = ["ResNet-50", "VGG-16", "ConvNeXt-T", "ViT-B/16", "Swin-T", "EfficientNet-B0", "CLIP ViT-B/16"]

GRADIENT = {
    "clean":      [0.781, 0.687, 0.787, 0.793, 0.775, 0.743, 0.621],
    "fgsm":       [0.408, 0.011, 0.292, 0.244, 0.191, 0.100, 0.058],
    "pgd":        [0.004, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000],
    "autoattack": [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    "square":     [0.010, 0.000, 0.010, 0.000, 0.000, 0.000, 0.000],
}
DEFENSE_GRADIENT = {"clean":0.060,"fgsm":0.024,"pgd":0.015,"autoattack":0.005,"square":0.015}

TRANSFER = {
    "clean":           [0.781, 0.695, 0.783, 0.796, 0.772, 0.748, 0.628],
    "transfer_robust": [0.003, 0.478, 0.638, 0.747, 0.661, 0.623, 0.532],
    "drop":            [0.778, 0.217, 0.145, 0.049, 0.111, 0.125, 0.096],
    "fooling":         [0.996, 0.327, 0.198, 0.079, 0.161, 0.186, 0.205],
}
DEFENSE_TRANSFER = {"clean":0.060,"transfer_robust":0.059,"drop":0.001,"fooling":0.017}

TYPO = {
    "clean":   [0.781, 0.687, 0.787, 0.793, 0.775, 0.743, 0.621],
    "robust":  [0.768, 0.663, 0.777, 0.773, 0.770, 0.722, 0.368],
    "drop":    [0.013, 0.024, 0.010, 0.020, 0.005, 0.021, 0.253],
    "fooling": [0.036, 0.076, 0.032, 0.039, 0.032, 0.055, 0.452],
    "tasr":    [0.001, 0.000, 0.000, 0.001, 0.002, 0.001, 0.342],
}
DEFENSE_TYPO = {"clean":0.060,"robust":0.058,"drop":0.002,"fooling":0.117,"tasr":0.002}

CORRUPTION_TYPES = [
    "gaussian_noise","shot_noise","impulse_noise","defocus_blur","glass_blur",
    "motion_blur","zoom_blur","snow","frost","fog","brightness","contrast",
    "elastic_transform","pixelate","jpeg_compression",
]
CORRUPTION_MATRIX = np.array([
    [0.487,0.457,0.358,0.447,0.088,0.129,0.393,0.664,0.618,0.384,0.721,0.700,0.771,0.594,0.660],
    [0.150,0.143,0.091,0.190,0.056,0.045,0.242,0.511,0.420,0.165,0.563,0.313,0.647,0.285,0.475],
    [0.567,0.554,0.565,0.489,0.130,0.160,0.417,0.730,0.656,0.387,0.750,0.712,0.769,0.643,0.701],
    [0.636,0.637,0.515,0.522,0.258,0.216,0.463,0.730,0.677,0.543,0.743,0.717,0.773,0.686,0.695],
    [0.592,0.566,0.563,0.490,0.134,0.181,0.408,0.727,0.684,0.549,0.738,0.728,0.773,0.583,0.671],
    [0.432,0.436,0.425,0.353,0.093,0.110,0.337,0.609,0.536,0.226,0.695,0.526,0.725,0.585,0.647],
    [0.403,0.400,0.368,0.396,0.147,0.141,0.316,0.533,0.511,0.355,0.579,0.543,0.596,0.481,0.488],
])
CORRUPTION_MEAN = CORRUPTION_MATRIX.mean(axis=1)
DEFENSE_CORRUPTION_MEAN = 0.035

CLEAN_ACC = np.array(GRADIENT["clean"])
MODEL_COLORS = ["#2196F3","#FF5722","#4CAF50","#9C27B0","#FF9800","#009688","#E91E63"]
DEFENSE_COLOR  = "#B71C1C"
BASELINE_COLOR = "#1565C0"
ATTACK_COLORS  = {"clean":"#37474F","fgsm":"#FFA726","pgd":"#EF5350","autoattack":"#B71C1C","square":"#7E57C2"}

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE  (17×11 landscape — tables stay inside)
# ──────────────────────────────────────────────────────────────────────────────
W, H = 17, 11   # inches — every figure uses this

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.titlesize":   11,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "figure.facecolor": "white",
    "axes.facecolor":   "#FAFAFA",
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

def rg(v):
    """Map [0,1] → RdYlGn colour."""
    return plt.cm.RdYlGn(np.clip(v, 0, 1))

def style_header(tbl, ncols, bg="#263238"):
    for j in range(ncols):
        tbl[0, j].set_facecolor(bg)
        tbl[0, j].set_text_props(color="white", fontweight="bold")

def style_rowlabel(tbl, nrows, is_defense_row=-1):
    for i in range(nrows):
        tbl[i+1, -1].set_facecolor("#FFCDD2" if i == is_defense_row else "#ECEFF1")
        tbl[i+1, -1].set_text_props(fontweight="bold")

def build_table(ax, cellText, rowLabels, colLabels, fontsize=8, scale_y=1.7, title=""):
    ax.axis("off")
    if title:
        ax.set_title(title, fontweight="bold", fontsize=10, pad=6)
    tbl = ax.table(cellText=cellText, rowLabels=rowLabels,
                   colLabels=colLabels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, scale_y)
    return tbl

# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT PATHS
# ──────────────────────────────────────────────────────────────────────────────
DEST = r"C:\Users\sachi\Desktop\Vision_Research_Paper\.claude"
PDF_PATH = DEST + r"\Adversarial_Robustness_Benchmark_Results.pdf"

# ══════════════════════════════════════════════════════════════════════════════
# BUILD PDF
# ══════════════════════════════════════════════════════════════════════════════
with PdfPages(PDF_PATH) as pdf:

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 1 — TITLE
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("#1A237E")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#1A237E"); ax.axis("off")

    ax.text(0.5, 0.84, "Adversarial Robustness Benchmark",
            ha="center", va="center", fontsize=30, fontweight="bold",
            color="white", transform=ax.transAxes)
    ax.text(0.5, 0.76, "7 Vision Classifiers × 4 Attack Axes + Madry Defense Model",
            ha="center", va="center", fontsize=17, color="#90CAF9", transform=ax.transAxes)
    ax.text(0.5, 0.69, "Full Results Report — All Tables, All Numbers",
            ha="center", va="center", fontsize=13, color="#BBDEFB", transform=ax.transAxes)

    ax.plot([0.1, 0.9], [0.63, 0.63], color="#90CAF9", linewidth=1.5, transform=ax.transAxes)

    info = [
        ("Threat Model",  "L∞,  ε = 8/255 ≈ 0.0314  in [0, 1] pixel space"),
        ("Dataset",       "ImageNet-1k  —  1,000-image class-balanced benchmark  (seed 42)"),
        ("Attacks",       "FGSM · PGD-20 · AutoAttack (standard) · Square (5000 q)"),
        ("Transfer",      "PGD-20 on ResNet-50 surrogate → 6 black-box targets"),
        ("Typographic",   "White-sticker overlay with wrong-class synonym text"),
        ("Corruptions",   "15 ImageNet-C-style types at severity 3"),
        ("Defense",       "Madry PGD-5 adversarial training (ResNet-50, 8 epochs, Kaggle T4)"),
        ("Hardware",      "Single Tesla T4 GPU (Kaggle free tier)"),
        ("Author",        "Sachit Jain  —  2026"),
    ]
    for k, (label, val) in enumerate(info):
        y = 0.56 - k * 0.055
        ax.text(0.22, y, label + ":", ha="right", va="center", fontsize=11,
                color="#90CAF9", fontweight="bold", transform=ax.transAxes)
        ax.text(0.24, y, val, ha="left", va="center", fontsize=11,
                color="white", transform=ax.transAxes)

    ax.text(0.5, 0.07, " · ".join(MODELS),
            ha="center", va="center", fontsize=9, color="#B0BEC5",
            transform=ax.transAxes, style="italic")
    ax.text(0.5, 0.025, "+ Defense ResNet-50 (Madry PGD-5 adversarial training — NULL RESULT, clean acc collapsed to 6%)",
            ha="center", va="center", fontsize=9, color="#EF9A9A",
            transform=ax.transAxes, style="italic")

    pdf.savefig(fig, dpi=150); plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 2 — EXECUTIVE SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(W, H))
    fig.suptitle("Executive Summary — Clean Accuracy & Gradient-Attack Overview",
                 fontsize=13, fontweight="bold", y=0.99)

    # left: clean accuracy
    ax = axes[0]
    x = np.arange(len(MODELS))
    bars = ax.bar(x, CLEAN_ACC * 100, color=MODEL_COLORS, edgecolor="white",
                  linewidth=0.8, width=0.6, zorder=3)
    ax.axhline(DEFENSE_GRADIENT["clean"] * 100, color=DEFENSE_COLOR,
               linestyle="--", linewidth=2,
               label=f"Defense R50 clean ({DEFENSE_GRADIENT['clean']*100:.1f}%)", zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(MODELS, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
    ax.set_title("Clean (Unattacked) Accuracy"); ax.legend(fontsize=9)
    for bar, v in zip(bars, CLEAN_ACC):
        ax.text(bar.get_x()+bar.get_width()/2, v*100+1, f"{v*100:.1f}",
                ha="center", fontsize=9, fontweight="bold")

    # right: grouped bars per attack
    ax = axes[1]
    attacks  = ["clean","fgsm","pgd","autoattack","square"]
    atk_lbls = ["Clean","FGSM","PGD-20","AutoAttack","Square"]
    width = 0.13
    xb = np.arange(len(MODELS))
    for i,(atk,lbl) in enumerate(zip(attacks,atk_lbls)):
        vals = [GRADIENT[atk][j]*100 for j in range(len(MODELS))]
        ax.bar(xb+(i-2)*width, vals, width, color=ATTACK_COLORS[atk],
               label=lbl, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(xb); ax.set_xticklabels(MODELS, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
    ax.set_title("All 4 Gradient Attacks — per Model")
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    pdf.savefig(fig, dpi=150); plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 3 — GRADIENT TABLE  (full width, generous height)
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(W, H))
    fig.suptitle("Axis 1 — Gradient Attacks: Full Accuracy Table (8 Models)",
                 fontsize=13, fontweight="bold", y=0.98)

    ax_tbl = fig.add_axes([0.01, 0.08, 0.98, 0.86])

    col_labels  = ["Clean", "FGSM", "PGD-20", "AutoAttack", "Square"]
    row_labels  = MODELS + ["★ Defense R50"]
    table_data  = []
    for i in range(len(MODELS)):
        table_data.append([GRADIENT["clean"][i], GRADIENT["fgsm"][i], GRADIENT["pgd"][i],
                           GRADIENT["autoattack"][i], GRADIENT["square"][i]])
    table_data.append([DEFENSE_GRADIENT["clean"], DEFENSE_GRADIENT["fgsm"],
                       DEFENSE_GRADIENT["pgd"], DEFENSE_GRADIENT["autoattack"],
                       DEFENSE_GRADIENT["square"]])

    cell_text = [[f"{v:.3f}   ({v*100:.1f}%)" for v in row] for row in table_data]

    tbl = build_table(ax_tbl, cell_text, row_labels, col_labels,
                      fontsize=10, scale_y=2.1,
                      title="Accuracy values: format  0.NNN  (NN.N%)  —  Defense row highlighted red  |  Color: green = robust, red = fragile")

    style_header(tbl, 5)
    style_rowlabel(tbl, len(table_data), is_defense_row=len(MODELS))

    for i, row in enumerate(table_data):
        is_def = (i == len(MODELS))
        for j, v in enumerate(row):
            if is_def:
                tbl[i+1, j].set_facecolor("#FFCDD2")
            else:
                tbl[i+1, j].set_facecolor(rg(v))

    # footnote
    fig.text(0.5, 0.02,
             "PGD: 20 steps, step=2/255, random start.  AutoAttack: APGD-CE + APGD-T + FAB-T + Square (200-img subset).  "
             "Square: 5000 queries, black-box.  ε = 8/255 L∞ for all attacks.",
             ha="center", fontsize=8.5, color="#424242",
             bbox=dict(boxstyle="round", facecolor="#F5F5F5", edgecolor="#BDBDBD"))

    pdf.savefig(fig, dpi=150); plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 4 — GRADIENT DROP HEATMAP + DEFENSE COMPARISON BARS
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(W, H))
    fig.suptitle("Axis 1 — Gradient Attacks: Drop Heatmap & Defense Comparison",
                 fontsize=13, fontweight="bold")

    # Heatmap (left 55%)
    ax_h = fig.add_axes([0.04, 0.14, 0.50, 0.76])
    drop_matrix = np.array([[GRADIENT["clean"][i] - GRADIENT[a][i]
                              for a in ["fgsm","pgd","autoattack","square"]]
                             for i in range(len(MODELS))]) * 100
    im = ax_h.imshow(drop_matrix, cmap="Reds", aspect="auto", vmin=0, vmax=85)
    ax_h.set_xticks(range(4))
    ax_h.set_xticklabels(["FGSM","PGD-20","AutoAttack","Square"], fontsize=10)
    ax_h.set_yticks(range(len(MODELS)))
    ax_h.set_yticklabels(MODELS, fontsize=10)
    ax_h.set_title("Accuracy Drop (%) — darker red = worse", fontsize=11)
    for i in range(len(MODELS)):
        for j in range(4):
            ax_h.text(j, i, f"{drop_matrix[i,j]:.1f}",
                      ha="center", va="center", fontsize=10, fontweight="bold",
                      color="white" if drop_matrix[i,j] > 50 else "black")
    plt.colorbar(im, ax=ax_h, shrink=0.85, label="Drop (%)")

    # Baseline vs Defense bar (right 40%)
    ax_b = fig.add_axes([0.60, 0.14, 0.37, 0.76])
    atk_keys  = ["clean","fgsm","pgd","autoattack","square"]
    atk_names = ["Clean","FGSM","PGD-20","AutoAttack","Square"]
    b_vals = [GRADIENT[a][0]*100 for a in atk_keys]
    d_vals = [DEFENSE_GRADIENT[a]*100 for a in atk_keys]
    x = np.arange(len(atk_keys)); w = 0.35
    ax_b.bar(x-w/2, b_vals, w, color=BASELINE_COLOR, label="Baseline R50", zorder=3)
    ax_b.bar(x+w/2, d_vals, w, color=DEFENSE_COLOR,  label="Defense R50",  zorder=3)
    ax_b.set_xticks(x); ax_b.set_xticklabels(atk_names, fontsize=9)
    ax_b.set_ylabel("Accuracy (%)"); ax_b.set_ylim(0, 100)
    ax_b.set_title("Baseline vs Defense ResNet-50", fontsize=11)
    for xi,(b,d) in enumerate(zip(b_vals,d_vals)):
        ax_b.text(xi-w/2, b+1.5, f"{b:.1f}", ha="center", fontsize=8.5)
        ax_b.text(xi+w/2, d+1.5, f"{d:.1f}", ha="center", fontsize=8.5, color=DEFENSE_COLOR)
    ax_b.legend(fontsize=9)

    fig.text(0.5, 0.03,
             "Defense 'wins' on PGD/AA/Square (+0.5–1.1 pp) are noise at floor — both models already at ≈0%.  "
             "Clean accuracy: 78.1% → 6.0%  (−72 pp).  Net result: total regression.",
             ha="center", fontsize=9, color="#B71C1C",
             bbox=dict(boxstyle="round", facecolor="#FFEBEE", edgecolor="#EF9A9A"))

    pdf.savefig(fig, dpi=150); plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 5 — TRANSFER TABLE
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(W, H))
    fig.suptitle("Axis 2 — Transfer Attacks: Full Table (Surrogate = ResNet-50 PGD-20)",
                 fontsize=13, fontweight="bold", y=0.98)

    ax_tbl = fig.add_axes([0.01, 0.10, 0.98, 0.82])

    t_cols = ["Clean", "Transfer-Robust", "Accuracy Drop", "Fooling Rate"]
    t_rows = MODELS + ["★ Defense R50"]
    t_data = []
    for i in range(len(MODELS)):
        t_data.append([TRANSFER["clean"][i], TRANSFER["transfer_robust"][i],
                       TRANSFER["drop"][i], TRANSFER["fooling"][i]])
    t_data.append([DEFENSE_TRANSFER["clean"], DEFENSE_TRANSFER["transfer_robust"],
                   DEFENSE_TRANSFER["drop"], DEFENSE_TRANSFER["fooling"]])

    tbl = build_table(ax_tbl,
                      [[f"{v:.3f}   ({v*100:.1f}%)" for v in row] for row in t_data],
                      t_rows, t_cols, fontsize=10, scale_y=2.1,
                      title="Transfer attack: PGD adversarials crafted on ResNet-50, evaluated on all 7 targets  |  Defense highlighted red")

    style_header(tbl, 4)
    style_rowlabel(tbl, len(t_data), is_defense_row=len(MODELS))
    for i, row in enumerate(t_data):
        is_def = (i == len(MODELS))
        for j, v in enumerate(row):
            if is_def:
                tbl[i+1, j].set_facecolor("#FFCDD2")
            else:
                # Clean / Robust → green = high.  Drop / Fooling → green = low
                if j in (0, 1):
                    tbl[i+1, j].set_facecolor(rg(v))
                else:
                    tbl[i+1, j].set_facecolor(rg(1 - v))

    fig.text(0.5, 0.02,
             "Fooling rate = fraction of originally-correct images that flipped under transfer attack.  "
             "ResNet-50 row = white-box self-transfer (should match gradient PGD: 0.003 ≈ 0.004 ✓).  "
             "CNN→CNN transfers better than CNN→Transformer (ViT-B/16 most resistant: 4.9% drop).",
             ha="center", fontsize=8.5, color="#424242",
             bbox=dict(boxstyle="round", facecolor="#F5F5F5", edgecolor="#BDBDBD"))

    pdf.savefig(fig, dpi=150); plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 6 — TRANSFER CHARTS
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(W, H))
    fig.suptitle("Axis 2 — Transfer Attack Charts", fontsize=13, fontweight="bold")

    # left: clean vs transfer-robust
    ax = axes[0]
    x = np.arange(len(MODELS)); w = 0.35
    ax.bar(x-w/2, [v*100 for v in TRANSFER["clean"]], w,
           color=MODEL_COLORS, label="Clean", edgecolor="white", linewidth=0.5, zorder=3)
    ax.bar(x+w/2, [v*100 for v in TRANSFER["transfer_robust"]], w,
           color=[c+"99" for c in MODEL_COLORS], label="Transfer-Robust",
           edgecolor="white", linewidth=0.5, zorder=3, hatch="///")
    ax.axhline(DEFENSE_TRANSFER["clean"]*100, color=DEFENSE_COLOR, linestyle="--",
               linewidth=1.8, label=f"Defense clean ({DEFENSE_TRANSFER['clean']*100:.1f}%)")
    ax.axhline(DEFENSE_TRANSFER["transfer_robust"]*100, color=DEFENSE_COLOR, linestyle=":",
               linewidth=1.8, label=f"Defense transfer-robust ({DEFENSE_TRANSFER['transfer_robust']*100:.1f}%)")
    ax.set_xticks(x); ax.set_xticklabels(MODELS, rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
    ax.set_title("Clean vs Transfer-Robust Accuracy"); ax.legend(fontsize=8)

    # right: fooling rate
    ax = axes[1]
    fool_vals = [v*100 for v in TRANSFER["fooling"]]
    ax.bar(x, fool_vals, color=MODEL_COLORS, edgecolor="white", linewidth=0.6, zorder=3)
    ax.axhline(DEFENSE_TRANSFER["fooling"]*100, color=DEFENSE_COLOR, linestyle="--",
               linewidth=2, label=f"Defense ({DEFENSE_TRANSFER['fooling']*100:.1f}%)")
    ax.set_xticks(x); ax.set_xticklabels(MODELS, rotation=30, ha="right")
    ax.set_ylabel("Fooling Rate (%)"); ax.set_ylim(0, 110)
    ax.set_title("Transfer Fooling Rate")
    for bar, v in zip(ax.patches, fool_vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+1.5, f"{v:.1f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    pdf.savefig(fig, dpi=150); plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 7 — TYPOGRAPHIC TABLE
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(W, H))
    fig.suptitle("Axis 3 — Typographic Attack: Full Table",
                 fontsize=13, fontweight="bold", y=0.98)

    ax_tbl = fig.add_axes([0.01, 0.10, 0.98, 0.82])

    ty_cols = ["Clean", "Robust (untargeted)", "Accuracy Drop", "Fooling Rate", "TASR (targeted)"]
    ty_rows = MODELS + ["★ Defense R50"]
    ty_data = []
    for i in range(len(MODELS)):
        ty_data.append([TYPO["clean"][i], TYPO["robust"][i], TYPO["drop"][i],
                        TYPO["fooling"][i], TYPO["tasr"][i]])
    ty_data.append([DEFENSE_TYPO["clean"], DEFENSE_TYPO["robust"], DEFENSE_TYPO["drop"],
                    DEFENSE_TYPO["fooling"], DEFENSE_TYPO["tasr"]])

    tbl = build_table(ax_tbl,
                      [[f"{v:.3f}   ({v*100:.1f}%)" for v in row] for row in ty_data],
                      ty_rows, ty_cols, fontsize=10, scale_y=2.1,
                      title="★ KEY FINDING: CLIP TASR = 34.2% vs pure-classifier mean ≈ 0.1%  |  Defense highlighted red  |  CLIP TASR cell highlighted orange")

    style_header(tbl, 5)
    style_rowlabel(tbl, len(ty_data), is_defense_row=len(MODELS))
    for i, row in enumerate(ty_data):
        is_def  = (i == len(MODELS))
        is_clip = (i == 6)
        for j, v in enumerate(row):
            if is_def:
                tbl[i+1, j].set_facecolor("#FFCDD2")
            elif is_clip and j == 4:           # CLIP TASR — orange highlight
                tbl[i+1, j].set_facecolor("#FF6F00")
                tbl[i+1, j].set_text_props(color="white", fontweight="bold")
            else:
                if j in (0, 1):
                    tbl[i+1, j].set_facecolor(rg(v))
                elif j == 4:                   # TASR — low = green
                    tbl[i+1, j].set_facecolor(rg(1 - min(v/0.4, 1)))
                else:
                    tbl[i+1, j].set_facecolor(rg(1 - min(v/0.3, 1)))

    fig.text(0.5, 0.02,
             "TASR (Targeted Attack Success Rate) = fraction of images where model predicted the exact class name written on the sticker.  "
             "Pure classifiers cannot read text → TASR ≈ 0.  "
             "CLIP is language-grounded → text is a first-class semantic feature → TASR = 34.2%.",
             ha="center", fontsize=8.5, color="#424242",
             bbox=dict(boxstyle="round", facecolor="#FFF8E1", edgecolor="#FFD54F"))

    pdf.savefig(fig, dpi=150); plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 8 — TYPOGRAPHIC CHARTS
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(W, H))
    fig.suptitle("Axis 3 — Typographic Attack Charts", fontsize=13, fontweight="bold")

    all_labels = MODELS + ["Defense R50"]
    all_colors = MODEL_COLORS + [DEFENSE_COLOR]

    ax = axes[0]
    tasr_vals = [v*100 for v in TYPO["tasr"]] + [DEFENSE_TYPO["tasr"]*100]
    bars = ax.bar(range(len(tasr_vals)), tasr_vals, color=all_colors,
                  edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xticks(range(len(tasr_vals)))
    ax.set_xticklabels(all_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("TASR — Targeted Attack Success Rate (%)")
    ax.set_title("★ CLIP TASR = 34.2%  (pure classifiers ≈ 0%)")
    for bar, v in zip(bars, tasr_vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.4, f"{v:.1f}%",
                ha="center", fontsize=9, fontweight="bold")

    ax = axes[1]
    drop_vals = [v*100 for v in TYPO["drop"]] + [DEFENSE_TYPO["drop"]*100]
    bars2 = ax.bar(range(len(drop_vals)), drop_vals, color=all_colors,
                   edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_xticks(range(len(drop_vals)))
    ax.set_xticklabels(all_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy Drop (clean − robust) %")
    ax.set_title("Typographic Accuracy Drop\n(CLIP −25.3 pp  vs  pure-classifier mean −1.6 pp)")
    for bar, v in zip(bars2, drop_vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.3, f"{v:.1f}%",
                ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    pdf.savefig(fig, dpi=150); plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 9 — CORRUPTION HEATMAP (full page)
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(W, H))
    fig.suptitle("Axis 4 — ImageNet-C-Style Corruptions: Full Heatmap (7 Models × 15 Types, Severity 3)",
                 fontsize=13, fontweight="bold", y=0.99)

    ax_h = fig.add_axes([0.07, 0.20, 0.87, 0.74])
    im = ax_h.imshow(CORRUPTION_MATRIX * 100, cmap="RdYlGn", aspect="auto", vmin=0, vmax=80)
    ax_h.set_xticks(range(15))
    ax_h.set_xticklabels([c.replace("_", "\n") for c in CORRUPTION_TYPES], fontsize=9)
    ax_h.set_yticks(range(7))
    ax_h.set_yticklabels(MODELS, fontsize=10)
    ax_h.set_title("Robust Accuracy (%)  —  green = robust, red = fragile", fontsize=11)
    for i in range(7):
        for j in range(15):
            v = CORRUPTION_MATRIX[i, j] * 100
            ax_h.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                      fontweight="bold", color="black" if 20 < v < 65 else "white")
    plt.colorbar(im, ax=ax_h, shrink=0.9, label="Robust Accuracy (%)")

    # Row summary bar (bottom)
    ax_bot = fig.add_axes([0.07, 0.04, 0.87, 0.12])
    mean_c = CORRUPTION_MATRIX.mean(axis=0) * 100
    ax_bot.bar(range(15), mean_c, color=plt.cm.tab20(np.linspace(0,1,15)),
               edgecolor="white", linewidth=0.5)
    ax_bot.set_xticks(range(15))
    ax_bot.set_xticklabels([c.replace("_","\n") for c in CORRUPTION_TYPES], fontsize=7.5)
    ax_bot.set_ylabel("Mean\nRobust\nAcc (%)", fontsize=7)
    ax_bot.set_title("Mean across 7 models per corruption type", fontsize=9)
    for j, v in enumerate(mean_c):
        ax_bot.text(j, v+0.5, f"{v:.0f}", ha="center", fontsize=7.5)

    pdf.savefig(fig, dpi=150); plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 10 — CORRUPTION TABLE
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(W, H))
    fig.suptitle("Axis 4 — Corruptions: Full Numeric Table (Robust Accuracy per Cell)",
                 fontsize=13, fontweight="bold", y=0.98)

    ax_tbl = fig.add_axes([0.01, 0.06, 0.98, 0.88])

    c_cols = [c.replace("_","\n") for c in CORRUPTION_TYPES] + ["Mean"]
    c_rows = MODELS + ["★ Defense R50"]
    c_data = []
    for i in range(7):
        row = list(CORRUPTION_MATRIX[i]) + [CORRUPTION_MEAN[i]]
        c_data.append(row)
    # defense row — per corruption not available; use mean for all cols
    c_data.append([DEFENSE_CORRUPTION_MEAN]*15 + [DEFENSE_CORRUPTION_MEAN])

    tbl = build_table(ax_tbl,
                      [[f"{v:.3f}" for v in row] for row in c_data],
                      c_rows, c_cols, fontsize=7.8, scale_y=1.65,
                      title="Robust accuracy per (model, corruption) at severity 3  |  Defense row: only mean available (3.5%), shown uniform  |  Color: green=robust, red=fragile")

    style_header(tbl, 16, bg="#263238")
    style_rowlabel(tbl, len(c_data), is_defense_row=7)
    for i, row in enumerate(c_data):
        is_def = (i == 7)
        for j, v in enumerate(row):
            if is_def:
                tbl[i+1, j].set_facecolor("#FFCDD2")
            else:
                tbl[i+1, j].set_facecolor(rg(v))

    fig.text(0.5, 0.01,
             "Sanity check: mean drop = 0.263 (outside literature window [0.05, 0.25]) — axis reported as FAIL by runner.  "
             "4/15 corruptions (motion_blur/snow/frost/fog) are NumPy approximations, not canonical ImageNet-C.  "
             "Model rankings are valid; absolute numbers not comparable to published ImageNet-C.",
             ha="center", fontsize=8, color="#424242",
             bbox=dict(boxstyle="round", facecolor="#F5F5F5", edgecolor="#BDBDBD"))

    pdf.savefig(fig, dpi=150); plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 11 — DEFENSE: ALL 4 AXES SIDE BY SIDE
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(W, H))
    fig.suptitle("Defense ResNet-50 vs Baseline ResNet-50 — All 4 Axes",
                 fontsize=13, fontweight="bold")

    w = 0.35

    # 11a Gradient
    ax = axes[0, 0]
    ak = ["clean","fgsm","pgd","autoattack","square"]
    an = ["Clean","FGSM","PGD-20","AutoAttack","Square"]
    bv = [GRADIENT[a][0]*100 for a in ak]
    dv = [DEFENSE_GRADIENT[a]*100 for a in ak]
    x = np.arange(5)
    ax.bar(x-w/2, bv, w, color=BASELINE_COLOR, label="Baseline R50", zorder=3)
    ax.bar(x+w/2, dv, w, color=DEFENSE_COLOR,  label="Defense R50",  zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(an, fontsize=9)
    ax.set_title("Gradient Attacks"); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0,100)
    ax.legend(fontsize=9)
    for xi,(b,d) in enumerate(zip(bv,dv)):
        ax.text(xi-w/2, b+1.5, f"{b:.1f}", ha="center", fontsize=8)
        ax.text(xi+w/2, d+1.5, f"{d:.1f}", ha="center", fontsize=8, color=DEFENSE_COLOR)

    # 11b Transfer
    ax = axes[0, 1]
    tk = ["clean","transfer_robust","drop","fooling"]
    tn = ["Clean","Transfer\nRobust","Drop","Fooling\nRate"]
    bv = [TRANSFER[a][0]*100 for a in tk]
    dv = [DEFENSE_TRANSFER[a]*100 for a in tk]
    x = np.arange(4)
    ax.bar(x-w/2, bv, w, color=BASELINE_COLOR, label="Baseline R50", zorder=3)
    ax.bar(x+w/2, dv, w, color=DEFENSE_COLOR,  label="Defense R50",  zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(tn, fontsize=9)
    ax.set_title("Transfer Attacks"); ax.set_ylabel("Accuracy / Rate (%)"); ax.set_ylim(0,110)
    ax.legend(fontsize=9)
    for xi,(b,d) in enumerate(zip(bv,dv)):
        ax.text(xi-w/2, b+1.5, f"{b:.1f}", ha="center", fontsize=8)
        ax.text(xi+w/2, d+1.5, f"{d:.1f}", ha="center", fontsize=8, color=DEFENSE_COLOR)

    # 11c Typographic
    ax = axes[1, 0]
    tyk = ["clean","robust","drop","fooling","tasr"]
    tyn = ["Clean","Robust","Drop","Fooling\nRate","TASR"]
    bv = [TYPO[a][0]*100 for a in tyk]
    dv = [DEFENSE_TYPO[a]*100 for a in tyk]
    x = np.arange(5)
    ax.bar(x-w/2, bv, w, color=BASELINE_COLOR, label="Baseline R50", zorder=3)
    ax.bar(x+w/2, dv, w, color=DEFENSE_COLOR,  label="Defense R50",  zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(tyn, fontsize=9)
    ax.set_title("Typographic Attack"); ax.set_ylabel("Accuracy / Rate (%)"); ax.set_ylim(0,100)
    ax.legend(fontsize=9)
    for xi,(b,d) in enumerate(zip(bv,dv)):
        ax.text(xi-w/2, b+1.5, f"{b:.1f}", ha="center", fontsize=8)
        ax.text(xi+w/2, d+1.5, f"{d:.1f}", ha="center", fontsize=8, color=DEFENSE_COLOR)

    # 11d Corruption
    ax = axes[1, 1]
    bc = CORRUPTION_MEAN[0]*100
    dc = DEFENSE_CORRUPTION_MEAN*100
    ax.bar(["Baseline R50","Defense R50"], [bc, dc],
           color=[BASELINE_COLOR, DEFENSE_COLOR], edgecolor="white", width=0.45, zorder=3)
    ax.set_title("Mean Corruption Robustness (15 types)")
    ax.set_ylabel("Mean Robust Acc (%)"); ax.set_ylim(0, 70)
    ax.text(0, bc+1, f"{bc:.1f}%", ha="center", fontsize=13, fontweight="bold")
    ax.text(1, dc+1, f"{dc:.1f}%", ha="center", fontsize=13, fontweight="bold", color=DEFENSE_COLOR)

    fig.text(0.5, 0.01,
             "Defense: PGD-5 adversarial training, 8 epochs from scratch, Kaggle T4.  "
             "Clean accuracy: 78.1% → 6.0% (−72 pp).  All apparent wins are floor-effect artifacts.",
             ha="center", fontsize=9, color="#B71C1C",
             bbox=dict(boxstyle="round", facecolor="#FFEBEE", edgecolor="#EF9A9A"))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    pdf.savefig(fig, dpi=150); plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 12 — MASTER TABLE (all 8 models × all axes)
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(W, H))
    fig.suptitle("Master Comparison Table — All 8 Models × All 4 Axes",
                 fontsize=13, fontweight="bold", y=0.98)

    ax = fig.add_axes([0.01, 0.08, 0.98, 0.86])
    ax.axis("off")

    m_cols = [
        "G:Clean","G:FGSM","G:PGD","G:AA","G:Sq",
        "T:Rob","T:Drop","T:Fool",
        "Ty:Rob","Ty:Drop","Ty:TASR",
        "C:Mean",
    ]
    m_rows = MODELS + ["★ Defense R50"]
    m_data = []
    for i in range(len(MODELS)):
        m_data.append([
            GRADIENT["clean"][i], GRADIENT["fgsm"][i], GRADIENT["pgd"][i],
            GRADIENT["autoattack"][i], GRADIENT["square"][i],
            TRANSFER["transfer_robust"][i], TRANSFER["drop"][i], TRANSFER["fooling"][i],
            TYPO["robust"][i], TYPO["drop"][i], TYPO["tasr"][i],
            CORRUPTION_MEAN[i],
        ])
    m_data.append([
        DEFENSE_GRADIENT["clean"],  DEFENSE_GRADIENT["fgsm"],  DEFENSE_GRADIENT["pgd"],
        DEFENSE_GRADIENT["autoattack"], DEFENSE_GRADIENT["square"],
        DEFENSE_TRANSFER["transfer_robust"], DEFENSE_TRANSFER["drop"], DEFENSE_TRANSFER["fooling"],
        DEFENSE_TYPO["robust"], DEFENSE_TYPO["drop"], DEFENSE_TYPO["tasr"],
        DEFENSE_CORRUPTION_MEAN,
    ])

    mt = ax.table(
        cellText=[[f"{v:.3f}" for v in row] for row in m_data],
        rowLabels=m_rows, colLabels=m_cols,
        loc="center", cellLoc="center",
    )
    mt.auto_set_font_size(False); mt.set_fontsize(9.5); mt.scale(1, 2.05)

    section_colors = {
        (0,4): "#1A237E", (5,7): "#1B5E20", (8,10): "#4A148C", (11,11): "#BF360C"
    }
    for j in range(12):
        mt[0, j].set_text_props(color="white", fontweight="bold")
        if j <= 4:   mt[0, j].set_facecolor("#1A237E")
        elif j <= 7: mt[0, j].set_facecolor("#1B5E20")
        elif j <= 10:mt[0, j].set_facecolor("#4A148C")
        else:        mt[0, j].set_facecolor("#BF360C")

    for i, row in enumerate(m_data):
        is_def = (i == len(MODELS))
        mt[i+1, -1].set_facecolor("#FFCDD2" if is_def else "#ECEFF1")
        mt[i+1, -1].set_text_props(fontweight="bold")
        for j, v in enumerate(row):
            if is_def:
                mt[i+1, j].set_facecolor("#FFCDD2")
            else:
                if j in (2,3,4):      mt[i+1,j].set_facecolor(rg(v*20))
                elif j in (6,7):      mt[i+1,j].set_facecolor(rg(1-v))
                elif j == 9:          mt[i+1,j].set_facecolor(rg(1-min(v/0.3,1)))
                elif j == 10:         mt[i+1,j].set_facecolor(rg(1-min(v/0.4,1)))
                else:                 mt[i+1,j].set_facecolor(rg(v))

    legend_patches = [
        mpatches.Patch(color="#1A237E", label="G: Gradient  (Clean/FGSM/PGD/AutoAttack/Square)"),
        mpatches.Patch(color="#1B5E20", label="T: Transfer  (Robust/Drop/FoolingRate)"),
        mpatches.Patch(color="#4A148C", label="Ty: Typographic  (Robust/Drop/TASR)"),
        mpatches.Patch(color="#BF360C", label="C: Corruption mean  (15 types, severity 3)"),
        mpatches.Patch(color="#FFCDD2", label="Defense R50  (highlighted — NULL RESULT, clean=6%)"),
    ]
    ax.legend(handles=legend_patches, loc="lower center", bbox_to_anchor=(0.5, -0.04),
              ncol=3, fontsize=8.5, framealpha=0.95)

    pdf.savefig(fig, dpi=150); plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 13 — RANKINGS + RADAR + KEY FINDINGS
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(W, H))
    fig.suptitle("Rankings & Key Findings", fontsize=13, fontweight="bold")

    # Radar (left)
    categories = ["Clean\nAccuracy","FGSM\nRobustness","Corruption\nRobustness",
                  "Typographic\nRobustness","Transfer\nRobustness"]
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]

    ax_r = fig.add_subplot(1, 2, 1, polar=True)
    ax_r.set_theta_offset(np.pi/2); ax_r.set_theta_direction(-1)
    ax_r.set_xticks(angles[:-1]); ax_r.set_xticklabels(categories, fontsize=8.5)
    ax_r.set_ylim(0, 1)
    ax_r.set_title("Robustness Radar — 7 Baselines\n(higher = better)", fontsize=10, pad=20)

    for i, (model, color) in enumerate(zip(MODELS, MODEL_COLORS)):
        vals = [
            GRADIENT["clean"][i],
            min(GRADIENT["fgsm"][i] / 0.45, 1),
            CORRUPTION_MEAN[i],
            max(0, 1 - TYPO["drop"][i] / 0.26),
            TRANSFER["transfer_robust"][i],
        ]
        v = vals + vals[:1]
        ax_r.plot(angles, v, color=color, linewidth=2, label=model)
        ax_r.fill(angles, v, alpha=0.07, color=color)
    ax_r.legend(loc="lower right", bbox_to_anchor=(1.45, -0.12), fontsize=8)

    # Key findings (right)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")

    findings = [
        ("★ KEY FINDING", "#E65100",
         "CLIP ViT-B/16: TASR = 34.2%  (pure-classifier mean ≈ 0.1%)"),
        ("", "#BF360C",
         "Language grounding is a liability, not a defense, under typographic attacks."),
        ("", "#BF360C",
         "CLIP drop = 25.3 pp  vs  pure-classifier mean = 1.6 pp  (16.3× larger)."),
        (None, None, None),
        ("GRADIENT", "#1A237E",
         "All 7 models → ≈0% under PGD-20 and AutoAttack (strong white-box attacks)."),
        ("", "#1A237E",
         "FGSM: ResNet-50 most resilient (40.8%);  VGG-16 worst (1.1%)."),
        ("", "#1A237E",
         "No gradient masking detected across all 7 models (AA ≤ PGD ✓)."),
        (None, None, None),
        ("TRANSFER", "#1B5E20",
         "CNN→Transformer transfers poorly: ViT-B/16 only 4.9% drop from ResNet-50 surrogate."),
        ("", "#1B5E20",
         "VGG-16 most susceptible to transfer (21.7% drop)."),
        ("", "#1B5E20",
         "Self-transfer R50 0.3% ≈ white-box PGD 0.4%  — harness validated ✓"),
        (None, None, None),
        ("CORRUPTION", "#4A148C",
         "ViT-B/16 most robust overall (mean 58.7%);  VGG-16 least (28.6%)."),
        ("", "#4A148C",
         "Glass-blur hardest (avg drop 61.2%);  elastic-transform easiest (avg drop 1.9%)."),
        (None, None, None),
        ("DEFENSE", "#B71C1C",
         "FAILED — clean accuracy collapsed: 78.1% → 6.0%  (−72 pp)."),
        ("", "#B71C1C",
         "Root cause: 100-class training scored on 1000-class eval label space."),
        ("", "#B71C1C",
         "All apparent wins are floor-effect artifacts (noise at 0%)."),
        (None, None, None),
        ("CLEAN RANKING",  "#263238", "#1 ViT-B/16 79.3%  #2 ConvNeXt-T 78.7%  #3 ResNet-50 78.1%"),
        ("CORRUPT RANKING","#263238", "#1 ViT-B/16 58.7%  #2 Swin-T 55.9%  #3 ConvNeXt-T 54.9%"),
        ("FGSM RANKING",   "#263238", "#1 ResNet-50 40.8%  #2 ConvNeXt-T 29.2%  #3 ViT-B/16 24.4%"),
    ]

    y = 0.98
    for label, lc, text in findings:
        if label is None:
            y -= 0.022; continue
        if label:
            ax2.text(0.0, y, label + ":", fontsize=9, fontweight="bold",
                     color=lc, transform=ax2.transAxes, va="top")
        ax2.text(0.22, y, text, fontsize=8.8, color="#212121" if label else "#424242",
                 transform=ax2.transAxes, va="top",
                 fontweight="bold" if label == "★ KEY FINDING" else "normal")
        y -= 0.042

    plt.tight_layout()
    pdf.savefig(fig, dpi=150); plt.close(fig)

print(f"PDF saved: {PDF_PATH}")
