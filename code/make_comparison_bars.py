"""
Headline comparison figure per cohort: C-index (higher=better) + IBS (lower=better)
across ALL methods, for the 3 missing scenarios (P/G/C), 0% missing. DCMD highlighted.
Cohort-aware -> reuse for future cohorts by adding to NUMBERS.

Usage: python3 make_comparison_bars.py            # all cohorts in NUMBERS
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/home/sbarua/Region_based_segmentation/missing_modality/results/figures"

# (C-index, IBS) per method per scenario; None = not applicable (MOTCat complete-only)
NUMBERS = {
    "KIRC": {
        "zero":     {"P": (0.715, 0.153), "G": (0.617, 0.146), "C": (0.729, 0.158)},
        "mean":     {"P": (0.694, 0.165), "G": (0.691, 0.137), "C": (0.729, 0.158)},
        "KNN":      {"P": (0.715, 0.162), "G": (0.660, 0.158), "C": (0.729, 0.158)},
        "Flex-MoE": {"P": (0.713, 0.162), "G": (0.688, 0.152), "C": (0.723, 0.166)},
        "MUSE":     {"P": (0.682, 0.181), "G": (0.690, 0.149), "C": (0.726, 0.160)},
        "MOTCat":   {"P": None,           "G": None,           "C": (0.724, 0.141)},
        "HEALNet":  {"P": (0.675, 0.160), "G": (0.657, 0.162), "C": (0.701, 0.143)},
        "ShaSpec":  {"P": (0.692, 0.167), "G": (0.683, 0.150), "C": (0.735, 0.166)},
        "DCMD":     {"P": (0.751, 0.121), "G": (0.705, 0.135), "C": (0.764, 0.128)},
    },
    "GBMLGG": {
        "zero":     {"P": (0.742, 0.160), "G": (0.804, 0.131), "C": (0.810, 0.144)},
        "mean":     {"P": (0.778, 0.149), "G": (0.796, 0.127), "C": (0.810, 0.144)},
        "KNN":      {"P": (0.805, 0.140), "G": (0.801, 0.148), "C": (0.810, 0.144)},
        "Flex-MoE": {"P": (0.739, 0.183), "G": (0.778, 0.175), "C": (0.798, 0.163)},
        "MUSE":     {"P": (0.778, 0.169), "G": (0.818, 0.131), "C": (0.814, 0.150)},
        "MOTCat":   {"P": None,           "G": None,           "C": (0.809, 0.133)},
        "HEALNet":  {"P": (0.775, 0.162), "G": (0.798, 0.186), "C": (0.811, 0.151)},
        "ShaSpec":  {"P": (0.740, 0.182), "G": (0.796, 0.143), "C": (0.797, 0.159)},
        "DCMD":     {"P": (0.817, 0.117), "G": (0.823, 0.117), "C": (0.823, 0.129)},
    },
}
ORDER = ["zero", "mean", "KNN", "Flex-MoE", "MUSE", "MOTCat", "HEALNet", "ShaSpec", "DCMD"]
SCEN = [("P", "Genes missing (image-only)"), ("G", "Image missing (gene-only)"), ("C", "Complete")]
BASE_C, OURS_C = "#9aa7b5", "#2f6db0"


def make_cohort(cohort):
    d = NUMBERS[cohort]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2))
    for col, (sk, sname) in enumerate(SCEN):
        for row, (metric, mi, better) in enumerate([("C-index", 0, "higher"), ("IBS", 1, "lower")]):
            ax = axes[row][col]
            names, vals, colors = [], [], []
            for m in ORDER:
                cell = d[m][sk]
                if cell is None:
                    continue
                names.append(m); vals.append(cell[mi])
                colors.append(OURS_C if m == "DCMD" else BASE_C)
            x = np.arange(len(names))
            bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
            # annotate DCMD value
            for b, m, v in zip(bars, names, vals):
                if m == "DCMD":
                    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center",
                            va="bottom" if better == "higher" else "top", fontsize=8, fontweight="bold", color=OURS_C)
            ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            if row == 0:
                ax.set_title(sname, fontsize=10.5)
                lo = min(vals) - 0.02
                ax.set_ylim(max(0.5, lo), max(vals) + 0.03)
            else:
                ax.set_ylim(0, max(vals) * 1.18)
            if col == 0:
                ax.set_ylabel(f"{metric}\n({'higher' if better=='higher' else 'lower'} = better)", fontsize=10)
    fig.tight_layout()   # no title (added as LaTeX caption)
    os.makedirs(FIG, exist_ok=True)
    out = f"{FIG}/fig_comparison_{cohort}.png"
    fig.savefig(out, dpi=165, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


if __name__ == "__main__":
    for c in NUMBERS:
        make_cohort(c)
