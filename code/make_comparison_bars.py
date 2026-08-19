"""
Headline comparison figure per cohort: C-index (higher=better) + IBS (lower=better)
across ALL methods, for the 3 missing scenarios (P/G/C), 0% missing. TRUST highlighted.
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
        "TRUST":     {"P": (0.751, 0.121), "G": (0.705, 0.135), "C": (0.764, 0.128)},
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
        "TRUST":     {"P": (0.817, 0.117), "G": (0.823, 0.117), "C": (0.823, 0.129)},
    },
    "LUAD": {
        "zero":     {"P": (0.552, 0.224), "G": (0.547, 0.207), "C": (0.566, 0.229)},
        "mean":     {"P": (0.554, 0.231), "G": (0.561, 0.206), "C": (0.566, 0.229)},
        "KNN":      {"P": (0.544, 0.240), "G": (0.566, 0.214), "C": (0.566, 0.229)},
        "Flex-MoE": {"P": (0.565, 0.228), "G": (0.575, 0.215), "C": (0.576, 0.242)},
        "MUSE":     {"P": (0.554, 0.228), "G": (0.566, 0.183), "C": (0.562, 0.209)},
        "MOTCat":   {"P": None,           "G": None,           "C": (0.559, 0.169)},
        "HEALNet":  {"P": (0.568, 0.203), "G": (0.517, 0.191), "C": (0.553, 0.194)},
        "ShaSpec":  {"P": (0.561, 0.234), "G": (0.581, 0.215), "C": (0.581, 0.237)},
        "TRUST":     {"P": (0.574, 0.164), "G": (0.560, 0.170), "C": (0.591, 0.161)},
    },
    "UCEC": {
        "zero":     {"P": (0.629, 0.104), "G": (0.604, 0.091), "C": (0.698, 0.084)},
        "mean":     {"P": (0.661, 0.093), "G": (0.643, 0.092), "C": (0.698, 0.084)},
        "KNN":      {"P": (0.675, 0.094), "G": (0.611, 0.098), "C": (0.698, 0.084)},
        "Flex-MoE": {"P": (0.654, 0.095), "G": (0.617, 0.108), "C": (0.675, 0.096)},
        "MUSE":     {"P": (0.637, 0.080), "G": (0.622, 0.084), "C": (0.659, 0.080)},
        "MOTCat":   {"P": None,           "G": None,           "C": (0.674, 0.090)},
        "HEALNet":  {"P": (0.660, 0.080), "G": (0.657, 0.085), "C": (0.684, 0.084)},
        "ShaSpec":  {"P": (0.657, 0.096), "G": (0.654, 0.099), "C": (0.680, 0.087)},
        "TRUST":     {"P": (0.669, 0.087), "G": (0.672, 0.088), "C": (0.702, 0.082)},
    },
    "BRCA": {
        "zero":     {"P": (0.596, 0.135), "G": (0.457, 0.118), "C": (0.581, 0.138)},
        "mean":     {"P": (0.594, 0.134), "G": (0.487, 0.119), "C": (0.581, 0.138)},
        "KNN":      {"P": (0.608, 0.132), "G": (0.518, 0.136), "C": (0.581, 0.138)},
        "Flex-MoE": {"P": (0.588, 0.136), "G": (0.528, 0.139), "C": (0.579, 0.140)},
        "MUSE":     {"P": (0.506, 0.115), "G": (0.524, 0.110), "C": (0.510, 0.104)},
        "MOTCat":   {"P": None,           "G": None,           "C": (0.578, 0.110)},
        "HEALNet":  {"P": (0.592, 0.117), "G": (0.508, 0.114), "C": (0.570, 0.110)},
        "ShaSpec":  {"P": (0.589, 0.140), "G": (0.491, 0.133), "C": (0.577, 0.143)},
        "TRUST":     {"P": (0.622, 0.102), "G": (0.542, 0.103), "C": (0.612, 0.103)},
    },
}
ORDER = ["zero", "mean", "KNN", "Flex-MoE", "MUSE", "MOTCat", "HEALNet", "ShaSpec", "TRUST"]
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
                colors.append(OURS_C if m == "TRUST" else BASE_C)
            x = np.arange(len(names))
            bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
            # annotate TRUST value
            for b, m, v in zip(bars, names, vals):
                if m == "TRUST":
                    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center",
                            va="bottom" if better == "higher" else "top", fontsize=8, fontweight="bold", color=OURS_C)
            ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            if row == 0:
                ax.set_title(sname, fontsize=10.5)
                lo = min(vals) - 0.02
                ax.set_ylim(max(0.4, lo), max(vals) + 0.03)  # 0.4 floor so sub-0.5 C-index bars (e.g. BRCA gene-only) stay visible
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
