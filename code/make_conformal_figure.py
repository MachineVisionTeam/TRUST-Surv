"""
Money figure for the conformal reliability STAR, from results/conformal_summary.csv.

Panel A: coverage at the 90% target in the MISSING-modality scenarios (avg of
         genes-missing + image-missing), per cohort — naive (raw model) vs
         conformal. Dashed line = 90% target. Shows naive dips BELOW target
         (overconfident) and conformal restores >= target.
Panel B: guarantee across levels — achieved coverage vs target (80/90/95),
         averaged over cohorts and missing scenarios, for naive vs conformal.
         Diagonal = ideal; shaded region below = invalid (overconfident).

Usage: python3 make_conformal_figure.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/home/sbarua/Region_based_segmentation/missing_modality/results"
CSV = os.path.join(RES, "conformal_summary.csv")
FIG = os.path.join(RES, "figures")
COHORTS = ["KIRC", "GBMLGG", "LUAD", "UCEC", "BRCA"]
MISS = ["genes_missing", "image_missing"]
NAIVE_C, OURS_C = "#9aa7b5", "#2f6db0"


def load():
    rows = {}
    with open(CSV) as f:
        for ln in f.readlines()[1:]:
            c, a, s, nv, cf, wd = ln.strip().split(",")
            rows[(c, float(a), s)] = (min(float(nv), 1.0), min(float(cf), 1.0))  # clip to 1.0
    return rows


def main():
    r = load()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Panel A: 90% target, missing scenarios averaged, per cohort ----
    a90 = 0.10
    naive = [np.mean([r[(c, a90, s)][0] for s in MISS]) for c in COHORTS]
    conf = [np.mean([r[(c, a90, s)][1] for s in MISS]) for c in COHORTS]
    x = np.arange(len(COHORTS)); w = 0.38
    axA.bar(x - w/2, naive, w, color=NAIVE_C, label="Naive (raw model)", edgecolor="white")
    axA.bar(x + w/2, conf, w, color=OURS_C, label="Conformal (ours)", edgecolor="white")
    axA.axhline(0.90, ls="--", color="#c0392b", lw=1.4, label="90% target")
    for xi, v in zip(x - w/2, naive):
        axA.text(xi, v - 0.03, f"{v:.2f}", ha="center", va="top", fontsize=7.5, color="#4a4a4a")
    axA.set_xticks(x); axA.set_xticklabels(COHORTS, fontsize=9)
    axA.set_ylim(0.6, 1.02); axA.set_ylabel("Empirical coverage")
    axA.set_title("Coverage at 90% target — missing-modality scenarios\n(avg of genes-miss + image-miss)",
                  fontsize=10.5)
    axA.legend(fontsize=8.5, loc="lower right"); axA.grid(axis="y", alpha=0.25)

    # ---- Panel B: achieved vs target across levels (avg cohorts+missing scen) ----
    targets = [0.80, 0.90, 0.95]; alphas = [0.20, 0.10, 0.05]
    naive_l, conf_l = [], []
    for a in alphas:
        naive_l.append(np.mean([r[(c, a, s)][0] for c in COHORTS for s in MISS]))
        conf_l.append(np.mean([r[(c, a, s)][1] for c in COHORTS for s in MISS]))
    axB.fill_between([0.70, 1.0], [0.70, 1.0], 0.0, color="#f2d7d5", alpha=0.5,
                     label="invalid (overconfident)")
    axB.plot([0.70, 1.0], [0.70, 1.0], "k--", lw=1.2, label="ideal")
    axB.plot(targets, naive_l, "o-", color=NAIVE_C, lw=2, ms=7, label="Naive (raw model)")
    axB.plot(targets, conf_l, "s-", color=OURS_C, lw=2, ms=7, label="Conformal (ours)")
    axB.set_xlim(0.78, 0.97); axB.set_ylim(0.70, 1.0)
    axB.set_xticks(targets); axB.set_xlabel("Target coverage")
    axB.set_ylabel("Achieved coverage")
    axB.set_title("Guarantee across levels (avg over cohorts, missing scenarios)", fontsize=10.5)
    axB.legend(fontsize=8.5, loc="upper left"); axB.grid(alpha=0.25)

    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    out = os.path.join(FIG, "fig_conformal_reliability.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
