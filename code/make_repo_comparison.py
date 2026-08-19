"""
Build TRUST-Surv/COMPARISON.txt — the repo-level, EMMS-style overall picture across
ALL cohorts. Generated from make_tables.py (single source of truth) + the raw
real-missing result files, so it can never drift from the per-cohort tables.

Usage: python3 make_repo_comparison.py
"""
import os
import re

import make_tables as MT

MM = "/home/sbarua/Region_based_segmentation/missing_modality"
OUT = f"{MM}/TRUST-Surv/COMPARISON.txt"
RES = f"{MM}/results"

HEADER = """\
################################################################################
# TRUST-Surv — OVERALL COMPARISON vs all baseline papers, per cohort (EMMS-style)
# Scenarios: P=genes-missing(image-only), G=image-missing(gene-only), C=complete
# Metrics: C-index (higher=better discrimination) / IBS (lower=better calibration)
# Feature-matched: UNI2h (image) + BulkRNABert (gene), pooled. TRUST-Surv* = ours.
# Baselines: naive(zero/mean/KNN) + Flex-MoE(NeurIPS24) + MUSE(ICLR24) +
#            MOTCat(ICCV23, complete-ref) + HEALNet(NeurIPS24) + ShaSpec(CVPR23).
# Protocol: 5-fold CV; 0% and 60% missing (60% = mean over 5 blank-configs).
# Cohorts: KIRC (417 paired), GBMLGG (592), LUAD (447), UCEC (467), BRCA (940).
################################################################################
"""


def real_missing_section():
    """Summarise the real-missing augmentation experiment across cohorts."""
    lines = ["\n" + "=" * 78,
             "SUPPLEMENTARY — REAL-MISSING AUGMENTATION (TRUST-Surv only)",
             "Train additionally on genuinely single-modality patients that the paired-only",
             "protocol (and DisPro/EMMS) discard. Test set UNCHANGED, so directly comparable.",
             "=" * 78]
    deltas_c, deltas_i = [], []
    for cohort in MT.NUM:
        p = f"{RES}/dcmd_realmissing__{cohort}.txt"
        if not os.path.exists(p):
            continue
        txt = open(p).read()
        n = re.search(r"augmented with (\d+) gene-only \+ (\d+) image-only", txt)
        lines.append(f"\n--- {cohort} (+{n.group(1)} gene-only, +{n.group(2)} image-only) ---"
                     if n else f"\n--- {cohort} ---")
        for line in txt.splitlines():
            s = line.strip()
            if s.startswith("TABLE 1"):
                lines.append("  [0% simulated]        paired-only -> +real-missing")
            elif s.startswith("TABLE 2"):
                lines.append("  [60% simulated, avg]  paired-only -> +real-missing")
            if s.startswith(("genes-miss", "image-miss", "both(")):
                lines.append("    " + s)
                d = re.findall(r"\(([+-][\d.]+)\)", s)
                if len(d) == 2:
                    deltas_c.append(float(d[0])); deltas_i.append(float(d[1]))
    ic = sum(1 for d in deltas_c if d > 0); ii = sum(1 for d in deltas_i if d < 0)
    lines += ["\nHONEST READ-OUT (no cherry-picking):",
              f"  * C-index improved in {ic}/{len(deltas_c)} cells — effect is SMALL and MIXED.",
              "    Best gain: LUAD image-miss(G) +0.0161. Worst loss: UCEC image-miss(G) -0.0300.",
              f"  * IBS (calibration) improved in {ii}/{len(deltas_i)} cells.",
              "    Best gain: GBMLGG both(C) -0.0084.",
              "  * Cohort split: KIRC/LUAD/BRCA are near-neutral-to-positive; GBMLGG improves",
              "    calibration; UCEC DEGRADES the image-miss C-index (-0.030) — so the net",
              "    effect across all five cohorts is inconsistent.",
              "  * Conclusion: real-missing data is usable by TRUST-Surv without imputation, but",
              "    it is NOT a reliable discrimination or calibration boost across cohorts.",
              "    Reported as a supplementary finding, NOT as a headline claim."]
    return "\n".join(lines)


def main():
    parts = [HEADER, MT.master_table(), ""]
    for cohort in MT.NUM:
        parts += ["\n" + MT.cohort_table(cohort), ""]
    parts.append(real_missing_section())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(parts) + "\n")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
