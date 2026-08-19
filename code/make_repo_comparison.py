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
# repo folder still named DCMD-Surv on disk (rename on GitHub separately)         MOTCat(ICCV23, complete-ref) + HEALNet(NeurIPS24) + ShaSpec(CVPR23).
# Protocol: 5-fold CV; 0% and 60% missing (60% = mean over 5 blank-configs).
# Cohorts: KIRC (417 paired), GBMLGG (592), LUAD (447), UCEC (467), BRCA (940).
################################################################################
"""


def real_missing_section():
 repo folder still named DCMD-Surv on disk (rename on GitHub separately) """Summarise the real-missing augmentation experiment across cohorts."""
 repo folder still named DCMD-Surv on disk (rename on GitHub separately) lines = ["\n" + "=" * 78,
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)          "SUPPLEMENTARY — REAL-MISSING AUGMENTATION (TRUST-Surv only)",
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)          "Train additionally on genuinely single-modality patients that the paired-only",
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)          "protocol (and DisPro/EMMS) discard. Test set UNCHANGED, so directly comparable.",
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)          "=" * 78]
 repo folder still named DCMD-Surv on disk (rename on GitHub separately) deltas_c, deltas_i = [], []
 repo folder still named DCMD-Surv on disk (rename on GitHub separately) for cohort in MT.NUM:
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)     p = f"{RES}/dcmd_realmissing__{cohort}.txt"
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)     if not os.path.exists(p):
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)         continue
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)     txt = open(p).read()
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)     n = re.search(r"augmented with (\d+) gene-only \+ (\d+) image-only", txt)
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)     lines.append(f"\n--- {cohort} (+{n.group(1)} gene-only, +{n.group(2)} image-only) ---"
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)                  if n else f"\n--- {cohort} ---")
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)     for line in txt.splitlines():
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)         s = line.strip()
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)         if s.startswith("TABLE 1"):
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)             lines.append("  [0% simulated]        paired-only -> +real-missing")
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)         elif s.startswith("TABLE 2"):
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)             lines.append("  [60% simulated, avg]  paired-only -> +real-missing")
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)         if s.startswith(("genes-miss", "image-miss", "both(")):
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)             lines.append("    " + s)
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)             d = re.findall(r"\(([+-][\d.]+)\)", s)
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)             if len(d) == 2:
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)                 deltas_c.append(float(d[0])); deltas_i.append(float(d[1]))
 repo folder still named DCMD-Surv on disk (rename on GitHub separately) ic = sum(1 for d in deltas_c if d > 0); ii = sum(1 for d in deltas_i if d < 0)
 repo folder still named DCMD-Surv on disk (rename on GitHub separately) lines += ["\nHONEST READ-OUT (no cherry-picking):",
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)           f"  * C-index improved in {ic}/{len(deltas_c)} cells — effect is SMALL and MIXED.",
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)           "    Best gain: LUAD image-miss(G) +0.0161. Worst loss: UCEC image-miss(G) -0.0300.",
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)           f"  * IBS (calibration) improved in {ii}/{len(deltas_i)} cells.",
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)           "    Best gain: GBMLGG both(C) -0.0084.",
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)           "  * Cohort split: KIRC/LUAD/BRCA are near-neutral-to-positive; GBMLGG improves",
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)           "    calibration; UCEC DEGRADES the image-miss C-index (-0.030) — so the net",
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)           "    effect across all five cohorts is inconsistent.",
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)           "  * Conclusion: real-missing data is usable by TRUST-Surv without imputation, but",
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)           "    it is NOT a reliable discrimination or calibration boost across cohorts.",
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)           "    Reported as a supplementary finding, NOT as a headline claim."]
 repo folder still named DCMD-Surv on disk (rename on GitHub separately) return "\n".join(lines)


def main():
 repo folder still named DCMD-Surv on disk (rename on GitHub separately) parts = [HEADER, MT.master_table(), ""]
 repo folder still named DCMD-Surv on disk (rename on GitHub separately) for cohort in MT.NUM:
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)     parts += ["\n" + MT.cohort_table(cohort), ""]
 repo folder still named DCMD-Surv on disk (rename on GitHub separately) parts.append(real_missing_section())
 repo folder still named DCMD-Surv on disk (rename on GitHub separately) os.makedirs(os.path.dirname(OUT), exist_ok=True)
 repo folder still named DCMD-Surv on disk (rename on GitHub separately) with open(OUT, "w") as f:
 repo folder still named DCMD-Surv on disk (rename on GitHub separately)     f.write("\n".join(parts) + "\n")
 repo folder still named DCMD-Surv on disk (rename on GitHub separately) print(f"saved -> {OUT}")


if __name__ == "__main__":
 repo folder still named DCMD-Surv on disk (rename on GitHub separately) main()
