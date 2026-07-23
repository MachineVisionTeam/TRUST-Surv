"""
Generate clean combined comparison tables (current baseline set, no DisPro):
  results/comparison_<COHORT>.txt   (per cohort, all methods, 0% + 60%)
  results/MASTER_comparison.txt      (both cohorts side by side, complete scenario)
DCMD marked with *. Numbers hardcoded (single source of truth) -> add cohorts here.
"""
import os

R = "/home/sbarua/Region_based_segmentation/missing_modality/results"
# NUM[cohort][setting][method][scenario] = (C-index, IBS); None = n/a (MOTCat)
NUM = {
    "KIRC": {
        "0%": {
            "zero-fill":  {"P": (0.715, 0.153), "G": (0.617, 0.146), "C": (0.729, 0.158)},
            "mean-imp":   {"P": (0.694, 0.165), "G": (0.691, 0.137), "C": (0.729, 0.158)},
            "KNN-imp":    {"P": (0.715, 0.162), "G": (0.660, 0.158), "C": (0.729, 0.158)},
            "Flex-MoE":   {"P": (0.713, 0.162), "G": (0.688, 0.152), "C": (0.723, 0.166)},
            "MUSE":       {"P": (0.682, 0.181), "G": (0.690, 0.149), "C": (0.726, 0.160)},
            "MOTCat":     {"P": None,           "G": None,           "C": (0.724, 0.141)},
            "HEALNet":    {"P": (0.675, 0.160), "G": (0.657, 0.162), "C": (0.701, 0.143)},
            "ShaSpec":    {"P": (0.692, 0.167), "G": (0.683, 0.150), "C": (0.735, 0.166)},
            "DCMD-Surv*": {"P": (0.751, 0.121), "G": (0.705, 0.135), "C": (0.764, 0.128)},
        },
        "60%": {
            "zero-fill":  {"P": (0.709, 0.148), "G": (0.663, 0.141), "C": (0.734, 0.145)},
            "mean-imp":   {"P": (0.706, 0.155), "G": (0.673, 0.147), "C": (0.736, 0.149)},
            "KNN-imp":    {"P": (0.705, 0.155), "G": (0.654, 0.162), "C": (0.717, 0.156)},
            "Flex-MoE":   {"P": (0.693, 0.169), "G": (0.638, 0.181), "C": (0.723, 0.161)},
            "MUSE":       {"P": (0.684, 0.159), "G": (0.646, 0.168), "C": (0.713, 0.149)},
            "HEALNet":    {"P": (0.668, 0.158), "G": (0.657, 0.155), "C": (0.705, 0.148)},
            "ShaSpec":    {"P": (0.689, 0.167), "G": (0.620, 0.173), "C": (0.710, 0.158)},
            "DCMD-Surv*": {"P": (0.738, 0.122), "G": (0.706, 0.135), "C": (0.759, 0.132)},
        },
    },
    "GBMLGG": {
        "0%": {
            "zero-fill":  {"P": (0.742, 0.160), "G": (0.804, 0.131), "C": (0.810, 0.144)},
            "mean-imp":   {"P": (0.778, 0.149), "G": (0.796, 0.127), "C": (0.810, 0.144)},
            "KNN-imp":    {"P": (0.805, 0.140), "G": (0.801, 0.148), "C": (0.810, 0.144)},
            "Flex-MoE":   {"P": (0.739, 0.183), "G": (0.778, 0.175), "C": (0.798, 0.163)},
            "MUSE":       {"P": (0.778, 0.169), "G": (0.818, 0.131), "C": (0.814, 0.150)},
            "MOTCat":     {"P": None,           "G": None,           "C": (0.809, 0.133)},
            "HEALNet":    {"P": (0.775, 0.162), "G": (0.798, 0.186), "C": (0.811, 0.151)},
            "ShaSpec":    {"P": (0.740, 0.182), "G": (0.796, 0.143), "C": (0.797, 0.159)},
            "DCMD-Surv*": {"P": (0.817, 0.117), "G": (0.823, 0.117), "C": (0.823, 0.129)},
        },
        "60%": {
            "zero-fill":  {"P": (0.791, 0.140), "G": (0.809, 0.130), "C": (0.816, 0.131)},
            "mean-imp":   {"P": (0.782, 0.144), "G": (0.791, 0.136), "C": (0.813, 0.138)},
            "KNN-imp":    {"P": (0.802, 0.134), "G": (0.806, 0.134), "C": (0.806, 0.138)},
            "Flex-MoE":   {"P": (0.763, 0.172), "G": (0.779, 0.169), "C": (0.803, 0.157)},
            "MUSE":       {"P": (0.787, 0.155), "G": (0.792, 0.147), "C": (0.814, 0.132)},
            "HEALNet":    {"P": (0.781, 0.162), "G": (0.789, 0.161), "C": (0.816, 0.147)},
            "ShaSpec":    {"P": (0.774, 0.164), "G": (0.788, 0.155), "C": (0.803, 0.148)},
            "DCMD-Surv*": {"P": (0.814, 0.118), "G": (0.823, 0.117), "C": (0.817, 0.133)},
        },
    },
    "LUAD": {
        "0%": {
            "zero-fill":  {"P": (0.552, 0.224), "G": (0.547, 0.207), "C": (0.566, 0.229)},
            "mean-imp":   {"P": (0.554, 0.231), "G": (0.561, 0.206), "C": (0.566, 0.229)},
            "KNN-imp":    {"P": (0.544, 0.240), "G": (0.566, 0.214), "C": (0.566, 0.229)},
            "Flex-MoE":   {"P": (0.565, 0.228), "G": (0.575, 0.215), "C": (0.576, 0.242)},
            "MUSE":       {"P": (0.554, 0.228), "G": (0.566, 0.183), "C": (0.562, 0.209)},
            "MOTCat":     {"P": None,           "G": None,           "C": (0.559, 0.169)},
            "HEALNet":    {"P": (0.568, 0.203), "G": (0.517, 0.191), "C": (0.553, 0.194)},
            "ShaSpec":    {"P": (0.561, 0.234), "G": (0.581, 0.215), "C": (0.581, 0.237)},
            "DCMD-Surv*": {"P": (0.574, 0.164), "G": (0.560, 0.170), "C": (0.591, 0.161)},
        },
        "60%": {
            "zero-fill":  {"P": (0.554, 0.215), "G": (0.542, 0.187), "C": (0.567, 0.208)},
            "mean-imp":   {"P": (0.562, 0.216), "G": (0.545, 0.200), "C": (0.570, 0.211)},
            "KNN-imp":    {"P": (0.559, 0.217), "G": (0.545, 0.213), "C": (0.569, 0.213)},
            "Flex-MoE":   {"P": (0.544, 0.230), "G": (0.537, 0.226), "C": (0.566, 0.217)},
            "MUSE":       {"P": (0.548, 0.223), "G": (0.525, 0.219), "C": (0.554, 0.215)},
            "HEALNet":    {"P": (0.555, 0.201), "G": (0.544, 0.186), "C": (0.553, 0.193)},
            "ShaSpec":    {"P": (0.553, 0.223), "G": (0.548, 0.227), "C": (0.566, 0.217)},
            "DCMD-Surv*": {"P": (0.558, 0.165), "G": (0.560, 0.170), "C": (0.571, 0.161)},
        },
    },
    "UCEC": {
        "0%": {
            "zero-fill":  {"P": (0.629, 0.104), "G": (0.604, 0.091), "C": (0.698, 0.084)},
            "mean-imp":   {"P": (0.661, 0.093), "G": (0.643, 0.092), "C": (0.698, 0.084)},
            "KNN-imp":    {"P": (0.675, 0.094), "G": (0.611, 0.098), "C": (0.698, 0.084)},
            "Flex-MoE":   {"P": (0.654, 0.095), "G": (0.617, 0.108), "C": (0.675, 0.096)},
            "MUSE":       {"P": (0.637, 0.080), "G": (0.622, 0.084), "C": (0.659, 0.080)},
            "MOTCat":     {"P": None,           "G": None,           "C": (0.674, 0.090)},
            "HEALNet":    {"P": (0.660, 0.080), "G": (0.657, 0.085), "C": (0.684, 0.084)},
            "ShaSpec":    {"P": (0.657, 0.096), "G": (0.654, 0.099), "C": (0.680, 0.087)},
            "DCMD-Surv*": {"P": (0.669, 0.087), "G": (0.672, 0.088), "C": (0.702, 0.082)},
        },
        "60%": {
            "zero-fill":  {"P": (0.646, 0.092), "G": (0.621, 0.096), "C": (0.672, 0.093)},
            "mean-imp":   {"P": (0.635, 0.093), "G": (0.622, 0.096), "C": (0.671, 0.094)},
            "KNN-imp":    {"P": (0.640, 0.090), "G": (0.615, 0.100), "C": (0.661, 0.092)},
            "Flex-MoE":   {"P": (0.624, 0.097), "G": (0.589, 0.110), "C": (0.661, 0.098)},
            "MUSE":       {"P": (0.665, 0.080), "G": (0.595, 0.084), "C": (0.670, 0.080)},
            "HEALNet":    {"P": (0.646, 0.086), "G": (0.614, 0.091), "C": (0.668, 0.084)},
            "ShaSpec":    {"P": (0.650, 0.096), "G": (0.595, 0.105), "C": (0.668, 0.098)},
            "DCMD-Surv*": {"P": (0.651, 0.081), "G": (0.657, 0.085), "C": (0.679, 0.083)},
        },
    },
    "BRCA": {
        "0%": {
            "zero-fill":  {"P": (0.596, 0.135), "G": (0.457, 0.118), "C": (0.581, 0.138)},
            "mean-imp":   {"P": (0.594, 0.134), "G": (0.487, 0.119), "C": (0.581, 0.138)},
            "KNN-imp":    {"P": (0.608, 0.132), "G": (0.518, 0.136), "C": (0.581, 0.138)},
            "Flex-MoE":   {"P": (0.588, 0.136), "G": (0.528, 0.139), "C": (0.579, 0.140)},
            "MUSE":       {"P": (0.506, 0.115), "G": (0.524, 0.110), "C": (0.510, 0.104)},
            "MOTCat":     {"P": None,           "G": None,           "C": (0.578, 0.110)},
            "HEALNet":    {"P": (0.592, 0.117), "G": (0.508, 0.114), "C": (0.570, 0.110)},
            "ShaSpec":    {"P": (0.589, 0.140), "G": (0.491, 0.133), "C": (0.577, 0.143)},
            "DCMD-Surv*": {"P": (0.622, 0.102), "G": (0.542, 0.103), "C": (0.612, 0.103)},
        },
        "60%": {
            "zero-fill":  {"P": (0.592, 0.125), "G": (0.526, 0.114), "C": (0.576, 0.128)},
            "mean-imp":   {"P": (0.594, 0.126), "G": (0.503, 0.121), "C": (0.576, 0.131)},
            "KNN-imp":    {"P": (0.580, 0.132), "G": (0.519, 0.128), "C": (0.557, 0.135)},
            "Flex-MoE":   {"P": (0.594, 0.133), "G": (0.502, 0.138), "C": (0.574, 0.137)},
            "MUSE":       {"P": (0.526, 0.104), "G": (0.542, 0.104), "C": (0.554, 0.104)},
            "HEALNet":    {"P": (0.613, 0.112), "G": (0.524, 0.118), "C": (0.573, 0.114)},
            "ShaSpec":    {"P": (0.598, 0.133), "G": (0.508, 0.139), "C": (0.579, 0.139)},
            "DCMD-Surv*": {"P": (0.624, 0.102), "G": (0.510, 0.104), "C": (0.619, 0.103)},
        },
    },
}
ORDER = ["zero-fill", "mean-imp", "KNN-imp", "Flex-MoE", "MUSE", "MOTCat", "HEALNet", "ShaSpec", "DCMD-Surv*"]
SCEN = [("P", "genes-miss(P)"), ("G", "image-miss(G)"), ("C", "both(C)")]


def cell(v):
    return f"{v[0]:.3f}/{v[1]:.3f}" if v else "   -   /  -  "


def cohort_table(cohort):
    lines = ["=" * 78,
             f"{cohort} — DCMD-Surv vs baselines. C-index/IBS (higher C / lower IBS = better).",
             "Feature-matched (UNI2h + BulkRNABert pooled). DCMD-Surv* = ours. MOTCat=complete-only ref.",
             "=" * 78]
    for setting in ["0%", "60%"]:
        d = NUM[cohort][setting]
        lines.append(f"\n--- {setting} MISSING ---")
        lines.append(f"  {'METHOD':13s} | {'P (C/IBS)':13s} | {'G (C/IBS)':13s} | {'C (C/IBS)':13s}")
        for m in ORDER:
            if m not in d:
                continue
            r = d[m]
            lines.append(f"  {m:13s} | {cell(r['P'])} | {cell(r['G'])} | {cell(r['C'])}")
    return "\n".join(lines)


def master_table():
    """All cohorts side by side. Cohort-agnostic -> new cohorts auto-included."""
    cohorts = list(NUM.keys())
    lines = ["=" * (18 + 16 * len(cohorts)),
             "MASTER COMPARISON — DCMD-Surv vs baselines across cohorts. C-index / IBS.",
             "=" * (18 + 16 * len(cohorts))]
    for setting in ["0%", "60%"]:
        lines.append(f"\n--- {setting} MISSING · COMPLETE scenario (C) ---")
        lines.append("  " + f"{'METHOD':13s} | " + " | ".join(f"{c:13s}" for c in cohorts))
        for m in ORDER:
            row = []
            any_val = False
            for c in cohorts:
                d = NUM[c][setting].get(m)
                if d:
                    row.append(cell(d["C"])); any_val = True
                else:
                    row.append("   -   /  -  ")
            if any_val:
                lines.append(f"  {m:13s} | " + " | ".join(row))
    lines.append("\nWHAT HOLDS (verified cell-by-cell against every baseline):")
    lines.append("  * COMPLETE (C) scenario: DCMD-Surv has the HIGHEST C-index on all five")
    lines.append("    cohorts at BOTH 0% and 60% missing (10/10 cells).")
    lines.append("  * IBS (calibration): DCMD-Surv is the LOWEST (or tied) in 23 of the 30 cells")
    lines.append("    across all cohorts/settings/scenarios. The exception cohort is UCEC, where")
    lines.append("    MUSE is better-calibrated (lower IBS) in all 6 cells.")
    lines.append("\nEXCEPTIONS (stated explicitly — DCMD is NOT best in these cells):")
    lines.append("  C-index losses:")
    lines.append("  - LUAD 0% image-miss(G) : ShaSpec 0.581 > DCMD 0.560")
    lines.append("  - LUAD 60% genes-miss(P): mean-imp 0.562 > DCMD 0.558")
    lines.append("  - UCEC 0% genes-miss(P) : KNN 0.675 > DCMD 0.669")
    lines.append("  - UCEC 60% genes-miss(P): MUSE 0.665 > DCMD 0.651")
    lines.append("  - BRCA 60% image-miss(G): MUSE 0.542 > DCMD 0.510")
    lines.append("  IBS losses:")
    lines.append("  - GBMLGG 60% complete   : zero-fill 0.131 < DCMD 0.133")
    lines.append("  - UCEC (all 6 cells)    : MUSE ~0.080-0.084 < DCMD 0.081-0.088")
    lines.append("  LUAD is the hardest cohort (all methods 0.52-0.59) and BRCA gene-only is")
    lines.append("  near-random for everyone (~0.51-0.54); those single-modality scenarios sit")
    lines.append("  within fold-noise of the imputation baselines. UCEC is well-calibrated for")
    lines.append("  every method (IBS ~0.08), so its calibration gaps are small in absolute terms.")
    lines.append("\nPer-scenario (P/G/C) detail for each cohort: comparison_<COHORT>.txt")
    lines.append("MOTCat is a complete-data reference (no missing scenarios), as in EMMS.")
    return "\n".join(lines)


def main():
    os.makedirs(R, exist_ok=True)
    for cohort in NUM:
        with open(os.path.join(R, f"comparison_{cohort}.txt"), "w") as f:
            f.write(cohort_table(cohort) + "\n")
        print(f"saved -> {R}/comparison_{cohort}.txt")
    with open(os.path.join(R, "MASTER_comparison.txt"), "w") as f:
        f.write(master_table() + "\n")
    print(f"saved -> {R}/MASTER_comparison.txt")


if __name__ == "__main__":
    main()
