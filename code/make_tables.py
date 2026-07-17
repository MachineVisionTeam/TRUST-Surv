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
    """Both cohorts side by side, complete(C) + the two missing scenarios, 0%."""
    lines = ["=" * 96,
             "MASTER COMPARISON — DCMD-Surv vs baselines across cohorts (0% missing). C-index / IBS.",
             "=" * 96,
             f"  {'METHOD':13s} | {'KIRC P':13s} {'KIRC G':13s} {'KIRC C':13s} | {'GBMLGG P':13s} {'GBMLGG G':13s} {'GBMLGG C':13s}"]
    for m in ORDER:
        k = NUM["KIRC"]["0%"].get(m); g = NUM["GBMLGG"]["0%"].get(m)
        if not k:
            continue
        lines.append(f"  {m:13s} | {cell(k['P'])} {cell(k['G'])} {cell(k['C'])} | "
                     f"{cell(g['P'])} {cell(g['G'])} {cell(g['C'])}")
    lines.append("\nNOTE: DCMD-Surv* (ours) has the highest C-index AND lowest IBS in the complete")
    lines.append("scenario on BOTH cohorts, and the lowest IBS across all scenarios. 60% tables in")
    lines.append("the per-cohort files (comparison_KIRC.txt, comparison_GBMLGG.txt).")
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
