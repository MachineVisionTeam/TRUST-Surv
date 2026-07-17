"""
Compute DisPro-native IBS (and cal-MAD) on KIRC from the dumped discrete-time
survival curves, using the SAME methodology as our DCMD calibration
(calibration_mm): train-fold censoring distribution + eval_time_grid.

DisPro is discrete-time (4 bins). S[:,k] = P(T > q_bins[k+1]); we turn it into a
right-continuous step survival function and evaluate on the shared time grid.

status -> scenario:  wsi = genes-missing(P) | omics = image-missing(G) | wsi-omics = both(C)
"""
import os, sys, glob
import numpy as np
import pandas as pd

CODE_DIR = "/home/sbarua/Region_based_segmentation/missing_modality/code"
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
from calibration_mm import eval_time_grid, compute_ibs, calibration_mad  # noqa: E402

import glob as _glob
_ARGV = sys.argv[1:] if len(sys.argv) > 1 else []
_RUN = "results_run" if "--uni2h" in _ARGV else "results_run_uni_prepath"
_TAG = "uni2h" if "--uni2h" in _ARGV else "native_prepath"
WSIOMICS = ("/home/sbarua/Region_based_segmentation/missing_modality/replication/dispro/"
            f"{_RUN}/WSI_Omics/KIRC")
CSV = ("/home/sbarua/Region_based_segmentation/missing_modality/replication/dispro/DisPro/"
       "splits/csv_missing_cleaned/KIRC/KIRC_fold_{fold}_{suffix}.csv")
RESULTS_DIR = "/home/sbarua/Region_based_segmentation/missing_modality/results"

STATUS2SCEN = {"wsi": "genes_missing", "omics": "image_missing", "wsi-omics": "both"}
NFOLDS = 5
CONFIGS_0 = ["W0_O0"]
CONFIGS_60 = ["W0_O60", "W20_O40", "W30_O30", "W40_O20", "W60_O0"]


def dump_dir(suffix):
    hits = _glob.glob(os.path.join(WSIOMICS, "[[]DisPro[]]-[[]{}[]]-*".format(suffix)))
    return os.path.join(hits[0], "ibs_dump") if hits else None


def step_survival(S, q_bins, times):
    """Right-continuous discrete survival at `times`.
    S: (n, nbins) with S[:,k] = survival prob at bin edge q_bins[k+1].
    Returns (n, len(times))."""
    S = np.asarray(S, float)
    n, nb = S.shape
    edges = np.asarray(q_bins, float)[1:nb + 1]        # upper edge of each bin
    out = np.ones((n, len(times)))
    for j, tau in enumerate(times):
        k = np.searchsorted(edges, tau, side="right") - 1   # last edge <= tau
        out[:, j] = 1.0 if k < 0 else S[:, min(k, nb - 1)]
    return out


def train_labels(fold, suffix):
    d = pd.read_csv(CSV.format(fold=fold, suffix=suffix))
    tr = d[d["Split"] == "train"]
    return tr["Event"].to_numpy(float), tr["Status"].to_numpy(float)


def eval_configs(configs, label):
    """Average IBS/cal-MAD over the given missing-configs x folds, per scenario."""
    per = {s: {"ibs": [], "mad": []} for s in STATUS2SCEN.values()}
    for suffix in configs:
        dd = dump_dir(suffix)
        if dd is None:
            print(f"[warn] no dump dir for {suffix}"); continue
        for status, scen in STATUS2SCEN.items():
            for fold in range(NFOLDS):
                f = os.path.join(dd, f"{status}_fold{fold}.npz")
                if not os.path.exists(f):
                    print(f"[warn] missing {status} fold{fold} {suffix}"); continue
                z = np.load(f)
                S, ev, cen, qb = z["S"], z["event"], z["censor"], z["q_bins"]
                te_t = ev.astype(float); te_e = (1.0 - cen).astype(float)
                tr_t, tr_e = train_labels(fold, suffix)
                times = eval_time_grid(te_t, te_e)
                teS = step_survival(S, qb, times)
                ibs = compute_ibs(tr_t, tr_e, te_t, te_e, teS, times)
                t0 = float(np.median(times))
                S_t0 = step_survival(S, qb, np.array([t0]))[:, 0]
                mad, _, _ = calibration_mad(S_t0, te_t, te_e, t0)
                per[scen]["ibs"].append(ibs); per[scen]["mad"].append(mad)
    sn = {"genes_missing": "genes-miss(P)", "image_missing": "image-miss(G)", "both": "both(C)"}
    out = [f"\n{label}"]
    res = {}
    for scen in ["genes_missing", "image_missing", "both"]:
        ibs = np.nanmean(per[scen]["ibs"]); mad = np.nanmean(per[scen]["mad"])
        res[scen] = (ibs, mad)
        out.append(f"  {sn[scen]:14s} IBS={ibs:.4f}  cal-MAD={mad:.4f}")
    return res, out


def main():
    header = ["=" * 64, "DisPro-NATIVE (PrePATH) — IBS + cal-MAD, KIRC 5-fold",
              "from dumped discrete-time survival curves (same grid/censoring as ours)", "=" * 64]
    print("\n".join(header))
    _, l0 = eval_configs(CONFIGS_0, "TABLE 1 — 0% missing (W0_O0)")
    _, l60 = eval_configs(CONFIGS_60, "TABLE 2 — 60% missing (avg of 5 configs)")
    lines = header + l0 + l60
    print("\n".join(l0)); print("\n".join(l60))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outp = os.path.join(RESULTS_DIR, f"dispro_{_TAG}_ibs__KIRC.txt")
    with open(outp, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nsaved -> {outp}")


if __name__ == "__main__":
    main()
