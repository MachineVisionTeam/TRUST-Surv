"""
Generate GBMLGG 5-fold splits + missing configs on the PAIRED set (image ∩ gene),
matching the EXACT KIRC/DisPro protocol:
  - config CSV columns: ID, Event, Status, WSI, Omics, Split
  - Status = event = 1 - censored (Paper-1 convention, roi_splits_gbmlgg.py L285)
  - Event  = Survival months (days; internally consistent)
  - missing = blank WSI/Omics cells (NaN) for a DISJOINT % of TRAIN patients
  - val ALWAYS complete (both present)
Configs (WSI-blank%, Omics-blank%): W0_O0(0,0) W0_O60(0,60) W20_O40(20,40)
  W30_O30(30,30) W40_O20(40,20) W60_O0(60,0)
"""
import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

OUT = "/home/sbarua/Region_based_segmentation/missing_modality/data_gbmlgg/splits"
IMG_PKL = "/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_patient_level/bulkrnabert_data/gbmlgg_uni2h_patient_1536d.pkl"
GENE_NPZ = "/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_patient_level/bulkrnabert_data/gbmlgg_bulkrnabert_256d.npz"
CLIN = "/mnt/storage7/Dataset_pathomicfusion/GBMLGG/data/TCGA_GBMLGG/all_dataset.csv"

CONFIGS = [("W0_O0", 0, 0), ("W0_O60", 0, 60), ("W20_O40", 20, 40),
           ("W30_O30", 30, 30), ("W40_O20", 40, 20), ("W60_O0", 60, 0)]
NFOLDS = 5


def main():
    os.makedirs(OUT, exist_ok=True)
    img = pickle.load(open(IMG_PKL, "rb"))
    img_ids = set(map(str, img.index))
    z = np.load(GENE_NPZ, allow_pickle=True)
    gene_ids = set(str(x) for x in z["patient_ids"])
    clin = pd.read_csv(CLIN)
    clin["pid"] = clin["TCGA ID"].astype(str)
    clin["t"] = clin["Survival months"].astype(float)
    clin["e"] = (1 - clin["censored"]).astype(int)       # Paper-1 convention
    clin = clin.dropna(subset=["t", "e"]).drop_duplicates("pid").set_index("pid")

    paired = sorted(img_ids & gene_ids & set(clin.index))
    print(f"paired patients: {len(paired)}  (events={int(clin.loc[paired,'e'].sum())})")
    base = pd.DataFrame({"ID": paired,
                         "Event": clin.loc[paired, "t"].to_numpy(float),
                         "Status": clin.loc[paired, "e"].to_numpy(int)}).reset_index(drop=True)

    skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=0)
    folds = list(skf.split(base["ID"], base["Status"]))

    for fold, (tr_pos, va_pos) in enumerate(folds):
        for ci, (cfg, wblank, oblank) in enumerate(CONFIGS):
            df = base.copy()
            df["WSI"] = "present"; df["Omics"] = "present"; df["Split"] = "train"
            df.loc[va_pos, "Split"] = "val"
            tr_pos_list = list(tr_pos)
            rng = np.random.RandomState(10_000 * fold + ci)   # deterministic per (fold,cfg)
            rng.shuffle(tr_pos_list)
            n = len(tr_pos_list)
            nw = int(round(n * wblank / 100.0))
            no = int(round(n * oblank / 100.0))
            w_ids = tr_pos_list[:nw]                 # WSI blanked
            o_ids = tr_pos_list[nw:nw + no]          # Omics blanked (disjoint)
            df.loc[w_ids, "WSI"] = np.nan
            df.loc[o_ids, "Omics"] = np.nan
            df[["ID", "Event", "Status", "WSI", "Omics", "Split"]].to_csv(
                os.path.join(OUT, f"GBMLGG_fold_{fold}_{cfg}.csv"), index=False)
    print(f"wrote {NFOLDS * len(CONFIGS)} config CSVs -> {OUT}")


if __name__ == "__main__":
    main()
