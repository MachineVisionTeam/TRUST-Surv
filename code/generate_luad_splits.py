"""
Generate LUAD 5-fold splits + missing configs on the PAIRED set (image ∩ gene ∩
valid survival), matching the EXACT KIRC/GBMLGG protocol:
  - config CSV columns: ID, Event, Status, WSI, Omics, Split
  - Status = event = 1 if OS_STATUS starts with '1' (1:DECEASED) else 0
  - Event  = OS_MONTHS (months)
  - missing = blank WSI/Omics cells (NaN) for a DISJOINT % of TRAIN patients
  - val ALWAYS complete
Drops patients with missing/<=0 OS_MONTHS.
"""
import os, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

OUT = "/home/sbarua/Region_based_segmentation/missing_modality/data_luad/splits"
IMG_PKL = "/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_patient_level/bulkrnabert_data/luad_uni2h_patient_1536d.pkl"
GENE_NPZ = "/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_patient_level/bulkrnabert_data/luad_bulkrnabert_256d.npz"
CLIN = "/mnt/storage7/Dataset_pathomicfusion/LUAD/data/TCGA_LUAD/cbioportal/luad_clinical_patient.tsv"

CONFIGS = [("W0_O0", 0, 0), ("W0_O60", 0, 60), ("W20_O40", 20, 40),
           ("W30_O30", 30, 30), ("W40_O20", 40, 20), ("W60_O0", 60, 0)]
NFOLDS = 5


def _pid(x):
    p = str(x).split("-")
    return "-".join(p[:3]) if len(p) >= 3 else str(x)


def main():
    os.makedirs(OUT, exist_ok=True)
    img = pickle.load(open(IMG_PKL, "rb"))
    img_ids = set(map(str, img.index))
    z = np.load(GENE_NPZ, allow_pickle=True)
    gene_ids = set(str(x) for x in z["patient_ids"])

    cl = pd.read_csv(CLIN, sep="\t", comment="#")
    cl["pid"] = cl["PATIENT_ID"].apply(_pid)
    cl["t"] = pd.to_numeric(cl["OS_MONTHS"], errors="coerce")
    cl["e"] = cl["OS_STATUS"].astype(str).str.startswith("1").astype(int)   # 1:DECEASED
    cl = cl.dropna(subset=["t"])
    cl = cl[cl["t"] > 0].drop_duplicates("pid").set_index("pid")

    paired = sorted(set(map(_pid, img_ids)) & set(map(_pid, gene_ids)) & set(cl.index))
    print(f"paired patients (with valid survival): {len(paired)}  events={int(cl.loc[paired,'e'].sum())}")
    base = pd.DataFrame({"ID": paired,
                         "Event": cl.loc[paired, "t"].to_numpy(float),
                         "Status": cl.loc[paired, "e"].to_numpy(int)}).reset_index(drop=True)

    skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=0)
    folds = list(skf.split(base["ID"], base["Status"]))
    for fold, (tr_pos, va_pos) in enumerate(folds):
        for ci, (cfg, wblank, oblank) in enumerate(CONFIGS):
            df = base.copy()
            df["WSI"] = "present"; df["Omics"] = "present"; df["Split"] = "train"
            df.loc[va_pos, "Split"] = "val"
            tr_list = list(tr_pos)
            rng = np.random.RandomState(10_000 * fold + ci)
            rng.shuffle(tr_list)
            n = len(tr_list)
            nw = int(round(n * wblank / 100.0)); no = int(round(n * oblank / 100.0))
            df.loc[tr_list[:nw], "WSI"] = np.nan
            df.loc[tr_list[nw:nw + no], "Omics"] = np.nan
            df[["ID", "Event", "Status", "WSI", "Omics", "Split"]].to_csv(
                os.path.join(OUT, f"LUAD_fold_{fold}_{cfg}.csv"), index=False)
    print(f"wrote {NFOLDS * len(CONFIGS)} config CSVs -> {OUT}")


if __name__ == "__main__":
    main()
