"""
Missing-modality dataset for LUAD. SAME interface as dataset_mm / dataset_gbmlgg:
load_gene_dict(), load_fold_config(), CONFIGS.

Per patient:
  X_img  (1536,) : pooled UNI2-h (luad_uni2h_patient_1536d.pkl)
  X_gene (256,)  : BulkRNABert embedding
  img_present / gene_present : 1.0 if present in CSV, else 0.0 (blanked)
  t (time)  : CSV 'Event'  (OS_MONTHS)
  e (event) : CSV 'Status' (1 = OS_STATUS 1:DECEASED, 0 = censored)

Paired set = 447 (image ∩ gene ∩ valid survival). Splits by generate_luad_splits.py.
"""
import os
import pickle
import numpy as np
import pandas as pd

LUAD_SPLITS = "/home/sbarua/Region_based_segmentation/missing_modality/data_luad/splits"
IMG_PKL = "/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_patient_level/bulkrnabert_data/luad_uni2h_patient_1536d.pkl"
BULK_NPZ = "/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_patient_level/bulkrnabert_data/luad_bulkrnabert_256d.npz"

CONFIGS = ["W0_O0", "W0_O60", "W20_O40", "W30_O30", "W40_O20", "W60_O0"]

_img_dict = None


def _pid(x):
    p = str(x).split("-")
    return "-".join(p[:3]) if len(p) >= 3 else str(x)


def load_gene_dict():
    z = np.load(BULK_NPZ, allow_pickle=True)
    emb = z["embeddings"]
    assert emb.shape[1] == 256, f"gene dim {emb.shape[1]} != 256"
    return {_pid(p): emb[i].astype(np.float32) for i, p in enumerate(z["patient_ids"])}


def _image_dict():
    global _img_dict
    if _img_dict is None:
        df = pickle.load(open(IMG_PKL, "rb"))
        assert df.shape[1] == 1536, f"image dim {df.shape[1]} != 1536"
        _img_dict = {_pid(pid): df.loc[pid].to_numpy(np.float32) for pid in df.index}
    return _img_dict


def _assemble(sub: pd.DataFrame, gene_dict: dict) -> dict:
    imgd = _image_dict()
    n = len(sub)
    X_img = np.zeros((n, 1536), np.float32)
    X_gene = np.zeros((n, 256), np.float32)
    img_present = np.zeros(n, np.float32)
    gene_present = np.zeros(n, np.float32)
    t = np.zeros(n, np.float32); e = np.zeros(n, np.float32)
    ids = []
    for i, (_, row) in enumerate(sub.iterrows()):
        pid = _pid(row["ID"])
        X_img[i] = imgd[pid]
        X_gene[i] = gene_dict[pid]
        img_present[i] = 0.0 if pd.isna(row["WSI"]) else 1.0
        gene_present[i] = 0.0 if pd.isna(row["Omics"]) else 1.0
        t[i] = float(row["Event"]); e[i] = float(row["Status"])
        ids.append(pid)
    return dict(X_img=X_img, X_gene=X_gene, img_present=img_present,
                gene_present=gene_present, t=t, e=e, ids=ids)


def load_fold_config(fold: int, cfg: str, gene_dict: dict) -> dict:
    csv = os.path.join(LUAD_SPLITS, f"LUAD_fold_{fold}_{cfg}.csv")
    df = pd.read_csv(csv)
    return {
        "train": _assemble(df[df["Split"] == "train"].reset_index(drop=True), gene_dict),
        "val":   _assemble(df[df["Split"] == "val"].reset_index(drop=True),   gene_dict),
    }


if __name__ == "__main__":
    gd = load_gene_dict()
    for cfg in ["W0_O0", "W20_O40"]:
        d = load_fold_config(0, cfg, gd)
        tr, va = d["train"], d["val"]
        print(f"[{cfg}] train n={len(tr['t'])} val n={len(va['t'])} | "
              f"img-miss={int((tr['img_present']==0).sum())} gene-miss={int((tr['gene_present']==0).sum())} "
              f"both-present={int(((tr['img_present']==1)&(tr['gene_present']==1)).sum())}")
        print(f"       X_img{tr['X_img'].shape} X_gene{tr['X_gene'].shape} events(tr)={int(tr['e'].sum())} "
              f"val-complete={bool((va['img_present']==1).all() and (va['gene_present']==1).all())}")
    print("SMOKE OK")
