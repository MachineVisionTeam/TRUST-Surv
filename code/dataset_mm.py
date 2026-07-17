"""
Missing-modality dataset for KIRC — reuses the EXACT DisPro split CSVs so our
method runs on identical 5-fold splits + identical missing patterns as our
DisPro reproduction (apples-to-apples).

Patient-level (pooled UNI2-h). Attention-MIL was tried and found neutral on
KIRC, so we keep the simpler mean-pooled image.

Per patient:
  X_img  (1536,) : mean-pooled UNI2-h over the patient's ROIs
  X_gene (256,)  : BulkRNABert embedding
  img_present    : 1.0 if WSI cell present in CSV, else 0.0 (missing)
  gene_present   : 1.0 if Omics cell present, else 0.0 (missing)
  t (time)       : CSV 'Event'  (OS months)
  e (event)      : CSV 'Status' (1 = DECEASED/death, 0 = censored)

Labels straight from the DisPro CSVs (Status=1=DECEASED) -> matches DisPro
exactly, sidesteps KIRC's inverted `censored` convention. Missingness is a MASK
(data is paired): we always load real image + gene; present-flags say which to
replace with an absent token (student). The gene TEACHER always uses real gene.
"""
import os
import numpy as np
import pandas as pd
import torch

DISPRO_SPLITS = "/home/sbarua/Region_based_segmentation/missing_modality/replication/dispro/DisPro/splits/csv_missing_cleaned/KIRC"
WSI_BAG_DIR   = "/home/sbarua/Region_based_segmentation/missing_modality/replication/dispro/data/wsi/TCGA__KIRC/pt_files/uni2h"
BULK_NPZ      = "/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_patient_level/bulkrnabert_data/kirc_bulkrnabert_256d.npz"

CONFIGS = ["W0_O0", "W0_O60", "W20_O40", "W30_O30", "W40_O20", "W60_O0"]


def load_gene_dict():
    z = np.load(BULK_NPZ, allow_pickle=False)
    emb = z["embeddings"]
    assert emb.shape[1] == 256, f"gene dim {emb.shape[1]} != 256"
    return {str(p): emb[i].astype(np.float32) for i, p in enumerate(z["patient_ids"])}


_pool_cache: dict[str, np.ndarray] = {}
def load_pooled_image(pid: str) -> np.ndarray:
    if pid in _pool_cache:
        return _pool_cache[pid]
    bag = torch.load(os.path.join(WSI_BAG_DIR, f"{pid}.pt"),
                     map_location="cpu", weights_only=False)
    vec = bag.float().mean(dim=0).numpy().astype(np.float32)
    assert vec.shape == (1536,), f"pooled shape {vec.shape} != (1536,)"
    _pool_cache[pid] = vec
    return vec


def _assemble(sub: pd.DataFrame, gene_dict: dict) -> dict:
    n = len(sub)
    X_img = np.zeros((n, 1536), np.float32)
    X_gene = np.zeros((n, 256), np.float32)
    img_present = np.zeros(n, np.float32)
    gene_present = np.zeros(n, np.float32)
    t = np.zeros(n, np.float32)
    e = np.zeros(n, np.float32)
    ids = []
    for i, (_, row) in enumerate(sub.iterrows()):
        pid = str(row["ID"])
        X_img[i] = load_pooled_image(pid)
        X_gene[i] = gene_dict[pid]
        img_present[i] = 0.0 if pd.isna(row["WSI"]) else 1.0
        gene_present[i] = 0.0 if pd.isna(row["Omics"]) else 1.0
        t[i] = float(row["Event"])
        e[i] = float(row["Status"])
        ids.append(pid)
    return dict(X_img=X_img, X_gene=X_gene, img_present=img_present,
               gene_present=gene_present, t=t, e=e, ids=ids)


def load_fold_config(fold: int, cfg: str, gene_dict: dict) -> dict:
    csv = os.path.join(DISPRO_SPLITS, f"KIRC_fold_{fold}_{cfg}.csv")
    df = pd.read_csv(csv)
    return {
        "train": _assemble(df[df["Split"] == "train"].reset_index(drop=True), gene_dict),
        "val":   _assemble(df[df["Split"] == "val"].reset_index(drop=True),   gene_dict),
    }


# --- cohort dispatch: `COHORT=GBMLGG` re-exports the GBMLGG loader; KIRC default ---
if os.environ.get("COHORT", "KIRC").upper() == "GBMLGG":
    from dataset_gbmlgg import (load_gene_dict, load_fold_config,  # noqa: F401,F811
                                CONFIGS)


if __name__ == "__main__":
    gd = load_gene_dict()
    d = load_fold_config(0, "W20_O40", gd)
    tr = d["train"]
    print(f"train n={len(tr['t'])}  X_img={tr['X_img'].shape}  "
          f"img-miss={int((tr['img_present']==0).sum())}  gene-miss={int((tr['gene_present']==0).sum())}  "
          f"both-present={int(((tr['img_present']==1)&(tr['gene_present']==1)).sum())}")
    print("SMOKE OK")
