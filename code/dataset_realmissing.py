"""
Loader for KIRC REAL-missing patients (not simulated): patients that genuinely
have only ONE modality. Used to AUGMENT the training set (test set is unchanged).

  gene-only  : BulkRNABert present, no WSI  -> img_present=0, gene_present=1
  image-only : pooled UNI2h present, no gene -> img_present=1, gene_present=0

Survival labels from the cBioPortal pan-cancer clinical TSV. Image placeholder is
zeros (masked out via img_present); same for absent gene.
"""
import os, glob
import numpy as np
import pandas as pd

from dataset_mm import (WSI_BAG_DIR, BULK_NPZ, load_pooled_image, load_gene_dict)

CLINICAL = "/mnt/storage7/Dataset_pathomicfusion/KIRC/data/TCGA_KIRC/kirc_tcga_pan_can_atlas_2018_clinical_data.tsv"
IMG_DIM, GENE_DIM = 1536, 256


def _pid(x):
    p = str(x).split("-")
    return "-".join(p[:3]) if len(p) >= 3 else str(x)


def _clinical():
    df = pd.read_csv(CLINICAL, sep="\t")
    idc = "Patient ID"
    df = df.dropna(subset=["Overall Survival (Months)", "Overall Survival Status"])
    t = df.set_index(idc)["Overall Survival (Months)"].astype(float)
    # status is "LIVING" / "DECEASED"
    e = (df.set_index(idc)["Overall Survival Status"].astype(str)
         .str.upper().str.contains("DECEASED")).astype(int)
    return t, e


def load_real_missing(gene_dict=None):
    """Return dict with gene-only + image-only patient arrays (X_img, X_gene, masks, t, e)."""
    if gene_dict is None:
        gene_dict = load_gene_dict()
    t_all, e_all = _clinical()

    wsi_ids = {_pid(os.path.basename(f)[:-3]): os.path.basename(f)[:-3]
               for f in glob.glob(WSI_BAG_DIR + "/*.pt")}
    z = np.load(BULK_NPZ, allow_pickle=True)
    gene_ids = {_pid(x): str(x) for x in z["patient_ids"]}
    emb = {str(pid): z["embeddings"][i] for i, pid in enumerate(z["patient_ids"])}

    wsi_p, gene_p = set(wsi_ids), set(gene_ids)
    gene_only = [p for p in (gene_p - wsi_p) if p in t_all.index]
    image_only = [p for p in (wsi_p - gene_p) if p in t_all.index]

    rows = {"X_img": [], "X_gene": [], "img_present": [], "gene_present": [], "t": [], "e": []}
    # gene-only
    for p in gene_only:
        rows["X_img"].append(np.zeros(IMG_DIM, np.float32))
        rows["X_gene"].append(emb[gene_ids[p]].astype(np.float32))
        rows["img_present"].append(0.0); rows["gene_present"].append(1.0)
        rows["t"].append(float(t_all[p])); rows["e"].append(float(e_all[p]))
    # image-only
    for p in image_only:
        rows["X_img"].append(load_pooled_image(wsi_ids[p]).astype(np.float32))
        rows["X_gene"].append(np.zeros(GENE_DIM, np.float32))
        rows["img_present"].append(1.0); rows["gene_present"].append(0.0)
        rows["t"].append(float(t_all[p])); rows["e"].append(float(e_all[p]))

    out = {k: np.asarray(v, np.float32) for k, v in rows.items()}
    out["n_gene_only"] = len(gene_only); out["n_image_only"] = len(image_only)
    return out


if __name__ == "__main__":
    d = load_real_missing()
    print(f"gene-only: {d['n_gene_only']}  image-only: {d['n_image_only']}")
    print(f"X_img {d['X_img'].shape}  X_gene {d['X_gene'].shape}")
    print(f"img_present sum={d['img_present'].sum():.0f}  gene_present sum={d['gene_present'].sum():.0f}")
    print(f"events={d['e'].sum():.0f}/{len(d['e'])}  median t={np.median(d['t']):.1f}")
