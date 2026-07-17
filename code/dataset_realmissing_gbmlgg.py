"""
GBMLGG REAL-missing patients (genuinely single-modality), to AUGMENT training.
GBMLGG has BALANCED real-missing (unlike KIRC): image-only + gene-only.

  image-only : pooled uni2h present, no gene  -> img_present=1, gene_present=0
               survival from all_dataset.csv ('Survival months' col is DAYS)
  gene-only  : BulkRNABert present, no image  -> img_present=0, gene_present=1
               survival from MCAT tcga_gbmlgg_all_clean.csv (survival_months -> DAYS x30.44)

Both survival sources are the SAME data in different units (verified: ratio=30.44,
corr=1.0000). event e = 1 - censored (Paper-1 convention). Times unified to DAYS to
match the paired loader (dataset_gbmlgg uses all_dataset days).
"""
import pickle
import numpy as np
import pandas as pd

IMG_PKL = "/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_patient_level/bulkrnabert_data/gbmlgg_uni2h_patient_1536d.pkl"
GENE_NPZ = "/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_patient_level/bulkrnabert_data/gbmlgg_bulkrnabert_256d.npz"
ALL_DATASET = "/mnt/storage7/Dataset_pathomicfusion/GBMLGG/data/TCGA_GBMLGG/all_dataset.csv"
MCAT_CLIN = "/mnt/storage7/Dataset_pathomicfusion/GBMLGG_mcat/genomics/tcga_gbmlgg_all_clean.csv"
DAYS_PER_MONTH = 30.44
IMG_DIM, GENE_DIM = 1536, 256


def _pid(x):
    p = str(x).split("-")
    return "-".join(p[:3]) if len(p) >= 3 else str(x)


def load_real_missing(gene_dict=None):
    from dataset_gbmlgg import load_gene_dict
    if gene_dict is None:
        gene_dict = load_gene_dict()
    # image dict (pkl already patient-level pooled)
    img_df = pickle.load(open(IMG_PKL, "rb"))
    img_dict = {str(pid): img_df.loc[pid].to_numpy(np.float32) for pid in img_df.index}
    z = np.load(GENE_NPZ, allow_pickle=True)
    gene_ids = {str(x) for x in z["patient_ids"]}

    img_p, gene_p = set(img_dict), gene_ids
    # survival: image patients (all_dataset, DAYS)
    ad = pd.read_csv(ALL_DATASET)
    ad["pid"] = ad["TCGA ID"].apply(_pid)
    ad = ad.dropna(subset=["Survival months", "censored"]).drop_duplicates("pid").set_index("pid")
    ad_t = ad["Survival months"].astype(float)                # DAYS
    ad_e = (1 - ad["censored"]).astype(int)
    # survival: gene-only patients (MCAT, MONTHS -> DAYS)
    mc = pd.read_csv(MCAT_CLIN, usecols=["case_id", "survival_months", "censorship"])
    mc["pid"] = mc["case_id"].apply(_pid)
    mc = mc.dropna(subset=["survival_months", "censorship"]).drop_duplicates("pid").set_index("pid")
    mc_t = mc["survival_months"].astype(float) * DAYS_PER_MONTH   # -> DAYS
    mc_e = (1 - mc["censorship"]).astype(int)

    image_only = [p for p in (img_p - gene_p) if p in ad_t.index]
    gene_only = [p for p in (gene_p - img_p) if p in mc_t.index]

    rows = {"X_img": [], "X_gene": [], "img_present": [], "gene_present": [], "t": [], "e": []}
    for p in image_only:
        rows["X_img"].append(img_dict[p]); rows["X_gene"].append(np.zeros(GENE_DIM, np.float32))
        rows["img_present"].append(1.0); rows["gene_present"].append(0.0)
        rows["t"].append(float(ad_t[p])); rows["e"].append(float(ad_e[p]))
    for p in gene_only:
        rows["X_img"].append(np.zeros(IMG_DIM, np.float32)); rows["X_gene"].append(gene_dict[p].astype(np.float32))
        rows["img_present"].append(0.0); rows["gene_present"].append(1.0)
        rows["t"].append(float(mc_t[p])); rows["e"].append(float(mc_e[p]))

    out = {k: np.asarray(v, np.float32) for k, v in rows.items()}
    out["n_image_only"] = len(image_only); out["n_gene_only"] = len(gene_only)
    return out


if __name__ == "__main__":
    d = load_real_missing()
    print(f"image-only: {d['n_image_only']}  gene-only: {d['n_gene_only']}")
    print(f"X_img{d['X_img'].shape} X_gene{d['X_gene'].shape}")
    print(f"img_present sum={d['img_present'].sum():.0f} gene_present sum={d['gene_present'].sum():.0f}")
    print(f"events={d['e'].sum():.0f}/{len(d['e'])} | t(days) median={np.median(d['t']):.0f} min={d['t'].min():.0f} max={d['t'].max():.0f}")
    # sanity: image-only rows have zero gene, gene-only rows have zero image
    io = d['img_present'] == 1
    print(f"image-only rows: gene allzero={bool((np.abs(d['X_gene'][io]).sum(1)==0).all())} img nonzero={bool((np.abs(d['X_img'][io]).sum(1)>0).all())}")
    go = d['gene_present'] == 1
    print(f"gene-only rows: img allzero={bool((np.abs(d['X_img'][go]).sum(1)==0).all())} gene nonzero={bool((np.abs(d['X_gene'][go]).sum(1)>0).all())}")
