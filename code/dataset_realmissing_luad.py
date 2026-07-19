"""
LUAD REAL-missing patients (genuinely single-modality), to AUGMENT training.
Simpler than GBMLGG: ONE survival source (cbioportal clinical tsv), consistent
units (OS_MONTHS). event e = 1 if OS_STATUS starts with '1' (1:DECEASED).

  image-only : pooled uni2h present, no gene  -> img_present=1, gene_present=0
  gene-only  : BulkRNABert present, no image  -> img_present=0, gene_present=1
Times in OS_MONTHS to match the paired loader (dataset_luad). Drops t<=0.
"""
import pickle
import numpy as np
import pandas as pd

IMG_PKL = "/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_patient_level/bulkrnabert_data/luad_uni2h_patient_1536d.pkl"
GENE_NPZ = "/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_patient_level/bulkrnabert_data/luad_bulkrnabert_256d.npz"
CLIN = "/mnt/storage7/Dataset_pathomicfusion/LUAD/data/TCGA_LUAD/cbioportal/luad_clinical_patient.tsv"
IMG_DIM, GENE_DIM = 1536, 256


def _pid(x):
    p = str(x).split("-")
    return "-".join(p[:3]) if len(p) >= 3 else str(x)


def load_real_missing(gene_dict=None):
    from dataset_luad import load_gene_dict
    if gene_dict is None:
        gene_dict = load_gene_dict()
    img_df = pickle.load(open(IMG_PKL, "rb"))
    img_dict = {_pid(pid): img_df.loc[pid].to_numpy(np.float32) for pid in img_df.index}
    z = np.load(GENE_NPZ, allow_pickle=True)
    gene_ids = {_pid(x) for x in z["patient_ids"]}

    img_p, gene_p = set(img_dict), gene_ids
    cl = pd.read_csv(CLIN, sep="\t", comment="#")
    cl["pid"] = cl["PATIENT_ID"].apply(_pid)
    cl["t"] = pd.to_numeric(cl["OS_MONTHS"], errors="coerce")
    cl["e"] = cl["OS_STATUS"].astype(str).str.startswith("1").astype(int)
    cl = cl.dropna(subset=["t"]); cl = cl[cl["t"] > 0].drop_duplicates("pid").set_index("pid")

    image_only = [p for p in (img_p - gene_p) if p in cl.index]
    gene_only = [p for p in (gene_p - img_p) if p in cl.index]

    rows = {"X_img": [], "X_gene": [], "img_present": [], "gene_present": [], "t": [], "e": []}
    for p in image_only:
        rows["X_img"].append(img_dict[p]); rows["X_gene"].append(np.zeros(GENE_DIM, np.float32))
        rows["img_present"].append(1.0); rows["gene_present"].append(0.0)
        rows["t"].append(float(cl.loc[p, "t"])); rows["e"].append(float(cl.loc[p, "e"]))
    for p in gene_only:
        rows["X_img"].append(np.zeros(IMG_DIM, np.float32)); rows["X_gene"].append(gene_dict[p].astype(np.float32))
        rows["img_present"].append(0.0); rows["gene_present"].append(1.0)
        rows["t"].append(float(cl.loc[p, "t"])); rows["e"].append(float(cl.loc[p, "e"]))

    out = {k: np.asarray(v, np.float32) for k, v in rows.items()}
    out["n_image_only"] = len(image_only); out["n_gene_only"] = len(gene_only)
    return out


if __name__ == "__main__":
    d = load_real_missing()
    print(f"image-only: {d['n_image_only']}  gene-only: {d['n_gene_only']}")
    print(f"X_img{d['X_img'].shape} X_gene{d['X_gene'].shape}")
    print(f"img_present sum={d['img_present'].sum():.0f} gene_present sum={d['gene_present'].sum():.0f}")
    print(f"events={d['e'].sum():.0f}/{len(d['e'])} | t(months) median={np.median(d['t']):.1f}")
    io = d['img_present'] == 1; go = d['gene_present'] == 1
    print(f"image-only rows: gene allzero={bool((np.abs(d['X_gene'][io]).sum(1)==0).all())} img nonzero={bool((np.abs(d['X_img'][io]).sum(1)>0).all())}")
    print(f"gene-only rows: img allzero={bool((np.abs(d['X_img'][go]).sum(1)==0).all())} gene nonzero={bool((np.abs(d['X_gene'][go]).sum(1)>0).all())}")
