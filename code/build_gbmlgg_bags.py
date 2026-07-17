"""
Aggregate GBMLGG per-patch uni2h features (all_st_patches_512_uni2h) into
per-PATIENT bags for MOTCat's co-attention: {pid: (n_patches, 1536) float32}.
Only the paired-592 patients are needed. Cached to a pkl.
"""
import os, glob, pickle
import numpy as np
import torch

PATCH_DIR = "/mnt/storage7/Dataset_pathomicfusion/GBMLGG/data/TCGA_GBMLGG/all_st_patches_512_uni2h"
OUT = "/home/sbarua/Region_based_segmentation/missing_modality/data_gbmlgg/gbmlgg_patient_bags_uni2h.pkl"
SPLIT0 = "/home/sbarua/Region_based_segmentation/missing_modality/data_gbmlgg/splits/GBMLGG_fold_0_W0_O0.csv"


def pid_of(fname):
    p = os.path.basename(fname).split("-")
    return "-".join(p[:3])


def main():
    import pandas as pd
    paired = set(pd.read_csv(SPLIT0)["ID"].astype(str))
    files = glob.glob(PATCH_DIR + "/*.pt")
    by_pid = {}
    for f in files:
        pid = pid_of(f)
        if pid in paired:
            by_pid.setdefault(pid, []).append(f)
    print(f"paired={len(paired)} | patients with patches={len(by_pid)}")
    missing = paired - set(by_pid)
    if missing:
        print(f"  WARNING: {len(missing)} paired patients have NO patches: {list(missing)[:5]}")

    bags = {}
    for i, (pid, fs) in enumerate(by_pid.items()):
        vecs = [torch.load(f, map_location="cpu", weights_only=False).float().numpy() for f in fs]
        bags[pid] = np.stack(vecs).astype(np.float32)      # (n_patches, 1536)
        if i % 100 == 0:
            print(f"  {i}/{len(by_pid)}  {pid}: {bags[pid].shape}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as fh:
        pickle.dump(bags, fh)
    npp = [v.shape[0] for v in bags.values()]
    print(f"saved {len(bags)} bags -> {OUT}")
    print(f"patches/patient: min={min(npp)} max={max(npp)} mean={np.mean(npp):.1f}")


if __name__ == "__main__":
    main()
