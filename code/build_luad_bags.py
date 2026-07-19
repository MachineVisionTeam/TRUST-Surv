"""
Aggregate LUAD per-slide uni2h patch features (LUAD_uni2h_slide/*_patches.pt, each
a dict with 'features' (n_patch, 1536)) into per-PATIENT bags for MOTCat:
{pid: (n_patches, 1536) float32}. Only the paired-447 patients. Cached to a pkl.
"""
import os, glob, pickle
import numpy as np
import torch

SLIDE_DIR = "/mnt/storage7/Dataset_pathomicfusion/LUAD/data/TCGA_LUAD/LUAD_uni2h_slide"
OUT = "/home/sbarua/Region_based_segmentation/missing_modality/data_luad/luad_patient_bags_uni2h.pkl"
SPLIT0 = "/home/sbarua/Region_based_segmentation/missing_modality/data_luad/splits/LUAD_fold_0_W0_O0.csv"


def _pid(fname):
    return "-".join(os.path.basename(fname).split("-")[:3])


def main():
    import pandas as pd
    paired = set(pd.read_csv(SPLIT0)["ID"].astype(str))
    files = glob.glob(SLIDE_DIR + "/*_patches.pt")
    by_pid = {}
    for f in files:
        p = _pid(f)
        if p in paired:
            by_pid.setdefault(p, []).append(f)
    print(f"paired={len(paired)} | patients with patch files={len(by_pid)}")
    missing = paired - set(by_pid)
    if missing:
        print(f"  WARNING: {len(missing)} paired patients have NO patch files: {list(missing)[:5]}")

    bags = {}
    for i, (pid, fs) in enumerate(by_pid.items()):
        feats = []
        for f in fs:
            d = torch.load(f, map_location="cpu", weights_only=False)
            feats.append(d["features"].float().numpy())     # (n_patch, 1536)
        bags[pid] = np.concatenate(feats, axis=0).astype(np.float32)
        if i % 100 == 0:
            print(f"  {i}/{len(by_pid)}  {pid}: {bags[pid].shape}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pickle.dump(bags, open(OUT, "wb"))
    npp = [v.shape[0] for v in bags.values()]
    print(f"saved {len(bags)} bags -> {OUT}")
    print(f"patches/patient: min={min(npp)} max={max(npp)} mean={np.mean(npp):.0f}")


if __name__ == "__main__":
    main()
