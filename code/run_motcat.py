"""
MOTCat (ICCV'23) reproduced as the COMPLETE-DATA fusion reference on KIRC, using
their ACTUAL model (models/model_motcat.py). MOTCat's OT co-attention requires
BOTH modalities -> it only produces the complete (C) scenario at 0% missing,
exactly as EMMS used it (no-missing reference row).

Inputs (feature-matched to our other methods):
  x_path  : uni2h WSI patch bag (27 x 1536)         [wsi_net patched to 1536-in]
  x_omic1..6 : BulkRNABert 256-d split into 6 groups [43,43,43,43,43,41]
Discrete-time survival (4 bins, NLL loss). Reports complete-data C-index + IBS.

Usage: python3 run_motcat.py --epochs 20 [--smoke]
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

MOTCAT = "/home/sbarua/Region_based_segmentation/missing_modality/replication/MOTCat"
CODE = "/home/sbarua/Region_based_segmentation/missing_modality/code"
HYPAL = "/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_sample_level"
for p in (MOTCAT, CODE, HYPAL):
    if p not in sys.path:
        sys.path.insert(0, p)
from models.model_motcat import MOTCAT_Surv  # noqa: E402
from genodistil_cpkf import cindex_lifelines  # noqa: E402
from dataset_mm import DISPRO_SPLITS, WSI_BAG_DIR, load_gene_dict  # noqa: E402
from calibration_mm import breslow_cumhaz  # noqa: E402
from compute_ibs_dispro import step_survival  # noqa: E402
from calibration_mm import eval_time_grid, compute_ibs  # noqa: E402

RESULTS_DIR = "/home/sbarua/Region_based_segmentation/missing_modality/results"
OMIC_SPLIT = [43, 86, 129, 172, 215, 256]   # cut points -> 6 groups of BulkRNABert
N_BINS = 4

COHORT = os.environ.get("COHORT", "KIRC").upper()
_MM = "/home/sbarua/Region_based_segmentation/missing_modality"
_BAGS = None
if COHORT == "GBMLGG":
    import pickle
    SPLITS_DIR = f"{_MM}/data_gbmlgg/splits"
    _BAGS = pickle.load(open(f"{_MM}/data_gbmlgg/gbmlgg_patient_bags_uni2h.pkl", "rb"))
elif COHORT == "LUAD":
    import pickle
    SPLITS_DIR = f"{_MM}/data_luad/splits"
    _BAGS = pickle.load(open(f"{_MM}/data_luad/luad_patient_bags_uni2h.pkl", "rb"))
else:
    SPLITS_DIR = DISPRO_SPLITS


def load_bag(pid, device):
    if _BAGS is not None:                                             # GBMLGG / LUAD: cached bags
        return torch.from_numpy(_BAGS[pid]).float().to(device)       # (n_patches, 1536)
    x = torch.load(os.path.join(WSI_BAG_DIR, f"{pid}.pt"), map_location="cpu", weights_only=False)
    if isinstance(x, dict):
        x = x.get("features", next(iter(x.values())))
    return x.float().to(device)


def disc_bins(t, e):
    ev = t[e == 1]
    qb = np.quantile(ev, np.linspace(0, 1, N_BINS + 1))
    qb[0] = t.min() - 1e-6; qb[-1] = t.max() + 1e-6
    y = np.clip(np.digitize(t, qb) - 1, 0, N_BINS - 1)
    return y.astype(int), qb


def nll_surv(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    Y = Y.view(-1, 1); c = c.view(-1, 1).float()
    S_pad = torch.cat([torch.ones_like(c), S], 1)
    unc = -(1 - c) * (torch.log(torch.gather(S_pad, 1, Y).clamp(min=eps)) +
                      torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
    cen = -c * torch.log(torch.gather(S_pad, 1, Y + 1).clamp(min=eps))
    return ((1 - alpha) * (cen + unc) + alpha * unc).mean()


def omic_groups(gene_vec, device):
    out = {}
    prev = 0
    for i, cut in enumerate(OMIC_SPLIT):
        out[f"x_omic{i+1}"] = torch.from_numpy(gene_vec[prev:cut]).float().to(device)
        prev = cut
    return out


def run_fold(fold, gd, device, epochs, seed):
    df = pd.read_csv(os.path.join(SPLITS_DIR, f"{COHORT}_fold_{fold}_W0_O0.csv"))
    tr = df[df["Split"] == "train"]; va = df[df["Split"] == "val"]
    ytr, qb = disc_bins(tr["Event"].to_numpy(float), tr["Status"].to_numpy(float))
    torch.manual_seed(seed); np.random.seed(seed)
    model = MOTCAT_Surv(omic_sizes=[43, 43, 43, 43, 43, 41], n_classes=N_BINS, fusion="concat").to(device)
    model.wsi_net[0] = nn.Linear(1536, 256).to(device)   # uni2h is 1536-d
    opt = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)

    def sample(row, y=None):
        pid = row["ID"]
        bag = load_bag(pid, device)
        g = gd[pid].astype(np.float32)
        d = {"x_path": bag, **omic_groups(g, device)}
        return d

    tr_rows = tr.to_dict("records"); ev_bin = dict(zip(tr["ID"], ytr))
    model.train()
    for _ in range(epochs):
        order = np.random.permutation(len(tr_rows))
        for j in order:
            row = tr_rows[j]
            if row["ID"] not in gd:
                continue
            opt.zero_grad()
            out = model(**sample(row))
            hazards, S = out[0], out[1]           # forward returns (hazards, S, Y_hat, attn)
            Y = torch.tensor([ev_bin[row["ID"]]], device=device)
            c = torch.tensor([1 - row["Status"]], device=device)
            loss = nll_surv(hazards, S, Y, c)
            loss.backward(); opt.step()

    # eval on val (complete)
    model.eval()
    risks, Ss, ts, es = [], [], [], []
    with torch.no_grad():
        for row in va.to_dict("records"):
            if row["ID"] not in gd:
                continue
            out = model(**sample(row))
            S = out[1]                            # S (survival) is the 2nd output
            risks.append(float(-torch.sum(S)))
            Ss.append(S.squeeze(0).cpu().numpy())
            ts.append(row["Event"]); es.append(row["Status"])
    risks = np.array(risks); ts = np.array(ts, float); es = np.array(es, float)
    c_index = cindex_lifelines(risks, es, ts)
    # IBS from discrete S curves
    S_arr = np.stack(Ss)
    times = eval_time_grid(ts, es)
    teS = step_survival(S_arr, qb, times)
    ibs = compute_ibs(tr["Event"].to_numpy(float), tr["Status"].to_numpy(float), ts, es, teS, times)
    return c_index, ibs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    gd = load_gene_dict()
    folds = 1 if args.smoke else 5
    cs, ibss = [], []
    for fold in range(folds):
        c, ibs = run_fold(fold, gd, device, args.epochs, fold)
        cs.append(c); ibss.append(ibs)
        print(f"[fold {fold}] complete(C) C-index={c:.4f}  IBS={ibs:.4f}")
    if args.smoke:
        return
    out = ("=" * 60 + f"\nMOTCat (complete-data reference) — {COHORT} 5-fold, 0% missing\n" +
           "OT co-attention needs BOTH modalities -> complete(C) scenario only\n" + "=" * 60 +
           f"\n  both(C)  C-index={np.mean(cs):.4f}+/-{np.std(cs):.4f}  IBS={np.mean(ibss):.4f}\n")
    print("\n" + out)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outp = os.path.join(RESULTS_DIR, f"baseline_motcat__{COHORT}.txt")
    with open(outp, "w") as f:
        f.write(out)
    print(f"saved -> {outp}")


if __name__ == "__main__":
    main()
