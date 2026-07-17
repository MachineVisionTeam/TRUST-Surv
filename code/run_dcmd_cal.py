"""
DCMD-Surv + CALIBRATION on KIRC. Reports, per scenario, discrimination (C-index)
AND calibration (IBS + calibration-MAD), raw and post-hoc recalibrated.

Routing (imputation-free):
  both          -> h_F   (img=1, gene=1)
  genes_missing -> h_I   (img=1, gene=0)   [gene-aligned image head]
  image_missing -> h_G   (img=0, gene=1)   [full-gene head]

Usage: python3 run_dcmd_cal.py --epochs 30 --gpu 0 [--la 1.0 --ld 0.3]
"""
import argparse, os, sys
import numpy as np
import torch

HYPAL_PATH = "/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_sample_level"
CODE_DIR = "/home/sbarua/Region_based_segmentation/missing_modality/code"
for p in (HYPAL_PATH, CODE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from genodistil_cpkf import cindex_lifelines  # noqa: E402
from dataset_mm import load_gene_dict, load_fold_config, CONFIGS  # noqa: E402
from run_dcmd import train_one_fold, _t  # reuse DCMD training  # noqa: E402
from calibration_mm import cox_calibration_report, fit_temperature  # noqa: E402

RESULTS_DIR = "/home/sbarua/Region_based_segmentation/missing_modality/results"


@torch.no_grad()
def _risks(model, X_img, X_gene, sc_i, sc_g, device, im, gm, n):
    Xi = _t(sc_i.transform(X_img).astype(np.float32), device)
    Xg = _t(sc_g.transform(X_gene).astype(np.float32), device)
    a = torch.ones(n, device=device) if im else torch.zeros(n, device=device)
    b = torch.ones(n, device=device) if gm else torch.zeros(n, device=device)
    h_F, h_I, h_G, _, _ = model(Xi, Xg, a, b)
    return h_F.cpu().numpy(), h_I.cpu().numpy(), h_G.cpu().numpy()


def eval_fold(model, tr, va, sc_i, sc_g, device):
    model.eval()
    ntr, nva = len(tr["t"]), len(va["t"])
    trF, _, _ = _risks(model, tr["X_img"], tr["X_gene"], sc_i, sc_g, device, 1, 1, ntr)
    _, trI, _ = _risks(model, tr["X_img"], tr["X_gene"], sc_i, sc_g, device, 1, 0, ntr)
    _, _, trG = _risks(model, tr["X_img"], tr["X_gene"], sc_i, sc_g, device, 0, 1, ntr)
    teF, _, _ = _risks(model, va["X_img"], va["X_gene"], sc_i, sc_g, device, 1, 1, nva)
    _, teI, _ = _risks(model, va["X_img"], va["X_gene"], sc_i, sc_g, device, 1, 0, nva)
    _, _, teG = _risks(model, va["X_img"], va["X_gene"], sc_i, sc_g, device, 0, 1, nva)
    out = {}
    for name, tr_r, te_r in [("both", trF, teF), ("genes_missing", trI, teI),
                             ("image_missing", trG, teG)]:
        c = cindex_lifelines(te_r, va["e"], va["t"])
        raw = cox_calibration_report(tr_r, tr["t"], tr["e"], te_r, va["t"], va["e"], temperature=1.0)
        T = fit_temperature(tr_r, tr["t"], tr["e"])
        rec = cox_calibration_report(tr_r, tr["t"], tr["e"], te_r, va["t"], va["e"], temperature=T)
        out[name] = {"cindex": c, "ibs": raw["ibs"], "cal_mad": raw["cal_mad"],
                     "ibs_recal": rec["ibs"], "cal_mad_recal": rec["cal_mad"], "T": T}
    return out


METRICS = ["cindex", "ibs", "cal_mad", "ibs_recal", "cal_mad_recal", "T"]


def run_config(cfg_name, gd, device, epochs, folds, la, ld, seed=0):
    acc = {s: {k: [] for k in METRICS} for s in ["both", "genes_missing", "image_missing"]}
    for fold in range(folds):
        data = load_fold_config(fold, cfg_name, gd)
        model, sc_i, sc_g = train_one_fold(data["train"], device, epochs, seed + fold, la, ld)
        r = eval_fold(model, data["train"], data["val"], sc_i, sc_g, device)
        for s in acc:
            for k in acc[s]:
                acc[s][k].append(r[s][k])
    return {s: {k: (float(np.nanmean(v)), float(np.nanstd(v))) for k, v in acc[s].items()} for s in acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--la", type=float, default=1.0)
    ap.add_argument("--ld", type=float, default=0.3)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    epochs = 40 if args.smoke else args.epochs
    folds = 1 if args.smoke else 5
    cohort = os.environ.get("COHORT", "KIRC").upper()
    print(f"[dcmd-cal] cohort={cohort} device={device} epochs={epochs} folds={folds} la={args.la} ld={args.ld}")
    gd = load_gene_dict()
    sn = {"genes_missing": "genes-miss(P)", "image_missing": "image-miss(G)", "both": "both(C)"}
    results = {}
    for cfg in (CONFIGS[:1] if args.smoke else CONFIGS):
        res = run_config(cfg, gd, device, epochs, folds, args.la, args.ld)
        results[cfg] = res
        for s in ["genes_missing", "image_missing", "both"]:
            m = res[s]
            print(f"[{cfg}] {sn[s]:14s} C={m['cindex'][0]:.3f} | IBS {m['ibs'][0]:.4f}->{m['ibs_recal'][0]:.4f} "
                  f"calMAD {m['cal_mad'][0]:.4f}->{m['cal_mad_recal'][0]:.4f} (T={m['T'][0]:.2f})")
    if args.smoke:
        return
    lines = ["=" * 72, f"DCMD-Surv + CALIBRATION — {cohort} 5-fold",
             "C-index (higher=better) | IBS, calMAD (lower=better calibrated)", "=" * 72]
    for label, cfgs in [("TABLE 1 (0% missing)", ["W0_O0"]), ("TABLE 2 (60% missing, avg)", CONFIGS[1:])]:
        lines.append("\n" + label)
        lines.append(f"  {'scenario':14s}  C-index | IBS(raw->recal) | calMAD(raw->recal)")
        for s in ["genes_missing", "image_missing", "both"]:
            C = np.mean([results[c][s]["cindex"][0] for c in cfgs])
            I = np.mean([results[c][s]["ibs"][0] for c in cfgs])
            Ir = np.mean([results[c][s]["ibs_recal"][0] for c in cfgs])
            M = np.mean([results[c][s]["cal_mad"][0] for c in cfgs])
            Mr = np.mean([results[c][s]["cal_mad_recal"][0] for c in cfgs])
            lines.append(f"  {sn[s]:14s}  {C:.4f}  | {I:.4f}->{Ir:.4f} | {M:.4f}->{Mr:.4f}")
    out = "\n".join(lines)
    print("\n" + out)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outp = os.path.join(RESULTS_DIR, f"ours_dcmd_calibration__{cohort}.txt")
    with open(outp, "w") as f:
        f.write(out + "\n")
    print(f"\nsaved -> {outp}")


if __name__ == "__main__":
    main()
