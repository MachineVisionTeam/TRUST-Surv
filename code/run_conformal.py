"""
Conformal reliability experiment (the STAR) — per cohort, 0% missing (W0_O0),
5-fold. For each inference scenario (complete / genes-missing / image-missing)
and each target coverage level (80% / 90% / 95%):

  - train DCMD backbone ONCE per fold (both modalities present)
  - split fold VAL into CALIBRATION half + TEST half (patient-disjoint)
  - Breslow S(t|x) + IPCW Ghat from TRAIN (of that scenario's head)
  - conformal LPB: naive vs conformal coverage on TEST (target 1-alpha)
  - pool TEST coverage across folds

Models are trained ONCE and reused across all coverage levels (conformal is
post-hoc), so evaluating 3 levels costs ~the same as one.

Headline: naive (raw model) coverage drifts below target under missing modalities;
the conformal layer restores the guaranteed 1-alpha coverage in EVERY scenario,
at EVERY level.

Writes:  results/conformal_reliability__<COHORT>.txt  (human table)
         results/conformal_summary.csv                (machine-readable, for the figure)

Usage: COHORT=KIRC python3 run_conformal.py --epochs 30 --gpu 0
"""
import argparse, os, sys
import numpy as np
import torch

CODE_DIR = "/home/sbarua/Region_based_segmentation/missing_modality/code"
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
from dataset_mm import load_gene_dict, load_fold_config     # noqa: E402  (COHORT-dispatched)
from run_dcmd import train_one_fold, _t                     # noqa: E402
from conformal_survival import conformal_lpb_report         # noqa: E402

RESULTS_DIR = "/home/sbarua/Region_based_segmentation/missing_modality/results"
CSV = os.path.join(RESULTS_DIR, "conformal_summary.csv")
SCEN = ["both", "genes_missing", "image_missing"]
SNAME = {"both": "complete (C)   ", "genes_missing": "genes-miss (P) ",
         "image_missing": "image-miss (G) "}
AVAIL = {"both": (1, 1), "genes_missing": (1, 0), "image_missing": (0, 1)}
LEVELS = [0.20, 0.10, 0.05]        # alpha -> target coverage 80% / 90% / 95%
NFOLDS = 5


@torch.no_grad()
def head_risk(model, X_img, X_gene, sc_i, sc_g, scenario, device):
    """Risk score for one scenario's routed head over all rows of X."""
    Xi = _t(sc_i.transform(X_img).astype(np.float32), device)
    Xg = _t(sc_g.transform(X_gene).astype(np.float32), device)
    n = Xi.shape[0]
    im, gm = AVAIL[scenario]
    a = torch.full((n,), float(im), device=device)
    b = torch.full((n,), float(gm), device=device)
    h_F, h_I, h_G, _, _ = model(Xi, Xg, a, b)
    return {"both": h_F, "genes_missing": h_I, "image_missing": h_G}[scenario].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--folds", type=int, default=NFOLDS)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    cohort = os.environ.get("COHORT", "KIRC").upper()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"[conformal] cohort={cohort} device={device} levels={[1-a for a in LEVELS]} "
          f"epochs={args.epochs} folds={args.folds}")
    gd = load_gene_dict()
    rng = np.random.RandomState(0)

    # ---- 1) train ONCE per fold, cache per-scenario risks + cal/test split ----
    cache = []   # list over folds: {scenario: (r_tr,t_tr,e_tr, r_cal,t_cal,e_cal, r_test,t_test,e_test)}
    for fold in range(args.folds):
        data = load_fold_config(fold, "W0_O0", gd)
        tr, va = data["train"], data["val"]
        model, sc_i, sc_g = train_one_fold(tr, device, args.epochs, fold, la=1.0, ld=0.3)
        model.eval()
        nv = len(va["t"]); perm = rng.permutation(nv); half = nv // 2
        ci, ti = perm[:half], perm[half:]          # calibration idx, test idx
        fold_c = {}
        for s in SCEN:
            r_tr = head_risk(model, tr["X_img"], tr["X_gene"], sc_i, sc_g, s, device)
            r_va = head_risk(model, va["X_img"], va["X_gene"], sc_i, sc_g, s, device)
            fold_c[s] = (r_tr, tr["t"], tr["e"],
                         r_va[ci], va["t"][ci], va["e"][ci],
                         r_va[ti], va["t"][ti], va["e"][ti])
        cache.append(fold_c)

    # ---- 2) evaluate every coverage level from the cached models (post-hoc) ----
    rows = []   # (alpha, scenario, naive, conf, width)
    for alpha in LEVELS:
        for s in SCEN:
            nv_l, cf_l, wd_l = [], [], []
            for fold_c in cache:
                (r_tr, t_tr, e_tr, r_cal, t_cal, e_cal, r_test, t_test, e_test) = fold_c[s]
                rep = conformal_lpb_report(r_tr, t_tr, e_tr, r_cal, t_cal, e_cal,
                                           r_test, t_test, e_test, alpha=alpha)
                nv_l.append(rep["cov_naive"]); cf_l.append(rep["cov_conformal"])
                wd_l.append(rep["median_width"])
            rows.append((alpha, s, float(np.mean(nv_l)), float(np.mean(cf_l)), float(np.mean(wd_l))))

    # ---- 3) human-readable table ----
    lines = ["", "=" * 82,
             f"{cohort} — CONFORMAL RELIABILITY (0% missing, {args.folds}-fold)",
             "  naive = raw model's own uncertainty   conformal = guaranteed LPB (IPCW split-conformal)",
             "  Guarantee is one-sided: coverage >= target (over-coverage valid; under-coverage fails).",
             "=" * 82]
    for alpha in LEVELS:
        target = 1 - alpha
        lines.append(f"\n--- target coverage {target:.2f} ---")
        lines.append(f"  {'scenario':15s} | naive cov | conformal cov | median width | naive holds? | conf holds?")
        for (a, s, nv, cf, wd) in rows:
            if a != alpha:
                continue
            nh = "yes" if nv >= target - 0.02 else "NO (overconf)"
            ch = "yes" if cf >= target - 0.02 else "NO"
            lines.append(f"  {SNAME[s]:15s} |   {nv:.3f}   |     {cf:.3f}     |   {wd:8.2f}   | "
                         f"{nh:13s}| {ch}")
    block = "\n".join(lines)
    print(block)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"conformal_reliability__{cohort}.txt"), "w") as f:
        f.write(block + "\n")

    # ---- 4) machine-readable summary (append; de-dup this cohort first) ----
    header = "cohort,alpha,scenario,naive,conformal,width\n"
    existing = []
    if os.path.exists(CSV):
        with open(CSV) as f:
            for ln in f.readlines()[1:]:
                if ln.strip() and not ln.startswith(cohort + ","):
                    existing.append(ln.rstrip("\n"))
    with open(CSV, "w") as f:
        f.write(header)
        for ln in existing:
            f.write(ln + "\n")
        for (a, s, nv, cf, wd) in rows:
            f.write(f"{cohort},{a:.2f},{s},{nv:.4f},{cf:.4f},{wd:.4f}\n")
    print(f"\nsaved -> results/conformal_reliability__{cohort}.txt  and  results/conformal_summary.csv")


if __name__ == "__main__":
    main()
