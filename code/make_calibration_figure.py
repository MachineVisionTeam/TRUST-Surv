"""
Calibration detail figure per cohort (DCMD-Surv): reliability curves (predicted vs
KM-observed survival) + time-dependent Brier score, for the 3 scenarios (P/G/C),
0% missing, 5-fold pooled. Shows DCMD's survival probabilities are well-calibrated.
Cohort-aware (COHORT env).

Usage: COHORT=KIRC python3 make_calibration_figure.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from sksurv.util import Surv
from sksurv.metrics import brier_score

CODE = "/home/sbarua/Region_based_segmentation/missing_modality/code"
HYPAL = "/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_sample_level"
for p in (CODE, HYPAL):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch
from dataset_mm import load_gene_dict, load_fold_config
from run_dcmd import train_one_fold, _t
from calibration_mm import breslow_cumhaz, survival_at_times

FIG = "/home/sbarua/Region_based_segmentation/missing_modality/results/figures"
SCEN = ["genes_missing", "image_missing", "both"]
SNAME = {"genes_missing": "Genes missing (image-only)",
         "image_missing": "Image missing (gene-only)", "both": "Complete"}
AVAIL = {"genes_missing": (1, 0), "image_missing": (0, 1), "both": (1, 1)}
COL = "#2f6db0"
NFOLDS = 5


def dcmd_curves(gd, device, grid, t0):
    out = {s: {"St0": [], "Sgrid": [], "t": [], "e": []} for s in SCEN}
    for fold in range(NFOLDS):
        data = load_fold_config(fold, "W0_O0", gd)
        tr, va = data["train"], data["val"]
        model, sc_i, sc_g = train_one_fold(tr, device, 30, fold, 1.0, 0.3)
        model.eval()
        from sklearn.preprocessing import StandardScaler  # scalers already fit inside train_one_fold? re-fit
        # train_one_fold returns fitted scalers
        def risk(X_img, X_gene, im, gm, n):
            Xi = _t(sc_i.transform(X_img).astype(np.float32), device)
            Xg = _t(sc_g.transform(X_gene).astype(np.float32), device)
            a = torch.ones(n, device=device) if im else torch.zeros(n, device=device)
            b = torch.ones(n, device=device) if gm else torch.zeros(n, device=device)
            with torch.no_grad():
                hF, hI, hG, _, _ = model(Xi, Xg, a, b)
            return {"both": hF, "genes_missing": hI, "image_missing": hG}
        ntr, nva = len(tr["t"]), len(va["t"])
        for s in SCEN:
            im, gm = AVAIL[s]
            r_tr = risk(tr["X_img"], tr["X_gene"], im, gm, ntr)[s].cpu().numpy()
            r_va = risk(va["X_img"], va["X_gene"], im, gm, nva)[s].cpu().numpy()
            ev, H0, rmax = breslow_cumhaz(r_tr, tr["t"], tr["e"])
            out[s]["Sgrid"].append(survival_at_times(r_va, ev, H0, rmax, grid))
            out[s]["St0"].append(survival_at_times(r_va, ev, H0, rmax, np.array([t0]))[:, 0])
            out[s]["t"].append(va["t"]); out[s]["e"].append(va["e"])
    for s in SCEN:
        for k in out[s]:
            out[s][k] = np.concatenate(out[s][k], axis=0)
    return out


def reliability(St0, t, e, t0, n_bins=5):
    edges = np.quantile(St0, np.linspace(0, 1, n_bins + 1))
    prd, obs = [], []
    for i in range(n_bins):
        m = (St0 >= edges[i]) & (St0 <= edges[i + 1]) if i == n_bins - 1 else (St0 >= edges[i]) & (St0 < edges[i + 1])
        if m.sum() < 8:
            continue
        km = KaplanMeierFitter().fit(t[m], e[m])
        obs.append(float(km.predict(t0))); prd.append(float(St0[m].mean()))
    return np.array(prd), np.array(obs)


def main():
    cohort = os.environ.get("COHORT", "KIRC").upper()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gd = load_gene_dict()
    # time grid + t0 from pooled event times
    d0 = load_fold_config(0, "W0_O0", gd)
    allt = np.concatenate([d0["train"]["t"], d0["val"]["t"]])
    alle = np.concatenate([d0["train"]["e"], d0["val"]["e"]])
    ev = allt[alle == 1]
    t0 = float(np.median(ev))
    grid = np.linspace(np.percentile(ev, 10), np.percentile(ev, 80), 14)
    unit = "days" if cohort == "GBMLGG" else "months"
    print(f"[cal] {cohort} t0={t0:.0f} {unit}, training DCMD...")
    dc = dcmd_curves(gd, device, grid, t0)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for col, s in enumerate(SCEN):
        # reliability
        ax = axes[0][col]
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Perfect")
        prd, obs = reliability(dc[s]["St0"], dc[s]["t"], dc[s]["e"], t0)
        ax.plot(prd, obs, "o-", color=COL, lw=2, ms=6, label="DCMD-Surv")
        ax.set_title(SNAME[s], fontsize=11); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect("equal"); ax.grid(alpha=0.25); ax.set_xlabel(f"Predicted S({t0:.0f} {unit})")
        if col == 0:
            ax.set_ylabel("Observed survival (KM)"); ax.legend(fontsize=8.5, loc="upper left")
        # brier(t)
        ax2 = axes[1][col]
        sv = Surv.from_arrays(event=dc[s]["e"].astype(bool), time=dc[s]["t"].astype(float))
        mask = grid < dc[s]["t"].max() - 1e-3
        tt, bs = brier_score(sv, sv, dc[s]["Sgrid"][:, mask], grid[mask])
        ax2.plot(tt, bs, "o-", color=COL, lw=2, ms=4)
        ax2.set_xlabel(f"Time ({unit})"); ax2.grid(alpha=0.25)
        if col == 0:
            ax2.set_ylabel("Brier score (lower=better)")
    fig.tight_layout()   # no title (added as LaTeX caption)
    os.makedirs(FIG, exist_ok=True)
    outp = f"{FIG}/fig_calibration_{cohort}.png"
    fig.savefig(outp, dpi=165, bbox_inches="tight")
    print(f"saved -> {outp}")


if __name__ == "__main__":
    main()
