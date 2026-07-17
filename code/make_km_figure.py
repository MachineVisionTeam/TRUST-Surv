"""
Journal-critical figure: Kaplan-Meier RISK STRATIFICATION for DCMD-Surv on KIRC
(0% missing, 5-fold pooled). Per scenario, split patients at the median predicted
risk into HIGH vs LOW risk; plot KM curves + log-rank p-value + at-risk counts.

This is the standard "does the model separate patients into clinically distinct
risk groups" figure expected by survival journals (MedIA / IEEE-TMI / MICCAI).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

CODE_DIR = "/home/sbarua/Region_based_segmentation/missing_modality/code"
HYPAL = "/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_sample_level"
for p in (CODE_DIR, HYPAL):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch
from dataset_mm import load_gene_dict, load_fold_config
from run_dcmd import train_one_fold, _t

FIG_DIR = "/home/sbarua/Region_based_segmentation/missing_modality/results/figures"
SCEN = ["genes_missing", "image_missing", "both"]
SNAME = {"genes_missing": "Genes missing (image-only)",
         "image_missing": "Image missing (gene-only)", "both": "Complete"}
AVAIL = {"genes_missing": (1, 0), "image_missing": (0, 1), "both": (1, 1)}
HEAD = {"genes_missing": "genes_missing", "image_missing": "image_missing", "both": "both"}
NFOLDS = 5
HI, LO = "#C44E52", "#4C72B0"


def collect_risks(gd, device):
    out = {s: {"risk": [], "t": [], "e": []} for s in SCEN}
    for fold in range(NFOLDS):
        data = load_fold_config(fold, "W0_O0", gd)
        tr, va = data["train"], data["val"]
        model, sc_i, sc_g = train_one_fold(tr, device, 30, fold, 1.0, 0.3)
        model.eval()
        nva = len(va["t"])
        Xi = _t(sc_i.transform(va["X_img"]).astype(np.float32), device)
        Xg = _t(sc_g.transform(va["X_gene"]).astype(np.float32), device)
        for s in SCEN:
            im, gm = AVAIL[s]
            a = torch.ones(nva, device=device) if im else torch.zeros(nva, device=device)
            b = torch.ones(nva, device=device) if gm else torch.zeros(nva, device=device)
            with torch.no_grad():
                hF, hI, hG, _, _ = model(Xi, Xg, a, b)
            risk = {"both": hF, "genes_missing": hI, "image_missing": hG}[HEAD[s]].cpu().numpy()
            out[s]["risk"].append(risk); out[s]["t"].append(va["t"]); out[s]["e"].append(va["e"])
    for s in SCEN:
        for k in out[s]:
            out[s][k] = np.concatenate(out[s][k])
    return out


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    cohort = os.environ.get("COHORT", "KIRC").upper()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("[km] training DCMD 5 folds + collecting risks...")
    R = collect_risks(load_gene_dict(), device)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    for ax, s in zip(axes, SCEN):
        risk, t, e = R[s]["risk"], R[s]["t"], R[s]["e"]
        med = np.median(risk)
        hi = risk > med                     # high risk = higher hazard score
        lo = ~hi
        lr = logrank_test(t[hi], t[lo], e[hi], e[lo])
        p = lr.p_value
        km = KaplanMeierFitter()
        km.fit(t[hi], e[hi], label=f"High risk (n={hi.sum()})")
        km.plot_survival_function(ax=ax, ci_show=True, color=HI, lw=2)
        km.fit(t[lo], e[lo], label=f"Low risk (n={lo.sum()})")
        km.plot_survival_function(ax=ax, ci_show=True, color=LO, lw=2)
        ptxt = "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"
        ax.text(0.05, 0.08, f"log-rank\n{ptxt}", transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        ax.set_title(SNAME[s], fontsize=11)
        ax.set_xlabel("Time (months)"); ax.grid(alpha=0.25); ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8.5, loc="upper right")
    axes[0].set_ylabel("Survival probability")
    unit = "days" if cohort == "GBMLGG" else "months"
    for ax in axes:
        ax.set_xlabel(f"Time ({unit})")
    fig.tight_layout()   # no title (added as LaTeX caption)
    outp = f"{FIG_DIR}/fig_km_stratification_{cohort}.png"
    fig.savefig(outp, dpi=170, bbox_inches="tight")
    print(f"saved -> {outp}")
    for s in SCEN:
        risk, t, e = R[s]["risk"], R[s]["t"], R[s]["e"]
        hi = risk > np.median(risk)
        lr = logrank_test(t[hi], t[~hi], e[hi], e[~hi])
        print(f"  {s:14s} log-rank p = {lr.p_value:.2e}")


if __name__ == "__main__":
    main()
