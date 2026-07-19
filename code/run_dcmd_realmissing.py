"""
REAL-MISSING experiment: does adding genuinely single-modality patients (that
DisPro/EMMS discard) to DCMD-Surv training help? Test set UNCHANGED (paired val,
3 scenarios), so directly comparable to the simulated-only main result.

For each fold: train DCMD twice with the SAME seed --
  (A) paired-only  (baseline, = main result)
  (B) paired + 94 real-missing (92 gene-only + 2 image-only)
Scalers fit on PRESENT-only features (avoid zero-placeholder corruption).
Reports C-index + IBS per scenario, both settings, 0% and 60%.

Usage: python3 run_dcmd_realmissing.py --epochs 30 [--smoke]
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

HYPAL = "/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_sample_level"
CODE = "/home/sbarua/Region_based_segmentation/missing_modality/code"
for p in (HYPAL, CODE):
    if p not in sys.path:
        sys.path.insert(0, p)
from genodistil_cpkf import (cox_partial_likelihood_loss, ranking_distillation_loss,  # noqa: E402
                             cindex_lifelines)
from dataset_mm import load_gene_dict, load_fold_config, CONFIGS  # noqa: E402
_RMC = os.environ.get("COHORT", "KIRC").upper()
if _RMC == "GBMLGG":
    from dataset_realmissing_gbmlgg import load_real_missing  # noqa: E402
elif _RMC == "LUAD":
    from dataset_realmissing_luad import load_real_missing  # noqa: E402
else:
    from dataset_realmissing import load_real_missing  # noqa: E402
from calibration_mm import cox_calibration_report  # noqa: E402
from model_dcmd import DCMDSurv  # noqa: E402

RESULTS_DIR = "/home/sbarua/Region_based_segmentation/missing_modality/results"
SCEN = {"genes_missing": (1, 0), "image_missing": (0, 1), "both": (1, 1)}


def _t(a, device):
    return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device)


def _augment(tr, rm):
    """Concatenate paired-train with real-missing patients."""
    out = {}
    for k in ["X_img", "X_gene", "img_present", "gene_present", "t", "e"]:
        out[k] = np.concatenate([tr[k], rm[k]], axis=0)
    return out


def train_fold(tr, device, epochs, seed, la=1.0, ld=0.3):
    # scalers on PRESENT-only (real-missing have zero placeholders)
    sc_i, sc_g = StandardScaler(), StandardScaler()
    im_mask = tr["img_present"] > 0.5
    gm_mask = tr["gene_present"] > 0.5
    sc_i.fit(tr["X_img"][im_mask])
    sc_g.fit(tr["X_gene"][gm_mask])
    Xi = _t(sc_i.transform(tr["X_img"]).astype(np.float32), device)
    Xg = _t(sc_g.transform(tr["X_gene"]).astype(np.float32), device)
    torch.manual_seed(seed); np.random.seed(seed)
    model = DCMDSurv().to(device)
    cfg = model.cfg
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    imp = _t(tr["img_present"], device); gmp = _t(tr["gene_present"], device)
    t = _t(tr["t"], device); e = _t(tr["e"], device)
    zero = torch.zeros((), device=device)
    both = (imp > 0.5) & (gmp > 0.5)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        h_F, h_I, h_G, G_img, G_gene = model(Xi, Xg, imp, gmp)
        L_F = cox_partial_likelihood_loss(h_F, t, e)
        L_G = cox_partial_likelihood_loss(h_G, t, e)
        mi = imp > 0.5
        if mi.sum() >= 2 and e[mi].sum() >= 1:
            L_I = cox_partial_likelihood_loss(h_I[mi], t[mi], e[mi])
            L_rank = ranking_distillation_loss(h_I[mi], h_G[mi].detach(), t[mi], e[mi])
        else:
            L_I, L_rank = zero, zero
        if both.sum() >= 2:
            gi = F.normalize(G_img[both], dim=1)
            gg = F.normalize(G_gene[both], dim=1).detach()
            L_align = F.mse_loss(gi, gg)
        else:
            L_align = zero
        L = L_F + cfg.lambda_aux * (L_G + L_I) + la * L_align + ld * L_rank
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        opt.step()
    return model, sc_i, sc_g


@torch.no_grad()
def eval_fold(model, sc_i, sc_g, tr, va, device):
    model.eval()
    out = {}
    for s, (im, gm) in SCEN.items():
        def risk(X_img, X_gene):
            n = len(X_img)
            Xi = _t(sc_i.transform(X_img).astype(np.float32), device)
            Xg = _t(sc_g.transform(X_gene).astype(np.float32), device)
            a = torch.full((n,), float(im), device=device); b = torch.full((n,), float(gm), device=device)
            hF, hI, hG, _, _ = model(Xi, Xg, a, b)
            return {"both": hF, "genes_missing": hI, "image_missing": hG}[s].cpu().numpy()
        te_r = risk(va["X_img"], va["X_gene"]); tr_r = risk(tr["X_img"], tr["X_gene"])
        c = cindex_lifelines(te_r, va["e"], va["t"])
        ibs = cox_calibration_report(tr_r, tr["t"], tr["e"], te_r, va["t"], va["e"], 1.0)["ibs"]
        out[s] = {"cindex": c, "ibs": ibs}
    return out


def run(cfgs, gd, rm, device, epochs, folds):
    res = {"paired": {}, "augmented": {}}
    for setting in res:
        for cfg in cfgs:
            acc = {s: {"cindex": [], "ibs": []} for s in SCEN}
            for fold in range(folds):
                data = load_fold_config(fold, cfg, gd)
                tr = data["train"]
                # for eval we need a PAIRED train subset for Breslow (use paired only)
                paired_tr = tr
                train_data = _augment(tr, rm) if setting == "augmented" else tr
                model, sc_i, sc_g = train_fold(train_data, device, epochs, fold)
                r = eval_fold(model, sc_i, sc_g, paired_tr, data["val"], device)
                for s in acc:
                    acc[s]["cindex"].append(r[s]["cindex"]); acc[s]["ibs"].append(r[s]["ibs"])
            res[setting][cfg] = {s: {k: float(np.nanmean(v)) for k, v in acc[s].items()} for s in acc}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    folds = 1 if args.smoke else 5
    cohort = os.environ.get("COHORT", "KIRC").upper()
    gd = load_gene_dict()
    rm = load_real_missing(gd)
    print(f"[real-missing] {rm['n_gene_only']} gene-only + {rm['n_image_only']} image-only added to training")
    cfgs = CONFIGS[:1] if args.smoke else CONFIGS
    res = run(cfgs, gd, rm, device, args.epochs, folds)
    sn = {"genes_missing": "genes-miss(P)", "image_missing": "image-miss(G)", "both": "both(C)"}

    def block(cs, label):
        lines = [f"\n{label}   (C-index / IBS)   paired-only  ->  +real-missing"]
        for s in ["genes_missing", "image_missing", "both"]:
            cp = np.mean([res["paired"][c][s]["cindex"] for c in cs])
            ca = np.mean([res["augmented"][c][s]["cindex"] for c in cs])
            ip = np.mean([res["paired"][c][s]["ibs"] for c in cs])
            ia = np.mean([res["augmented"][c][s]["ibs"] for c in cs])
            lines.append(f"  {sn[s]:14s} C {cp:.4f}->{ca:.4f} ({ca-cp:+.4f}) | IBS {ip:.4f}->{ia:.4f} ({ia-ip:+.4f})")
        return lines
    if args.smoke:
        print("\n".join(block(["W0_O0"], "SMOKE 0%"))); return
    lines = ["=" * 72, f"DCMD-Surv REAL-MISSING experiment — {cohort} 5-fold (test set unchanged)",
             f"training augmented with {rm['n_gene_only']} gene-only + {rm['n_image_only']} image-only real cases",
             "=" * 72]
    lines += block(["W0_O0"], "TABLE 1 (0% simulated)")
    lines += block(CONFIGS[1:], "TABLE 2 (60% simulated, avg)")
    out = "\n".join(lines); print("\n" + out)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outp = os.path.join(RESULTS_DIR, f"dcmd_realmissing__{cohort}.txt")
    with open(outp, "w") as f:
        f.write(out + "\n")
    print(f"\nsaved -> {outp}")


if __name__ == "__main__":
    main()
