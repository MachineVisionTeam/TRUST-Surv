"""
Train + eval DCMD-Surv on KIRC. Same imputation-free harness as run_mm, but the
distillation is now FEATURE-LEVEL cross-modal alignment (the ceiling-breaker):

  L = Cox(h_F) + lg*Cox(h_G) + li*Cox(h_I) + la*L_align [+ ld*rank-distill]
  L_align = MSE( normalize(G_img), normalize(G_gene).detach() )  on both-present
            -> pulls the image general component toward the gene general component
            -> head_I(G_img) becomes gene-informed -> lifts genes-missing beyond
               the naive image-only ceiling.

Eval routing (imputation-free): both->h_F | genes-missing->h_I | image-missing->h_G

Usage: python3 run_dcmd.py --epochs 30 [--la 1.0] [--ld 0.3]
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

HYPAL_PATH = "/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_sample_level"
CODE_DIR = "/home/sbarua/Region_based_segmentation/missing_modality/code"
for p in (HYPAL_PATH, CODE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
from genodistil_cpkf import (cox_partial_likelihood_loss,  # noqa: E402
                             ranking_distillation_loss, cindex_lifelines)
from dataset_mm import load_gene_dict, load_fold_config, CONFIGS  # noqa: E402
from model_dcmd import DCMDSurv  # noqa: E402

RESULTS_DIR = "/home/sbarua/Region_based_segmentation/missing_modality/results"


def _t(a, device):
    return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device)


def train_one_fold(tr, device, epochs, seed, la, ld, bidir=True, absent="learned"):
    sc_i, sc_g = StandardScaler(), StandardScaler()
    Xi = sc_i.fit_transform(tr["X_img"]).astype(np.float32)
    Xg = sc_g.fit_transform(tr["X_gene"]).astype(np.float32)
    torch.manual_seed(seed); np.random.seed(seed)
    model = DCMDSurv(absent=absent).to(device)
    cfg = model.cfg
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    Xi_t, Xg_t = _t(Xi, device), _t(Xg, device)
    imp = _t(tr["img_present"], device); gmp = _t(tr["gene_present"], device)
    t = _t(tr["t"], device); e = _t(tr["e"], device)
    zero = torch.zeros((), device=device)
    both = (imp > 0.5) & (gmp > 0.5)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        h_F, h_I, h_G, G_img, G_gene = model(Xi_t, Xg_t, imp, gmp)
        L_F = cox_partial_likelihood_loss(h_F, t, e)
        L_G = cox_partial_likelihood_loss(h_G, t, e)          # teacher (real gene)
        mi = imp > 0.5
        if mi.sum() >= 2 and e[mi].sum() >= 1:
            L_I = cox_partial_likelihood_loss(h_I[mi], t[mi], e[mi])
            # unidirectional gene->image rank distillation (bidirectional/fused-teacher
            # was tested and REJECTED: it corrupts the gene-only head -> negative transfer)
            L_rank = ranking_distillation_loss(h_I[mi], h_G[mi].detach(), t[mi], e[mi])
        else:
            L_I, L_rank = zero, zero
        # FEATURE-level cross-modal distillation (the ceiling-breaker)
        if both.sum() >= 2:
            gi = F.normalize(G_img[both], dim=1)
            gg = F.normalize(G_gene[both], dim=1).detach()
            L_align = F.mse_loss(gi, gg)
        else:
            L_align = zero
        _ = bidir  # kept for API compat; fused-teacher variant rejected (negative transfer)
        L = L_F + cfg.lambda_aux * (L_G + L_I) + la * L_align + ld * L_rank
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        opt.step()
    return model, sc_i, sc_g


@torch.no_grad()
def eval_three(model, va, sc_i, sc_g, device):
    model.eval()
    Xi = _t(sc_i.transform(va["X_img"]).astype(np.float32), device)
    Xg = _t(sc_g.transform(va["X_gene"]).astype(np.float32), device)
    n = len(va["t"]); ones = torch.ones(n, device=device); zeros = torch.zeros(n, device=device)
    h_F, _, _, _, _ = model(Xi, Xg, ones, ones)
    _, h_I, _, _, _ = model(Xi, Xg, ones, zeros)
    _, _, h_G, _, _ = model(Xi, Xg, zeros, ones)
    return {"both": cindex_lifelines(h_F.cpu().numpy(), va["e"], va["t"]),
            "genes_missing": cindex_lifelines(h_I.cpu().numpy(), va["e"], va["t"]),
            "image_missing": cindex_lifelines(h_G.cpu().numpy(), va["e"], va["t"])}


def run_config(cfg_name, gd, device, epochs, folds, la, ld, bidir=True, seed=0, absent="learned"):
    acc = {k: [] for k in ["both", "genes_missing", "image_missing"]}
    for fold in range(folds):
        data = load_fold_config(fold, cfg_name, gd)
        model, sc_i, sc_g = train_one_fold(data["train"], device, epochs, seed + fold, la, ld, bidir, absent)
        sc = eval_three(model, data["val"], sc_i, sc_g, device)
        for k in acc:
            acc[k].append(sc[k])
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--la", type=float, default=1.0, help="feature-alignment weight")
    ap.add_argument("--ld", type=float, default=0.3, help="rank-distill weight")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    epochs = 40 if args.smoke else args.epochs
    folds = 1 if args.smoke else 5
    print(f"[dcmd] device={device} epochs={epochs} folds={folds} la={args.la} ld={args.ld}")
    gd = load_gene_dict()
    sn = {"genes_missing": "genes-miss(P)", "image_missing": "image-miss(G)", "both": "both(C)"}
    results = {}
    for cfg in (CONFIGS[:1] if args.smoke else CONFIGS):
        res = run_config(cfg, gd, device, epochs, folds, args.la, args.ld)
        results[cfg] = res
        print(f"[{cfg}] " + "  ".join(f"{sn[k]}={res[k][0]:.4f}" for k in ["genes_missing", "image_missing", "both"]))
    if args.smoke:
        return
    lines = ["="*60, f"DCMD-Surv (feature-align la={args.la}, rank ld={args.ld}) — KIRC 5-fold", "="*60]
    for label, cfgs in [("TABLE 1 (0% missing)", ["W0_O0"]), ("TABLE 2 (60% missing, avg)", CONFIGS[1:])]:
        lines.append("\n"+label)
        for k in ["genes_missing", "image_missing", "both"]:
            lines.append(f"  {sn[k]:14s} {np.mean([results[c][k][0] for c in cfgs]):.4f}")
    out = "\n".join(lines); print("\n"+out)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "ours_dcmd__KIRC.txt"), "w") as f:
        f.write(out+"\n")
    print(f"\nsaved -> {RESULTS_DIR}/ours_dcmd__KIRC.txt")


if __name__ == "__main__":
    main()
