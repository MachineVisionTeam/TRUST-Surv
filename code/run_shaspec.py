"""
Train + eval ShaSpec-Surv on KIRC/GBMLGG (COHORT env). Cox + aux losses
(shared alignment + domain classification). C-index + IBS per scenario, 0% + 60%.

Usage: COHORT=KIRC python3 run_shaspec.py --epochs 30 [--smoke]
"""
import argparse, os, sys
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

HYPAL = "/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_sample_level"
CODE = "/home/sbarua/Region_based_segmentation/missing_modality/code"
for p in (HYPAL, CODE):
    if p not in sys.path:
        sys.path.insert(0, p)
from genodistil_cpkf import cox_partial_likelihood_loss, cindex_lifelines  # noqa: E402
from dataset_mm import load_gene_dict, load_fold_config, CONFIGS  # noqa: E402
from calibration_mm import cox_calibration_report  # noqa: E402
from shaspec_surv import ShaSpecSurv  # noqa: E402

RESULTS_DIR = "/home/sbarua/Region_based_segmentation/missing_modality/results"
SCEN = {"genes_missing": (1, 0), "image_missing": (0, 1), "both": (1, 1)}


def _t(a, device):
    return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device)


def train_fold(tr, device, epochs, seed, la=0.5, ld=0.1, lr=1e-3, wd=1e-4):
    sc_i, sc_g = StandardScaler(), StandardScaler()
    im_m, gm_m = tr["img_present"] > 0.5, tr["gene_present"] > 0.5
    sc_i.fit(tr["X_img"][im_m]); sc_g.fit(tr["X_gene"][gm_m])
    Xi = _t(sc_i.transform(tr["X_img"]).astype(np.float32), device)
    Xg = _t(sc_g.transform(tr["X_gene"]).astype(np.float32), device)
    torch.manual_seed(seed); np.random.seed(seed)
    model = ShaSpecSurv().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    imp = _t(tr["img_present"], device); gmp = _t(tr["gene_present"], device)
    t = _t(tr["t"], device); e = _t(tr["e"], device)
    both = (imp > 0.5) & (gmp > 0.5)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        risk, sh_i, sh_g, sp_i, sp_g = model(Xi, Xg, imp, gmp)
        cox = cox_partial_likelihood_loss(risk, t, e)
        align, domain = model.aux_losses(sh_i, sh_g, sp_i, sp_g, both)
        loss = cox + la * align + ld * domain
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model, sc_i, sc_g


@torch.no_grad()
def scenario_risk(model, sc_i, sc_g, X_img, X_gene, im_flag, gm_flag, device):
    n = len(X_img)
    Xi = _t(sc_i.transform(X_img).astype(np.float32), device)
    Xg = _t(sc_g.transform(X_gene).astype(np.float32), device)
    a = torch.full((n,), float(im_flag), device=device)
    b = torch.full((n,), float(gm_flag), device=device)
    model.eval()
    return model(Xi, Xg, a, b)[0].cpu().numpy()


def eval_fold(model, sc_i, sc_g, tr, va, device):
    out = {}
    for s, (im, gm) in SCEN.items():
        tr_r = scenario_risk(model, sc_i, sc_g, tr["X_img"], tr["X_gene"], im, gm, device)
        te_r = scenario_risk(model, sc_i, sc_g, va["X_img"], va["X_gene"], im, gm, device)
        c = cindex_lifelines(te_r, va["e"], va["t"])
        ibs = cox_calibration_report(tr_r, tr["t"], tr["e"], te_r, va["t"], va["e"], 1.0)["ibs"]
        out[s] = {"cindex": c, "ibs": ibs}
    return out


def run_config(cfg, gd, device, epochs, folds, seed=0):
    acc = {s: {"cindex": [], "ibs": []} for s in SCEN}
    for fold in range(folds):
        data = load_fold_config(fold, cfg, gd)
        model, sc_i, sc_g = train_fold(data["train"], device, epochs, seed + fold)
        r = eval_fold(model, sc_i, sc_g, data["train"], data["val"], device)
        for s in acc:
            acc[s]["cindex"].append(r[s]["cindex"]); acc[s]["ibs"].append(r[s]["ibs"])
    return {s: {k: float(np.nanmean(v)) for k, v in acc[s].items()} for s in acc}


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
    sn = {"genes_missing": "genes-miss(P)", "image_missing": "image-miss(G)", "both": "both(C)"}
    cfgs = CONFIGS[:1] if args.smoke else CONFIGS
    res = {c: run_config(c, gd, device, args.epochs, folds) for c in cfgs}
    for c in cfgs:
        print(f"[{c}] " + " ".join(f"{sn[s]}=C{res[c][s]['cindex']:.3f}/IBS{res[c][s]['ibs']:.3f}"
                                   for s in ["genes_missing", "image_missing", "both"]))
    if args.smoke:
        return
    lines = ["=" * 64, f"BASELINE: ShaSpec-Surv (CVPR'23, shared-specific, faithful) — {cohort} 5-fold", "=" * 64]
    for label, cs in [("TABLE 1 (0%)", ["W0_O0"]), ("TABLE 2 (60% avg)", CONFIGS[1:])]:
        lines.append("\n" + label + "\n  scenario         C-index | IBS")
        for s in ["genes_missing", "image_missing", "both"]:
            C = np.mean([res[c][s]["cindex"] for c in cs]); I = np.mean([res[c][s]["ibs"] for c in cs])
            lines.append(f"  {sn[s]:14s}  {C:.4f}  | {I:.4f}")
    out = "\n".join(lines); print("\n" + out)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outp = os.path.join(RESULTS_DIR, f"baseline_shaspec__{cohort}.txt")
    with open(outp, "w") as f:
        f.write(out + "\n")
    print(f"\nsaved -> {outp}")


if __name__ == "__main__":
    main()
