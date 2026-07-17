"""
Tier-1 imputation baselines on KIRC (same harness/splits/features as DCMD-Surv):
  - zero    : missing modality -> zeros
  - mean    : missing modality -> training-set mean (over present samples)
  - knn     : missing modality -> mean of k cross-modal nearest neighbours
              (EMMS's 'KNN' imputation baseline; Troyanskaya et al. 2001)
Fusion: concat(imputed image, imputed gene) -> MLP -> Cox. Reports C-index + IBS
+ cal-MAD per scenario (P/G/C), raw & recalibrated — identical protocol to ours.

Missingness is applied via the dataset masks (train blanking per config); for a
blanked modality the REAL feature is discarded and replaced by the imputation.

Usage: python3 baseline_impute.py --strategy knn [--epochs 30]
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

HYPAL = "/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_sample_level"
CODE = "/home/sbarua/Region_based_segmentation/missing_modality/code"
for p in (HYPAL, CODE):
    if p not in sys.path:
        sys.path.insert(0, p)
from genodistil_cpkf import cox_partial_likelihood_loss, cindex_lifelines  # noqa: E402
from dataset_mm import load_gene_dict, load_fold_config, CONFIGS  # noqa: E402
from calibration_mm import cox_calibration_report, fit_temperature  # noqa: E402

RESULTS_DIR = "/home/sbarua/Region_based_segmentation/missing_modality/results"
K = 5


def _t(a, device):
    return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device)


class Imputer:
    """Fit on TRAIN samples that have BOTH modalities; impute a missing modality
    from the present one. Features are raw (scaling happens later on the concat)."""
    def __init__(self, strategy):
        self.s = strategy

    def fit(self, Xi, Xg, im, gm):
        both = (im > 0.5) & (gm > 0.5)
        self.mean_i = Xi[im > 0.5].mean(0)
        self.mean_g = Xg[gm > 0.5].mean(0)
        if self.s == "knn":
            self.ci, self.cg = Xi[both], Xg[both]           # complete pool
            self.sc_i = StandardScaler().fit(self.ci)
            self.sc_g = StandardScaler().fit(self.cg)
            self.nn_i = NearestNeighbors(n_neighbors=K).fit(self.sc_i.transform(self.ci))  # query by image
            self.nn_g = NearestNeighbors(n_neighbors=K).fit(self.sc_g.transform(self.cg))  # query by gene
        return self

    def impute(self, Xi, Xg, im, gm):
        """Return (Xi', Xg') with blanked modalities filled per strategy."""
        Xi, Xg = Xi.copy(), Xg.copy()
        need_g = gm < 0.5   # gene missing -> impute from image
        need_i = im < 0.5   # image missing -> impute from gene
        if self.s == "zero":
            Xi[need_i] = 0.0; Xg[need_g] = 0.0
        elif self.s == "mean":
            Xi[need_i] = self.mean_i; Xg[need_g] = self.mean_g
        elif self.s == "knn":
            if need_g.any():
                d, idx = self.nn_i.kneighbors(self.sc_i.transform(Xi[need_g]))
                Xg[need_g] = self.cg[idx].mean(1)
            if need_i.any():
                d, idx = self.nn_g.kneighbors(self.sc_g.transform(Xg[need_i]))
                Xi[need_i] = self.ci[idx].mean(1)
        return Xi, Xg


class ConcatCox(nn.Module):
    def __init__(self, di=1536, dg=256, h=256, p=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(di + dg, h), nn.ReLU(True), nn.Dropout(p),
                                 nn.Linear(h, h // 2), nn.ReLU(True), nn.Dropout(p),
                                 nn.Linear(h // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_fold(tr, strategy, device, epochs, seed):
    imp = Imputer(strategy).fit(tr["X_img"], tr["X_gene"], tr["img_present"], tr["gene_present"])
    Xi, Xg = imp.impute(tr["X_img"], tr["X_gene"], tr["img_present"], tr["gene_present"])
    sc = StandardScaler().fit(np.concatenate([Xi, Xg], 1))
    X = sc.transform(np.concatenate([Xi, Xg], 1)).astype(np.float32)
    torch.manual_seed(seed); np.random.seed(seed)
    model = ConcatCox(Xi.shape[1], Xg.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = _t(X, device); t = _t(tr["t"], device); e = _t(tr["e"], device)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = cox_partial_likelihood_loss(model(Xt), t, e)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model, imp, sc


@torch.no_grad()
def scenario_risk(model, imp, sc, X_img, X_gene, im_flag, gm_flag, device):
    n = len(X_img)
    im = np.full(n, im_flag, float); gm = np.full(n, gm_flag, float)
    Xi, Xg = imp.impute(X_img, X_gene, im, gm)
    X = sc.transform(np.concatenate([Xi, Xg], 1)).astype(np.float32)
    model.eval()
    return model(_t(X, device)).cpu().numpy()


SCEN = {"genes_missing": (1, 0), "image_missing": (0, 1), "both": (1, 1)}
METRICS = ["cindex", "ibs", "cal_mad", "ibs_recal", "cal_mad_recal"]


def eval_fold(model, imp, sc, tr, va, device):
    out = {}
    for s, (im, gm) in SCEN.items():
        tr_r = scenario_risk(model, imp, sc, tr["X_img"], tr["X_gene"], im, gm, device)
        te_r = scenario_risk(model, imp, sc, va["X_img"], va["X_gene"], im, gm, device)
        c = cindex_lifelines(te_r, va["e"], va["t"])
        raw = cox_calibration_report(tr_r, tr["t"], tr["e"], te_r, va["t"], va["e"], 1.0)
        T = fit_temperature(tr_r, tr["t"], tr["e"])
        rec = cox_calibration_report(tr_r, tr["t"], tr["e"], te_r, va["t"], va["e"], T)
        out[s] = {"cindex": c, "ibs": raw["ibs"], "cal_mad": raw["cal_mad"],
                  "ibs_recal": rec["ibs"], "cal_mad_recal": rec["cal_mad"]}
    return out


def run_config(cfg, gd, strategy, device, epochs, folds, seed=0):
    acc = {s: {k: [] for k in METRICS} for s in SCEN}
    for fold in range(folds):
        data = load_fold_config(fold, cfg, gd)
        model, imp, sc = train_fold(data["train"], strategy, device, epochs, seed + fold)
        r = eval_fold(model, imp, sc, data["train"], data["val"], device)
        for s in acc:
            for k in acc[s]:
                acc[s][k].append(r[s][k])
    return {s: {k: float(np.nanmean(v)) for k, v in acc[s].items()} for s in acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="knn", choices=["zero", "mean", "knn"])
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
    res = {c: run_config(c, gd, args.strategy, device, args.epochs, folds) for c in cfgs}
    for c in cfgs:
        print(f"[{c}] " + " ".join(f"{sn[s]}=C{res[c][s]['cindex']:.3f}/IBS{res[c][s]['ibs']:.3f}"
                                   for s in ["genes_missing", "image_missing", "both"]))
    if args.smoke:
        return
    lines = ["=" * 64, f"BASELINE: {args.strategy}-impute + concat-MLP-Cox — {cohort} 5-fold", "=" * 64]
    for label, cs in [("TABLE 1 (0%)", ["W0_O0"]), ("TABLE 2 (60% avg)", CONFIGS[1:])]:
        lines.append("\n" + label + "\n  scenario         C-index | IBS(raw->recal) | calMAD(raw->recal)")
        for s in ["genes_missing", "image_missing", "both"]:
            C = np.mean([res[c][s]["cindex"] for c in cs]); I = np.mean([res[c][s]["ibs"] for c in cs])
            Ir = np.mean([res[c][s]["ibs_recal"] for c in cs]); M = np.mean([res[c][s]["cal_mad"] for c in cs])
            Mr = np.mean([res[c][s]["cal_mad_recal"] for c in cs])
            lines.append(f"  {sn[s]:14s}  {C:.4f}  | {I:.4f}->{Ir:.4f} | {M:.4f}->{Mr:.4f}")
    out = "\n".join(lines); print("\n" + out)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outp = os.path.join(RESULTS_DIR, f"baseline_{args.strategy}__{cohort}.txt")
    with open(outp, "w") as f:
        f.write(out + "\n")
    print(f"\nsaved -> {outp}")


if __name__ == "__main__":
    main()
