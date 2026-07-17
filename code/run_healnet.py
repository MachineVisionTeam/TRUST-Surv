"""
HEALNet (NeurIPS'24) reproduced for our missing-modality survival protocol, using
its ACTUAL model (healnet/models/healnet.py). Missing modality = pass None (its
native mechanism). Pooled features (uni2h 1536 + BulkRNABert 256), discrete-time
survival (4-bin NLL, same as MOTCat). Cohort via COHORT env. Reports C-index +
IBS per scenario (P/G/C), 0% and 60%.

Training (full-batch): split samples by missing-pattern into groups
(complete / image-missing / gene-missing); each group forwarded with None for its
absent modality; NLL summed. Eval: 3 scenarios via None.

Usage: COHORT=KIRC python3 run_healnet.py --epochs 40 [--smoke]
"""
import argparse, os, sys, importlib.util
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

HEALNET_MODEL = "/home/sbarua/Region_based_segmentation/missing_modality/replication/healnet/healnet/models/healnet.py"
HEALNET_ROOT = "/home/sbarua/Region_based_segmentation/missing_modality/replication/healnet"
HYPAL = "/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_sample_level"
CODE = "/home/sbarua/Region_based_segmentation/missing_modality/code"
for p in (HYPAL, CODE, HEALNET_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
from genodistil_cpkf import cindex_lifelines  # noqa: E402
from dataset_mm import load_gene_dict, load_fold_config, CONFIGS  # noqa: E402
from run_motcat import disc_bins, nll_surv  # reuse discrete-survival helpers  # noqa: E402
from compute_ibs_dispro import step_survival  # noqa: E402
from calibration_mm import eval_time_grid, compute_ibs  # noqa: E402

RESULTS_DIR = "/home/sbarua/Region_based_segmentation/missing_modality/results"
SCEN = {"genes_missing": (1, 0), "image_missing": (0, 1), "both": (1, 1)}
N_BINS = 4


def _load_healnet():
    spec = importlib.util.spec_from_file_location("healnet_model", HEALNET_MODEL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HealNet


def _t(a, device):
    return torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device)


def _haz_S(logits):
    hazards = torch.sigmoid(logits)
    S = torch.cumprod(1 - hazards, dim=1)
    return hazards, S


def train_fold(tr, device, epochs, seed):
    sc_i, sc_g = StandardScaler(), StandardScaler()
    im_m, gm_m = tr["img_present"] > 0.5, tr["gene_present"] > 0.5
    sc_i.fit(tr["X_img"][im_m]); sc_g.fit(tr["X_gene"][gm_m])
    Xi = _t(sc_i.transform(tr["X_img"]).astype(np.float32), device).unsqueeze(1)  # (n,1,1536)
    Xg = _t(sc_g.transform(tr["X_gene"]).astype(np.float32), device).unsqueeze(1)  # (n,1,256)
    y, qb = disc_bins(tr["t"], tr["e"])
    Y = torch.from_numpy(y).long().to(device)
    c = _t(1.0 - tr["e"], device)                      # censorship = 1 - event
    im = _t(tr["img_present"], device); gm = _t(tr["gene_present"], device)
    torch.manual_seed(seed); np.random.seed(seed)
    HealNet = _load_healnet()
    model = HealNet(n_modalities=2, channel_dims=[1536, 256],
                    num_spatial_axes=[1, 1], out_dims=N_BINS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    complete = (im > 0.5) & (gm > 0.5)
    imgmiss = (im < 0.5) & (gm > 0.5)                  # image absent, gene present
    genemiss = (im > 0.5) & (gm < 0.5)                 # gene absent, image present
    groups = [("both", complete), ("imgmiss", imgmiss), ("genemiss", genemiss)]
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.zeros((), device=device)
        for name, mask in groups:
            if mask.sum() < 1:
                continue
            idx = mask
            ti = None if name == "imgmiss" else Xi[idx]
            tg = None if name == "genemiss" else Xg[idx]
            logits = model([ti, tg])
            hazards, S = _haz_S(logits)
            loss = loss + nll_surv(hazards, S, Y[idx], c[idx]) * float(mask.sum())
        loss = loss / len(tr["t"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model, sc_i, sc_g, qb


@torch.no_grad()
def _scenario(model, sc_i, sc_g, X_img, X_gene, im_flag, gm_flag, device):
    Xi = _t(sc_i.transform(X_img).astype(np.float32), device).unsqueeze(1)
    Xg = _t(sc_g.transform(X_gene).astype(np.float32), device).unsqueeze(1)
    ti = Xi if im_flag else None
    tg = Xg if gm_flag else None
    model.eval()
    _, S = _haz_S(model([ti, tg]))
    return S.cpu().numpy()


def eval_fold(model, sc_i, sc_g, qb, tr, va, device):
    out = {}
    for s, (im, gm) in SCEN.items():
        Sva = _scenario(model, sc_i, sc_g, va["X_img"], va["X_gene"], im, gm, device)
        risk = -Sva.sum(1)
        c = cindex_lifelines(risk, va["e"], va["t"])
        times = eval_time_grid(va["t"], va["e"])
        teS = step_survival(Sva, qb, times)
        ibs = compute_ibs(tr["t"], tr["e"], va["t"], va["e"], teS, times)
        out[s] = {"cindex": c, "ibs": ibs}
    return out


def run_config(cfg, gd, device, epochs, folds, seed=0):
    acc = {s: {"cindex": [], "ibs": []} for s in SCEN}
    for fold in range(folds):
        data = load_fold_config(fold, cfg, gd)
        model, sc_i, sc_g, qb = train_fold(data["train"], device, epochs, seed + fold)
        r = eval_fold(model, sc_i, sc_g, qb, data["train"], data["val"], device)
        for s in acc:
            acc[s]["cindex"].append(r[s]["cindex"]); acc[s]["ibs"].append(r[s]["ibs"])
    return {s: {k: float(np.nanmean(v)) for k, v in acc[s].items()} for s in acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
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
    lines = ["=" * 64, f"BASELINE: HEALNet (NeurIPS'24, iterative fusion, None-missing) — {cohort} 5-fold", "=" * 64]
    for label, cs in [("TABLE 1 (0%)", ["W0_O0"]), ("TABLE 2 (60% avg)", CONFIGS[1:])]:
        lines.append("\n" + label + "\n  scenario         C-index | IBS")
        for s in ["genes_missing", "image_missing", "both"]:
            C = np.mean([res[c][s]["cindex"] for c in cs]); I = np.mean([res[c][s]["ibs"] for c in cs])
            lines.append(f"  {sn[s]:14s}  {C:.4f}  | {I:.4f}")
    out = "\n".join(lines); print("\n" + out)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outp = os.path.join(RESULTS_DIR, f"baseline_healnet__{cohort}.txt")
    with open(outp, "w") as f:
        f.write(out + "\n")
    print(f"\nsaved -> {outp}")


if __name__ == "__main__":
    main()
