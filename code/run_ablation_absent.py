"""
Absent-token ablation for Goal 1 (imputation-free). ONE cohort per process
(COHORT env selects it via dataset_mm dispatch).

WHY THE 60% SETTING (not 0%): the learned absent token only acts when the
TRAINING set contains patients with a missing modality — it fills the missing
modality's latent in the FUSION path so the complete-head h_F can still train on
those incomplete patients. At 0% missing the token is never triggered, so the
ablation must be run on the blank configs (the "60% missing" setting) and read on
the COMPLETE scenario (h_F), which is the head the token affects.

Two conditions, same seed/folds, differ ONLY by the absent token:
  LEARNED : absent="learned"  (full DCMD-Surv)
  ZERO    : absent="zero"     (fixed zero-fill = latent-space zero imputation)

Reports both(C) [complete = the head under test] as headline, plus genes-miss(P)
and image-miss(G) as controls (those route to h_I/h_G, which never use the token,
so their deltas should be ~0 -> a sanity check that the ablation is isolated).

Averages over the 5 blank configs (CONFIGS[1:]) = the 60%-missing setting, 5-fold.
Appends one block per cohort to results/ablation_absent_token.txt.

Usage: COHORT=KIRC python3 run_ablation_absent.py --epochs 30 --gpu 0
"""
import argparse, os, sys
import numpy as np
import torch

CODE_DIR = "/home/sbarua/Region_based_segmentation/missing_modality/code"
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
from dataset_mm import load_gene_dict, CONFIGS   # noqa: E402  (COHORT-dispatched)
from run_dcmd import run_config                  # noqa: E402

RESULTS_DIR = "/home/sbarua/Region_based_segmentation/missing_modality/results"
OUT = os.path.join(RESULTS_DIR, "ablation_absent_token.txt")
SN = {"both":          "both(C)       [complete, uses token]",
      "genes_missing": "genes-miss(P) [image-only, control] ",
      "image_missing": "image-miss(G) [gene-only, control]  "}


def avg_over_blank_configs(gd, device, epochs, folds, absent):
    """Mean C-index over the 5 blank (60%-missing) configs, per scenario."""
    blank = CONFIGS[1:]                       # W0_O60, W20_O40, W30_O30, W40_O20, W60_O0
    per = {k: [] for k in ["both", "genes_missing", "image_missing"]}
    for cfg in blank:
        res = run_config(cfg, gd, device, epochs, folds, la=1.0, ld=0.3, seed=0, absent=absent)
        for k in per:
            per[k].append(res[k][0])
    return {k: float(np.mean(v)) for k, v in per.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    cohort = os.environ.get("COHORT", "KIRC").upper()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"[ablation-absent] cohort={cohort} device={device} epochs={args.epochs} "
          f"folds={args.folds} (60%-missing setting, {len(CONFIGS)-1} blank configs)")
    gd = load_gene_dict()

    learned = avg_over_blank_configs(gd, device, args.epochs, args.folds, absent="learned")
    zero    = avg_over_blank_configs(gd, device, args.epochs, args.folds, absent="zero")

    lines = ["", "=" * 74,
             f"{cohort} — absent-token ablation (60%-missing setting, {args.folds}-fold, "
             f"{len(CONFIGS)-1}-config avg)",
             "  LEARNED = learned absent token (full)   ZERO = fixed zero-fill token",
             "=" * 74,
             f"  {'scenario':36s} | LEARNED  |  ZERO    |  delta (L-Z)"]
    for k in ["both", "genes_missing", "image_missing"]:
        d = learned[k] - zero[k]
        mark = "  <-- headline" if k == "both" else ""
        lines.append(f"  {SN[k]:36s} | {learned[k]:.4f}  | {zero[k]:.4f}  | {d:+.4f}{mark}")
    block = "\n".join(lines)
    print(block)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT, "a") as f:
        f.write(block + "\n")
    print(f"\nappended -> {OUT}")


if __name__ == "__main__":
    main()
