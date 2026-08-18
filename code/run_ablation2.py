"""
Two-step ablation for Goal 2 (does feature-level distillation break the
single-modality ceiling?). ONE cohort per process (COHORT env selects it, via
dataset_mm dispatch). Runs the SAME model twice at 0% missing (W0_O0), 5-fold:

  ON  : la=1.0, ld=0.3   (full DCMD-Surv, distillation enabled)
  OFF : la=0.0, ld=0.0   (distillation disabled -> ablated ceiling)

The headline number is genes-miss(P) = image-only head h_I. image-miss(G) and
both(C) are printed as CONTROLS: distillation should mainly lift genes-miss and
leave the others roughly unchanged.

Appends one block per cohort to results/ablation_distillation.txt so a bash loop
over cohorts builds the full table. No model code changed — only --la/--ld.

Usage: COHORT=KIRC python3 run_ablation2.py --epochs 30 --gpu 0
"""
import argparse, os, sys
import numpy as np
import torch

CODE_DIR = "/home/sbarua/Region_based_segmentation/missing_modality/code"
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
from dataset_mm import load_gene_dict          # noqa: E402  (COHORT-dispatched)
from run_dcmd import run_config                # noqa: E402

RESULTS_DIR = "/home/sbarua/Region_based_segmentation/missing_modality/results"
OUT = os.path.join(RESULTS_DIR, "ablation_distillation.txt")
CFG = "W0_O0"                                  # 0% missing (clean ceiling test)
SN = {"genes_missing": "genes-miss(P) [image-only]",
      "image_missing": "image-miss(G) [gene-only] ",
      "both":          "both(C)       [complete]  "}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    cohort = os.environ.get("COHORT", "KIRC").upper()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"[ablation2] cohort={cohort} device={device} epochs={args.epochs} folds={args.folds}")
    gd = load_gene_dict()

    # same seed=0, same folds -> ON and OFF differ ONLY by the distillation switch
    on  = run_config(CFG, gd, device, args.epochs, args.folds, la=1.0, ld=0.3, seed=0)
    off = run_config(CFG, gd, device, args.epochs, args.folds, la=0.0, ld=0.0, seed=0)

    lines = ["", "=" * 72,
             f"{cohort} — distillation ablation (0% missing, {args.folds}-fold mean C-index)",
             "  ON  = la=1.0 ld=0.3 (full)   OFF = la=0.0 ld=0.0 (no distillation)",
             "=" * 72,
             f"  {'scenario':30s} |   ON     |  OFF     |  delta (ON-OFF)"]
    for k in ["genes_missing", "image_missing", "both"]:
        d = on[k][0] - off[k][0]
        mark = "  <-- ceiling" if k == "genes_missing" else ""
        lines.append(f"  {SN[k]:30s} | {on[k][0]:.4f}  | {off[k][0]:.4f}  | {d:+.4f}{mark}")
    block = "\n".join(lines)
    print(block)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT, "a") as f:
        f.write(block + "\n")
    print(f"\nappended -> {OUT}")


if __name__ == "__main__":
    main()
