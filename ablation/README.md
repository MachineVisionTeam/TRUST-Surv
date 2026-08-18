# Component ablations (honest)

Two components carry the "DCMD" name; we ablate both and report — honestly — that
each has only a **marginal** effect. They are therefore **not** claimed as the
source of the gains. The contributions are the conformal reliability guarantee
(`../conformal/`) and calibration; the fusion backbone is reused from prior work.

### 1. Cross-modal distillation — `ablation_distillation.txt`
Distillation ON (`la=1.0, ld=0.3`) vs OFF (`la=0, ld=0`), genes-missing (image-only)
C-index, 0% missing, 5-fold, all 5 cohorts.

| Cohort | ON | OFF | Δ |
|---|---|---|---|
| KIRC | 0.7509 | 0.7430 | +0.0079 |
| GBMLGG | 0.8165 | 0.8176 | −0.0012 |
| LUAD | 0.5739 | 0.5648 | +0.0091 |
| UCEC | 0.6691 | 0.6723 | −0.0033 |
| BRCA | 0.6209 | 0.6265 | −0.0056 |

Average Δ ≈ **+0.001** — within fold noise. Distillation does not break the ceiling.

### 2. Learned absent token — `ablation_absent_token.txt`
Learned token vs fixed zero-fill, complete-scenario C-index (the head the token
affects), 60%-missing setting, 5-fold, all 5 cohorts. Deltas: +0.0001 / +0.0004 /
+0.0014 / +0.0001 / +0.0007 — average ≈ **+0.0005**. The learned token is barely
better than zero-fill.

**Takeaway.** These are lightweight components. The imputation-free routing
(specialized heads + availability-based selection) and the post-hoc conformal
reliability layer are what matter — see the main [`README`](../README.md) and
[`conformal/`](../conformal/).

Reproduce: `COHORT=KIRC python3 code/run_ablation2.py --epochs 30` and
`COHORT=KIRC python3 code/run_ablation_absent.py --epochs 30`.
