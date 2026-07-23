# DCMD-Surv

### Decomposed Cross-Modal Distillation for Imputation-Free Missing-Modality Survival Prediction

A multimodal (histology WSI + genomics) cancer **survival-prediction** model that keeps working when one modality is missing — **without imputing or reconstructing** the missing data.

> **Full name:** DCMD-Surv = *"Decomposed Cross-Modal Distillation for Survival"*

---

## Main Goal — why we built it

In the real clinic, patients often have only **one** modality: a pathology slide but no sequencing, or sequencing but no usable slide. Existing strong methods either **discard** single-modality patients, or **impute** the missing modality (e.g., DisPro uses an LLM to "recover" it) — which can hallucinate signal and produce over-confident, **poorly-calibrated** risk scores.

Our goals:

1. **Truly imputation-free** — never reconstruct the missing modality; use a learned *absent token* + a survival head specialized for each scenario.
2. **Break the single-modality ceiling** — via **feature-level cross-modal distillation**: decompose each modality into a modality-*general* component and pull the image-general toward the gene-general, so the image-only head becomes *gene-informed*.
3. **Trustworthy probabilities** — optimize not just ranking (**C-index**) but **calibration (IBS)**: predicted survival probabilities should be reliable, which matters most in the clinic.
4. **Exploit real incomplete cohorts** — train on genuinely single-modality patients that prior methods throw away.

---

## How the Model Works

**The problem in one sentence.** A patient with only a slide (no sequencing) gets a worse prediction than a patient with both — because histology alone has a hard ceiling. Genes carry prognostic signal the slide simply doesn't show.

**The usual fix, and why we avoid it.** Most methods *invent* the missing genes (imputation / reconstruction / an LLM "recovering" the modality). But invented genes are a guess presented as a measurement — the model then becomes over-confident, and its survival probabilities stop being trustworthy.

**Our fix — move the knowledge, not the data.** We never reconstruct the missing modality. Instead, *during training* (when both modalities are present), we teach the image branch to organise its features the way the gene branch does. At test time the image branch keeps that gene-shaped structure even with no genes in sight.

### The four pieces

**1. Two adapters → a shared space.**
Image (UNI2h, 1536-d) and genes (BulkRNABert, 256-d) each pass through a small adapter into a common 256-d space (`z_img`, `z_gene`). Different sizes, now comparable.

**2. Decomposition into a "general" component.**
Each modality is projected a second time into a **modality-general** component — `G_img` and `G_gene` — meant to hold what the two modalities *agree* about (the shared biology), as opposed to what is unique to a slide or to a transcriptome.

**3. Cross-modal distillation (the key step).**
We pull `G_img` toward `G_gene`:

```
L_align = MSE( normalize(G_img), normalize(G_gene).detach() )     # both-present patients only
```

`.detach()` makes this **one-directional**: genes teach the image, never the reverse. The image branch must move to meet the gene branch. Because `L_align` only compares *general components*, the image branch is never asked to reproduce actual gene values — **no reconstruction, so no imputation.**

> We tested making it bidirectional (fused-teacher → gene head). It **hurt** — the gene-only head started relying on image-derived patterns it cannot access when the image is absent (image-missing C-index 0.706 → 0.685). We kept it one-directional and report this honestly.

**4. Absent tokens + three specialised heads.**
A missing modality is replaced by a **learned "absent" vector** — an explicit *"this is not here"* signal, not a fake measurement:

```
z_img  = present ? z_img  : absent_image      (learned parameter)
z_gene = present ? z_gene : absent_gene
```

Three heads then predict risk, each for its own situation:

| Head | Used when | Reads |
|---|---|---|
| `h_F` | both present | HGBF-fused representation |
| `h_I` | **genes missing** | `G_img` — the *gene-aligned* image component |
| `h_G` | **image missing** | `z_gene` — the full gene features |

`h_I` is where the gain comes from: it reads the component that was trained to imitate genes, so an image-only patient gets a **gene-informed** prediction.

### Training objective

```
L = Cox(h_F) + λ_aux · [ Cox(h_G) + Cox(h_I) ] + λ_a · L_align + λ_d · L_rank
```

Cox partial-likelihood on each head, plus the feature alignment, plus a small gene→image ranking-distillation term. **≈2.5 M parameters** — light enough to train in minutes.

```
x_img (1536) ──adapter──> z_img ──> G_img ──┐(pull toward)     ──> h_I   [genes missing]
                             │              │  L_align
x_gene (256) ──adapter──> z_gene ──> G_gene ─┘(teacher, detached) ──> h_G  [image missing]
                             │
                    [absent tokens] ──> HGBF fusion ──────────────> h_F   [complete]
```

*(Full detail in [`ARCHITECTURE.txt`](ARCHITECTURE.txt); full head-to-head numbers in [`COMPARISON.txt`](COMPARISON.txt).)*

---

## Main Results

**The two metrics, in plain terms:**

- **C-index ↑** — *ranking*. "Of two patients, did the model correctly say which one dies sooner?" 0.5 = coin flip, 1.0 = perfect.
- **IBS ↓** — *calibration*. "When the model says *72% chance of surviving 3 years*, does that actually happen 72% of the time?" A model can rank well yet still state wrong probabilities — and it is the probability a clinician acts on.

**Setup.** 5-fold CV. Scenarios: **P** = genes missing (image-only) · **G** = image missing (gene-only) · **C** = complete. `0%` = complete training data · `60%` = 60% of training patients missing a modality (avg of 5 configurations).
**Feature-matched:** every method gets the *identical* UNI2h image + BulkRNABert gene features — so differences come from the method, not the features.

> **DCMD-Surv has the highest C-index in the complete scenario on all five cohorts (0% and 60%; 10/10 cells), and the lowest (or tied-lowest) IBS in 23 of 30 cells** — beating Flex-MoE (NeurIPS'24), MUSE (ICLR'24), HEALNet (NeurIPS'24), ShaSpec (CVPR'23), MOTCat (ICCV'23), and naive imputation.
>
> The main calibration exception is **UCEC**, where MUSE has lower IBS across every scenario. All exceptions are stated explicitly below rather than omitted.

### KIRC (n = 417 paired)

| Setting | Scenario | **DCMD-Surv (ours)** C / IBS | Best baseline C / IBS |
|---|---|---|---|
| 0% | genes-miss (P) | **0.751 / 0.121** | 0.715 / 0.153 (KNN) |
| 0% | image-miss (G) | **0.705 / 0.135** | 0.691 / 0.137 (MUSE/mean) |
| 0% | complete (C) | **0.764 / 0.128** | 0.735 / 0.141 (ShaSpec/MOTCat) |
| 60% | genes-miss (P) | **0.738 / 0.122** | 0.709 / 0.148 |
| 60% | image-miss (G) | **0.706 / 0.135** | 0.673 / 0.141 |
| 60% | complete (C) | **0.759 / 0.132** | 0.736 / 0.145 |

### GBMLGG (n = 592 paired)

| Setting | Scenario | **DCMD-Surv (ours)** C / IBS | Best baseline C / IBS |
|---|---|---|---|
| 0% | genes-miss (P) | **0.817 / 0.117** | 0.805 / 0.140 (KNN) |
| 0% | image-miss (G) | **0.823 / 0.117** | 0.818 / 0.127 (MUSE/mean) |
| 0% | complete (C) | **0.823 / 0.129** | 0.814 / 0.133 (MUSE/MOTCat) |
| 60% | genes-miss (P) | **0.814 / 0.118** | 0.802 / 0.134 (KNN) |
| 60% | image-miss (G) | **0.823 / 0.117** | 0.809 / 0.130 (zero) |
| 60% | complete (C) | **0.817** / 0.133 | 0.816 / **0.131** (zero) |

### LUAD (n = 447 paired)

| Setting | Scenario | **DCMD-Surv (ours)** C / IBS | Best baseline C / IBS |
|---|---|---|---|
| 0% | genes-miss (P) | **0.574 / 0.164** | 0.568 / 0.203 (HEALNet) |
| 0% | image-miss (G) | 0.560 / **0.170** | **0.581** / 0.183 (ShaSpec/MUSE) |
| 0% | complete (C) | **0.591 / 0.161** | 0.581 / 0.169 (ShaSpec/MOTCat) |
| 60% | genes-miss (P) | 0.558 / **0.165** | **0.562** / 0.201 (mean/HEALNet) |
| 60% | image-miss (G) | **0.560 / 0.170** | 0.548 / 0.186 (ShaSpec/HEALNet) |
| 60% | complete (C) | **0.571 / 0.161** | 0.570 / 0.193 (mean/HEALNet) |

### UCEC (n = 467 paired)

| Setting | Scenario | **DCMD-Surv (ours)** C / IBS | Best baseline C / IBS |
|---|---|---|---|
| 0% | genes-miss (P) | 0.669 / 0.087 | **0.675 / 0.080** (KNN / MUSE) |
| 0% | image-miss (G) | **0.672** / 0.088 | 0.657 / **0.084** (HEALNet / MUSE) |
| 0% | complete (C) | **0.702** / 0.082 | 0.698 / **0.080** (zero / MUSE) |
| 60% | genes-miss (P) | 0.651 / 0.081 | **0.665 / 0.080** (MUSE) |
| 60% | image-miss (G) | **0.657** / 0.085 | 0.621 / **0.084** (zero / MUSE) |
| 60% | complete (C) | **0.679** / 0.083 | 0.672 / **0.080** (zero / MUSE) |

*UCEC is the calibration counter-example:* DCMD wins the complete and image-miss C-index, but **MUSE has lower IBS in every cell** (~0.080–0.084 vs ours 0.081–0.088), and KNN/MUSE edge the genes-miss C-index. All methods are well-calibrated here (IBS ~0.08), so the absolute gaps are small — but the "lowest IBS" claim does not hold on UCEC.

### BRCA (n = 940 paired)

| Setting | Scenario | **DCMD-Surv (ours)** C / IBS | Best baseline C / IBS |
|---|---|---|---|
| 0% | genes-miss (P) | **0.622 / 0.102** | 0.608 / 0.115 (KNN / MUSE) |
| 0% | image-miss (G) | **0.542 / 0.103** | 0.528 / 0.110 (Flex-MoE / MUSE) |
| 0% | complete (C) | **0.612 / 0.103** | 0.581 / 0.104 (zero / MUSE) |
| 60% | genes-miss (P) | **0.624 / 0.102** | 0.613 / 0.104 (HEALNet / MUSE) |
| 60% | image-miss (G) | 0.510 / **0.104** | **0.542 / 0.104** (MUSE) |
| 60% | complete (C) | **0.619 / 0.103** | 0.579 / 0.104 (ShaSpec / MUSE) |

*BRCA is DCMD's strongest cohort on discrimination* (complete C-index +3–4 points over every baseline, lowest/tied IBS throughout). The one weak spot is **gene-only (image-miss): C-index is near-random (~0.51–0.54) for all methods**, MUSE edges DCMD at 60%, and DCMD's KM risk stratification for that scenario is non-significant (log-rank p = 0.35) — a genuine limitation, since BRCA transcriptome-only signal is simply weakly prognostic.

### Summary — does it win on the *missing* scenarios, and on calibration?

Because this is a **missing-modality** method, the scenarios that matter are the ones where a modality is actually absent (**P** = genes-missing / image-only, **G** = image-missing / gene-only). The complete scenario is where the method does *not* apply, so it is a sanity check, not the headline.

**1. Discrimination in the MISSING scenarios (C-index, P+G, 0%+60% → 4 cells/cohort):**

| Cohort | Missing-scenario C-index wins vs best baseline | Note |
|---|---|---|
| KIRC | **4 / 4** | clear, large margins |
| GBMLGG | **4 / 4** | clear |
| BRCA | **3 / 4** | loses gene-only 60% to MUSE |
| LUAD | 2 / 4 | loses to imputation/ShaSpec; hardest cohort |
| UCEC | 2 / 4 | loses genes-miss to KNN/MUSE |
| **Total** | **15 / 20 missing-scenario cells** | competitive-to-best, **not** a clean sweep |

So: **DCMD wins the majority of missing-modality cells (15/20)**, dominant on KIRC/GBMLGG, mixed on LUAD/UCEC where *naive imputation* is a genuine rival. Honest read: strong but not universal on discrimination.

**2. Calibration (IBS) — the more consistent and more novel win:**

- **Best-or-tied IBS on 4 of 5 cohorts** (KIRC, GBMLGG, LUAD, BRCA), 23/30 cells overall.
- **UCEC is the one calibration exception** — MUSE (also imputation-free) has lower IBS in all 6 UCEC cells; but every method is trivially well-calibrated there (IBS ~0.08), so the gaps are tiny.
- Unlike most missing-modality papers (which report only C-index/AUC), we show that **staying imputation-free yields better-calibrated survival probabilities than methods that invent the missing modality** — the finding most relevant to clinical use.

**One-line takeaway:** *DCMD-Surv wins most missing-modality cells (15/20) and is competitive-to-best on discrimination, but its clearest and most novel advantage is calibration — best on 4 of 5 cohorts, UCEC being the honest exception.*

**Key takeaways**

- DCMD-Surv has the **highest C-index in the complete scenario on all five cohorts**, at both 0% and 60% missing (10/10 cells) — a sanity check (method does not apply when nothing is missing).
- DCMD-Surv has the **lowest (or tied-lowest) IBS in 23 of 30 cells** — the most consistent advantage, and the one that matters clinically. The exception cohort is **UCEC**, where MUSE is better-calibrated across all 6 cells.
- **Where it does *not* win (stated openly):** GBMLGG 60% complete IBS (zero-fill 0.131 < ours 0.133); LUAD 0% image-miss C-index (ShaSpec 0.581 > ours 0.560); LUAD 60% genes-miss C-index (mean-imp 0.562 > ours 0.558); **UCEC genes-miss C-index** (KNN 0.675 at 0%, MUSE 0.665 at 60%) and **UCEC IBS throughout** (MUSE lower in all 6 cells); **BRCA 60% gene-only C-index** (MUSE 0.542 > ours 0.510).
- **LUAD is the hardest cohort** — *every* method lands in 0.52–0.59, so the single-modality scenarios there sit within fold-noise of the imputation baselines. The calibration gap (0.161 vs 0.19–0.24) is the clear separation.
- **BRCA gene-only is near-random for everyone** (~0.51–0.54); DCMD leads on the complete and image-only scenarios by a wide margin but does not rescue the weakly-prognostic transcriptome-only signal.
- **Real-missing experiment:** training on genuinely single-modality patients is *usable without imputation*. On the original three cohorts (KIRC/GBMLGG/LUAD) it mainly improved **calibration** (IBS better in 15/18 cells) with a small, mixed C-index effect (13/18, best +0.0156 on LUAD image-miss, worst −0.0090 on GBMLGG complete). Extending to UCEC/BRCA weakens this: **on UCEC it degrades the image-miss C-index (−0.030 at 0%, −0.029 at 60%)**, while BRCA is near-neutral (±0.008). Net across all five cohorts the effect is inconsistent — reported strictly as a supplementary finding, not a headline. Per-cohort deltas are in each `comparison_<COHORT>.txt` (and `dcmd_realmissing__*.txt` for the original three).

*Full per-method numbers (every baseline): [`COMPARISON.txt`](COMPARISON.txt) and each cohort's `comparison_*.txt`.*

---

## 📁 Folder Structure

```
DCMD-Surv/
├── ARCHITECTURE.txt        # main DCMD-Surv model architecture (detailed)
├── COMPARISON.txt          # overall comparison vs ALL baselines, per cohort (EMMS-style)
├── README.md               # this file
├── code/                   # all code (SHARED across cohorts; cohort via COHORT env var)
│   ├── model_dcmd.py                  # DCMD-Surv model
│   ├── run_dcmd_cal.py                # train/eval ours (C-index + IBS + cal-MAD)
│   ├── run_dcmd_realmissing.py        # real-missing (exploit single-modality cases)
│   ├── genodistil_cpkf.py             # HGBF fusion backbone (from HyPAL-Surv)
│   ├── dataset_mm.py / dataset_gbmlgg.py / dataset_luad.py   # loaders (COHORT dispatch)
│   ├── generate_{gbmlgg,luad}_splits.py / build_{gbmlgg,luad}_bags.py   # data prep
│   ├── baseline_impute.py             # naive zero/mean/KNN imputation
│   ├── flexmoe_surv.py, run_flexmoe.py         # Flex-MoE (NeurIPS'24)
│   ├── muse_surv.py, run_muse.py               # MUSE (ICLR'24)
│   ├── run_motcat.py                           # MOTCat (ICCV'23, complete-only ref)
│   ├── run_healnet.py                          # HEALNet (NeurIPS'24)
│   ├── shaspec_surv.py, run_shaspec.py         # ShaSpec (CVPR'23)
│   ├── calibration_mm.py / compute_ibs_dispro.py   # IBS / survival-curve utils
│   └── make_comparison_bars.py / make_km_figure.py / make_calibration_figure.py /
│       make_tables.py / make_repo_comparison.py      # figures + tables
├── KIRC/
│   ├── results/            # per-method .txt (C-index + IBS, 0% + 60%) + comparison_KIRC.txt
│   └── figures/            # fig_comparison / fig_km_stratification / fig_calibration
├── GBMLGG/
│   ├── results/            # (same layout)
│   └── figures/            # (same layout)
├── LUAD/
│   ├── results/            # (same layout)
│   └── figures/            # (same layout)
├── UCEC/
│   ├── results/            # comparison_UCEC.txt (comparison-only)
│   └── figures/            # fig_comparison / fig_km_stratification / fig_calibration
└── BRCA/
    ├── results/            # comparison_BRCA.txt (comparison-only)
    └── figures/            # (same layout)
```

---

## 🚀 How to Run

Cohort is selected by the `COHORT` env var (KIRC is the default).

```bash
# our method (C-index + IBS + cal-MAD), both 0% and 60% missing:
COHORT=KIRC   python3 code/run_dcmd_cal.py --epochs 30
COHORT=GBMLGG python3 code/run_dcmd_cal.py --epochs 30
COHORT=LUAD   python3 code/run_dcmd_cal.py --epochs 30
COHORT=UCEC   python3 code/run_dcmd_cal.py --epochs 30
COHORT=BRCA   python3 code/run_dcmd_cal.py --epochs 30

# a baseline (example):
COHORT=GBMLGG python3 code/run_healnet.py --epochs 40

# real-missing experiment:
COHORT=GBMLGG python3 code/run_dcmd_realmissing.py --epochs 30

# figures + tables:
COHORT=KIRC python3 code/make_km_figure.py
COHORT=KIRC python3 code/make_calibration_figure.py
python3 code/make_comparison_bars.py     # all cohorts (numbers in NUMBERS dict)
python3 code/make_tables.py              # comparison_*.txt + MASTER_comparison.txt
python3 code/make_repo_comparison.py     # rebuild COMPARISON.txt from make_tables.py
```

---

## 🧪 Protocol & Dependencies

- **Protocol:** 5-fold CV; scenarios P/G/C; missing settings 0% and 60% (5 blank-configs avg). Metrics: C-index (discrimination) + IBS (calibration).
- **Dependencies:** `torch`, `numpy`, `pandas`, `scikit-learn`, `scikit-survival`, `lifelines`, `torch_geometric` (MUSE), `pot` (MOTCat), `python-box` + `einops` (HEALNet).
- **Cohorts done:** KIRC (417 paired), GBMLGG (592), LUAD (447), UCEC (467), BRCA (940). Add another cohort by adding a `dataset_<cohort>.py` + a `COHORT` branch, then rerun the same commands.

---

## Baselines Compared

| Method | Venue | Type |
|---|---|---|
| zero / mean / KNN impute | — | naive imputation |
| Flex-MoE | NeurIPS 2024 | imputation-free (MoE) |
| MUSE | ICLR 2024 | imputation-free (bipartite GNN) |
| HEALNet | NeurIPS 2024 | imputation-free (iterative fusion) |
| ShaSpec | CVPR 2023 | imputation-free (shared-specific) |
| MOTCat | ICCV 2023 | complete-data fusion (reference) |
| **DCMD-Surv** | **(ours)** | **imputation-free (decomposed distillation)** |
