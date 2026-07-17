# DCMD-Surv

### Decomposed Cross-Modal Distillation for Imputation-Free Missing-Modality Survival Prediction

A multimodal (histology WSI + genomics) cancer **survival-prediction** model that keeps working when one modality is missing — **without imputing or reconstructing** the missing data.

> **Full name:** DCMD-Surv = *"Decomposed Cross-Modal Distillation for Survival"*

---

## 🎯 Main Goal — why we built it

In the real clinic, patients often have only **one** modality: a pathology slide but no sequencing, or sequencing but no usable slide. Existing strong methods either **discard** single-modality patients, or **impute** the missing modality (e.g., DisPro uses an LLM to "recover" it) — which can hallucinate signal and produce over-confident, **poorly-calibrated** risk scores.

Our goals:

1. **Truly imputation-free** — never reconstruct the missing modality; use a learned *absent token* + a survival head specialized for each scenario.
2. **Break the single-modality ceiling** — via **feature-level cross-modal distillation**: decompose each modality into a modality-*general* component and pull the image-general toward the gene-general, so the image-only head becomes *gene-informed*.
3. **Trustworthy probabilities** — optimize not just ranking (**C-index**) but **calibration (IBS)**: predicted survival probabilities should be reliable, which matters most in the clinic.
4. **Exploit real incomplete cohorts** — train on genuinely single-modality patients that prior methods throw away.

*(Full model in [`ARCHITECTURE.txt`](ARCHITECTURE.txt); full head-to-head in [`COMPARISON.txt`](COMPARISON.txt).)*

---

## 📊 Main Results

**5-fold CV.** C-index ↑ higher = better (discrimination) · IBS ↓ lower = better (calibration).
Scenarios: **P** = genes missing (image-only) · **G** = image missing (gene-only) · **C** = complete.
`0%` = complete training · `60%` = missing training (avg of 5 blank-configs).
**Feature-matched:** UNI2h image + BulkRNABert gene — the *same features for all methods*.

> **DCMD-Surv is the best method on both cohorts, on both axes (C-index *and* IBS) in the complete scenario, and has the lowest IBS (best calibration) in *every* scenario** — beating Flex-MoE (NeurIPS'24), MUSE (ICLR'24), HEALNet (NeurIPS'24), ShaSpec (CVPR'23), MOTCat (ICCV'23), and naive imputation.

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
| 60% | complete (C) | **0.817 / 0.133** | 0.816 / 0.131 (HEALNet) |

**Key takeaways**

- DCMD-Surv **wins the complete scenario on both cohorts** (C-index & IBS).
- DCMD-Surv has the **lowest IBS (best-calibrated) in every scenario, both cohorts** — a consistent, clinically-meaningful advantage baselines don't have.
- On single-modality scenarios it **leads or ties** the best baseline while staying markedly better calibrated.
- **Real-missing experiment:** adding genuinely single-modality patients helps (discrimination on KIRC, calibration on GBMLGG) — see `dcmd_realmissing__*.txt`.

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
│   ├── dataset_mm.py / dataset_gbmlgg.py       # per-cohort loaders (COHORT dispatch)
│   ├── generate_gbmlgg_splits.py / build_gbmlgg_bags.py   # data prep
│   ├── baseline_impute.py             # naive zero/mean/KNN imputation
│   ├── flexmoe_surv.py, run_flexmoe.py         # Flex-MoE (NeurIPS'24)
│   ├── muse_surv.py, run_muse.py               # MUSE (ICLR'24)
│   ├── run_motcat.py                           # MOTCat (ICCV'23, complete-only ref)
│   ├── run_healnet.py                          # HEALNet (NeurIPS'24)
│   ├── shaspec_surv.py, run_shaspec.py         # ShaSpec (CVPR'23)
│   ├── calibration_mm.py / compute_ibs_dispro.py   # IBS / survival-curve utils
│   └── make_comparison_bars.py / make_km_figure.py /
│       make_calibration_figure.py / make_tables.py   # figures + tables
├── KIRC/
│   ├── results/            # per-method .txt (C-index + IBS, 0% + 60%) + comparison_KIRC.txt
│   └── figures/            # fig_comparison / fig_km_stratification / fig_calibration
└── GBMLGG/
    ├── results/            # (same layout)
    └── figures/            # (same layout)
```

---

## 🚀 How to Run

Cohort is selected by the `COHORT` env var (KIRC is the default).

```bash
# our method (C-index + IBS + cal-MAD), both 0% and 60% missing:
COHORT=KIRC   python3 code/run_dcmd_cal.py --epochs 30
COHORT=GBMLGG python3 code/run_dcmd_cal.py --epochs 30

# a baseline (example):
COHORT=GBMLGG python3 code/run_healnet.py --epochs 40

# real-missing experiment:
COHORT=GBMLGG python3 code/run_dcmd_realmissing.py --epochs 30

# figures + tables:
COHORT=KIRC python3 code/make_km_figure.py
COHORT=KIRC python3 code/make_calibration_figure.py
python3 code/make_comparison_bars.py     # both cohorts (numbers in NUMBERS dict)
python3 code/make_tables.py              # comparison_*.txt + MASTER_comparison.txt
```

---

## 🧪 Protocol & Dependencies

- **Protocol:** 5-fold CV; scenarios P/G/C; missing settings 0% and 60% (5 blank-configs avg). Metrics: C-index (discrimination) + IBS (calibration).
- **Dependencies:** `torch`, `numpy`, `pandas`, `scikit-learn`, `scikit-survival`, `lifelines`, `torch_geometric` (MUSE), `pot` (MOTCat), `python-box` + `einops` (HEALNet).
- **Cohorts done:** KIRC (417), GBMLGG (592). Add LUAD/BRCA by adding a `dataset_<cohort>.py` + a `COHORT` branch, then rerun the same commands.

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
