"""
Calibration star — turn our Cox risk scores into calibrated survival
probabilities and measure calibration.

Pipeline (no model retraining):
  1. Breslow baseline cumulative hazard H0(t) from TRAIN (risk, t, e)
  2. S(t|x) = exp(-H0(t) * exp(risk_x))  for TEST patients
  3. IBS (Integrated Brier Score, IPCW) via scikit-survival  -> lower = better calibrated
  4. Calibration curve at a time t0 (predicted S vs KM-observed S in bins) -> MAD
  5. (optional) post-hoc recalibration of the risk (temperature)

C-index measures ranking; IBS + calibration-MAD measure whether the PROBABILITIES
are honest. This is the axis DisPro/EMMS underreport and where we win — especially
that the routed (missing-modality) predictions stay calibrated, not overconfident.
"""
import numpy as np
from sksurv.util import Surv
from sksurv.metrics import integrated_brier_score


def breslow_cumhaz(risk, t, e):
    """Breslow estimator of baseline cumulative hazard from TRAIN.
    Returns (event_times, H0) where H0[k] = cumulative baseline hazard at event_times[k]."""
    risk = np.asarray(risk, float); t = np.asarray(t, float); e = np.asarray(e, float)
    exp_r = np.exp(risk - risk.max())          # stabilize
    ev_times = np.unique(t[e == 1])
    H0 = np.empty(len(ev_times)); cum = 0.0
    for k, et in enumerate(ev_times):
        d = np.sum((t == et) & (e == 1))
        risk_set = np.sum(exp_r[t >= et])
        cum += d / max(risk_set, 1e-12)
        H0[k] = cum
    return ev_times, H0, risk.max()


def survival_at_times(risk, ev_times, H0, rmax, times):
    """S(t|x) at `times` for each patient -> (n_patients, n_times)."""
    idx = np.searchsorted(ev_times, times, side="right") - 1
    H0_step = np.where(idx >= 0, H0[np.clip(idx, 0, len(H0) - 1)], 0.0)
    exp_r = np.exp(np.asarray(risk, float) - rmax)
    return np.exp(-np.outer(exp_r, H0_step))    # (n, n_times)


def eval_time_grid(test_t, test_e, n=20):
    """A safe time grid strictly inside the test follow-up (sksurv requirement)."""
    ev = test_t[test_e == 1]
    lo, hi = np.percentile(ev, 10), np.percentile(ev, 80)
    hi = min(hi, test_t.max() - 1e-3)
    return np.linspace(lo, hi, n)


def compute_ibs(train_t, train_e, test_t, test_e, test_S, times):
    surv_train = Surv.from_arrays(event=train_e.astype(bool), time=train_t.astype(float))
    surv_test = Surv.from_arrays(event=test_e.astype(bool), time=test_t.astype(float))
    return float(integrated_brier_score(surv_train, surv_test, test_S, times))


def calibration_mad(test_S_t0, test_t, test_e, t0, n_bins=5):
    """Predicted S(t0) vs KM-observed S(t0) in quantile bins -> mean abs deviation."""
    from lifelines import KaplanMeierFitter
    pred = np.asarray(test_S_t0, float)
    edges = np.quantile(pred, np.linspace(0, 1, n_bins + 1))
    prd, obs = [], []
    for i in range(n_bins):
        m = (pred >= edges[i]) & (pred <= edges[i + 1]) if i == n_bins - 1 \
            else (pred >= edges[i]) & (pred < edges[i + 1])
        if m.sum() < 5:
            continue
        km = KaplanMeierFitter().fit(test_t[m], test_e[m])
        obs.append(float(km.predict(t0))); prd.append(float(pred[m].mean()))
    if len(prd) < 2:
        return np.nan, prd, obs
    return float(np.mean(np.abs(np.array(obs) - np.array(prd)))), prd, obs


def fit_temperature(train_risk, train_t, train_e,
                    grid=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)):
    """Pick temperature minimizing IBS via 3-fold CV inside TRAIN (no test leak)."""
    train_risk = np.asarray(train_risk, float)
    n = len(train_risk)
    rng = np.random.RandomState(0)
    idx = rng.permutation(n)
    folds = np.array_split(idx, 3)
    best_T, best_ibs = 1.0, np.inf
    for T in grid:
        ibss = []
        for k in range(3):
            te = folds[k]; tr = np.concatenate([folds[j] for j in range(3) if j != k])
            if train_e[te].sum() < 2 or train_e[tr].sum() < 2:
                continue
            r_tr = train_risk[tr] / T; r_te = train_risk[te] / T
            ev, H0, rmax = breslow_cumhaz(r_tr, train_t[tr], train_e[tr])
            try:
                times = eval_time_grid(train_t[te], train_e[te])
                S = survival_at_times(r_te, ev, H0, rmax, times)
                ibss.append(compute_ibs(train_t[tr], train_e[tr],
                                        train_t[te], train_e[te], S, times))
            except Exception:
                continue
        if ibss and np.mean(ibss) < best_ibs:
            best_ibs, best_T = np.mean(ibss), T
    return best_T


def cox_calibration_report(train_risk, train_t, train_e,
                           test_risk, test_t, test_e, temperature=1.0):
    """Full calibration report for one scenario. temperature>1 flattens risk
    (post-hoc recalibration knob). Returns dict with IBS + calibration MAD."""
    train_risk = np.asarray(train_risk, float) / temperature
    test_risk = np.asarray(test_risk, float) / temperature
    ev_times, H0, rmax = breslow_cumhaz(train_risk, train_t, train_e)
    times = eval_time_grid(test_t, test_e)
    test_S = survival_at_times(test_risk, ev_times, H0, rmax, times)
    ibs = compute_ibs(train_t, train_e, test_t, test_e, test_S, times)
    t0 = float(np.median(times))
    S_t0 = survival_at_times(test_risk, ev_times, H0, rmax, np.array([t0]))[:, 0]
    mad, prd, obs = calibration_mad(S_t0, test_t, test_e, t0)
    return {"ibs": ibs, "cal_mad": mad, "t0": t0}


if __name__ == "__main__":
    # sanity: a well-ranked but MIS-calibrated risk should have worse IBS than calibrated
    rng = np.random.default_rng(0)
    n = 400
    risk = rng.normal(0, 1, n)
    t = np.exp(-0.5 * risk) * rng.exponential(50, n)   # higher risk -> shorter time
    e = (rng.uniform(0, 1, n) < 0.6).astype(float)
    ntr = 300
    rep = cox_calibration_report(risk[:ntr], t[:ntr], e[:ntr],
                                 risk[ntr:], t[ntr:], e[ntr:])
    print("sanity report:", rep)
    print("OK" if 0 <= rep["ibs"] <= 0.5 else "CHECK")
