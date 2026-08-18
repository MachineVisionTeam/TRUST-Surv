"""
Conformal reliability layer for missing-modality survival (the STAR).

Turns each Cox risk score into a LOWER PREDICTIVE BOUND (LPB) on survival time
with a DISTRIBUTION-FREE coverage guarantee:  P( T >= L(x) ) >= 1 - alpha.

Method: split-conformal LPB with inverse-probability-of-censoring weighting
(IPCW), following Candes, Lei & Ramdas, "Conformalized Survival Analysis" (2023).
Post-hoc, no model retraining, wraps ANY risk score.

Assumption: completely-independent censoring (C _|_ (X,T)); the censoring
distribution G(t)=P(C>t) is estimated marginally by Kaplan-Meier on the censoring
indicator. This is the standard baseline version and what we state in the paper.

Pipeline per (fold, scenario):
  base predictor  : q(x) = model's own (1-alpha) LPB read off the Breslow S(t|x)
                    = largest event time t with S(t|x) >= 1-alpha
  calibrate       : eta = IPCW-weighted (1-alpha)-quantile of s_i = q(x_i) - T_i
                    over UNCENSORED calibration points
  conformal LPB   : L(x) = q(x) - eta
  coverage (IPCW) : mean_i  1{ Ttilde_i >= L_i } / Ghat(L_i)     (unbiased for P(T>=L))

Key comparison for the paper:
  NAIVE coverage  (eta=0, the model's raw uncertainty)  vs
  CONFORMAL coverage (guaranteed)  -> naive breaks under missing modalities,
  conformal restores the target 1-alpha.  Width = median (q - L) = the correction.
"""
import numpy as np
from lifelines import KaplanMeierFitter
from calibration_mm import breslow_cumhaz, survival_at_times


def km_censoring_survival(t, e):
    """Ghat(tau) = P(C > tau), KM on the CENSORING indicator (event = 1 - e).
    Returns a callable evaluating Ghat just-before tau (left-continuous), floored
    at a small positive value so IPCW weights never blow up to inf."""
    t = np.asarray(t, float); e = np.asarray(e, float)
    kmf = KaplanMeierFitter().fit(t, 1.0 - e)          # censoring as the 'event'
    gmin = max(1e-3, float(kmf.survival_function_.values.min()))

    def Ghat(tau):
        tau = np.asarray(tau, float)
        g = kmf.predict(np.clip(tau - 1e-9, 0, None)).values if tau.ndim else \
            float(kmf.predict(max(tau - 1e-9, 0.0)))
        return np.clip(g, gmin, 1.0)
    return Ghat


def base_lpb(risk, ev_times, H0, rmax, alpha):
    """q(x) = largest event time with predicted S(t|x) >= 1-alpha (the model's own
    (1-alpha) lower bound on survival time). If even the first event time already
    has S < 1-alpha (very high risk), q(x)=0."""
    if len(ev_times) == 0:
        return np.zeros(len(np.atleast_1d(risk)))
    S = survival_at_times(risk, ev_times, H0, rmax, ev_times)   # (n, n_ev)
    ok = S >= (1.0 - alpha)                                     # (n, n_ev) boolean
    q = np.zeros(S.shape[0])
    for i in range(S.shape[0]):
        idx = np.where(ok[i])[0]
        q[i] = ev_times[idx[-1]] if idx.size else 0.0
    return q


def _weighted_quantile(values, weights, level):
    """Weighted empirical `level`-quantile (level in [0,1])."""
    values = np.asarray(values, float); weights = np.asarray(weights, float)
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w) / np.sum(w)
    k = np.searchsorted(cw, min(level, 1.0), side="left")
    return v[min(k, len(v) - 1)]


def conformal_correction(q_cal, t_cal, e_cal, alpha, Ghat):
    """eta = IPCW-weighted (1-alpha)-quantile of s_i = q_i - T_i on UNCENSORED
    calibration points (weights 1/Ghat(T_i) undo the censoring selection bias).
    Standard finite-sample conformal level inflation (1-alpha)(1+1/n)."""
    m = e_cal > 0.5
    if m.sum() < 5:
        return 0.0
    s = q_cal[m] - t_cal[m]
    w = 1.0 / Ghat(t_cal[m])
    n = m.sum()
    level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n))
    return float(_weighted_quantile(s, w, level))


def ipcw_coverage(L, t, e, Ghat):
    """Unbiased IPCW estimate of P(T >= L(X)):  mean_i 1{Ttilde_i >= L_i}/Ghat(L_i).
    (Under independent censoring, E[1{Ttilde>=L}/G(L)|X] = P(T>=L|X).)"""
    L = np.asarray(L, float); t = np.asarray(t, float)
    covered = (t >= L).astype(float)
    w = 1.0 / Ghat(L)
    return float(np.mean(covered * w))


def conformal_lpb_report(train_risk, train_t, train_e,
                         cal_risk, cal_t, cal_e,
                         test_risk, test_t, test_e, alpha=0.1):
    """Full conformal LPB for one scenario. Base predictor + Breslow from TRAIN;
    conformal correction from CAL; coverage measured on TEST. Ghat from TRAIN.
    Returns naive vs conformal coverage (target 1-alpha) and median width."""
    ev, H0, rmax = breslow_cumhaz(train_risk, train_t, train_e)
    Ghat = km_censoring_survival(train_t, train_e)

    q_cal = base_lpb(cal_risk, ev, H0, rmax, alpha)
    q_test = base_lpb(test_risk, ev, H0, rmax, alpha)

    eta = conformal_correction(q_cal, cal_t, cal_e, alpha, Ghat)
    L_conf = q_test - eta

    cov_naive = ipcw_coverage(q_test, test_t, test_e, Ghat)     # eta = 0
    cov_conf = ipcw_coverage(L_conf, test_t, test_e, Ghat)
    width = float(np.median(q_test - L_conf))                   # = eta
    return {"target": 1.0 - alpha,
            "cov_naive": cov_naive, "cov_conformal": cov_conf,
            "eta": eta, "median_width": width}
